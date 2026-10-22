"""Tile-crop segmentation dataset for the dysplasia U-Net experiment.

Crops `patch_size` x `patch_size` RGB patches at the tiling grid (level 2,
224/112) of tiling_186_merged and aligns the dysplasia mask by cropping the
level-5 annotation-mask TIFFs at 1/8 scale and up-sampling nearest. Uses only
existing artifacts -- no re-tiling, no new masks:

  * image  <- OpenSlide read_region at the tiling level (white outside slide)
  * target <- union of the per-class annotation masks (lowgrade|highgrade),
              the same EMPIA JSONs tiling consumed
  * valid  <- where the patch actually sits inside the slide (padding is
              excluded from loss and metrics)

Selection: tiles with annotation > 0 are positives (their annotated region is
inside the 224 tile, hence inside the 512 crop, by construction); everything
else is negative, sampled at `neg_pos_ratio`. Positives can be capped with
`max_pos_per_epoch` (kept proportional across slides so no annotated slide is
dropped). The split is a deterministic slide-level hash (same val slides every
run), so a patient's slides never straddle train/val.

OpenSlide/pyvips handles are NOT fork-safe: they are opened lazily inside
workers (first __getitem__), cached per worker in a small LRU. The sample list
is grouped per slide (slide blocks shuffled, not individual patches) so the
LRU hits almost every access.
"""

from __future__ import annotations

import glob
import hashlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
import torch
from openslide import OpenSlide
from torch.utils.data import Dataset


class _SlideCache:
    """Per-worker lazy (OpenSlide, pyvips mask pair) handles, small LRU."""

    def __init__(
        self,
        slides: pd.DataFrame,
        masks_dir: Path,
        classes: tuple[str, ...],
        level: int,
        mask_level: int,
        lru: int = 2,
    ) -> None:
        self._path_by_id = dict(zip(slides["id"], slides["path"]))
        self._masks_dir = masks_dir
        self._classes = classes
        self._level = level
        self._mask_level = mask_level
        self._lru: OrderedDict[str, tuple] = OrderedDict()
        self._lru_max = lru

    def open(self, slide_id: str) -> tuple:
        if slide_id in self._lru:
            self._lru.move_to_end(slide_id)
            return self._lru[slide_id]
        path = self._path_by_id[slide_id]
        stem = Path(path).stem
        slide = OpenSlide(path)
        masks = {}
        for name in self._classes:
            #: 0/255 uint8 canvas; do NOT cast to "char" -- signed int8 wraps
            #: 255 to -1 and silently loses every annotated pixel.
            img = pyvips.Image.new_from_file(
                str(self._masks_dir / name / f"{stem}.tiff")
            )
            #: The mask TIF's top level must be the slide's level-`mask_level`
            #: canvas (write_big_tiff pyramids from that canvas); a mismatch
            #: here means every target is silently mis-registered, so fail.
            top = (img.width, img.height)
            dims = slide.level_dimensions[self._mask_level]
            if top != (dims[0], dims[1]):
                raise RuntimeError(
                    f"{name}/{stem}.tiff top level {top} != slide level "
                    f"{self._mask_level} dims {dims} -- mask level assumption broken"
                )
            masks[name] = img
        self._lru[slide_id] = (slide, masks)
        if len(self._lru) > self._lru_max:
            _, (old_slide, _) = self._lru.popitem(last=False)
            old_slide.close()
        return self._lru[slide_id]


class DysplasiaPatchDataset(Dataset):
    """(image, target, valid) patches for tile-level dysplasia segmentation.

    image:  float32 (3, P, P), RGB, /255, white where outside the slide.
    target: float32 (1, P, P), 0/1 dysplasia from the annotation masks.
    valid:  float32 (1, P, P), 1 inside the slide extent.
    """

    def __init__(
        self,
        tiling_dir: str | Path,
        masks_dir: str | Path,
        patch_size: int = 512,
        level: int = 2,
        mask_level: int = 5,
        classes: tuple[str, ...] = ("lowgrade_dysplasia", "highgrade_dysplasia"),
        max_pos_per_epoch: int | None = None,
        neg_pos_ratio: float = 1.0,
        split: int = 0,
        val_fraction: float = 0.1,
        seed: int = 0,
        augment: bool = True,
        cache_slides: int = 2,
    ) -> None:
        tiling_dir = Path(tiling_dir)
        self.scale = 2 ** (mask_level - level)
        if patch_size % self.scale:
            raise ValueError(
                f"patch_size {patch_size} must be divisible by the level ratio "
                f"2**(mask_level-level) = {self.scale}"
            )
        self.patch_size = patch_size
        self.level = level
        self.classes = tuple(classes)
        self.augment = augment
        rng = np.random.default_rng(seed + split)

        slides = pd.read_parquet(tiling_dir / "slides" / "slides.parquet")

        #: Deterministic slide-level split (sha256 of the stem): the val set is
        #: identical every run and a slide's fate never depends on the trainer
        #: seed. One slide per patient in hunCRC, so this is patient-level.
        val_cut = int(round(val_fraction * 100))
        def side(stem: str) -> int:
            return 1 if int(hashlib.sha256(stem.encode()).hexdigest(), 16) % 100 < val_cut else 0

        slides["split"] = slides["path"].map(lambda p: side(Path(p).stem))
        my_slides = slides[slides["split"] == split]
        if my_slides.empty:
            raise SystemExit(f"no slides in split {split} under {tiling_dir}")

        tiles = pd.concat(
            [
                pd.read_parquet(f, columns=["slide_id", "x", "y", "annotation"])
                for f in sorted(glob.glob(str(tiling_dir / "tiles" / "*.parquet")))
            ]
        )
        #: my_slides keeps the default RangeIndex; the join key is the id column
        #: (SHA-256), which is what tiles.slide_id holds.
        tiles = tiles[tiles["slide_id"].isin(my_slides["id"])]
        pos = tiles[tiles["annotation"] > 0]
        neg = tiles[tiles["annotation"] <= 0]
        if pos.empty:
            raise SystemExit(f"split {split} has no annotated tiles -- nothing to learn")

        def pick(grp: pd.DataFrame, quota: int, rng: np.random.Generator) -> pd.DataFrame:
            return grp.sample(min(quota, len(grp)), random_state=rng)

        pos_sel: list[tuple] = []
        if max_pos_per_epoch is not None:
            quota = {
                sid: max(1, round(n * max_pos_per_epoch / len(pos)))
                for sid, n in pos.groupby("slide_id").size().items()
            }
            for sid, grp in pos.groupby("slide_id"):
                sel = pick(grp, quota[sid], rng)
                pos_sel += list(zip(sel["slide_id"], sel["x"], sel["y"]))
        else:
            pos_sel = list(zip(pos["slide_id"], pos["x"], pos["y"]))

        neg_quota = int(round(neg_pos_ratio * len(pos_sel)))
        neg_sel: list[tuple] = []
        if neg_quota and not neg.empty:
            for sid, grp in neg.groupby("slide_id"):
                take = pick(grp, round(len(grp) * neg_quota / len(neg)), rng)
                neg_sel += list(zip(take["slide_id"], take["x"], take["y"]))

        #: Group per slide (block shuffle) so the worker LRU stays hot.
        blocks: dict[str, list[tuple]] = {}
        for s in pos_sel + neg_sel:
            blocks.setdefault(s[0], []).append(s)
        order = list(blocks)
        rng.shuffle(order)
        self.samples: list[tuple] = [s for sid in order for s in blocks[sid]]
        self.n_pos, self.n_neg = len(pos_sel), len(neg_sel)
        self._slides = my_slides
        self._cache = _SlideCache(
            my_slides, Path(masks_dir), self.classes, level, mask_level, cache_slides
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        slide_id, x, y = self.samples[i]
        P, s = self.patch_size, self.scale
        slide, masks = self._cache.open(slide_id)
        w, h = slide.level_dimensions[self.level]

        #: image: read the in-bounds rect, white-pad the rest, track `valid`.
        #: read_region's location is in LEVEL-0 coordinates (OpenSlide API);
        #: the tiling grid is at `level`, so scale by that level's downsample.
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + P, w), min(y + P, h)
        img = np.full((P, P, 3), 255, np.uint8)
        valid = np.zeros((P, P), np.float32)
        if x1 > x0 and y1 > y0:
            d = int(round(slide.level_downsamples[self.level]))
            region = np.asarray(
                slide.read_region((x0 * d, y0 * d), self.level, (x1 - x0, y1 - y0))
            )
            img[y0 - y : y0 - y + (y1 - y0), x0 - x : x0 - x + (x1 - x0)] = region[..., :3]
            valid[y0 - y : y0 - y + (y1 - y0), x0 - x : x0 - x + (x1 - x0)] = 1.0

        #: target: crop each class mask at 1/scale (pyvips pads 0 outside the
        #: canvas), union, nearest-neighbor up-sample.
        mw = P // s
        m = np.zeros((mw, mw), bool)
        for name, mimg in masks.items():
            m |= (
                np.frombuffer(
                    mimg.crop(x // s, y // s, mw, mw).write_to_memory(),
                    dtype=np.uint8,
                ).reshape(mw, mw)
                > 0
            )
        target = np.repeat(np.repeat(m, s, axis=0), s, axis=1).astype(np.float32)[None]

        img_t = torch.from_numpy(
            np.ascontiguousarray(img, dtype=np.float32) / 255.0
        ).permute(2, 0, 1)
        tgt_t = torch.from_numpy(target)
        val_t = torch.from_numpy(valid)[None]

        if self.augment:
            #: Per-worker seed: torch.initial_seed() differs across workers and
            #: epochs, so flips are not identical in every worker copy.
            r = np.random.default_rng(torch.initial_seed() + i)
            if r.random() < 0.5:
                img_t, tgt_t, val_t = img_t.flip(-1), tgt_t.flip(-1), val_t.flip(-1)
            if r.random() < 0.5:
                img_t, tgt_t, val_t = img_t.flip(-2), tgt_t.flip(-2), val_t.flip(-2)
        return img_t, tgt_t, val_t
