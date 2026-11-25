"""Per-class annotation mask pyramids, from the EMPIA JSONs of per-tile labels.

    uv run python -m preprocessing.annotation_masks +dataset=raw

Writes `<output_dir>/<class>/<slide>.tiff` -- one 0/255 tiled pyramid per (slide, class)
-- and logs the tree to MLflow under `config.mlflow_artifact_path`.

Why one file per class instead of the single class-id map the IBD generator writes:
this viewer draws a heatmap layer per file, and the library's class-map path
(`colormap_layer`) bins pixel values over 0..1, so class ids 1..9 all fall in the
first bin and every class renders in one colour. Proved, not assumed. One file per
class is what scripts/make_dysplasia_masks.py publishes and what the viewer is known
to overlay. See scripts/make_huncrc_annotation_report.py.

Source of truth is the annotation JSON, NOT dataset.csv. create_dataset's hunCRC
version selects [case_id, slide_path, tiles_path] and drops annot_path
(preprocessing/create_dataset.py:106, vs master:82-85 which filled it), so there is no
annot column to read. The JSONs are also what tiling consumes, so both stages derive
labels from the same files.

The slide is opened only for `slide_resolution` -- canvas and mpp come from the slide,
never a constant -- so a mask registers on its own slide rather than on whatever size
was typed in.
"""

import re
from pathlib import Path

import hydra
import numpy as np
import pyvips
import ray
from omegaconf import DictConfig
from openslide import OpenSlide
from rasterio.features import rasterize
from rationai.masks import process_items, slide_resolution, write_big_tiff
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.parsers import EMPAIAParser
from shapely.affinity import scale as shapely_scale


#: Class name -> mask value. Mirrors TARGET_GROUPS in scripts/make_annotation_mask.py,
#: the local one-slide tool. Duplicated deliberately: the job clones this repo from
#: GitHub at BRANCH, and scripts/ is untracked, so an import from there would
#: ModuleNotFoundError in the pod. Every other payload here (tissue_masks, qc,
#: tile_masks) is self-contained for the same reason. Keep the two in step -- the
#: values, not just the names, because a mask written with one numbering and read with
#: the other is a plausible image of the wrong thing.
TARGET_GROUPS: dict[str, int] = {
    "normal": 1,
    "inflammation": 2,
    "lowgrade_dysplasia": 3,
    "highgrade_dysplasia": 4,
    "suspicious_for_invasion": 5,
    "adenocarcinoma": 6,
    "tumor_necrosis": 7,
    "resection_edge": 8,
    "artifact": 9,
}


def create_annotation_mask(
    annot_path: Path,
    mask_dim: tuple[int, int],
    scale: float,
    target_groups: dict[str, int],
) -> np.ndarray:
    """Rasterize the named classes from one EMPIA JSON onto a `mask_dim` canvas.

    Coordinates are level-0 px in the JSON and the canvas is level `level`, hence
    `scale`. Same call sequence as the IBD generator: EMPAIAParser matches names with
    an anchored regex, so a label absent from `target_groups` draws nothing rather
    than something.

    Returns uint8 (height, width). A missing or unreadable JSON yields an all-zero
    mask, NOT an error -- which is why the caller checks the file exists first.
    """
    parser = EMPAIAParser(annot_path)
    width, height = mask_dim

    shapes: list[tuple] = []
    for label, value in target_groups.items():
        for poly in parser.get_polygons(name=rf"^{re.escape(label)}$"):
            if poly.is_empty or poly.area == 0:
                continue
            scaled = shapely_scale(poly, xfact=scale, yfact=scale, origin=(0, 0))
            if not scaled.is_valid:
                scaled = scaled.buffer(0)
            if not scaled.is_empty:
                shapes.append((scaled, value))

    if not shapes:
        return np.zeros((height, width), dtype=np.uint8)

    return rasterize(
        shapes, out_shape=(height, width), fill=0, all_touched=True, dtype=np.uint8
    )


@ray.remote(memory=2 * 1024**3)
def process_slide(
    item: tuple[str, str],
    classes: list[str],
    level: int,
    output_path: str,
) -> None:
    """Rasterize each requested class for one slide.

    One rasterize pass per class with a single-entry `target_groups`, so the mask for
    class X cannot contain class Y's polygons by construction rather than depending on
    a split of a combined array being done right. Costs re-reading the JSON per class,
    which is a few hundred KB against a 300 MB slide read.

    An empty class prints a warning instead of raising: a slide legitimately lacking
    some class is normal, so only the aggregate is worth failing on.
    """
    slide_path, annot_path = item
    slide_name = Path(slide_path).stem

    with OpenSlide(slide_path) as slide:
        full_dim = slide.level_dimensions[0]
        mask_dim = slide.level_dimensions[level]
        mpp_x, mpp_y = slide_resolution(slide, level)
    scale = mask_dim[0] / full_dim[0]

    for name in classes:
        coded = create_annotation_mask(
            Path(annot_path), mask_dim, scale, {name: TARGET_GROUPS[name]}
        )
        labelled = int((coded > 0).sum())
        binary = coded.astype("uint8") * 255
        dest = Path(output_path) / name / f"{slide_name}.tiff"
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_big_tiff(
            pyvips.Image.new_from_array(binary), dest, mpp_x=mpp_x, mpp_y=mpp_y
        )
        if not labelled:
            print(f"WARNING {slide_name}: {name} mask is empty (0 labelled px)")


@with_cli_args(["+preprocessing=annotation_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    annot_dir = Path(config.annot_dir)
    #: Under `dataset`, not global -- configs/dataset/raw.yaml is a `# @package`-less
    #: group installed as `dataset`, the same way create_dataset reads it
    #: (preprocessing/create_dataset.py:117). Verified by composing the config: a
    #: global `config.slides_path` is MISSING and would raise in the pod, not here.
    slides_dir = Path(config.dataset.slides_path)

    for path, what in ((annot_dir, "annot_dir"), (slides_dir, "dataset.slides_path")):
        if not path.is_dir():
            raise SystemExit(
                f"{what} is not a directory: {path} -- a wrong path here means zero "
                f"pairs, which the check below would otherwise only print"
            )

    #: Every requested class needs a value in TARGET_GROUPS, or rasterize draws nothing
    #: and the job goes green with a run full of empty layers.
    missing = [c for c in config.classes if c not in TARGET_GROUPS]
    if missing:
        raise SystemExit(
            f"classes {missing} have no value in TARGET_GROUPS "
            f"(known: {sorted(TARGET_GROUPS)})"
        )

    #: Match JSONs to slides rather than iterating either folder: a JSON with no slide
    #: or a slide with no JSON is worth printing, since create_annotation_mask returns
    #: an all-zero mask for a file it cannot find and would write it happily.
    pairs: list[tuple[str, str]] = []
    for annot in sorted(annot_dir.glob("*.json")):
        slide = slides_dir / f"{annot.stem}.mrxs"
        if slide.exists():
            pairs.append((str(slide), str(annot)))
        else:
            print(f"skipping {annot.name}: no slide at {slide}")
    if not pairs:
        raise SystemExit(f"no slide/annotation pairs under {annot_dir} + {slides_dir}")

    #: Written straight into the project dir, so a crash mid-run keeps the slides
    #: already done; MLflow gets the same tree afterwards. A tempdir would lose every
    #: completed mask after an hour of work.
    staging = Path(config.output_dir)
    staging.mkdir(parents=True, exist_ok=True)

    print(
        f"{len(pairs)} slides x {len(config.classes)} classes "
        f"= {len(pairs) * len(config.classes)} masks -> {staging}"
    )

    process_items(
        pairs,
        process_item=process_slide,
        fn_kwargs={
            "classes": list(config.classes),
            "level": config.level,
            "output_path": str(staging),
        },
        max_concurrent=config.max_concurrent,
    )

    #: process_items returns None, so the file count is the completion check: a task
    #: that died with the warning suppressed still shows up here as a missing file.
    expected = len(pairs) * len(config.classes)
    written = len(list(staging.glob("*/*.tiff")))
    print(f"wrote {written} of {expected} expected masks")
    if written != expected:
        for name in config.classes:
            print(f"  {name}: {len(list((staging / name).glob('*.tiff')))} files")
        raise SystemExit(
            f"{expected - written} mask(s) missing; not logging a partial tree to MLflow"
        )

    logger.log_artifacts(str(staging), config.mlflow_artifact_path)


if __name__ == "__main__":
    #: Same as tissue_masks/tile_masks/downscale_epithelium: process_items calls
    #: .remote() per item, which needs a running ray, and `excludes` stops git and the
    #: 6 GB .venv being shipped to every worker.
    with ray.init(runtime_env={"excludes": [".git", ".venv"]}):  # type: ignore[call-arg]
        main()
