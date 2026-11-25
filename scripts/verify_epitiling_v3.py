#!/usr/bin/env python
"""Verify the finished epi-200 masks and tiling_186_v3 against their manifests.

Read-only. Prints a short OK/PROBLEM summary per section.
"""
import glob
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tifffile import TiffFile

TILE_BASE = "/mnt/projects/huncrc_dysplasia/tiling_186_v3"
EPI_DIR = "/mnt/projects/huncrc_dysplasia/epithelium_masks/raw"
TISSUE_DIR = "/mnt/projects/huncrc_dysplasia/tissue_masks"


def tiling() -> None:
    manifests = {s: json.load(open(f"{TILE_BASE}/shard{s}/manifest.json")) for s in (1, 2, 3)}
    sd = pd.concat(
        [pq.read_table(f"{TILE_BASE}/shard{s}/slides/slides.parquet").to_pandas()
         for s in (1, 2, 3)], ignore_index=True)
    h2n = dict(zip(sd["id"], sd["path"].str.extract(r"(\d{3})\.mrxs$")[0]))

    g = None
    for s in (1, 2, 3):
        files = sorted(glob.glob(f"{TILE_BASE}/shard{s}/tiles/*.parquet"))
        n = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        print(f"shard{s}: {n} rows vs manifest {manifests[s]['tiles']} "
              f"-> {'OK' if n == manifests[s]['tiles'] else 'MISMATCH'}")
        t = pd.concat([pq.read_table(f, columns=["slide_id"]).to_pandas() for f in files],
                      ignore_index=True)
        g = t if g is None else pd.concat([g, t], ignore_index=True)
    g["num"] = g["slide_id"].map(h2n)
    print("unmapped tile slides:", int(g["num"].isna().sum()))
    cnt = g["num"].value_counts()
    all_per_slide = [p for m in manifests.values() for p in m["tiles_per_slide"].items()]
    mism = {sl: (int(cnt.get(sl, 0)), m) for sl, m in all_per_slide if int(cnt.get(sl, 0)) != m}
    print("per-slide count mismatches:", mism if mism else "NONE (all 186 match)")
    print(f"tiles/slide: min {int(cnt.min())} median {int(cnt.median())} "
          f"max {int(cnt.max())} total {int(cnt.sum())}")

    one = g[g["num"] == "114"]
    xs, ys = np.sort(one["x"].unique()), np.sort(one["y"].unique())
    print("slide 114 grid gaps x:", sorted(set(np.diff(xs))), " y:", sorted(set(np.diff(ys))))

    full = pd.concat(
        [pq.read_table(f).to_pandas()
         for f in sorted(glob.glob(f"{TILE_BASE}/shard1/tiles/*.parquet"))], ignore_index=True)
    full["num"] = full["slide_id"].map(h2n)
    print("\nshard1 column stats:")
    print(full[["tissue", "annotation", "epithelium", "blur", "artifacts"]].describe()
          .round(4).to_string())
    ann = full[full["annotation"] > 0]
    print("annotation>0:", len(ann), "tiles in", ann["num"].nunique(), "slides")
    print("tissue==epithelium on", int((full["tissue"] == full["epithelium"]).sum()),
          "of", len(full), "tiles (expected: run used tissue masks as epi stand-in)")
    print("clarity:", full["clarity"].value_counts(dropna=False).to_dict())
    print("fold:", full["fold"].value_counts(dropna=False).to_dict())


def epi() -> None:
    for slide in ("001", "030", "114", "169", "200"):
        with TiffFile(f"{EPI_DIR}/{slide}.tiff") as tf:
            assert tf.pages[0].shape == (147200, 136384)  # slide level 2
            m = tf.asarray(0)
            m4 = m[::4, ::4]  # 36800x34096 = level 4
        with TiffFile(f"{TISSUE_DIR}/{slide}.tiff") as tf:
            t = tf.asarray(0)  # 73600x68192 = level 3
        t_small = t[::2, ::2]  # 36800x34096 = level 4, same grid as m4
        epi_pos = m4 > 32
        tissue_pos = t_small > 32
        out = int((epi_pos & ~tissue_pos).sum())
        print(f"{slide}: epi nonzero {int(epi_pos.mean()*100):.1f}% of slide, "
              f"tissue {int(tissue_pos.mean()*100):.1f}%, "
              f"epi-outside-tissue {out/epi_pos.sum()*100:.2f}% of epi pixels, "
              f"max {int(m.max())} mean-over-epi {float(m4[epi_pos].mean()):.1f}")


if __name__ == "__main__":
    tiling()
    print("\n=== epi masks vs tissue ===")
    epi()
