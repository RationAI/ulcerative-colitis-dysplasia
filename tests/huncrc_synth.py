"""Synthetic HunCRC-like cohort + feature caches for script smoke tests.

15 train slides (3 per fold: 2 adenoma + 1 negative), 4 test_preliminary
slides, 1 test_final slide whose parquet is deliberately NOT written (reading
it would fail). 16-d embeddings with a class signal so heads can learn.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.huncrc_features import build_features
from ml.data.huncrc_labels import HG, LG, LabelConfig, load_slide_table


DIM = 16
REPO = Path(__file__).resolve().parents[1]


def cohort() -> list[tuple[str, str, float | None, str]]:
    rows: list[tuple[str, str, float | None, str]] = []
    n = 1
    for fold in range(5):
        for cat in ("adenoma", "adenoma", "negative"):
            rows.append((f"{n:03d}", "train", float(fold), cat))
            n += 1
    for cat in ("adenoma", "adenoma", "negative", "non_neoplastic_lesion", "CRC"):
        rows.append((f"{n:03d}", "test_preliminary", None, cat))
        n += 1
    rows.append((f"{n:03d}", "test_final", None, "negative"))
    return rows


def build(tmp: Path, n_tiles: int = 60, seed: int = 0) -> dict[str, Path]:
    rng = np.random.default_rng(seed)
    rows = cohort()
    slides = pd.DataFrame(
        {
            "id": [f"id{r[0]}" for r in rows],
            "path": [f"/wsi/{r[0]}.mrxs" for r in rows],
            "split": [r[1] for r in rows],
            "fold": [r[2] for r in rows],
        }
    )
    csv = pd.DataFrame(
        {
            "slideID": [r[0] for r in rows],
            "CATEGORY": [r[3] for r in rows],
            "Polyp type": [
                "hyperplastic" if r[3] == "non_neoplastic_lesion" else "Not_applicable"
                for r in rows
            ],
        },
        dtype=str,
    )
    (tmp / "tiling" / "slides").mkdir(parents=True)
    slides.to_parquet(tmp / "tiling" / "slides" / "slides.parquet")
    csv.to_csv(tmp / "slide_level_annotations.csv", index=False)
    table = load_slide_table(slides, csv)
    emb = tmp / "emb"
    emb.mkdir(parents=True)
    direction = rng.normal(size=DIM)
    for name, split, _, cat in rows:
        if split == "test_final":
            continue
        lg = np.zeros(n_tiles)
        if cat == "adenoma":
            lg[: n_tiles // 2] = rng.uniform(0.05, 1.0, n_tiles // 2)
        pos = (lg > 0).astype(float)
        e = rng.normal(size=(n_tiles, DIM)) + np.outer(pos * 1.5, direction)
        pd.DataFrame(
            {
                "slide_id": f"id{name}",
                "x": np.arange(n_tiles) * 112,
                "y": 0,
                LG: lg,
                HG: 0.0,
                "fold": 7.0,  # bogus tile-side fold, must be ignored
                "embedding": list(e),
            }
        ).to_parquet(emb / f"{name}.parquet")
    cache = tmp / "cache"
    out = {
        s: build_features(emb, table, (s,), LabelConfig(), cache, dim=DIM)
        for s in ("train", "test_preliminary")
    }
    out["test_preliminary_all"] = build_features(
        emb, table, ("test_preliminary",), LabelConfig(), cache, dim=DIM, mode="all"
    )
    out["tiling"] = tmp / "tiling"
    out["slide_csv"] = tmp / "slide_level_annotations.csv"
    return out


def run_script(name: str, *args: object) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "huncrc" / name), *map(str, args)],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc
