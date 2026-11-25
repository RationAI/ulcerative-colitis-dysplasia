"""Add a `split` column to dataset.csv and to tiling_186_merged.

The split_dataset stage (preprocessing/split_dataset.py) uploads three CSVs to
MLflow, and the IBD-era data configs then read per-split tiling artifacts. For
hunCRC the cohort was tiled in one pass, so instead we compute the identical
case-level split once and write it into the artifacts directly:

  1. Read dataset.csv (local path or mlflow-artifacts: URI; default is the
     hunCRC create_dataset run).
  2. Run the exact split logic of preprocessing/split_dataset.py (train 0.7 /
     test_preliminary 0.15 / test_final 0.15, grouped by case_id, random_state
     and folds from configs/preprocessing/split_dataset.yaml) -- the same
     partition the split_dataset job would produce.
  3. Write <out-csv>: dataset.csv + `split` (+ `fold` for CV on train rows).
  4. Add the `split` column to the tiling, in place: slides.parquet (joined on
     slide stem) and tiles/*.parquet (joined on slide_id).

Needs the public MLflow env when --dataset-csv is an artifact URI (pod env
injects the in-cluster one):
  export MLFLOW_TRACKING_URI=https://mlflow.rationai.cloud.e-infra.cz
  export MLFLOW_TRACKING_USERNAME=... MLFLOW_TRACKING_PASSWORD=...

    python scripts/add_split_column.py --dry-run
    python scripts/add_split_column.py
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent

#: dataset.csv from the hunCRC create_dataset run (experiment 168) -- same
#: default as scripts/preprocessing/split_dataset.py.
DEFAULT_DATASET_CSV = "mlflow-artifacts:/168/2ca0c1928c2c48acb56a5110efefb957/artifacts/dataset.csv"
DEFAULT_TILING_DIR = "/mnt/projects/huncrc_dysplasia/tiling_186_merged"


def load_dataset(src: str) -> pd.DataFrame:
    if src.startswith("mlflow-artifacts:"):
        from mlflow.artifacts import download_artifacts

        src = download_artifacts(src)
    #: The stems are zero-padded ("059.mrxs"); without dtype=str read_csv
    #: infers int64 for the index and the slide-stem join fails for every row.
    df = pd.read_csv(
        src, index_col="slide_id", dtype={"slide_id": str, "case_id": str}
    )
    if "case_id" not in df.columns:
        sys.exit(f"dataset.csv has no case_id column: {list(df.columns)}")
    return df


def compute_split(df: pd.DataFrame) -> pd.DataFrame:
    import sys

    from omegaconf import OmegaConf

    #: Run as `python scripts/add_split_column.py`, sys.path[0] is scripts/, where
    #: `preprocessing/` holds the JOB SUBMITTERS -- importing one executes
    #: submit_job. The repo root (the real `preprocessing` package) must win.
    sys.path.insert(0, str(REPO_ROOT))

    from preprocessing.split_dataset import add_folds, split_dataset

    cfg = OmegaConf.load(REPO_ROOT / "configs/preprocessing/split_dataset.yaml")
    train, test_preliminary, test_final = split_dataset(
        df, cfg["splits"], cfg["random_state"]
    )
    train = add_folds(train, cfg["n_folds"])

    out = df.copy()
    out["split"] = None
    out["fold"] = None
    for name, part in (
        ("train", train),
        ("test_preliminary", test_preliminary),
        ("test_final", test_final),
    ):
        out.loc[part.index, "split"] = name
    out.loc[train.index, "fold"] = train["fold"].to_numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-csv", default=DEFAULT_DATASET_CSV)
    ap.add_argument("--tiling-dir", default=DEFAULT_TILING_DIR)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tiling_dir = Path(args.tiling_dir)
    out_csv = Path(args.out_csv) if args.out_csv else tiling_dir.parent / "dataset_split.csv"

    split_df = compute_split(load_dataset(args.dataset_csv))

    print(f"dataset.csv: {len(split_df)} cases")
    print(split_df["split"].value_counts().to_string())
    train_folds = sorted(split_df.loc[split_df["split"] == "train", "fold"].unique())
    print(f"train folds: {train_folds}")

    slides = pd.read_parquet(tiling_dir / "slides" / "slides.parquet")
    stems = slides["path"].map(lambda p: Path(p).stem)
    unknown = sorted(set(stems) - set(split_df.index))
    if unknown:
        sys.exit(
            f"{len(unknown)} tiled slides missing from dataset.csv: {unknown[:10]}"
        )

    n_tile_files = len(glob.glob(str(tiling_dir / "tiles" / "*.parquet")))
    if args.dry_run:
        print(
            f"\ndry run: all {len(slides)} tiled slides match dataset.csv; "
            f"would write {out_csv} and add 'split' to slides.parquet "
            f"+ {n_tile_files} tile files."
        )
        return

    split_df.to_csv(out_csv)
    print(f"wrote {out_csv}")

    slides["split"] = stems.map(lambda s: split_df.loc[s, "split"])
    slides.to_parquet(tiling_dir / "slides" / "slides.parquet", index=False)
    print(f"added split to slides.parquet: {slides['split'].value_counts().to_dict()}")

    slide_split = slides.set_index("id")["split"]
    for f in sorted(glob.glob(str(tiling_dir / "tiles" / "*.parquet"))):
        t = pd.read_parquet(f)
        t["split"] = t["slide_id"].map(slide_split)
        if t["split"].isna().any():
            sys.exit(f"{f}: unmapped slide_ids")
        tmp = f + ".tmp"
        t.to_parquet(tmp, index=False)
        Path(tmp).replace(f)
        print(f"  {Path(f).name}: {len(t)} tiles, {t['split'].value_counts().to_dict()}")

    print("done")


if __name__ == "__main__":
    main()
