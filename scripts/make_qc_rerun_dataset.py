"""Build dataset_qc_14.csv -- only the slides that failed QC (in dataset.csv
but absent from tiling_186_merged) -- and upload it to MLflow.

The QC stage (preprocessing/quality_control.py) checks EVERY row of the CSV
it is given and has no slide-subset flag, so a targeted re-check means
pointing the submitter at a filtered CSV:

    export DATASET_ARTIFACT_URI=<URI printed at the end>
    export MLFLOW_TRACKING_PASSWORD=<access key>
    uv run python scripts/preprocessing/quality_control.py

Needs the public MLflow env (the pod's injected in-cluster URI only works
from job pods):
    export MLFLOW_TRACKING_URI=https://mlflow.rationai.cloud.e-infra.cz
    export MLFLOW_TRACKING_USERNAME=519192@mail.muni.cz
    export MLFLOW_TRACKING_PASSWORD=<access key>

    python scripts/make_qc_rerun_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


#: dataset_split.csv is the original dataset.csv (all 200 hunCRC slides) with
#: split/fold appended -- same slide_path values, so it is a fine source.
SPLIT_CSV = "/mnt/projects/huncrc_dysplasia/dataset_split.csv"
TILING_DIR = Path("/mnt/projects/huncrc_dysplasia/tiling_186_merged")
OUT_CSV = "/mnt/projects/huncrc_dysplasia/dataset_qc_14.csv"
#: The hunCRC experiment -- same ID the other hunCRC submitters use.
EXPERIMENT_ID = "168"


def main() -> None:
    #: dtype str: the stems are zero-padded, read_csv infers int64 otherwise.
    df = pd.read_csv(SPLIT_CSV, index_col="slide_id", dtype={"slide_id": str})
    slides = pd.read_parquet(TILING_DIR / "slides" / "slides.parquet")
    tiled = {Path(p).stem for p in slides["path"]}

    missing = sorted(set(df.index) - tiled)
    print(f"{len(missing)} slides in dataset.csv but absent from tiling (failed QC):")
    for s in missing:
        print(f"  {s}  {df.loc[s, 'slide_path']}")

    out = df.loc[missing, ["case_id", "slide_path", "tiles_path"]]
    out.to_csv(OUT_CSV)
    print(f"wrote {OUT_CSV} ({len(out)} rows)")

    import mlflow
    from mlflow.tracking import MlflowClient

    #: set_experiment() treats its argument as a NAME, so "168" would create a
    #: new experiment; resolve the real name from the ID first.
    exp = MlflowClient().get_experiment(EXPERIMENT_ID)
    mlflow.set_experiment(exp.name)
    with mlflow.start_run(run_name="qc-rerun-14-slides") as run:
        mlflow.log_artifact(OUT_CSV)
        print(
            f"DATASET_ARTIFACT_URI=mlflow-artifacts:/{exp.experiment_id}/"
            f"{run.info.run_id}/artifacts/dataset_qc_14.csv"
        )


if __name__ == "__main__":
    main()
