#: Provenance cut-down of the tiling step.
#:
#: The original preprocessing/tiling.py is left UNTOUCHED. This copy reuses
#: its tiling() ray pipeline (one source of truth for the logic) on the
#: 2-slide test chain and wraps it in a provenance-logged MLflow run.
#:
#:   1. verify_manifest — the registered cohort still matches the pinned run
#:   2. tiling          — the original logic on the 2 slides from the chain
#:   3. log_provenance  — env + hardware, config params, tile stats +
#:                        pins for the WHOLE chain (QC, tissue, split,
#:                        create_dataset runs)
#:
#: RUNS LAST in the fast wave — it consumes the artifacts of the split,
#: QC and tissue_masks cut-down runs. The three *_RUN_ID constants below are
#: filled in (by the maintainer of this folder) after those jobs finish, and
#: the script refuses to run while any of them is still a placeholder.
#:
#: Geometry = the proven pilot values from the production tiling submitter
#: (224 px @ 0.52 mpp, stride 112, tissue_threshold 0.5).
#:
#: Run as a kube-job (submitter: provenance/submitters/tiling.py; needs
#: JOB_USER + MLflow credentials, both exported there):
#:
#:     uv run python -m provenance.tiling
import os
import tempfile
from pathlib import Path

import mlflow
import mlflow.artifacts
import pandas as pd
import ray
from mlflow.artifacts import download_artifacts
from rationai.provenance import log_provenance
from rationai.provenance.dataset import verify_manifest

#: Reuse the original logic — the original module stays byte-identical.
from preprocessing.tiling import tiling

#: Where the cut-down test runs live, isolated from the production runs.
EXPERIMENT = "provenance_huncrc_test"
#: Pinned registration run of the huncrc cohort (verify gate).
DATASET_RUN_ID = "4ef2fb98adb8432b9ee224fc5bdd5231"
#: Pinned create_dataset cut-down run — root of this test chain.
CREATE_DATASET_RUN_ID = "bc5160ba96064230b375cb15fdf78320"
#: ── FILLED IN after the fast-wave jobs finish (run ids from their logs) ──
SPLIT_RUN_ID = "TODO_FILL_AFTER_SPLIT_JOB"
QC_RUN_ID = "TODO_FILL_AFTER_QC_JOB"
TISSUE_RUN_ID = "TODO_FILL_AFTER_TISSUE_JOB"
# ─────────────────────────────────────────────────────────────────────────

#: Same values as the production tiling submitter (proven pilot geometry).
MPP = 0.52
TILE_EXTENT = 224
STRIDE = 112
TISSUE_THRESHOLD = 0.5
#: Must match configs/preprocessing/tiling.yaml target_groups EXACTLY.
TARGET_GROUPS = ["lowgrade_dysplasia", "highgrade_dysplasia"]
ROW_PER_FILE = 50_000

PROJECT_PATH = "/mnt/projects/huncrc_dysplasia"
#: Per-slide EMPIA JSONs on the project share (config.dataset.annot_path).
ANNOT_FOLDER = f"{PROJECT_PATH}/annotations"

QC_URI = f"mlflow-artifacts:/212/{QC_RUN_ID}/artifacts/qc_output"
TISSUE_URI = f"mlflow-artifacts:/212/{TISSUE_RUN_ID}/artifacts/tissue_masks"
SPLIT_URIS = {
    name: f"mlflow-artifacts:/212/{SPLIT_RUN_ID}/artifacts/{name}.csv"
    for name in ("train", "test_preliminary", "test_final")
}


def main() -> None:
    if "TODO" in (SPLIT_RUN_ID, QC_RUN_ID, TISSUE_RUN_ID).__str__():
        raise SystemExit(
            "[prov tiling] the split/QC/tissue run ids are not pinned yet — "
            "fill SPLIT_RUN_ID / QC_RUN_ID / TISSUE_RUN_ID from the wave-1 job logs first"
        )

    # 1. verify the registered cohort before touching anything
    manifest_path = os.path.join(PROJECT_PATH, "preprocessing", "manifest.csv")
    check = verify_manifest(manifest_path, os.path.dirname(manifest_path), DATASET_RUN_ID)
    for detail in check["details"]:
        print(f"[prov tiling] verify: {detail}")
    if not check["verified"]:
        raise SystemExit(
            f"[prov tiling] dataset verification FAILED (run {DATASET_RUN_ID}) — refusing to process"
        )

    # 2. the chain's artifacts
    qc_folder = Path(download_artifacts(QC_URI))
    tissue_folder = Path(download_artifacts(TISSUE_URI))
    annot_folder = Path(ANNOT_FOLDER)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="🧩 Tiling (provenance, subset)") as run:
        tiles_frames = []
        with ray.init(runtime_env={"excludes": [".git", ".venv"]}):
            with tempfile.TemporaryDirectory() as tmpdir:
                for name, split_uri in SPLIT_URIS.items():
                    split = pd.read_csv(
                        mlflow.artifacts.download_artifacts(split_uri),
                        index_col="slide_id",
                    )
                    if split.empty:
                        print(f"[prov tiling] split '{name}' is empty (2-slide subset) — skipped")
                        continue

                    ds_slides, ds_tiles = tiling(
                        split,
                        qc_folder=qc_folder,
                        tissue_folder=tissue_folder,
                        epithelium_folder=tissue_folder,  #: no episeg yet — tissue stand-in, same as production
                        annot_folder=annot_folder,
                        tile_extent=TILE_EXTENT,
                        stride=STRIDE,
                        mpp=MPP,
                        tissue_threshold=TISSUE_THRESHOLD,
                        target_groups=TARGET_GROUPS,
                    )

                    save_dir = Path(tmpdir) / name
                    save_dir.mkdir(parents=True, exist_ok=True)
                    ds_slides.write_parquet(str(save_dir / "slides"), min_rows_per_file=ROW_PER_FILE)
                    ds_tiles.write_parquet(str(save_dir / "tiles"), min_rows_per_file=ROW_PER_FILE)

                    tiles_frames.append(pd.read_parquet(save_dir / "tiles"))

                # 3. upload (same layout as production) + log provenance
                mlflow.log_artifacts(tmpdir, "tiling")

                tiles = pd.concat(tiles_frames, ignore_index=True)
                log_provenance(
                    output=tiles,
                    config={
                        "dataset": {"name": "huncrc-cohort", "version": "1.0.0"},
                        "mpp": MPP,
                        "tile_extent": TILE_EXTENT,
                        "stride": STRIDE,
                        "tissue_threshold": TISSUE_THRESHOLD,
                        "target_groups": TARGET_GROUPS,
                        "provenance": {
                            "create_dataset_run_id": CREATE_DATASET_RUN_ID,
                            "split_run_id": SPLIT_RUN_ID,
                            "qc_run_id": QC_RUN_ID,
                            "tissue_run_id": TISSUE_RUN_ID,
                        },
                    },
                    inputs=[
                        f"runs:/{CREATE_DATASET_RUN_ID}",
                        f"runs:/{SPLIT_RUN_ID}",
                        f"runs:/{QC_RUN_ID}",
                        f"runs:/{TISSUE_RUN_ID}",
                    ],
                )
        print(f"[prov tiling] done — run {run.info.run_id} in experiment {EXPERIMENT}")


if __name__ == "__main__":
    main()
