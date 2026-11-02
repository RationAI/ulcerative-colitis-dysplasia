#: Provenance cut-down of the tile-masks step.
#:
#: The original preprocessing/tile_masks.py is left UNTOUCHED. This copy
#: reuses its process_slide() ray task (one source of truth for the logic)
#: on the 2-slide test dataset and wraps it in a provenance-logged MLflow
#: run:
#:
#:   1. verify_manifest  — the registered cohort still matches the pinned run
#:   2. process_slide    — the original logic: per-tile percentage masks
#:                         (epithelium/blur/artifacts + the two dysplasia
#:                         labels) + tile outlines, at level 3
#:   3. log_provenance   — env + hardware, config params, slide stats +
#:                         the pinned input run (runs:/<tiling run id>)
#:
#: RUNS LAST in the chain — it consumes the tiling cut-down's parquet
#: output. TILING_RUN_ID below is filled in (by the maintainer of this
#: folder) after the tiling job finishes, and the script refuses to run
#: while it is still a placeholder. Empty splits (test_preliminary on the
#: 2-slide subset) are simply absent from the tiling artifact and are
#: skipped.
#:
#: SLOW: MaskBuilder accumulates full tile spans per column — the slowest
#: LOCAL step in the chain, though still minutes-to-an-hour for 2 slides.
#:
#: Run as a kube-job (submitter: provenance/submitters/tile_masks.py; needs
#: JOB_USER + MLflow credentials, both exported there):
#:
#:     uv run python -m provenance.tile_masks
import os
import tempfile
from pathlib import Path

import mlflow
import mlflow.artifacts
import pandas as pd
import ray
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient
from rationai.masks import process_items
from rationai.provenance import log_provenance
from rationai.provenance.dataset import verify_manifest

#: Reuse the original logic — the original module stays byte-identical.
from preprocessing.tile_masks import process_slide

#: Where the cut-down test runs live, isolated from the production runs.
EXPERIMENT = "provenance_huncrc_test"
#: Pinned registration run of the huncrc cohort (verify gate).
DATASET_RUN_ID = "4ef2fb98adb8432b9ee224fc5bdd5231"
#: Pinned create_dataset cut-down run — root of this test chain (transitive
#: input, pinned by the tiling run).
CREATE_DATASET_RUN_ID = "bc5160ba96064230b375cb15fdf78320"
#: ── FILLED IN as the chain jobs finish (run ids from their logs) ──
SPLIT_RUN_ID = "2385fd8162d746188c5c3092e2a1b605"
QC_RUN_ID = "d9af25eaf0044e6d911887c69cc8e18b"
#: Compute mode; finished 2026-11-02, verified (provenance logged, both tiffs).
TISSUE_RUN_ID = "5d812b44fbc94d3cbf6391a1683297e6"
TILING_RUN_ID = "TODO_FILL_AFTER_TILING_JOB"
# ─────────────────────────────────────────────────────────────────────────

#: Same values as configs/preprocessing/tile_masks.yaml — except the
#: percentage names, which there are the IBD cohort's ("LG Dysplasia" …).
#: The hunCRC tiling output (target_groups) names the labels
#: lowgrade_dysplasia / highgrade_dysplasia.
LEVEL = 3
MAX_CONCURRENT = 2
PERCENTAGE_COLS = [
    "epithelium",
    "blur",
    "artifacts",
    "lowgrade_dysplasia",
    "highgrade_dysplasia",
]

PROJECT_PATH = "/mnt/projects/huncrc_dysplasia"
TILING_BASE_URI = f"mlflow-artifacts:/212/{TILING_RUN_ID}/artifacts/tiling"


def main() -> None:
    if "TODO" in (TILING_RUN_ID, TISSUE_RUN_ID).__str__():
        raise SystemExit(
            "[prov tile-masks] upstream run ids are not pinned yet — "
            "fill TILING_RUN_ID / TISSUE_RUN_ID from the job logs first"
        )

    # 1. verify the registered cohort before touching anything
    manifest_path = os.path.join(PROJECT_PATH, "preprocessing", "manifest.csv")
    check = verify_manifest(manifest_path, os.path.dirname(manifest_path), DATASET_RUN_ID)
    for detail in check["details"]:
        print(f"[prov tile-masks] verify: {detail}")
    if not check["verified"]:
        raise SystemExit(
            f"[prov tile-masks] dataset verification FAILED (run {DATASET_RUN_ID}) — refusing to process"
        )

    #: The tiling cut-down only wrote the NON-empty splits (the 2-slide subset
    #: has an empty test_preliminary) — process exactly what is there.
    client = MlflowClient()
    splits = [a.path for a in client.list_artifacts(TILING_RUN_ID, "tiling")]
    if not splits:
        raise SystemExit(f"[prov tile-masks] run {TILING_RUN_ID} has no `tiling` artifact")

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="🧱 Tile masks (provenance, subset)") as run:
        slides_frames = []
        with tempfile.TemporaryDirectory() as tmpdir:
            #: num_cpus caps the node at the pod's CPU limit — without it the raylet
            #: sees the NODE's 192 CPUs and prestarts 192 idle workers (observed:
            #: 13.7 GiB + ~0.6 core in the 2-slide tissue job). Scheduling only;
            #: the per-task logic is the original, untouched.
            with ray.init(num_cpus=4, runtime_env={"excludes": [".git", ".venv"]}):
                for split in splits:
                    local_path = Path(download_artifacts(f"{TILING_BASE_URI}/{split}"))
                    slides = pd.read_parquet(local_path / "slides")
                    tiles = pd.read_parquet(local_path / "tiles")
                    slides_frames.append(slides)
                    print(f"[prov tile-masks] {split}: {len(slides)} slides, {len(tiles)} tiles")

                    process_items(
                        (slide for _, slide in slides.iterrows()),
                        process_slide,
                        fn_kwargs={
                            "percentage_cols": PERCENTAGE_COLS,
                            "output_path": Path(tmpdir),
                            "tiles": tiles,
                            "level": LEVEL,
                        },
                        max_concurrent=MAX_CONCURRENT,
                    )

            # 2. upload (same layout as production) + log provenance
            mlflow.log_artifacts(tmpdir, "tile_masks")

            slides = pd.concat(slides_frames, ignore_index=True)
            log_provenance(
                output=slides,
                config={
                    "dataset": {"name": "huncrc-cohort", "version": "1.0.0"},
                    "level": LEVEL,
                    "max_concurrent": MAX_CONCURRENT,
                    "percentage_cols": PERCENTAGE_COLS,
                    #: the FULL upstream chain, named — so this run is
                    #: self-describing without following any link.
                    "provenance": {
                        "dataset_registration_run_id": DATASET_RUN_ID,
                        "create_dataset_run_id": CREATE_DATASET_RUN_ID,
                        "split_run_id": SPLIT_RUN_ID,
                        "qc_run_id": QC_RUN_ID,
                        "tissue_run_id": TISSUE_RUN_ID,
                        "tiling_run_id": TILING_RUN_ID,
                    },
                },
                #: every ancestor run, so the chain is interconnected from
                #: any node (the user run is auto-stamped as user_run_id).
                inputs=[
                    f"runs:/{DATASET_RUN_ID}",
                    f"runs:/{CREATE_DATASET_RUN_ID}",
                    f"runs:/{SPLIT_RUN_ID}",
                    f"runs:/{QC_RUN_ID}",
                    f"runs:/{TISSUE_RUN_ID}",
                    f"runs:/{TILING_RUN_ID}",
                ],
            )
        print(f"[prov tile-masks] done — run {run.info.run_id} in experiment {EXPERIMENT}")


if __name__ == "__main__":
    main()
