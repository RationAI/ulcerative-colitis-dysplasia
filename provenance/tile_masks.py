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
#: ── FILLED IN after the tiling job finishes (run id from its logs) ──
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
    if "TODO" in TILING_RUN_ID:
        raise SystemExit(
            "[prov tile-masks] the tiling run id is not pinned yet — "
            "fill TILING_RUN_ID from the tiling job logs first"
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
            with ray.init(runtime_env={"excludes": [".git", ".venv"]}):
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
                    "provenance": {
                        "create_dataset_run_id": CREATE_DATASET_RUN_ID,
                        "tiling_run_id": TILING_RUN_ID,
                    },
                },
                inputs=[f"runs:/{TILING_RUN_ID}"],
            )
        print(f"[prov tile-masks] done — run {run.info.run_id} in experiment {EXPERIMENT}")


if __name__ == "__main__":
    main()
