#: Provenance cut-down of the QC step — REUSE MODE.
#:
#: WHY NOT RECOMPUTE: slides 155/059 already went through the production QC
#: service (the 200-slide run `0e3d8a66…`, exp 168, "QC output (200 slides,
#: huncrc)" — it processed the whole cohort into the staging dir on the
#: project share, then FAILED before uploading anything to MLflow, so the
#: share is the only copy). Recomputing the 2 slides would burn ~7.2 h of
#: QC service time to produce the same masks for a test of the PROVENANCE
#: LIBRARY, not of QC. So this script:
#:
#:   1. verify_manifest  — the registered cohort still matches the pinned run
#:   2. selects the 2 slides from the chain's dataset.csv (bc5160ba run)
#:   3. COPIES the production masks for exactly those 2 slides into a
#:      prov-dedicated staging dir (the production tree is never touched)
#:   4. runs the same FAIL-LOUD gate as compute mode would (metrics rows +
#:      all 4 mask families present for both slides)
#:   5. log_provenance   — with the reuse recorded EXPLICITLY so the run is
#:      not a lie about where its bytes came from:
#:        * run name says "masks reused from production"
#:        * cfg_masks_source / cfg_reused_from_production_run params
#:        * inputs: the create_dataset run AND the production QC run
#:
#: The original preprocessing/quality_control.py is left UNTOUCHED.
#:
#: Run as a kube-job (submitter: provenance/submitters/quality_control.py;
#: needs JOB_USER + MLflow credentials, both exported there):
#:
#:     uv run python -m provenance.quality_control
import os
import shutil
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.artifacts import download_artifacts
from rationai.provenance import log_provenance
from rationai.provenance.dataset import verify_manifest

#: Where the cut-down test runs live, isolated from the production runs.
EXPERIMENT = "provenance_huncrc_test"
#: Pinned create_dataset cut-down run (provenance/create_dataset.py) — its
#: dataset.csv artifact IS the 2-slide input for this whole test chain.
CREATE_DATASET_RUN_ID = "bc5160ba96064230b375cb15fdf78320"
#: Pinned registration run of the huncrc cohort (verify gate).
DATASET_RUN_ID = "4ef2fb98adb8432b9ee224fc5bdd5231"
#: The production run that PRODUCED the reused masks (exp 168). It processed
#: all 200 slides into the staging below, then failed before the MLflow
#: upload — the share is the only copy of the bytes.
PRODUCTION_QC_RUN_ID = "0e3d8a66a7084e0f8a492ba6186c1a80"

#: The production QC parameters these masks were created with
#: (configs/preprocessing/qc.yaml) — still true of the reused bytes.
QC_PARAMETERS = {
    "mask_level": 3,
    "sample_level": 1,
    "check_residual": True,
    "check_folding": False,
    "check_focus": True,
    "wb_correction": True,
}
#: The 4 mask families get_qc_masks produces with those parameters.
MASK_SUBDIRS = ("blur_per_pixel", "blur_per_tile", "artifacts_per_pixel", "artifacts_per_tile")

PROJECT_PATH = "/mnt/projects/huncrc_dysplasia"
#: Production staging — read-only for this script.
PRODUCTION_QC_STAGING = f"{PROJECT_PATH}/preprocessing/quality_control"
#: Prov-dedicated staging dir the copied masks are logged from.
QC_OUTPUT_DIR = f"{PROJECT_PATH}/preprocessing/prov_test/qc"
DATASET_CSV_URI = (
    f"mlflow-artifacts:/212/{CREATE_DATASET_RUN_ID}/artifacts/dataset.csv"
)


def main() -> None:
    # 1. verify the registered cohort before touching anything
    manifest_path = os.path.join(PROJECT_PATH, "preprocessing", "manifest.csv")
    check = verify_manifest(manifest_path, os.path.dirname(manifest_path), DATASET_RUN_ID)
    for detail in check["details"]:
        print(f"[prov qc] verify: {detail}")
    if not check["verified"]:
        raise SystemExit(
            f"[prov qc] dataset verification FAILED (run {DATASET_RUN_ID}) — refusing to process"
        )

    # 2. the 2-slide input, straight from the create_dataset cut-down run
    dataset = pd.read_csv(download_artifacts(DATASET_CSV_URI))
    slides = dataset["slide_path"].to_list()
    stems = [Path(s).stem for s in slides]
    print(f"[prov qc] reusing production masks for: {stems}")

    # 3. copy exactly those 2 slides' masks out of the production staging
    output_path = Path(QC_OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    prod_metrics = pd.read_csv(Path(PRODUCTION_QC_STAGING) / "qc_metrics.csv")
    selected = prod_metrics[
        prod_metrics["slide_name"].astype(str).str.zfill(3).isin(stems)
    ]
    missing = set(stems) - set(selected["slide_name"].astype(str).str.zfill(3))
    if missing:
        raise SystemExit(f"[prov qc] production qc_metrics.csv has no rows for {sorted(missing)}")
    selected.to_csv(output_path / "qc_metrics.csv", index=False)

    for subdir in MASK_SUBDIRS:
        (output_path / subdir).mkdir(parents=True, exist_ok=True)
        for stem in stems:
            src = Path(PRODUCTION_QC_STAGING) / subdir / f"{stem}.tiff"
            if not src.exists():
                raise SystemExit(f"[prov qc] production mask missing: {src}")
            shutil.copy2(src, output_path / subdir / f"{stem}.tiff")

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="🎭 QC (provenance, subset — masks reused from production)") as run:
        # 4. the gate: same fail-loud checks compute mode uses (plus the masks)
        metrics_path = output_path / "qc_metrics.csv"
        if not metrics_path.exists():
            raise SystemExit("[prov qc] qc_metrics.csv missing — nothing was copied")
        metrics = pd.read_csv(metrics_path)
        missing = set(stems) - set(metrics["slide_name"].astype(str).str.zfill(3))
        if missing:
            raise SystemExit(f"[prov qc] QC rows missing for slides {sorted(missing)} — not logging provenance")
        for subdir in MASK_SUBDIRS:
            for stem in stems:
                if not (output_path / subdir / f"{stem}.tiff").exists():
                    raise SystemExit(f"[prov qc] mask missing after copy: {subdir}/{stem}.tiff")

        # 5. upload (what tiling reads) + log provenance, reuse stated
        mlflow.log_artifacts(str(output_path), "qc_output")

        log_provenance(
            output=metrics,
            config={
                "dataset": {"name": "huncrc-cohort", "version": "1.0.0"},
                "masks_source": PRODUCTION_QC_STAGING,
                "reused_from_production_run": PRODUCTION_QC_RUN_ID,
                "reuse_reason": "masks already produced by the production 200-slide QC service run; a provenance-library test does not need a recompute",
                "qc_parameters": QC_PARAMETERS,
                "mask_subdirs": list(MASK_SUBDIRS),
                "subset_slides": slides,
                "provenance": {"create_dataset_run_id": CREATE_DATASET_RUN_ID},
            },
            inputs=[
                f"runs:/{CREATE_DATASET_RUN_ID}",
                f"runs:/{PRODUCTION_QC_RUN_ID}",
            ],
        )
        print(f"[prov qc] done — run {run.info.run_id} in experiment {EXPERIMENT}")


if __name__ == "__main__":
    main()
