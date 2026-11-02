#: Provenance cut-down of the embeddings step — REUSE MODE.
#:
#: WHY NOT RECOMPUTE: the prov-gigapath embeddings of the full cohort are
#: already on the project share (preprocessing/embeddings/prov-gigapath,
#: one parquet per slide), and slides 155/059 are in it — with row counts
#: matching the production tiling_200 grid 1:1 (48,686 / 128,545), i.e. the
#: vectors were computed from exactly the tiling the chain reuses. Re-running
#: gigapath inference for 2 slides for a test of the PROVENANCE LIBRARY
#: would burn GPU hours to reproduce the same vectors. So this script:
#:
#:   1. verify_manifest — the registered cohort still matches the pinned run
#:   2. selects the 2 slides from the chain's dataset.csv (bc5160ba run)
#:   3. COPIES the 2 per-slide parquets into a prov-dedicated staging dir
#:      (the production tree is never touched)
#:   4. gates on both files present AND row counts matching the tiling_200
#:      grid — the same check the training dataset enforces at load time
#:   5. log_provenance  — the reuse recorded EXPLICITLY (run name,
#:      cfg_masks_source, cfg_reuse_reason)
#:
#: SOURCE NOTE: no MLflow run of the production gigapath inference exists
#: (every experiment was checked) — the share dir is the only record of
#: where the bytes came from, so it is pinned as a param instead of a run id.
#:
#: The original preprocessing/embeddings.py is left UNTOUCHED.
#:
#: FAST: copies ~170 MB + a predicate-pushed grid check — minutes.
#:
#: Run as a kube-job (submitter: provenance/submitters/embeddings.py; needs
#: JOB_USER + MLflow credentials, both exported there):
#:
#:     uv run python -m provenance.embeddings
import os
import shutil
from pathlib import Path

import mlflow
import pandas as pd
import pyarrow.parquet as pq
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

#: The production model these per-slide parquets hold.
MODEL = "prov-gigapath"

PROJECT_PATH = "/mnt/projects/huncrc_dysplasia"
#: Production embeddings — read-only for this script.
PRODUCTION_EMBEDDINGS_DIR = f"{PROJECT_PATH}/preprocessing/embeddings/{MODEL}"
#: Prov-dedicated staging dir the copied parquets are logged from.
EMBEDDINGS_OUTPUT_DIR = f"{PROJECT_PATH}/preprocessing/prov_test/embeddings"
#: The tiling the embeddings were computed from (grid cross-check).
PRODUCTION_TILING_DIR = f"{PROJECT_PATH}/preprocessing/tiling_200"
DATASET_CSV_URI = (
    f"mlflow-artifacts:/212/{CREATE_DATASET_RUN_ID}/artifacts/dataset.csv"
)


def main() -> None:
    # 1. verify the registered cohort before touching anything
    manifest_path = os.path.join(PROJECT_PATH, "preprocessing", "manifest.csv")
    check = verify_manifest(manifest_path, os.path.dirname(manifest_path), DATASET_RUN_ID)
    for detail in check["details"]:
        print(f"[prov embeddings] verify: {detail}")
    if not check["verified"]:
        raise SystemExit(
            f"[prov embeddings] dataset verification FAILED (run {DATASET_RUN_ID}) — refusing to process"
        )

    # 2. the 2-slide input, straight from the create_dataset cut-down run
    dataset = pd.read_csv(download_artifacts(DATASET_CSV_URI))
    slides = dataset["slide_path"].to_list()
    stems = [Path(s).stem for s in slides]
    print(f"[prov embeddings] reusing production embeddings for: {stems}")

    # 3. copy exactly those 2 slides' parquets out of the production dir
    output_path = Path(EMBEDDINGS_OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    row_counts: dict[str, int] = {}
    for stem in stems:
        src = Path(PRODUCTION_EMBEDDINGS_DIR) / f"{stem}.parquet"
        if not src.exists():
            raise SystemExit(f"[prov embeddings] production parquet missing: {src}")
        row_counts[stem] = pq.ParquetFile(src).metadata.num_rows
        if row_counts[stem] == 0:
            raise SystemExit(f"[prov embeddings] empty parquet: {src}")
        shutil.copy2(src, output_path / f"{stem}.parquet")

    # 4. grid cross-check: embedding rows must equal the tiling_200 grid
    #    (predicate pushdown — only the 2 slides' tile rows are loaded)
    slides_df = pd.read_parquet(Path(PRODUCTION_TILING_DIR) / "slides" / "slides.parquet")
    slides_df["stem"] = slides_df["path"].map(lambda p: Path(p).stem)
    id_of = dict(zip(slides_df["stem"], slides_df["id"]))
    missing = set(stems) - set(id_of)
    if missing:
        raise SystemExit(f"[prov embeddings] tiling_200 has no slide rows for {sorted(missing)}")
    grid: dict[str, int] = {}
    for f in sorted(Path(PRODUCTION_TILING_DIR, "tiles").glob("*.parquet")):
        table = pq.read_table(f, columns=["slide_id"],
                              filters=[("slide_id", "in", {id_of[s] for s in stems})])
        for sid, n in table.to_pandas()["slide_id"].value_counts().items():
            grid[sid] = n
    for stem in stems:
        expected, got = grid.get(id_of[stem], 0), row_counts[stem]
        if expected != got:
            raise SystemExit(
                f"[prov embeddings] {stem}: {got} embedding rows != {expected} tiling tiles — "
                "embeddings were computed from a different tiling"
            )
    print(f"[prov embeddings] grid check OK: { {s: row_counts[s] for s in stems} }")

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="🧬 Embeddings (provenance, subset — reused from production gigapath run)") as run:
        # gate: both files landed
        for stem in stems:
            if not (output_path / f"{stem}.parquet").exists():
                raise SystemExit(f"[prov embeddings] mask missing after copy: {stem}.parquet")

        # 5. upload (same per-slide layout as production) + log provenance
        mlflow.log_artifacts(str(output_path), "embeddings")

        log_provenance(
            output=pd.DataFrame({"slide": stems, "embedding_rows": row_counts}),
            config={
                "dataset": {"name": "huncrc-cohort", "version": "1.0.0"},
                "model": MODEL,
                "masks_source": PRODUCTION_EMBEDDINGS_DIR,
                "reuse_reason": "vectors already computed by the production full-cohort gigapath run; a provenance-library test does not need inference",
                "derived_from_tiling": PRODUCTION_TILING_DIR,
                "subset_slides": stems,
                "provenance": {"create_dataset_run_id": CREATE_DATASET_RUN_ID},
            },
            inputs=[f"runs:/{CREATE_DATASET_RUN_ID}"],
        )
        print(f"[prov embeddings] done — run {run.info.run_id} in experiment {EXPERIMENT}")


if __name__ == "__main__":
    main()
