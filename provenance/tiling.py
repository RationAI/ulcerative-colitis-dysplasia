#: Provenance cut-down of the tiling step — REUSE MODE.
#:
#: WHY NOT RECOMPUTE: slides 155/059 are already tiled — the production
#: tiling of the full cohort (preprocessing/tiling_200) contains both
#: (059 → 128,545 tiles, 155 → 48,686 tiles, at the proven pilot geometry
#: 224 px / 0.52 mpp / stride 112). The prov-gigapath embeddings were
#: computed from exactly this tiling (row counts match 1:1), so the reused
#: bytes are also the bytes the chain's embeddings sit on. Recomputing the
#: grid for a test of the PROVENANCE LIBRARY would burn ray + openslide
#: hours to reproduce the same tiles. So this script:
#:
#:   1. verify_manifest — the registered cohort still matches the pinned run
#:   2. selects the 2 slides from the chain's split CSVs (run 2385fd81) —
#:      train=059, test_final=155 (test_preliminary is empty by design on
#:      the 2-slide subset)
#:   3. FILTERS the production tiling_200 to exactly those 2 slides and
#:      RESTAMPS split/fold to the chain's own split — the production
#:      split/fold columns are stale for this test chain
#:   4. writes the same per-split parquet layout the tile_masks cut-down
#:      consumes ({split}/slides + {split}/tiles)
#:   5. log_provenance  — the reuse recorded EXPLICITLY (run name,
#:      cfg_masks_source, cfg_reuse_reason) and the geometry read from the
#:      reused data itself
#:
#: SOURCE NOTE: no MLflow run of the production tiling exists (every
#: experiment was checked) — the share dir is the only record of where the
#: bytes came from, so it is pinned as a param instead of a run id.
#:
#: CHAIN NOTE: this run pins the runs it actually consumes (create_dataset,
#: split). The QC and tissue runs are siblings in the chain — they are
#: pinned by the tile_masks and training runs, which carry the full
#: ancestor list.
#:
#: The original preprocessing/tiling.py is left UNTOUCHED.
#:
#: FAST: pandas/pyarrow only, no ray/openslide — minutes.
#:
#: Run as a kube-job (submitter: provenance/submitters/tiling.py; needs
#: JOB_USER + MLflow credentials, both exported there):
#:
#:     uv run python -m provenance.tiling
import os
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import pyarrow.parquet as pq
from mlflow.artifacts import download_artifacts
from rationai.provenance import log_provenance
from rationai.provenance.dataset import verify_manifest

#: Where the cut-down test runs live, isolated from the production runs.
EXPERIMENT = "provenance_huncrc_test"
#: Pinned registration run of the huncrc cohort (verify gate).
DATASET_RUN_ID = "4ef2fb98adb8432b9ee224fc5bdd5231"
#: Pinned create_dataset cut-down run — root of this test chain.
CREATE_DATASET_RUN_ID = "bc5160ba96064230b375cb15fdf78320"
#: Pinned split cut-down run — its CSVs ARE the split this step restamps.
SPLIT_RUN_ID = "2385fd8162d746188c5c3092e2a1b605"

PROJECT_PATH = "/mnt/projects/huncrc_dysplasia"
#: Production tiling of the full cohort — read-only for this script.
PRODUCTION_TILING_DIR = f"{PROJECT_PATH}/preprocessing/tiling_200"
#: test_preliminary is empty on the 2-slide subset (degenerate split) —
#: only the two non-empty splits are read.
SPLIT_URIS = {
    name: f"mlflow-artifacts:/212/{SPLIT_RUN_ID}/artifacts/{name}.csv"
    for name in ("train", "test_final")
}


def main() -> None:
    # 1. verify the registered cohort before touching anything
    manifest_path = os.path.join(PROJECT_PATH, "preprocessing", "manifest.csv")
    check = verify_manifest(manifest_path, os.path.dirname(manifest_path), DATASET_RUN_ID)
    for detail in check["details"]:
        print(f"[prov tiling] verify: {detail}")
    if not check["verified"]:
        raise SystemExit(
            f"[prov tiling] dataset verification FAILED (run {DATASET_RUN_ID}) — refusing to process"
        )

    # 2. the chain's own split (NOT the production one).
    #    The split CSVs carry no slide_id column (written index=False) —
    #    join on slide_path.
    split_of: dict[str, tuple[str, float]] = {}  # stem -> (split, fold)
    for name in ("train", "test_final"):
        df = pd.read_csv(download_artifacts(SPLIT_URIS[name]))
        has_fold = "fold" in df.columns
        for _, row in df.iterrows():
            stem = Path(row["slide_path"]).stem
            fold = float(row["fold"]) if has_fold and pd.notna(row["fold"]) else float("nan")
            split_of[stem] = (name, fold)
    if len(split_of) != 2:
        raise SystemExit(f"[prov tiling] expected 2 slides in the chain split, got {sorted(split_of)}")
    print(f"[prov tiling] chain split: {split_of}")

    # 3. the production tiling for exactly those 2 slides
    slides = pd.read_parquet(Path(PRODUCTION_TILING_DIR) / "slides" / "slides.parquet")
    slides["stem"] = slides["path"].map(lambda p: Path(p).stem)
    sel = slides[slides["stem"].isin(split_of)].copy()
    missing = set(split_of) - set(sel["stem"])
    if missing:
        raise SystemExit(f"[prov tiling] tiling_200 has no slide rows for {sorted(missing)}")
    keep_ids = set(sel["id"])

    #: Predicate pushdown: only the 2 slides' tile rows are ever loaded.
    tiles = pd.concat(
        [
            pq.read_table(
                f, filters=[("slide_id", "in", keep_ids)]
            ).to_pandas()
            for f in sorted(Path(PRODUCTION_TILING_DIR, "tiles").glob("*.parquet"))
        ],
        ignore_index=True,
    )

    # 4. restamp split/fold to the chain's split (production values are stale)
    stamp = {sid: split_of[stem] for sid, stem in zip(sel["id"], sel["stem"])}
    sel["split"] = sel["id"].map(lambda i: stamp[i][0])
    sel["fold"] = sel["id"].map(lambda i: stamp[i][1])
    tiles["split"] = tiles["slide_id"].map(lambda i: stamp[i][0])
    tiles["fold"] = tiles["slide_id"].map(lambda i: stamp[i][1])

    # gate: every chain slide is tiled, non-trivial grid
    for stem, (name, _fold) in split_of.items():
        if int((sel["split"] == name).sum()) != 1 or int((tiles["split"] == name).sum()) == 0:
            raise SystemExit(f"[prov tiling] no/slide rows for {stem} — refusing to log")
    print(f"[prov tiling] {len(sel)} slides, {len(tiles)} tiles")

    #: Geometry as the reused bytes actually encode it (same for both slides).
    geo = sel.iloc[0]
    geometry = {
        "mpp": float(geo["mpp_x"]),
        "tile_extent": int(geo["tile_extent_x"]),
        "stride": int(geo["stride_x"]),
        "tissue_threshold": 0.5,  #: production tiling value (configs/preprocessing/tiling.yaml)
    }

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="🧩 Tiling (provenance, subset — reused from production tiling_200)") as run:
        with tempfile.TemporaryDirectory() as tmpdir:
            # 5. per-split layout the tile_masks cut-down consumes
            base = Path(tmpdir) / "tiling"
            for name in ("train", "test_final"):
                out = base / name
                out.mkdir(parents=True, exist_ok=True)
                sel[sel["split"] == name].drop(columns=["stem"]).to_parquet(str(out / "slides"))
                tiles[tiles["split"] == name].to_parquet(str(out / "tiles"))

            mlflow.log_artifacts(str(base), "tiling")

            log_provenance(
                output=tiles,
                config={
                    "dataset": {"name": "huncrc-cohort", "version": "1.0.0"},
                    "masks_source": PRODUCTION_TILING_DIR,
                    "reuse_reason": "both slides already tiled in the production full-cohort tiling at the proven pilot geometry; a provenance-library test does not need a recompute",
                    **geometry,
                    "restamped_split": {stem: name for stem, (name, _f) in split_of.items()},
                    "subset_slides": sorted(split_of),
                    "provenance": {
                        "dataset_registration_run_id": DATASET_RUN_ID,
                        "create_dataset_run_id": CREATE_DATASET_RUN_ID,
                        "split_run_id": SPLIT_RUN_ID,
                    },
                },
                #: the runs this step actually consumes (the reuse bytes have
                #: no producer run to pin — see the header note).
                inputs=[
                    f"runs:/{CREATE_DATASET_RUN_ID}",
                    f"runs:/{SPLIT_RUN_ID}",
                ],
            )
        print(f"[prov tiling] done — run {run.info.run_id} in experiment {EXPERIMENT}")


if __name__ == "__main__":
    main()
