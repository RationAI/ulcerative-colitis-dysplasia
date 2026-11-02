#: Provenance cut-down of the TRAINING step — the real compute leg of the
#: test chain.
#:
#: Runs the ORIGINAL training entry point (ml/__main__.py: @hydra.main +
#: @autolog, DysplasiaEmbeddingsClassifier on precomputed tile embeddings)
#: on the 2-slide chain, in-process, and appends the provenance log to the
#: SAME run the training itself creates:
#:
#:   1. verify_manifest — the registered cohort still matches the pinned run
#:   2. build the 2-slide subset data dir (job-local tmp, never the share):
#:        * tiling: tiling_200 filtered to the 2 slides, split/fold
#:          restamped to the chain's split (train=059, test_final=155)
#:        * embeddings: the 2 per-slide parquets copied in
#:   3. the original main() — hydra composes the UNMODIFIED experiment
#:      config (+experiment=embeddings_huncrc200, byte-identical); the
#:      prov-only deviations go in as CLI overrides, no config file is
#:      added or modified:
#:        * subset data paths
#:        * val split = test_final (test_preliminary is empty on the
#:          2-slide subset)
#:        * cluster_k = null (both subset slides are annotated and every
#:          annotated tile is dysplasia — the serial-sections filter would
#:          leave single-class splits and an undefined AUC)
#:        * max_epochs = 2 (a pipeline test, not a model)
#:        * checkpoint dir = prov-dedicated (training/embeddings_huncrc2prov
#:          — the production training/embeddings_huncrc200 stays untouched)
#:        * MLflow experiment (provenance_huncrc_test) + run name
#:   4. log_provenance — into the training run, which is still the active
#:      run after main() returns (the logger creates and starts it but
#:      never ends it — atexit does) — full ancestor chain as inputs
#:
#: The original ml/ module, configs/ and training/ are left UNTOUCHED.
#:
#: SLOW: real GPU training on ~177k tile vectors (2 epochs, A100) — the
#: only compute step of the chain besides tile_masks.
#:
#: Run as a kube-job (submitter: provenance/submitters/training.py; needs
#: JOB_USER + MLflow credentials, both exported there):
#:
#:     uv run python -m provenance.training
import os
import shutil
import sys
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
#: Pinned split cut-down run — its CSVs define the chain's split.
SPLIT_RUN_ID = "2385fd8162d746188c5c3092e2a1b605"
#: Pinned QC cut-down run (reuse mode).
QC_RUN_ID = "d9af25eaf0044e6d911887c69cc8e18b"
#: Pinned tissue cut-down run (compute mode; verified healthy 2026-11-02).
TISSUE_RUN_ID = "5d812b44fbc94d3cbf6391a1683297e6"
#: ── FILLED IN after the reuse-wave jobs finish (run ids from their logs) ──
TILING_RUN_ID = "TODO_FILL_AFTER_TILING_JOB"
EMBEDDINGS_RUN_ID = "TODO_FILL_AFTER_EMBEDDINGS_JOB"
# ─────────────────────────────────────────────────────────────────────────

#: The ORIGINAL experiment config — reused byte-identical; prov deviations
#: are CLI overrides in main(), not config edits.
EXPERIMENT_CONFIG = "embeddings_huncrc200"
#: A pipeline test, not a model — two full passes over ~177k tiles.
MAX_EPOCHS = 2
RUN_NAME = "📊 Training (provenance, 2-slide subset)"

PROJECT_PATH = "/mnt/projects/huncrc_dysplasia"
#: Production data the subset is built from — read-only for this script.
PRODUCTION_TILING_DIR = f"{PROJECT_PATH}/preprocessing/tiling_200"
PRODUCTION_EMBEDDINGS_DIR = f"{PROJECT_PATH}/preprocessing/embeddings/prov-gigapath"
#: Prov-dedicated checkpoint dir (production's stays untouched).
CHECKPOINT_DIR = f"{PROJECT_PATH}/training/embeddings_huncrc2prov"
#: test_preliminary is empty on the 2-slide subset (degenerate split) —
#: only the two non-empty splits are read.
SPLIT_URIS = {
    name: f"mlflow-artifacts:/212/{SPLIT_RUN_ID}/artifacts/{name}.csv"
    for name in ("train", "test_final")
}


def _build_subset(tmpdir: Path) -> tuple[Path, Path, pd.DataFrame, list[str]]:
    """The 2-slide subset data dir: restamped tiling + the 2 embeddings."""
    #: The chain's own split (the split CSVs carry no slide_id column —
    #: they are written index=False — join on slide_path).
    split_of: dict[str, tuple[str, float]] = {}  # stem -> (split, fold)
    for name in ("train", "test_final"):
        df = pd.read_csv(download_artifacts(SPLIT_URIS[name]))
        has_fold = "fold" in df.columns
        for _, row in df.iterrows():
            stem = Path(row["slide_path"]).stem
            fold = float(row["fold"]) if has_fold and pd.notna(row["fold"]) else float("nan")
            split_of[stem] = (name, fold)
    if len(split_of) != 2:
        raise SystemExit(f"[prov training] expected 2 slides in the chain split, got {sorted(split_of)}")

    slides = pd.read_parquet(Path(PRODUCTION_TILING_DIR) / "slides" / "slides.parquet")
    slides["stem"] = slides["path"].map(lambda p: Path(p).stem)
    sel = slides[slides["stem"].isin(split_of)].copy()
    missing = set(split_of) - set(sel["stem"])
    if missing:
        raise SystemExit(f"[prov training] tiling_200 has no slide rows for {sorted(missing)}")
    keep_ids = set(sel["id"])

    tiles = pd.concat(
        [
            pq.read_table(f, filters=[("slide_id", "in", keep_ids)]).to_pandas()
            for f in sorted(Path(PRODUCTION_TILING_DIR, "tiles").glob("*.parquet"))
        ],
        ignore_index=True,
    )
    if tiles.empty:
        raise SystemExit("[prov training] no tiles found for the 2 slides — refusing to train")

    stamp = {sid: split_of[stem] for sid, stem in zip(sel["id"], sel["stem"])}
    sel["split"] = sel["id"].map(lambda i: stamp[i][0])
    sel["fold"] = sel["id"].map(lambda i: stamp[i][1])
    tiles["split"] = tiles["slide_id"].map(lambda i: stamp[i][0])
    tiles["fold"] = tiles["slide_id"].map(lambda i: stamp[i][1])

    tiling_dir = tmpdir / "tiling"
    (tiling_dir / "slides").mkdir(parents=True)
    (tiling_dir / "tiles").mkdir(parents=True)
    sel.drop(columns=["stem"]).to_parquet(str(tiling_dir / "slides" / "slides.parquet"))
    tiles.to_parquet(str(tiling_dir / "tiles" / "tiles_0000.parquet"))

    emb_dir = tmpdir / "embeddings"
    emb_dir.mkdir(parents=True)
    for stem in split_of:
        src = Path(PRODUCTION_EMBEDDINGS_DIR) / f"{stem}.parquet"
        if not src.exists():
            raise SystemExit(f"[prov training] embedding parquet missing: {src}")
        shutil.copy2(src, emb_dir / f"{stem}.parquet")

    #: Per-split summary — the "output" frame logged with the provenance.
    rows = []
    for name in ("train", "test_final"):
        t = tiles[tiles["split"] == name]
        rows.append(
            {
                "split": name,
                "slides": int((sel["split"] == name).sum()),
                "tiles": int(len(t)),
                "dysplasia_tiles": int(((t["lowgrade_dysplasia"] > 0) | (t["highgrade_dysplasia"] > 0)).sum()),
            }
        )
    return tiling_dir, emb_dir, pd.DataFrame(rows), sorted(split_of)


def main() -> None:
    if "TODO" in (TILING_RUN_ID, EMBEDDINGS_RUN_ID).__str__():
        raise SystemExit(
            "[prov training] upstream run ids are not pinned yet — "
            "fill TILING_RUN_ID / EMBEDDINGS_RUN_ID from the job logs first"
        )

    # 1. verify the registered cohort before touching anything
    manifest_path = os.path.join(PROJECT_PATH, "preprocessing", "manifest.csv")
    check = verify_manifest(manifest_path, os.path.dirname(manifest_path), DATASET_RUN_ID)
    for detail in check["details"]:
        print(f"[prov training] verify: {detail}")
    if not check["verified"]:
        raise SystemExit(
            f"[prov training] dataset verification FAILED (run {DATASET_RUN_ID}) — refusing to train"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        # 2. the 2-slide subset (job-local — the share is never written)
        tiling_dir, emb_dir, summary, stems = _build_subset(Path(tmpdir))
        print(f"[prov training] subset: {summary.to_dict('records')}")

        # 3. the original entry point, with the prov deviations as CLI
        #    overrides (the experiment config itself stays byte-identical).
        #: hydra's CLI grammar cannot address list elements (no
        #: `callbacks[0].dirpath`), so the checkpoint callback is overridden
        #: as a whole list — same fields as the production experiment config,
        #: only the dirpath repointed at the prov-dedicated dir.
        sys.argv = [
            sys.argv[0],
            f"+experiment={EXPERIMENT_CONFIG}",
            f"++dataset.tiling_dir={tiling_dir}",
            f"++dataset.embeddings_dir={emb_dir}",
            #: test_preliminary is empty on the 2-slide subset.
            "++data.datasets.val.split=test_final",
            #: Both subset slides are annotated and every annotated tile is
            #: dysplasia — the serial-sections filter would leave
            #: single-class splits and an undefined AUC.
            "++dataset.cluster_k=null",
            f"++trainer.max_epochs={MAX_EPOCHS}",
            f'++trainer.callbacks=[{{_target_: lightning.pytorch.callbacks.ModelCheckpoint, dirpath: {CHECKPOINT_DIR}, filename: "emb-dysplasia-{{epoch:02d}}", monitor: validation/auc, mode: max, save_last: true}}]',
            f"++metadata.experiment_name={EXPERIMENT}",
            #: quoted: hydra's lexer rejects the emoji unquoted on the CLI.
            f'++metadata.run_name="{RUN_NAME}"',
            '++metadata.description="2-slide provenance test chain (train: 059, test_final: 155)"',
        ]
        from ml.__main__ import main as training_main  #: in-process: same process, same active run

        training_main()

        # 4. the training run is still active (the logger never ends it) —
        #    append the provenance to it.
        active = mlflow.active_run()
        if active is None or active.data.tags.get("mlflow.runName") != RUN_NAME:
            raise SystemExit(
                "[prov training] the training run is not the active run after main() — "
                "refusing to log provenance into the wrong run"
            )
        log_provenance(
            output=summary,
            config={
                "dataset": {"name": "huncrc-cohort", "version": "1.0.0"},
                "experiment_config": EXPERIMENT_CONFIG,
                "max_epochs": MAX_EPOCHS,
                "cluster_k_null_reason": "both subset slides are annotated; the serial-sections filter would leave single-class splits (undefined AUC)",
                "val_split_reason": "test_preliminary is empty on the 2-slide subset — val uses test_final",
                "checkpoint_dir": CHECKPOINT_DIR,
                "tiling_source": PRODUCTION_TILING_DIR,
                "embeddings_source": PRODUCTION_EMBEDDINGS_DIR,
                "subset_slides": stems,
                "provenance": {
                    "dataset_registration_run_id": DATASET_RUN_ID,
                    "create_dataset_run_id": CREATE_DATASET_RUN_ID,
                    "split_run_id": SPLIT_RUN_ID,
                    "qc_run_id": QC_RUN_ID,
                    "tissue_run_id": TISSUE_RUN_ID,
                    "tiling_run_id": TILING_RUN_ID,
                    "embeddings_run_id": EMBEDDINGS_RUN_ID,
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
                f"runs:/{EMBEDDINGS_RUN_ID}",
            ],
        )
        print(f"[prov training] done — run {active.info.run_id} in experiment {EXPERIMENT}")


if __name__ == "__main__":
    main()
