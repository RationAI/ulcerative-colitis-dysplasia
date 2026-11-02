#: Provenance cut-down of the split-dataset step.
#:
#: The original preprocessing/split_dataset.py is left UNTOUCHED. This copy
#: reuses its split_dataset() / add_folds() (one source of truth for the
#: logic) on the 2-slide test dataset and wraps it in a provenance-logged
#: MLflow run:
#:
#:   1. verify_manifest  — the registered cohort still matches the pinned run
#:   2. split_dataset    — the original logic (0.7 / 0.15 / 0.15, rs=42)
#:   3. log_provenance   — env + hardware, config params, split sizes +
#:                         the pinned input run (runs:/<create_dataset run>)
#:
#: INSTANT: pure pandas on 2 rows — the fastest step in the chain.
#:
#: KNOWN DEGENERATE CASE (2 slides = 2 case groups):
#:   * the first split is 1 train / 1 test — fine;
#:   * preliminary_size = round(1 * 0.15/0.3) = 0 rows, and ratiopath's
#:     train_test_split rejects int train_size=0 (its @validate_params floor
#:     is 1) — so the original split_dataset() raises TypeError on exactly
#:     2 rows. The fallback below performs the same first split and leaves
#:     test_preliminary empty (the tiling cut-down skips empty splits).
#:   * train then holds a SINGLE case group, so GroupKFold(5) would raise
#:     "n_splits > n_groups" — with fewer groups than folds, fold 0 is
#:     assigned directly instead.
#: On a real cohort (200 rows) none of the fallbacks trigger.
#:
#: Run as a kube-job (submitter: provenance/submitters/split_dataset.py;
#: needs JOB_USER + MLflow credentials, both exported there):
#:
#:     uv run python -m provenance.split_dataset
import os
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.artifacts import download_artifacts
from omegaconf import OmegaConf
from rationai.provenance import log_provenance
from rationai.provenance.dataset import verify_manifest
from ratiopath.model_selection import train_test_split

#: Reuse the original logic — the original module stays byte-identical.
from preprocessing.split_dataset import add_folds, split_dataset

#: Where the cut-down test runs live, isolated from the production runs.
EXPERIMENT = "provenance_huncrc_test"
#: Pinned create_dataset cut-down run (provenance/create_dataset.py) — its
#: dataset.csv artifact IS the 2-slide input for this whole test chain.
CREATE_DATASET_RUN_ID = "bc5160ba96064230b375cb15fdf78320"
#: Pinned registration run of the huncrc cohort (verify gate).
DATASET_RUN_ID = "4ef2fb98adb8432b9ee224fc5bdd5231"

#: Same values as configs/preprocessing/split_dataset.yaml.
SPLITS = OmegaConf.create({"train": 0.7, "test_preliminary": 0.15, "test_final": 0.15})
N_FOLDS = 5
RANDOM_STATE = 42

PROJECT_PATH = "/mnt/projects/huncrc_dysplasia"
DATASET_CSV_URI = (
    f"mlflow-artifacts:/212/{CREATE_DATASET_RUN_ID}/artifacts/dataset.csv"
)


def main() -> None:
    # 1. verify the registered cohort before touching anything
    manifest_path = os.path.join(PROJECT_PATH, "preprocessing", "manifest.csv")
    check = verify_manifest(manifest_path, os.path.dirname(manifest_path), DATASET_RUN_ID)
    for detail in check["details"]:
        print(f"[prov split] verify: {detail}")
    if not check["verified"]:
        raise SystemExit(
            f"[prov split] dataset verification FAILED (run {DATASET_RUN_ID}) — refusing to process"
        )

    # 2. the 2-slide input, straight from the create_dataset cut-down run
    dataset = pd.read_csv(download_artifacts(DATASET_CSV_URI))
    print(f"[prov split] {len(dataset)} rows: {dataset['slide_id'].tolist()}")

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="🔀 Split dataset (provenance, subset)") as run:
        # 3. original logic — with the degenerate-2-row fallbacks documented above
        try:
            train, test_preliminary, test_final = split_dataset(
                dataset, SPLITS, RANDOM_STATE
            )
            degenerate = False
        except TypeError:
            #: 2 rows → preliminary_size = 0 → ratiopath rejects int train_size=0.
            #: Same first split, preliminary left empty.
            print("[prov split] degenerate 2-slide case — preliminary will be empty")
            train, test_final = train_test_split(
                dataset,
                train_size=SPLITS["train"],
                random_state=RANDOM_STATE,
                groups=dataset["case_id"],
            )
            test_preliminary = test_final.iloc[0:0]
            degenerate = True

        if train["case_id"].nunique() >= N_FOLDS:
            train = add_folds(train, N_FOLDS)
        else:
            #: Fewer case groups in train than folds — GroupKFold would raise.
            train["fold"] = 0

        print(
            f"[prov split] train={len(train)} preliminary={len(test_preliminary)} "
            f"final={len(test_final)} (degenerate={degenerate})"
        )

        # 4. write + upload (same artifact names as production) + log provenance
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, df in (
                ("train", train),
                ("test_preliminary", test_preliminary),
                ("test_final", test_final),
            ):
                output_path = Path(tmpdir) / f"{name}.csv"
                df.to_csv(output_path, index=False)
                mlflow.log_artifact(str(output_path))

            log_provenance(
                output=train,
                config={
                    "dataset": {"name": "huncrc-cohort", "version": "1.0.0"},
                    "splits": {"train": 0.7, "test_preliminary": 0.15, "test_final": 0.15},
                    "n_folds": N_FOLDS,
                    "random_state": RANDOM_STATE,
                    "split_sizes": {
                        "train": int(len(train)),
                        "test_preliminary": int(len(test_preliminary)),
                        "test_final": int(len(test_final)),
                    },
                    "subset_slides": [str(s) for s in dataset["slide_id"]],
                    "degenerate_two_slide_fallback": degenerate,
                    "provenance": {"create_dataset_run_id": CREATE_DATASET_RUN_ID},
                },
                inputs=[f"runs:/{CREATE_DATASET_RUN_ID}"],
            )
        print(f"[prov split] done — run {run.info.run_id} in experiment {EXPERIMENT}")


if __name__ == "__main__":
    main()
