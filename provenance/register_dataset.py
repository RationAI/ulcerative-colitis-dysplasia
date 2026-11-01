#: Register the project dataset in MLflow — per project, once per dataset
#: version. Writes the canonical manifest.csv next to the cohort csv (data
#: stays in place) and stores the per-file fingerprint (size + mtime) in a
#: registration run inside the lib-global Dataset_Registry experiment —
#: later runs pin it by run_id.
#:
#: Run as a kube-job AFTER create_user (capture_environment hard-fails for an
#: unregistered JOB_USER):
#:
#:     uv run python -m provenance.register_dataset
#:
#: The job needs PROJECTS storage (read the cohort csv, write manifest.csv)
#: and DATA storage (stat the slide files for the fingerprint).
from rationai.provenance import register_dataset

# The lib-global dataset registry — where lookup_dataset_run() and the lib's
# auto-tagging expect registrations to live (rationai/provenance/dataset.py:29).
EXPERIMENT = "Dataset_Registry"

PROJECT_DIR = "/mnt/projects/huncrc_dysplasia"

#: One entry per registered dataset. The cohort is the raw-slide registry.
#: When an embeddings/features csv exists, add it here the same way
#: (registering features as data is the deck's step 8b), e.g.:
#:     {
#:         "csv": f"{PROJECT_DIR}/preprocessing/embeddings/embeddings.csv",
#:         "path_column": "feature_path",
#:         "dataset_name": "huncrc-embeddings",
#:         "version": "1.0.0",
#:     },
DATASETS = [
    {
        "csv": f"{PROJECT_DIR}/preprocessing/dataset_split.csv",
        "path_column": "slide_path",
        "dataset_name": "huncrc-cohort",
        "version": "1.0.0",
    },
]


def main() -> None:
    for spec in DATASETS:
        print(f"[register] {spec['dataset_name']} v{spec['version']}  <- {spec['csv']}")
        run_id = register_dataset(
            spec["csv"],
            path_column=spec["path_column"],
            experiment_name=EXPERIMENT,
            dataset_name=spec["dataset_name"],
            version=spec["version"],
        )
        print(f"[register] {spec['dataset_name']}: pinned registration run_id = {run_id}")
        print("[register] -> copy this run_id into the config of any run that uses this data")


if __name__ == "__main__":
    main()
