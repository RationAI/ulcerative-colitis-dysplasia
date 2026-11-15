from kube_jobs import storage, submit_job


# MLflow run (Step 1) that holds the per-model val predictions:
# artifacts/valfold/<training_run_id>/val_predictions.parquet. Fill this in once
# the valfold job has finished.
valfold_run_id = "cee90311d3ef4dc78c0fff97ac40beee"

# Step 4 is CPU-only (reads the saved predictions parquets + slide geometry and
# rasterizes heatmaps), so no GPU is requested.
submit_job(
    job_name="ulcerative-colitis-dysplasia-heatmap",
    username="borisim",
    public=False,
    cpu=8,
    memory="32Gi",
    shm="4Gi",
    script=[
        "git clone -b feature/ml-cnn https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run python -m ml +experiment=ml/heatmap/virchow2 "
        "heatmap.valfold_run_id='" + valfold_run_id + "'",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
