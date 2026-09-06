from kube_jobs import storage, submit_job


# Step 2 is CPU-only (it just reads the saved predictions parquets and sweeps
# thresholds), so no GPU is requested.
# valfold_run_id (level 2 / 1.55mpp): holds the per-model val predictions under
# artifacts/valfold/<training_run_id>/val_predictions.parquet.
submit_job(
    job_name="ulcerative-colitis-dysplasia-valthreshold",
    username="borisim",
    public=False,
    cpu=4,
    memory="8Gi",
    shm="2Gi",
    script=[
        "git clone -b feature/ml-cnn https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run python -m ml +experiment=ml/valthreshold/virchow2_l2 "
        "valthreshold.valfold_run_id='974946bd720f4becb74272732532c786'",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
