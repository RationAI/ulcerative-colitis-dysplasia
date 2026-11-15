from kube_jobs import storage, submit_job


# Step 3a run (artifacts/test_preliminary/<train_run_id>/test_predictions.parquet).
testprelim_run_id = "a4b3a79326d24e3bb33a43e34fe842be"
# Step 2 run (artifacts/valthreshold/thresholds.parquet).
thresholds_run_id = "6dc852a7429b4304b6d13da2368c0206"

# Step 3b is CPU-only (reads saved parquets + thresholds, computes F1).
submit_job(
    job_name="ulcerative-colitis-dysplasia-testf1",
    username="borisim",
    public=False,
    cpu=4,
    memory="8Gi",
    shm="2Gi",
    script=[
        "git clone -b feature/ml-cnn https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run python -m ml +experiment=ml/testf1/virchow2 "
        "testf1.testprelim_run_id='" + testprelim_run_id + "' "
        "testf1.thresholds_run_id='" + thresholds_run_id + "'",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
