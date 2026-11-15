from kube_jobs import storage, submit_job


# The trained model to run inference with (MLflow artifact URI).
# Choose the embedding level to match the checkpoint: level1_extent224 (0.52 mpp)
# or level2_extent224 (1.55 mpp).
checkpoint = "mlflow-artifacts:/111/<run_id>/artifacts/checkpoints/<epoch=-step=>/checkpoint.ckpt"

submit_job(
    job_name="ulcerative-colitis-dysplasia-predict",
    username="borisim",
    public=False,
    cpu=8,
    memory="32Gi",
    shm="16Gi",
    gpu="mig-1g.10gb",
    script=[
        "git clone -b feature/ml-cnn https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run python -m ml +experiment=ml/predict/virchow2_test_preliminary "
        "+dataset=embeddings/level2_extent224 "
        "checkpoint='" + checkpoint + "'",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
