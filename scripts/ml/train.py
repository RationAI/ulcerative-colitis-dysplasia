from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-dysplasia-train-new",
    username="...",
    public=False,
    cpu=16,
    memory="64Gi",
    shm="64Gi",
    gpu="mig-1g.10gb",
    script=[
        "git clone -b feature/ml-cnn https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run python -m ml '+experiment=ml/train/virchow2_l...' val_fold=0,1,2,3,4 --multirun",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
