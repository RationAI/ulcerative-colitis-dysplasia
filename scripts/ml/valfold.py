from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-dysplasia-valfold",
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
        "uv run python -m ml +experiment=ml/valfold/virchow2 valfold.manifest='ml/manifests/level2_high.yaml'",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
