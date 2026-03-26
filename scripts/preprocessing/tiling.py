from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-dysplasia-tiling",
    username="borisim",
    public=False,
    cpu=8,
    memory="32Gi",
    shm="48Gi",
    script=[
        "git clone -b feat/tiling --single-branch https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run --active -m preprocessing.tiling",
    ],
    storage=[storage.secure.DATA],
)