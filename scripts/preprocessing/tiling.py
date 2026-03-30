from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-dysplasia-tiling",
    username="...",
    public=False,
    cpu=8,
    memory="16Gi",
    shm="16Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run python -m preprocessing.tiling"
        "+dataset=..."
        "+experiment/preprocessing/tiling=...",
    ],
    storage=[storage.secure.DATA],
)
