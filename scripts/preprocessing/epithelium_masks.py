from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-dysplasia-epithelium-masks",
    username="...",
    public=False,
    cpu=8,
    memory="16Gi",
    shm="16Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run python -m preprocessing.epithelium_masks +dataset=...",
    ],
    storage=[storage.secure.DATA],
)
