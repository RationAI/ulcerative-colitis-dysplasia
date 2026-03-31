from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-dysplasia-quality-control",
    username=...,
    cpu=8,
    memory="16Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        "uv sync",
        "uv run python -m preprocessing.quality_control +dataset=...",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
