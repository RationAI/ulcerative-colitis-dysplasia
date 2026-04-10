from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-dataset-split-dataset",
    username=...,
    public=False,
    cpu=2,
    memory="4Gi",
    storage=[storage.secure.DATA],
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.split_dataset +dataset=...",
    ],
)
