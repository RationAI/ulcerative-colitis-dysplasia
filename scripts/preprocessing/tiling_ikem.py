from kube_jobs import storage, submit_job

for mpp in ("mpp052", "mpp155"):
    submit_job(
        job_name=f"ulcerative-colitis-dysplasia-tiling-ikem-{mpp}",
        username=...,
        public=False,
        cpu=8,
        memory="16Gi",
        shm="16Gi",
        script=[
            "git clone https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
            "cd workdir",
            "uv sync --frozen",
            (
                "uv run python -m preprocessing.tiling "
                "+dataset=ikem/processed_w_masks "
                f"+experiment/preprocessing/tiling/ikem={mpp}"
            ),
        ],
        storage=[storage.secure.DATA, storage.secure.PROJECTS],
    )
