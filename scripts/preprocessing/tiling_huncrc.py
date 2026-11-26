from kube_jobs import storage, submit_job

#: Native pyramid levels of the TCIA_HunCRC slides (0.1213 um/px at level 0).
for mpp in ("mpp024", "mpp049", "mpp097", "mpp194"):
    submit_job(
        job_name=f"ulcerative-colitis-dysplasia-tiling-huncrc-{mpp}",
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
                "+dataset=huncrc/processed_w_masks "
                f"+experiment/preprocessing/tiling/huncrc={mpp}"
            ),
        ],
        storage=[storage.secure.DATA, storage.secure.PROJECTS],
    )
