#: Submit tissue-mask creation for TCIA_HunCRC as a cluster job.
#:
#: Same hunCRC setup as quality_control.py / annotation_masks.py -- public cloud,
#: BRANCH pinned, MLflow token exported into the job. Replaces the IBD template this file
#: was (username "...", public=False, storage.secure.DATA, "+dataset=..." placeholder).
#:
#: MOUNTS: DATA only, and that asymmetry is the whole design of this stage.
#:   * public.DATA -- REQUIRED. Unlike QC, this payload opens every slide itself:
#:     OpenSlide at preprocessing/tissue_masks.py:28 and pyvips.new_from_file(:31) both
#:     take the slide_path string from dataset.csv, so it must resolve INSIDE the pod.
#:     public.DATA mounts rationai/Data at /mnt/data/Public, which is exactly the prefix
#:     dataset.csv's slide_path column carries (configs/dataset/raw.yaml:2).
#:   * PROJECTS -- deliberately NOT mounted. Nothing is written to /mnt/projects: output
#:     goes to a TemporaryDirectory(:48) that is uploaded and then deleted(:61). Mounting
#:     it would change nothing and invite someone to "fix" a nonexistent path problem.
#:     Contrast annotation_masks, which needs both because it reads JSONs from the project
#:     share and writes masks there.
#:
#: The IBD template's storage.secure.DATA is wrong twice over for hunCRC: secure is the
#: certified/Sensitive cloud (see the warning in create_dataset.py -- exposing VPN tunnel
#: endpoints to public traffic "will have consequences"), and hunCRC lives on the PUBLIC
#: rationai/Data share, which secure does not mount. public=False would also schedule the
#: pod onto the sensitive cluster where the slides are not mounted at all.
#:
#: +dataset=processed with the artifact URI overridden, exactly as quality_control.py
#: does. It cannot be +dataset=raw: this payload really does read the CSV
#: (config.dataset.mlflow_uris.dataset at :42-44, then dataset["slide_path"] at :52), and
#: configs/dataset/raw.yaml defines no mlflow_uris at all -- it would fail composition.
#: processed.yaml still hardcodes the IBD cohort's dataset.csv
#: (configs/dataset/processed.yaml:6, experiment 111), hence the ++ override. Override
#: rather than edit that file, which other stages still reference.
#:
#: SIZING. memory=24Gi: each ray task asks for 4Gi (preprocessing/tissue_masks.py:22) and
#: configs/preprocessing/tissue_masks.yaml:4 runs max_concurrent 4, so 16 GiB is the
#: tasks alone; the rest is the driver, pyvips and libvips temp files. cpu=8 matches the
#: other mask submitters. The IBD template's shm=16Gi is kept -- it is ray's object store
#: and pyvips temp space for whole-slide reads at level 3 (68k x 73k px).
#:
#: WATCH ONE THING IF IT FAILS: all 200 masks accumulate in the TemporaryDirectory before
#: the single log_artifacts at :61, and that dir is pod-local emptyDir. Tissue masks are
#: mostly-black level-3 pyramids so deflate keeps them small, but if the pod dies near the
#: END you lose everything at once -- there is no incremental upload and no partial-run
#: guard in this payload (unlike preprocessing/annotation_masks.py, which counts expected
#: files and refuses to log a partial tree). Check the artifact count after the run.
#:
#: Needs exported in the shell that runs this:
#:   export MLFLOW_TRACKING_PASSWORD=<access key>
#:   export DATASET_ARTIFACT_URI=...   (optional; defaults to the hunCRC create_dataset run)
#:
#: The job clones configs/ and preprocessing/ from GitHub at BRANCH, so a local edit to
#: configs/preprocessing/tissue_masks.yaml or the payload has no effect until pushed.
#: base.yaml's uncommitted hunCRC paths do NOT matter here -- no output path derives from
#: project_path, and configs/hydra/default.yaml sets output_subdir: null, run.dir: ., so
#: Hydra writes nothing to disk either.
import os

from kube_jobs import storage, submit_job


BRANCH = "huncrc"

#: Same values as create_dataset.py -- see the notes there for why the k8s label and the
#: MLflow identity differ, and why the token comes from the environment.
MLFLOW_USER = "519192@mail.muni.cz"
MLFLOW_TOKEN = os.environ.get("MLFLOW_TRACKING_PASSWORD")
DATASET_ARTIFACT_URI = os.environ.get(
    "DATASET_ARTIFACT_URI",
    "mlflow-artifacts:/168/2ca0c1928c2c48acb56a5110efefb957/artifacts/dataset.csv",
)

if not MLFLOW_TOKEN:
    raise SystemExit(
        "MLFLOW_TRACKING_PASSWORD is unset. Mint an access key in the MLflow UI "
        "(Permission -> create access key) and export it. A login password is rejected."
    )

submit_job(
    job_name="ulcerative-colitis-dysplasia-tissue-masks-huncrc",
    username="borisim",
    image="cerit.io/rationai/base:2.0.6",
    public=True,
    cpu=8,
    memory="24Gi",
    shm="16Gi",
    script=[
        f"git clone --branch {BRANCH} "
        "https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        f"set +x; export MLFLOW_TRACKING_USERNAME='{MLFLOW_USER}'; "
        f"export MLFLOW_TRACKING_PASSWORD='{MLFLOW_TOKEN}'; set -x",
        "uv sync --frozen",
        "uv run python -m preprocessing.tissue_masks +dataset=processed "
        f"++dataset.mlflow_uris.dataset={DATASET_ARTIFACT_URI}",
    ],
    storage=[storage.public.DATA],
)
