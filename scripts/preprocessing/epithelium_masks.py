#!/usr/bin/env python
#: Submit the EPITHELIUM SEGMENTATION stage for TCIA_HunCRC.
#:
#: Same hunCRC setup as quality_control.py / tissue_masks.py / annotation_masks.py --
#: public cloud, BRANCH pinned, MLflow token exported into the job. Replaces the IBD
#: template this file was (username "...", public=False, storage.secure.DATA, a
#: "+dataset=..." placeholder that made the file unrunnable as committed).
#:
#: BOTH storage mounts are required here, which is what distinguishes this stage:
#:   * public.DATA -- the payload hands `slide_path` strings from dataset.csv to the model
#:     service, which opens the slide ITSELF. Those strings are /mnt/data/Public/... (the
#:     prefix configs/dataset/raw.yaml:slides_path produces), and the model pod resolves
#:     them against its own mount of the same rationai-data-ro-pvc. public.DATA is also
#:     what the local download_artifacts of the dataset csv reads from.
#:   * public.PROJECTS -- TWO reasons, and the second one is fatal if missed:
#:       1. output_dir is ${project_path}/epithelium_masks/raw (configs/preprocessing/
#:          epithelium_masks.yaml), and /mnt/projects is the only read-write mount;
#:       2. preprocessing/epithelium_masks.py:79 downloads the tissue masks with
#:          dst_path=config.project_path, then passes tissue_mask_path to the service. Both
#:          paths must be visible to the SERVICE, so they cannot live in a pod-local tmpdir
#:          the way preprocessing/tissue_masks.py does. That is the same reason QC needs
#:          PROJECTS and tissue masks do not.
#:
#: The IBD template's storage.secure.DATA is wrong twice over: secure is the Sensitive
#: Cloud, a certified environment for patient data whose VPN tunnel endpoints must not be
#: exposed to public traffic, and hunCRC lives on the PUBLIC rationai/Data share anyway.
#:
#: PREREQUISITE -- this stage cannot start before the tissue-mask job finishes.
#: preprocessing/epithelium_masks.py:76 downloads config.dataset.mlflow_uris.tissue and
#: indexes it by slide name (:34), so it needs the hunCRC `tissue_masks` artifact, which
#: does not exist until scripts/preprocessing/tissue_masks.py has run and you have its run
#: id. The URI is a REQUIRED env var below on purpose -- configs/dataset/
#: processed_w_tissue.yaml still carries the IBD cohort's tissue artifact (experiment 111),
#: so with a default here a silent run would segment the wrong cohort's slides and log
#: plausible-looking masks. Export it as
#:   export TISSUE_ARTIFACT_URI=mlflow-artifacts:/<exp>/<run>/artifacts/tissue_masks
#:
#: `++` (not `+`) on the two mlflow_uris overrides: +dataset=processed already creates
#: dataset.mlflow_uris.dataset, and processed_w_tissue.yaml adds .tissue. `+` would crash
#: on "key already exists". `+dataset=raw` is not an option -- raw.yaml defines no
#: mlflow_uris at all, so neither key exists to override.
#:
#: SIZING. The job is request-bound: it POSTs one heatmap-builder job per slide and waits
#: (AsyncClient timeout=3000 s, preprocessing/epithelium_masks.py:17). The IBD template's
#: cpu=8/16Gi is kept; shm=16Gi likewise (ray's object store on /dev/shm). max_concurrent
#: is 1 in the config, so this is serial and slow -- with QC at ~3.6 h/slide on the same
#: kind of 321 Gpx slides, raise max_concurrent deliberately, not here.
#:
#: only_missing: true in the config makes a re-run resume from output_dir, so a crashed
#: run is restarted rather than salvaged. Note the log_artifacts at :90 only happens at
#: the END, so an interrupted run logs nothing even though output_dir has masks on disk --
#: they survive on PROJECTS and the next run uploads them.
#:
#: Needs exported in the shell that runs this:
#:   export MLFLOW_TRACKING_PASSWORD=<access key>
#:   export TISSUE_ARTIFACT_URI=mlflow-artifacts:/<exp>/<run>/artifacts/tissue_masks
#:   export DATASET_ARTIFACT_URI=...   (optional; defaults to the hunCRC create_dataset run)
#:
#: The job clones configs/ and preprocessing/ from GitHub at BRANCH, so a local edit to
#: configs/preprocessing/epithelium_masks.yaml (the new `model:` key) or to the payload has
#: NO EFFECT until pushed. base.yaml's uncommitted hunCRC paths do NOT need to be pushed,
#: because project_path is overridden on the command line below and every output path here
#: derives from it. Without that override the pushed base.yaml would send both the tissue
#: download and output_dir to /mnt/projects/inflammatory_bowel_disease/... -- the IBD
#: project, absent on this cluster, so it would fail loudly rather than silently. Override
#: it anyway.
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
TISSUE_ARTIFACT_URI = os.environ.get("TISSUE_ARTIFACT_URI")

if not MLFLOW_TOKEN:
    raise SystemExit(
        "MLFLOW_TRACKING_PASSWORD is unset. Mint an access key in the MLflow UI "
        "(Permission -> create access key) and export it. A login password is rejected."
    )
if not TISSUE_ARTIFACT_URI:
    raise SystemExit(
        "TISSUE_ARTIFACT_URI is unset, and there is no safe default: "
        "configs/dataset/processed_w_tissue.yaml still points at the IBD cohort's tissue "
        "masks (mlflow-artifacts:/111/...), which would silently segment the wrong slides. "
        "Run the tissue-mask job first, then export the run id it prints as "
        "mlflow-artifacts:/<exp>/<run>/artifacts/tissue_masks."
    )

submit_job(
    job_name="ulcerative-colitis-dysplasia-epithelium-masks-huncrc",
    username="borisim",
    image="cerit.io/rationai/base:2.0.6",
    public=True,
    cpu=8,
    memory="16Gi",
    shm="16Gi",
    script=[
        f"git clone --branch {BRANCH} "
        "https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        f"set +x; export MLFLOW_TRACKING_USERNAME='{MLFLOW_USER}'; "
        f"export MLFLOW_TRACKING_PASSWORD='{MLFLOW_TOKEN}'; set -x",
        "uv sync --frozen",
        "uv run python -m preprocessing.epithelium_masks +dataset=processed_w_tissue "
        f"++dataset.mlflow_uris.dataset={DATASET_ARTIFACT_URI} "
        f"++dataset.mlflow_uris.tissue={TISSUE_ARTIFACT_URI} "
        "project_path=/mnt/projects/huncrc_dysplasia",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
