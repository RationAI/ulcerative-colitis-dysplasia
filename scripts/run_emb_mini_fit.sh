#!/usr/bin/env bash
#: Local CPU smoke test for the hunCRC embedding classifier (2 epochs,
#: 3 slides per split, checkpoints to /tmp/emb_huncrc_test). Run from the
#: repo root:
#:
#:     bash scripts/run_emb_mini_fit.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export MLFLOW_TRACKING_URI=https://mlflow.rationai.cloud.e-infra.cz
export MLFLOW_TRACKING_USERNAME=519192@mail.muni.cz
export MLFLOW_TRACKING_PASSWORD="$(cat ~/.mlflow_key)"

.venv/bin/python -m ml +experiment=embeddings_huncrc \
  data.num_workers=0 data.batch_size=128 trainer.max_epochs=2 \
  ++dataset.max_slides=3 trainer.callbacks.0.dirpath=/tmp/emb_huncrc_test
