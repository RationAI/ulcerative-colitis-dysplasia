# TODO

## 1. Make provenance logging automatic (zero per-script code)

**Goal:** provenance should be logged the way MLflow logging is — the user writes
nothing per script (no callback, no explicit `log_provenance(...)` call); it just
happens for every run.

**How MLflow auto-logging works today (findings):**
- `configs/base.yaml` → `logger: mlflow` → `configs/logger/mlflow.yaml` instantiates
  `rationai.mlkit.lightning.loggers.MLFlowLogger` (subclass of lightning's
  MLFlowLogger; starts the run lazily on first `experiment` access, stamps git
  commit/branch/repo in `__init__`).
- `@autolog` (`rationai/mlkit/autolog.py`) is the only per-script code: instantiates
  `config.logger`, logs the config files as artifacts, logs
  `metadata.hyperparams`, captures stdout/stderr to `console.log`.
- The kube-jobs submitter injects `MLFLOW_TRACKING_URI` + credentials + `JOB_USER`
  env vars into every job.
- ⇒ every run flows through ONE choke point: mlkit's `@autolog` + `MLFlowLogger`.
  Entry points (e.g. `ml/__main__.py`) contain zero mlflow imports.

**Why provenance is not automatic today:**
- Requires an explicit per-script `log_provenance(output=..., config=..., inputs=[...])`
  — see `provenance/training.py:235`: the output frame AND the full ancestor run
  list are passed by hand.
- Lightning callbacks exist in the lib (`EnvironmentCallback`, `ProvenanceCallback`,
  `DatasetVerificationCallback` in `rationai/provenance/lightning/`) but must be
  added to `Trainer(callbacks=[...])` by hand, and `ProvenanceCallback` is
  training-shaped (split re-computation, checkpoint/optimizer summaries, dataset
  verification that assumes a manifest layout the production preprocessing outputs
  don't have).

**Plan (recommended: hook `@autolog`):**
1. In `rationai/mlkit/autolog.py`, after `_log_config(config, logger)` (the run is
   active by then), add an opt-in block (config key `metadata.provenance` or env
   `RATIONAI_PROVENANCE`): call `log_provenance(config=config, output=<entry
   function return value>)`, wrapped in `try/except ImportError` so mlkit still
   works where the provenance lib is not installed.
2. Convention change: entry functions should `return` their output DataFrame (they
   currently return `None`) so frame stats + manifests get logged; `None` → skip
   frame stats (env + config leaves + input URIs are still logged — that is the
   90 % of the value, and `input_uris` is the chain-edge data the PROV document is
   built from; it is auto-detected from `mlflow-artifacts:/` / `runs:/` URIs in the
   config, which production configs already carry).
3. Extend `_find_input_uris` (`rationai/provenance/log_provenance.py`) to turn bare
   run-id config fields (e.g. `dataset.run_id: <32-hex>`) into `runs:/<id>` inputs —
   only URI strings are detected today.
4. Decide where the change lands: a patch to the mlkit source repo (mlkit is a pip
   install in this venv's site-packages, so a real fix means a version bump there),
   vs. a repo-local wrapper decorator that replaces the
   `from rationai.mlkit import autolog` line in the ~15 entry scripts (works in this
   repo immediately, touches every entry point).

**Rejected:** global monkey-patch of `mlflow.start_run` — global side effects,
ordering fragility, double-logs when the callbacks are also active.

**Open questions:**
- Opt-in gate: config key vs. env var?
- Output-frame return convention: accepted (touches ~15 scripts) or is
  env + config + inputs sufficient?
- Patch lands in mlkit (needs a release) or repo-local wrapper?
