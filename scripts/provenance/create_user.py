#: Register YOURSELF in the User_Registry experiment — once, ever.
#:
#: Run as a kube-job: JOB_USER comes from the submitter (new kube-jobs derives
#: it from YOUR kube config; our submitter exports it explicitly), so the
#: username can never be a typo (provenance looks up every later run by
#: exactly this name — an unregistered user fails at startup).
#:
#:     uv run python -m scripts.provenance.create_user
#:
#: Needs MLflow credentials as usual: MLFLOW_TRACKING_USERNAME /
#: MLFLOW_TRACKING_PASSWORD.
import os

import mlflow
from rationai.provenance import register_new_user
from rationai.provenance.dataset import lookup_experiment


def main() -> None:
    username = os.environ.get("JOB_USER") or os.environ.get("MLFLOW_USER")
    if not username:
        raise SystemExit(
            "JOB_USER is unset — this must run as a kube-job (kube-jobs fills "
            "it from your kube config). Do not hardcode a username."
        )

    # once, ever — skip if this username already has a User_Registry run
    exp_id = lookup_experiment("User_Registry")
    if exp_id is not None:
        runs = mlflow.search_runs(experiment_ids=[exp_id], output_format="pandas")
        if "tags.username" in runs.columns and (runs["tags.username"] == username).any():
            print(f"[create_user] '{username}' already registered — nothing to do")
            return

    register_new_user(
        username=username,
        real_name="Šimon Bořil",
        email="519192@mail.muni.cz",
        organization="RationAI",
        lead_name="Tomáš Brázdil",
        lead_email="brazdil@muni.cz",
    )


if __name__ == "__main__":
    main()
