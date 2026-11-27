"""Optional, never-fatal MLflow tracking for the HunCRC tile classifier.

The RationAI MLflow server sits behind an OIDC login: a request without an
access token gets the login page. A Kubernetes job has no token, and a
notebook has one only when it is typed in. So for the tile classifier MLflow
is a best-effort extra:

- :class:`OptionalMLFlowLogger` (a subclass of the house ``MLFlowLogger``, so
  ``@autolog`` keeps working) checks for credentials and makes one probe call
  when it is created. Without credentials, or if the server refuses or times
  out, it turns itself off and every later call is a no-op. The same happens
  on the first failure in the middle of a run. Training never sees an
  exception from it; results always go to disk.
- :func:`setup_tracking` picks the credentials (typed via getpass with
  ``login``, else MLFLOW_TRACKING_TOKEN, else username + password from the
  environment) and puts them in this process's environment only. Values are
  never printed, logged, stored or put in a config.
- :func:`find_secrets` refuses a config (or Hydra overrides) that could carry
  a credential, because ``@autolog`` uploads the config as an artifact.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from rationai.mlkit.lightning.loggers import MLFlowLogger

from ml.huncrc_mlflow_auth import (
    AUTH_VARS,
    PUBLIC_MLFLOW,
    AuthMissingError,
    find_secrets,
    mask,
    redact_uri,
    setup_tracking,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

__all__ = [
    "AUTH_VARS",
    "PUBLIC_MLFLOW",
    "AuthMissingError",
    "OptionalMLFlowLogger",
    "find_secrets",
    "mask",
    "redact_uri",
    "setup_tracking",
]


def _say(msg: str) -> None:
    # the real stderr: sys.stdout/stderr may be captured and sent to MLflow
    print(f"[mlflow] {msg}", file=sys.__stderr__, flush=True)


def _safe(method: Callable[..., Any]) -> Callable[..., Any]:
    """Run an MLflow call only while active; the first failure turns logging off."""

    def wrapper(self: OptionalMLFlowLogger, *args: Any, **kwargs: Any) -> Any:
        if not self.active:
            return None
        try:
            return method(self, *args, **kwargs)
        except Exception as e:
            self.disable(
                f"{method.__name__} failed: {type(e).__name__}: {mask(str(e))[:300]}"
            )
            return None

    wrapper.__name__ = method.__name__
    wrapper.__doc__ = method.__doc__
    return wrapper


class OptionalMLFlowLogger(MLFlowLogger):
    """House MLFlowLogger that never raises: off without credentials or server.

    ``enabled=False`` never touches the network (cluster jobs). Checkpoints are
    never uploaded (``log_model`` is forced to False).
    """

    def __init__(
        self,
        *,
        tracking_uri: str = PUBLIC_MLFLOW,
        enabled: bool = True,
        login: bool = False,
        timeout_s: int = 20,
        log_model: Any = False,
        **kwargs: Any,
    ) -> None:
        self.active = False
        self.reason = "disabled in the config (mlflow.enabled=false)"
        self.auth = "not used"
        self._logged_steps: set[tuple[str, int]] = set()
        if enabled:
            try:
                self.auth = setup_tracking(tracking_uri, login, timeout_s=timeout_s)
                self.reason = ""
            except AuthMissingError as e:
                self.reason = str(e)
        if log_model:
            _say("log_model ignored: checkpoints are never uploaded")
        super().__init__(tracking_uri=tracking_uri, log_model=False, **kwargs)
        if not self.reason:
            try:
                self.active = True
                _ = super().experiment  # creates the run (and experiment if new)
                _ = self.run_id
            except Exception as e:
                self.active = False
                self.reason = (
                    f"server not usable: {type(e).__name__}: {mask(str(e))[:300]}"
                )
        if self.active:
            _say(
                f"logging to {redact_uri(tracking_uri)} run {self.run_id}; credentials: {self.auth}"
            )
        else:
            _say(f"logging OFF, results go to disk only: {self.reason}")

    def disable(self, reason: str) -> None:
        if self.active:
            self.active = False
            self.reason = reason
            _say(f"logging turned OFF for the rest of the run: {reason}")

    # every method that talks to the server is wrapped
    log_metrics = _safe(MLFlowLogger.log_metrics)
    log_hyperparams = _safe(MLFlowLogger.log_hyperparams)
    log_stream = _safe(MLFlowLogger.log_stream)
    log_artifact = _safe(MLFlowLogger.log_artifact)
    log_artifacts = _safe(MLFlowLogger.log_artifacts)
    log_table = _safe(MLFlowLogger.log_table)
    finalize = _safe(MLFlowLogger.finalize)
    after_save_checkpoint = _safe(MLFlowLogger.after_save_checkpoint)

    @_safe
    def set_tags(self, tags: dict[str, str]) -> None:
        for k, v in tags.items():
            self.experiment.set_tag(self.run_id, k, v)

    @_safe
    def log_series(self, name: str, points: Iterable[tuple[int, float]]) -> None:
        """Stepped metric; a (name, step) already logged in this run is skipped."""
        from mlflow.entities import Metric

        new = [(s, v) for s, v in points if (name, s) not in self._logged_steps]
        if not new:
            return
        ts = _now_ms()
        self.experiment.log_batch(
            self.run_id, metrics=[Metric(name, float(v), ts, int(s)) for s, v in new]
        )
        self._logged_steps.update((name, s) for s, _ in new)

    @_safe
    def log_text_artifact(self, text: str, name: str) -> None:
        self.experiment.log_text(self.run_id, text, name)


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)
