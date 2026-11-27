"""MLflow credential handling with no mlflow import (safe for dry runs).

:func:`setup_tracking` picks the credentials (typed via getpass with
``login``, else MLFLOW_TRACKING_TOKEN, else username + password from the
environment) and puts them in this process's environment only. Values are
never printed, logged, stored or put in a config. :func:`find_secrets`
refuses config entries that could carry a credential.
"""

from __future__ import annotations

import getpass
import os
import re
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


PUBLIC_MLFLOW = "https://mlflow.rationai.cloud.e-infra.cz"
AUTH_VARS = ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD",
             "MLFLOW_TRACKING_TOKEN")  # fmt: skip
_SECRETS: list[str] = []  # in memory only, to mask them in messages

SECRET_KEY = re.compile(r"(?i)(password|passwd|token|secret|credential|api_?key)")


class AuthMissingError(RuntimeError):
    """No usable MLflow credentials in this process."""


def redact_uri(uri: str) -> str:
    return re.sub(r"//[^/@]+@", "//***@", uri)


def mask(text: str) -> str:
    for secret in _SECRETS:
        text = text.replace(secret, "***")
    return re.sub(r"(?i)(bearer|token|password)\S*\s*\S+", r"\1 ***", text)


def _needs_auth(uri: str) -> bool:
    return uri.startswith(("http://", "https://"))


def setup_tracking(
    uri: str,
    login: bool,
    ask: Callable[[str], str] = input,
    ask_hidden: Callable[[str], str] = getpass.getpass,
    timeout_s: int = 20,
) -> str:
    """Point this process at ``uri`` and pick the credentials; returns which.

    Only os.environ of this process changes. Nothing is printed except the
    prompts; callers report the returned description, never a value.
    Raises :class:`AuthMissingError` when an http(s) server has no credentials.
    """
    os.environ["MLFLOW_TRACKING_URI"] = uri
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", str(timeout_s))
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    if not _needs_auth(uri):
        return "none needed (local store)"
    if login:
        user = ask("MLflow login email: ").strip()
        secret = ask_hidden("MLflow API access token (hidden; not your SSO password): ")
        if not user or not secret:
            raise AuthMissingError("login email and access token are required")
        _SECRETS.append(secret)
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = secret
        os.environ.pop("MLFLOW_TRACKING_TOKEN", None)
        return "username + password typed at the prompt (--login)"
    if os.environ.get("MLFLOW_TRACKING_TOKEN"):
        _SECRETS.append(os.environ["MLFLOW_TRACKING_TOKEN"])
        os.environ.pop("MLFLOW_TRACKING_USERNAME", None)
        os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)
        return "MLFLOW_TRACKING_TOKEN from the environment"
    if os.environ.get("MLFLOW_TRACKING_USERNAME") and os.environ.get(
        "MLFLOW_TRACKING_PASSWORD"
    ):
        _SECRETS.append(os.environ["MLFLOW_TRACKING_PASSWORD"])
        return "MLFLOW_TRACKING_USERNAME + _PASSWORD from the environment"
    raise AuthMissingError(
        "no MLflow credentials (a username alone is not enough); use login in a "
        "Jupyter terminal or set MLFLOW_TRACKING_TOKEN"
    )


def find_secrets(items: Iterable[tuple[str, Any]]) -> list[str]:
    """Keys or values that could carry a credential.

    Key names like *token*, and any ``oc.env`` interpolation (it could read
    MLFLOW_TRACKING_PASSWORD).
    """
    bad = []
    for key, value in items:
        leaf = key.rsplit(".", 1)[-1]
        if SECRET_KEY.search(leaf):
            bad.append(f"{key}: key name looks like a credential")
        if isinstance(value, str) and "oc.env" in value:
            bad.append(f"{key}: environment interpolation ({value})")
    return bad
