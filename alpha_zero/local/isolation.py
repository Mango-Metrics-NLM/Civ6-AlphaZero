"""Fail-closed isolation of local runs from GCP (LF-E5).

A "local" run on a developer workstation can see Application Default
Credentials, a user-scope ``CIV6AZ_GCS_BUCKET`` and an auto-loaded ``.env``.
Left alone, it could silently write to production GCS. This module makes GCS
construction *impossible* under local mode rather than merely unlikely:

* :func:`enter_local_mode` scrubs the cloud-reachable variables, points the
  gcloud config at an empty directory, and returns the child environment:
  the current one minus a deny list (``*_KEY``, ``*_TOKEN``, ...). It is not
  an allowlist; ``*_SECRET``, ``*_PASSWORD``, ``*_CREDENTIALS`` and ``*_PAT``
  pass through to the pinned child (final review M-7).
* :func:`guard_gcs` is called by ``GCSBackend.__init__`` and its lazy
  ``client`` property and raises :class:`LocalModeViolation`.

Stdlib-only on purpose: ``alpha_zero.cloud.gcp.storage`` imports this module,
so it must not pull in torch, numpy or ``alpha_zero.utils``.
"""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Mapping
from pathlib import Path

LOCAL_RUN_ENV = "CIV6AZ_LOCAL_RUN"
SKIP_DOTENV_ENV = "CIV6AZ_SKIP_DOTENV"

# Popped from the CURRENT process (the spec's GCP list, plus Vertex AIP_*).
_PARENT_POP_PATTERNS: tuple[str, ...] = (
    "GOOGLE_*",
    "VERTEX_*",
    "CIV6AZ_VERTEX_*",
    "CIV6AZ_GCS_*",
    "CIV6_GCS_*",
    "*CONTAINER_IMAGE_URI*",
    "AIP_*",
    "GCLOUD_*",
    "GCP_*",
    "GCS_*",
    "CLOUDSDK_*",
    "NO_GCE_CHECK",
)

# Additionally withheld from the returned CHILD environment only.
_CHILD_ONLY_DENY_PATTERNS: tuple[str, ...] = (
    "*_TOKEN",
    "*_KEY",
    "SSLKEYLOGFILE",
    "CIV6AZ_ALLOW_UNSAFE_CHECKPOINT",
)


class LocalModeViolation(BaseException):
    """Raised when local mode would reach GCP.

    Subclasses ``BaseException`` so the ``except Exception`` blocks around
    state saves (``orchestrator_state``, ``vertex_orchestrator``) cannot
    swallow it and carry on as if nothing happened.
    """


def is_local_mode(env: Mapping[str, str] | None = None) -> bool:
    """Return True when ``CIV6AZ_LOCAL_RUN=1`` in ``env`` (default ``os.environ``)."""
    source = os.environ if env is None else env
    return source.get(LOCAL_RUN_ENV) == "1"


def guard_gcs(what: str = "GCS") -> None:
    """Raise :class:`LocalModeViolation` if the process is in local mode."""
    if is_local_mode():
        raise LocalModeViolation(
            f"{what} is unreachable under {LOCAL_RUN_ENV}=1: a local run must never touch GCP"
        )


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    upper = name.upper()
    return any(fnmatch.fnmatchcase(upper, pat) for pat in patterns)


def enter_local_mode(run_id: str, run_dir: str | Path) -> dict[str, str]:
    """Put this process into local mode and return the child environment.

    The current ``os.environ`` is mutated: only the GCP variables (GOOGLE_*,
    VERTEX_*, CIV6AZ_GCS_*, *CONTAINER_IMAGE_URI*, AIP_*, ...) are popped and
    the local flags set, so in-process code is fenced. ``*_KEY``, ``*_TOKEN``,
    ``SSLKEYLOGFILE`` and ``CIV6AZ_ALLOW_UNSAFE_CHECKPOINT`` stay in the current
    process and are removed from the returned child environment only. The
    returned dict is the environment for subprocesses: a deny list, not an allowlist, so
    any other variable passes through. ``run_id`` is recorded as
    ``CIV6AZ_RUN_ID``.
    """
    run_path = Path(run_dir)
    empty_gcloud = run_path / "gcloud-empty"
    empty_gcloud.mkdir(parents=True, exist_ok=True)

    for name in [n for n in os.environ if _matches(n, _PARENT_POP_PATTERNS)]:
        os.environ.pop(name, None)

    overrides = {
        "CLOUDSDK_CONFIG": str(empty_gcloud),
        "NO_GCE_CHECK": "True",
        "CIV6AZ_STORAGE_BACKEND": "local",
        LOCAL_RUN_ENV: "1",
        SKIP_DOTENV_ENV: "1",
        "CIV6AZ_RUN_ID": run_id,
    }
    os.environ.update(overrides)

    child = {
        k: v
        for k, v in os.environ.items()
        if not _matches(k, _PARENT_POP_PATTERNS + _CHILD_ONLY_DENY_PATTERNS)
    }
    child.update(overrides)
    return child
