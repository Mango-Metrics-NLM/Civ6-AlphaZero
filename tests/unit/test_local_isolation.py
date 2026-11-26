"""Fail-closed GCP isolation for local runs (LF-E5)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from alpha_zero.cloud.gcp.storage import GCSBackend, LocalBackend, get_backend
from alpha_zero.local.isolation import (
    LocalModeViolation,
    enter_local_mode,
    is_local_mode,
)

SENSITIVE = {
    "GOOGLE_APPLICATION_CREDENTIALS": "/x/adc.json",
    "GOOGLE_CLOUD_PROJECT": "prod",
    "VERTEX_REGION": "us",
    "CIV6AZ_GCS_BUCKET": "prod-bucket",
    "CIV6AZ_CONTAINER_IMAGE_URI": "gcr.io/x",
    "GITHUB_TOKEN": "t",
    "WANDB_API_KEY": "k",
    "SSLKEYLOGFILE": "/tmp/k",
    "CIV6AZ_ALLOW_UNSAFE_CHECKPOINT": "1",
}


@pytest.fixture
def cloud_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in SENSITIVE.items():
        monkeypatch.setenv(k, v)


@pytest.fixture
def restore_environ():
    """Snapshot os.environ and restore it exactly (enter_local_mode mutates it)."""
    snapshot = os.environ.copy()
    yield snapshot
    os.environ.clear()
    os.environ.update(snapshot)


@pytest.fixture
def local_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIV6AZ_LOCAL_RUN", "1")


def test_gcsbackend_ctor_raises(cloud_env: None, local_mode: None) -> None:
    with pytest.raises(LocalModeViolation):
        GCSBackend("prod")


def test_get_backend_gcs_branch_raises(
    cloud_env: None, local_mode: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CIV6AZ_STORAGE_BACKEND", "gcs")
    with pytest.raises(LocalModeViolation):
        get_backend("prod")


def test_client_property_raises(cloud_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = GCSBackend("prod")  # local mode off: constructible
    monkeypatch.setenv("CIV6AZ_LOCAL_RUN", "1")
    with pytest.raises(LocalModeViolation):
        _ = backend.client


def test_except_exception_does_not_swallow(cloud_env: None, local_mode: None) -> None:
    def swallow() -> None:
        try:
            GCSBackend("prod")
        except Exception:
            return

    with pytest.raises(LocalModeViolation):
        swallow()
    assert not issubclass(LocalModeViolation, Exception)


def test_local_mode_off_is_constructible(cloud_env: None) -> None:
    assert not is_local_mode()
    assert isinstance(GCSBackend("prod"), GCSBackend)


def test_local_backend_still_works_in_local_mode(
    local_mode: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CIV6AZ_STORAGE_BACKEND", "local")
    assert isinstance(get_backend(), LocalBackend)


def test_enter_local_mode_child_env(
    cloud_env: None, tmp_path: Path, restore_environ: dict[str, str]
) -> None:
    for k in ("CIV6AZ_STORAGE_BACKEND", "CIV6AZ_SKIP_DOTENV", "CIV6AZ_RUN_ID", "CLOUDSDK_CONFIG"):
        os.environ.pop(k, None)
    os.environ.pop("NO_GCE_CHECK", None)
    os.environ["AIP_CHECKPOINT_DIR"] = "gs://x/y"

    child = enter_local_mode("run1", tmp_path)

    for name in child:
        up = name.upper()
        assert not up.startswith(("GOOGLE_", "VERTEX_", "AIP_")), name
        assert not up.endswith(("_TOKEN", "_KEY")), name
        assert "CONTAINER_IMAGE_URI" not in up, name
    for banned in ("CIV6AZ_GCS_BUCKET", "SSLKEYLOGFILE", "CIV6AZ_ALLOW_UNSAFE_CHECKPOINT"):
        assert banned not in child
    assert child["CLOUDSDK_CONFIG"] == str(tmp_path / "gcloud-empty")
    assert (tmp_path / "gcloud-empty").is_dir()
    assert child["NO_GCE_CHECK"] == "True"
    assert child["CIV6AZ_STORAGE_BACKEND"] == "local"
    assert child["CIV6AZ_LOCAL_RUN"] == "1"
    assert child["CIV6AZ_SKIP_DOTENV"] == "1"
    assert child["CIV6AZ_RUN_ID"] == "run1"
    # current process: GCP vars popped, local flags set ...
    assert is_local_mode()
    for gone in ("CIV6AZ_GCS_BUCKET", "GOOGLE_APPLICATION_CREDENTIALS", "AIP_CHECKPOINT_DIR"):
        assert gone not in os.environ
    # ... but secrets and unsafe-checkpoint opt-in are NOT popped from the parent
    for kept in (
        "GITHUB_TOKEN",
        "WANDB_API_KEY",
        "SSLKEYLOGFILE",
        "CIV6AZ_ALLOW_UNSAFE_CHECKPOINT",
    ):
        assert kept in os.environ
    with pytest.raises(LocalModeViolation):
        GCSBackend("prod")


def test_restore_environ_leaves_no_trace(cloud_env: None, tmp_path: Path) -> None:
    """Run enter_local_mode under the restore helper; environ must be identical after."""
    before = os.environ.copy()
    snapshot = os.environ.copy()
    try:
        enter_local_mode("r", tmp_path)
        assert os.environ != before
    finally:
        os.environ.clear()
        os.environ.update(snapshot)
    assert dict(os.environ) == before


def test_gcs_storage_backend_raises_in_local_mode(local_mode: None) -> None:
    from alpha_zero.storage.config import StorageConfig
    from alpha_zero.storage.gcs_backend import GCSStorageBackend

    with pytest.raises(LocalModeViolation):
        GCSStorageBackend(StorageConfig.for_gcs("bucket"))


def test_aip_checkpoint_dir_cannot_switch_local_to_gcs(
    local_mode: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from alpha_zero.storage import StorageConfig, create_storage_backend

    monkeypatch.setenv("AIP_CHECKPOINT_DIR", "gs://x/y")
    with pytest.raises(LocalModeViolation):
        create_storage_backend(StorageConfig.for_local(str(tmp_path)))


def test_skip_dotenv_honoured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from alpha_zero.utils.env_loader import load_civ6az_env

    env_file = tmp_path / ".env"
    env_file.write_text("CIV6AZ_GCS_BUCKET=from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("CIV6AZ_GCS_BUCKET", raising=False)
    monkeypatch.setenv("CIV6AZ_SKIP_DOTENV", "1")
    load_civ6az_env(env_file, warn_missing=False)
    assert "CIV6AZ_GCS_BUCKET" not in os.environ
    # negative control: without the switch the file is loaded
    monkeypatch.delenv("CIV6AZ_SKIP_DOTENV")
    load_civ6az_env(env_file, warn_missing=False)
    assert os.environ["CIV6AZ_GCS_BUCKET"] == "from-dotenv"
    monkeypatch.delenv("CIV6AZ_GCS_BUCKET")


def test_isolation_module_is_stdlib_only() -> None:
    import ast

    src = (Path(__file__).parents[2] / "alpha_zero" / "local" / "isolation.py").read_text(
        encoding="utf-8"
    )
    roots = {
        n.names[0].name.split(".")[0] if isinstance(n, ast.Import) else (n.module or "")
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Import | ast.ImportFrom)
    }
    assert not roots & {"torch", "numpy", "alpha_zero"}


# --- A backend built BEFORE local mode must not keep an open path to GCS ----


class _FakeBlob:
    def __init__(self, store: dict[str, bytes], name: str) -> None:
        self._store, self.name = store, name

    def upload_from_string(self, data: bytes, **_: object) -> None:
        self._store[self.name] = data

    def download_as_bytes(self, **_: object) -> bytes:
        return self._store[self.name]

    def exists(self) -> bool:
        return self.name in self._store

    def delete(self) -> None:
        self._store.pop(self.name, None)


class _FakeBucket:
    def __init__(self, store: dict[str, bytes]) -> None:
        self._store = store

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(self._store, name)


class _FakeClient:
    def __init__(self, **_: object) -> None:
        self.store: dict[str, bytes] = {}

    def bucket(self, _name: str) -> _FakeBucket:
        return _FakeBucket(self.store)

    def list_blobs(self, _bucket: str, **_kw: object) -> list[_FakeBlob]:
        return [_FakeBlob(self.store, n) for n in self.store]

    def close(self) -> None:
        pass


@pytest.fixture
def prebuilt_gcs_backend(monkeypatch: pytest.MonkeyPatch):
    """A GCSStorageBackend constructed with local mode OFF, using a fake client."""
    from types import SimpleNamespace

    from alpha_zero.storage import gcs_backend as mod
    from alpha_zero.storage.config import StorageConfig

    monkeypatch.setattr(mod, "HAS_GCS", True)
    monkeypatch.setattr(mod, "gcs", SimpleNamespace(Client=_FakeClient))
    monkeypatch.setattr(mod, "gcs_retry", SimpleNamespace(Retry=lambda **_: None), raising=False)
    backend = mod.GCSStorageBackend(StorageConfig.for_gcs("bucket"))
    # Sanity: it works while local mode is off (so the raises below are the guard).
    backend.save_config({"a": 1}, 1)
    assert backend.load_config(1) == {"a": 1}
    return backend


@pytest.mark.parametrize(
    "call",
    [
        lambda b: b.save_model({}, 1),
        lambda b: b.load_model(1),
        lambda b: b.list_files("models"),
        lambda b: b.exists("models/x"),
        lambda b: b.delete("models/x"),
        lambda b: b.save_config({"a": 1}, 2),
        lambda b: b.load_config(1),
        lambda b: b.list_checkpoints(),
        lambda b: b.close(),
    ],
    ids=[
        "save_model",
        "load_model",
        "list_files",
        "exists",
        "delete",
        "save_config",
        "load_config",
        "list_checkpoints",
        "close",
    ],
)
def test_prebuilt_gcs_storage_backend_io_raises_after_local_mode(
    prebuilt_gcs_backend, monkeypatch: pytest.MonkeyPatch, call
) -> None:
    monkeypatch.setenv("CIV6AZ_LOCAL_RUN", "1")
    with pytest.raises(LocalModeViolation):
        call(prebuilt_gcs_backend)


def test_cloud_gcp_backend_cached_client_is_guarded(
    cloud_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cloud/gcp/storage.py: a client cached before local mode is not reusable."""
    backend = GCSBackend("prod")
    backend._client = _FakeClient()  # type: ignore[assignment]
    assert backend.upload_blob("b", "p", b"x").startswith("gs://")
    monkeypatch.setenv("CIV6AZ_LOCAL_RUN", "1")
    for op in (
        lambda: backend.upload_blob("b", "p", b"x"),
        lambda: backend.download_blob("b", "p"),
        lambda: backend.blob_exists("b", "p"),
        lambda: backend.list_blobs("b", "p"),
        lambda: backend.delete_blob("b", "p"),
    ):
        with pytest.raises(LocalModeViolation):
            op()


def test_git_sha_passes_through_to_the_child(
    cloud_env: None, tmp_path: Path, restore_environ: dict[str, str]
) -> None:
    """``CIV6AZ_GIT_SHA`` is how a pinned child learns its commit (T10 review I1).

    A pinned child runs from a ``git archive`` export with no ``.git``, so if the
    allowlist ever swallowed this name every checkpoint would record ``git_sha: null``.
    """
    os.environ["CIV6AZ_GIT_SHA"] = "0123456789abcdef0123456789abcdef01234567"
    child = enter_local_mode("run1", tmp_path)
    assert child["CIV6AZ_GIT_SHA"] == "0123456789abcdef0123456789abcdef01234567"
