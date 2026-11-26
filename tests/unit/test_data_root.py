"""Data-root resolution and hardening (LF-E4).

Large artifacts must live under ``CIV6AZ_DATA_ROOT``, never in the OneDrive-synced
repo. Nothing here touches a real system path: roots are ``tmp_path`` children,
the Windows volume probes are patched, and ``icacls`` is a recording fake.

Negative controls performed when this file was written (each reverted):

* the inside-repo check removed from ``assert_safe_data_root`` ->
  ``test_refuses_inside_repo`` fails;
* the ``OneDrive*`` component check removed -> ``test_refuses_onedrive_named_component`` fails;
* the length check removed -> ``test_refuses_long_path`` fails.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from alpha_zero.utils import data_root as dr


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial", dr.ENV_VAR):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    return path


@pytest.fixture
def good_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("dr") / "data"


@pytest.fixture
def windows_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dr, "_is_windows", lambda: True)
    monkeypatch.setattr(dr, "_volume_filesystem", lambda path: "NTFS")
    monkeypatch.setattr(dr, "_is_removable_drive", lambda path: False)
    monkeypatch.setattr(dr, "_free_bytes", lambda path: 500 * 1024**3)


def test_resolve_unset_is_none() -> None:
    assert dr.resolve_data_root() is None


def test_resolve_required_raises_when_unset() -> None:
    with pytest.raises(dr.DataRootError, match=dr.ENV_VAR):
        dr.resolve_data_root(required=True)


def test_resolve_returns_absolute_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(dr.ENV_VAR, str(tmp_path / "x"))
    assert dr.resolve_data_root() == (tmp_path / "x").resolve()


def test_resolve_blank_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(dr.ENV_VAR, "   ")
    assert dr.resolve_data_root() is None


def test_storage_root_is_under_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(dr.ENV_VAR, str(tmp_path))
    assert dr.storage_root() == tmp_path.resolve() / "storage"


def test_storage_root_requires_root() -> None:
    with pytest.raises(dr.DataRootError):
        dr.storage_root()


@pytest.mark.parametrize("run_id", ["a", "train-42-20261111T000000Z", "A1._-x", "a" * 64])
def test_run_id_accepts(run_id: str) -> None:
    assert dr.validate_run_id(run_id) == run_id


@pytest.mark.parametrize("run_id", ["", "-a", ".a", "a/b", "a\\b", "a b", "a" * 65, "../x"])
def test_run_id_rejects(run_id: str) -> None:
    with pytest.raises(dr.DataRootError):
        dr.validate_run_id(run_id)


def test_default_run_id_format() -> None:
    now = datetime(2026, 11, 11, 1, 2, 3, tzinfo=UTC)
    assert dr.default_run_id(42, now=now) == "train-42-20261111T010203Z"
    assert dr.default_run_id(None, now=now) == "train-unseeded-20261111T010203Z"


def test_default_checkpoint_dir(tmp_path: Path) -> None:
    assert dr.default_checkpoint_dir(tmp_path, "r1") == tmp_path / "runs" / "r1" / "checkpoints"


def test_default_checkpoint_dir_validates_run_id(tmp_path: Path) -> None:
    with pytest.raises(dr.DataRootError):
        dr.default_checkpoint_dir(tmp_path, "../evil")


def test_accepts_good_root(good_root: Path, repo: Path, windows_ok: None) -> None:
    dr.assert_safe_data_root(good_root, repo)


def test_refuses_inside_repo(repo: Path, windows_ok: None) -> None:
    with pytest.raises(dr.DataRootError, match="repo"):
        dr.assert_safe_data_root(repo / "data", repo)


def test_refuses_repo_itself(repo: Path, windows_ok: None) -> None:
    with pytest.raises(dr.DataRootError, match="repo"):
        dr.assert_safe_data_root(repo, repo)


def test_refuses_under_onedrive_env(
    monkeypatch: pytest.MonkeyPatch, good_root: Path, repo: Path, windows_ok: None
) -> None:
    monkeypatch.setenv("OneDriveConsumer", str(good_root.parent))
    with pytest.raises(dr.DataRootError, match="OneDrive"):
        dr.assert_safe_data_root(good_root, repo)


def test_refuses_onedrive_named_component(tmp_path: Path, repo: Path, windows_ok: None) -> None:
    with pytest.raises(dr.DataRootError, match="OneDrive"):
        dr.assert_safe_data_root(tmp_path / "OneDrive - Contoso" / "d", repo)


def test_refuses_long_path(tmp_path: Path, repo: Path, windows_ok: None) -> None:
    with pytest.raises(dr.DataRootError, match="120"):
        dr.assert_safe_data_root(tmp_path / ("a" * 130), repo)


def test_refuses_non_ntfs(
    monkeypatch: pytest.MonkeyPatch, good_root: Path, repo: Path, windows_ok: None
) -> None:
    monkeypatch.setattr(dr, "_volume_filesystem", lambda path: "exFAT")
    with pytest.raises(dr.DataRootError, match="NTFS"):
        dr.assert_safe_data_root(good_root, repo)


def test_accepts_refs(
    monkeypatch: pytest.MonkeyPatch, good_root: Path, repo: Path, windows_ok: None
) -> None:
    monkeypatch.setattr(dr, "_volume_filesystem", lambda path: "ReFS")
    dr.assert_safe_data_root(good_root, repo)


def test_refuses_removable(
    monkeypatch: pytest.MonkeyPatch, good_root: Path, repo: Path, windows_ok: None
) -> None:
    monkeypatch.setattr(dr, "_is_removable_drive", lambda path: True)
    with pytest.raises(dr.DataRootError, match="removable"):
        dr.assert_safe_data_root(good_root, repo)


def test_warns_on_low_free_space(
    monkeypatch: pytest.MonkeyPatch,
    good_root: Path,
    repo: Path,
    windows_ok: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(dr, "_free_bytes", lambda path: 10 * 1024**3)
    with caplog.at_level(logging.WARNING):
        dr.assert_safe_data_root(good_root, repo)
    assert any("free" in r.message for r in caplog.records)


def test_non_windows_skips_volume_checks(
    monkeypatch: pytest.MonkeyPatch, good_root: Path, repo: Path
) -> None:
    monkeypatch.setattr(dr, "_is_windows", lambda: False)

    def boom(path: Path) -> str:
        raise AssertionError("volume probe must be skipped off Windows")

    monkeypatch.setattr(dr, "_volume_filesystem", boom)
    monkeypatch.setattr(dr, "_is_removable_drive", boom)
    monkeypatch.setattr(dr, "_free_bytes", lambda path: 500 * 1024**3)
    dr.assert_safe_data_root(good_root, repo)


def test_harden_creates_layout_and_calls_icacls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(dr, "_is_windows", lambda: True)
    monkeypatch.setenv("USERNAME", "tester")
    monkeypatch.setenv("USERDOMAIN", "BOX")
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    root = tmp_path / "data"
    dr.harden_data_root(root, runner=fake_run)
    for sub in dr.SUBDIRS:
        assert (root / sub).is_dir()
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == "icacls" and cmd[1] == str(root)
    assert "/inheritance:r" in cmd
    joined = " ".join(cmd)
    assert "*S-1-5-18" in joined and "*S-1-5-32-544" in joined and "BOX\\tester" in joined


def test_harden_failure_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dr, "_is_windows", lambda: True)

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 5, "", "denied")

    with pytest.raises(dr.DataRootError, match="icacls"):
        dr.harden_data_root(tmp_path / "d", runner=fake_run)


def test_scoped_temp_env(tmp_path: Path) -> None:
    env = dr.scoped_temp_env(tmp_path)
    assert env["TMP"] == env["TEMP"] == str(tmp_path / "tmp")
