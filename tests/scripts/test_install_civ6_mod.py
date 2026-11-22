"""Unit tests for ``scripts/install_civ6_mod.py``.

Focus: the in-place overwrite fallback used when ``shutil.rmtree`` fails
with ``WinError 5`` / ``PermissionError`` — OneDrive holds persistent locks
on synced Mods directories, so the remove-then-copy strategy cannot delete
the existing install even though individual files remain replaceable.
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Module loader - the script is not a package, so import it by file path.
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "install_civ6_mod.py"


def _load_installer():
    spec = importlib.util.spec_from_file_location("install_civ6_mod", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def installer():
    return _load_installer()


@pytest.fixture(autouse=True)
def _restore_root_logging() -> Iterator[None]:
    """Prevent setup_logging() from leaking a closed capsys StreamHandler."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        yield
    finally:
        for handler in list(root.handlers):
            if handler not in saved_handlers:
                root.removeHandler(handler)
                handler.close()
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------


def _make_tree(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _tree_contents(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
        for p in root.rglob("*")
        if p.is_file()
    }


SOURCE_FILES = {
    "AlphaZeroAI.modinfo": "<Mod id='new'/>",
    "Config/AlphaZeroAI_Config.sql": "INSERT INTO new;",
    "Scripts/AlphaZeroAI_Bridge.lua": "print('bridge')",
}

TARGET_FILES = {
    "AlphaZeroAI.modinfo": "<Mod id='old'/>",
    "Config/AlphaZeroAI_Config.sql": "INSERT INTO old;",
    "Config/Stale_Leftover.sql": "DROP TABLE stale;",
}


# ---------------------------------------------------------------------------
# overwrite_in_place() behaviour
# ---------------------------------------------------------------------------


def test_overwrite_in_place_replaces_updates_and_prunes(installer, tmp_path) -> None:
    """Files are replaced per-file and destination-only files are removed."""
    source = tmp_path / "source"
    target = tmp_path / "target"
    _make_tree(source, SOURCE_FILES)
    _make_tree(target, TARGET_FILES)

    installer.overwrite_in_place(source, target)

    assert _tree_contents(target) == _tree_contents(source)


def test_overwrite_in_place_raises_on_hash_mismatch(installer, tmp_path) -> None:
    """A corrupted copy is caught by the per-file SHA256 verification."""
    source = tmp_path / "source"
    target = tmp_path / "target"
    _make_tree(source, {"AlphaZeroAI.modinfo": "<Mod id='new'/>"})
    target.mkdir()

    def _corrupting_copy(src, dst, **kwargs):
        Path(dst).write_text("corrupted", encoding="utf-8")

    with patch.object(installer.shutil, "copy2", _corrupting_copy):
        with pytest.raises(installer.InstallVerificationError):
            installer.overwrite_in_place(source, target)


# ---------------------------------------------------------------------------
# install_mod() behaviour
# ---------------------------------------------------------------------------


def test_install_mod_falls_back_when_rmtree_denied(installer, tmp_path) -> None:
    """WinError 5 from rmtree triggers the in-place overwrite fallback."""
    source = tmp_path / "source"
    target = tmp_path / "Mods" / "AlphaZeroAI"
    _make_tree(source, SOURCE_FILES)
    _make_tree(target, TARGET_FILES)

    denied = PermissionError(13, "Access is denied")
    with patch.object(installer.shutil, "rmtree", side_effect=denied):
        result = installer.install_mod(source, target)

    assert result == 0
    assert _tree_contents(target) == _tree_contents(source)


def test_install_mod_fresh_copy_when_target_missing(installer, tmp_path) -> None:
    """Without an existing install, the plain copytree path still works."""
    source = tmp_path / "source"
    target = tmp_path / "Mods" / "AlphaZeroAI"
    _make_tree(source, SOURCE_FILES)

    result = installer.install_mod(source, target)

    assert result == 0
    assert _tree_contents(target) == _tree_contents(source)


def test_install_mod_removes_cleanly_when_rmtree_succeeds(installer, tmp_path) -> None:
    """When rmtree works, the original remove-then-copy strategy is kept."""
    source = tmp_path / "source"
    target = tmp_path / "Mods" / "AlphaZeroAI"
    _make_tree(source, SOURCE_FILES)
    _make_tree(target, TARGET_FILES)

    result = installer.install_mod(source, target)

    assert result == 0
    assert _tree_contents(target) == _tree_contents(source)


def test_install_mod_fails_when_fallback_also_denied(installer, tmp_path) -> None:
    """If per-file replacement is denied as well, install reports failure."""
    source = tmp_path / "source"
    target = tmp_path / "Mods" / "AlphaZeroAI"
    _make_tree(source, SOURCE_FILES)
    _make_tree(target, TARGET_FILES)

    denied = PermissionError(13, "Access is denied")
    with (
        patch.object(installer.shutil, "rmtree", side_effect=denied),
        patch.object(installer.shutil, "copy2", side_effect=denied),
    ):
        result = installer.install_mod(source, target)

    assert result == 1


def test_install_mod_fails_when_copytree_raises(installer, tmp_path) -> None:
    """copytree failure on a fresh install (no existing target) is caught and
    reported as exit code 1, not left to propagate as an uncaught exception."""
    source = tmp_path / "source"
    target = tmp_path / "Mods" / "AlphaZeroAI"
    _make_tree(source, SOURCE_FILES)

    denied = OSError(13, "Access is denied")
    with patch.object(installer.shutil, "copytree", side_effect=denied):
        result = installer.install_mod(source, target)

    assert result == 1
    assert not target.exists()


# ---------------------------------------------------------------------------
# overwrite_in_place() - stale directory removal best-effort swallow
# ---------------------------------------------------------------------------


def test_overwrite_in_place_swallows_rmdir_error_on_stale_directory(
    installer, tmp_path, caplog
) -> None:
    """A stale directory that fails to rmdir() (e.g. OneDrive lock) is logged
    as a warning and does not abort the overwrite."""
    source = tmp_path / "source"
    target = tmp_path / "target"
    _make_tree(source, {"AlphaZeroAI.modinfo": "<Mod id='new'/>"})
    # Stale directory only exists in target, with a file inside so it isn't
    # implicitly removed by rglob logic before the rmdir attempt.
    stale_dir = target / "StaleDir"
    _make_tree(target, {"AlphaZeroAI.modinfo": "<Mod id='old'/>"})
    stale_dir.mkdir(parents=True, exist_ok=True)
    (stale_dir / "leftover.txt").write_text("x", encoding="utf-8")

    real_rmdir = Path.rmdir

    def _failing_rmdir(self):
        if self.name == "StaleDir":
            raise OSError(13, "Access is denied")
        return real_rmdir(self)

    with caplog.at_level(logging.WARNING):
        with patch.object(Path, "rmdir", _failing_rmdir):
            # The leftover file inside stale_dir is removed by the stale-file
            # cleanup step first, so rmdir is attempted on an (now) empty dir.
            (stale_dir / "leftover.txt").unlink()
            installer.overwrite_in_place(source, target)

    assert "Could not remove stale directory" in caplog.text
    # The rest of the overwrite must still have completed successfully.
    assert (target / "AlphaZeroAI.modinfo").read_text(encoding="utf-8") == "<Mod id='new'/>"


# ---------------------------------------------------------------------------
# get_default_mod_dir() - path auto-detection
# ---------------------------------------------------------------------------


def test_get_default_mod_dir_prefers_onedrive_when_present(
    installer, tmp_path, monkeypatch
) -> None:
    """When the OneDrive-redirected Civ6 folder exists, it wins over bare
    Documents even if both are present."""
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    onedrive_civ6 = tmp_path / "OneDrive" / "Documents" / "My Games" / "Sid Meier's Civilization VI"
    bare_civ6 = tmp_path / "Documents" / "My Games" / "Sid Meier's Civilization VI"
    onedrive_civ6.mkdir(parents=True)
    bare_civ6.mkdir(parents=True)

    result = installer.get_default_mod_dir()

    assert result == onedrive_civ6 / "Mods"


def test_get_default_mod_dir_falls_back_to_bare_documents(installer, tmp_path, monkeypatch) -> None:
    """When only the bare-Documents Civ6 folder exists, it is used."""
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    bare_civ6 = tmp_path / "Documents" / "My Games" / "Sid Meier's Civilization VI"
    bare_civ6.mkdir(parents=True)

    result = installer.get_default_mod_dir()

    assert result == bare_civ6 / "Mods"


def test_get_default_mod_dir_defaults_to_onedrive_when_neither_exists(
    installer, tmp_path, monkeypatch, caplog
) -> None:
    """When Civ6 has never been run yet (no candidate dir exists), the
    installer falls back to the OneDrive path and warns the user."""
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    with caplog.at_level(logging.WARNING):
        result = installer.get_default_mod_dir()

    expected = (
        tmp_path / "OneDrive" / "Documents" / "My Games" / "Sid Meier's Civilization VI" / "Mods"
    )
    assert result == expected
    assert "Civ6 user-data directory not found" in caplog.text


def test_get_default_mod_dir_warns_on_non_windows(installer, tmp_path, monkeypatch, caplog) -> None:
    """On non-Windows platforms a warning directs the user to --mod-dir, but
    the same path-resolution logic still runs and returns a usable path."""
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(installer.sys, "platform", "linux")

    with caplog.at_level(logging.WARNING):
        result = installer.get_default_mod_dir()

    assert "designed for Windows" in caplog.text
    expected = (
        tmp_path / "OneDrive" / "Documents" / "My Games" / "Sid Meier's Civilization VI" / "Mods"
    )
    assert result == expected


def test_get_default_mod_dir_uses_expanduser_when_userprofile_unset(
    installer, tmp_path, monkeypatch
) -> None:
    """Without USERPROFILE set (non-Windows CI), the function falls back to
    os.path.expanduser("~") rather than raising."""
    monkeypatch.delenv("USERPROFILE", raising=False)
    monkeypatch.setattr(installer.os.path, "expanduser", lambda p: str(tmp_path))

    result = installer.get_default_mod_dir()

    expected = (
        tmp_path / "OneDrive" / "Documents" / "My Games" / "Sid Meier's Civilization VI" / "Mods"
    )
    assert result == expected


# ---------------------------------------------------------------------------
# main() - CLI entry point
# ---------------------------------------------------------------------------


def test_main_returns_1_when_source_dir_missing(installer, tmp_path, monkeypatch) -> None:
    """An invalid --source-dir is rejected before any filesystem mutation."""
    missing_source = tmp_path / "does-not-exist"
    monkeypatch.setattr(sys, "argv", ["install_civ6_mod.py", "--source-dir", str(missing_source)])

    result = installer.main()

    assert result == 1


def test_main_returns_1_when_source_dir_is_a_file(installer, tmp_path, monkeypatch) -> None:
    """A --source-dir that resolves to a file (not a directory) is rejected."""
    source_file = tmp_path / "not_a_dir.txt"
    source_file.write_text("x", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["install_civ6_mod.py", "--source-dir", str(source_file)])

    result = installer.main()

    assert result == 1


def test_main_creates_missing_mods_directory_then_installs(
    installer, tmp_path, monkeypatch
) -> None:
    """When --mod-dir points at a directory that doesn't exist yet, main()
    creates it (mkdir parents=True) before installing, rather than failing."""
    source = tmp_path / "source"
    _make_tree(source, SOURCE_FILES)
    mods_dir = tmp_path / "brand_new" / "Mods"
    assert not mods_dir.exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "install_civ6_mod.py",
            "--source-dir",
            str(source),
            "--mod-dir",
            str(mods_dir),
        ],
    )

    result = installer.main()

    assert result == 0
    assert mods_dir.exists()
    assert _tree_contents(mods_dir / installer.MOD_NAME) == _tree_contents(source)


def test_main_returns_1_when_mods_directory_creation_fails(
    installer, tmp_path, monkeypatch
) -> None:
    """If the Mods directory doesn't exist and mkdir() fails (e.g. permission
    denied), main() reports failure instead of crashing."""
    source = tmp_path / "source"
    _make_tree(source, SOURCE_FILES)
    mods_dir = tmp_path / "brand_new" / "Mods"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "install_civ6_mod.py",
            "--source-dir",
            str(source),
            "--mod-dir",
            str(mods_dir),
        ],
    )

    denied = OSError(13, "Access is denied")
    with patch.object(installer.Path, "mkdir", side_effect=denied):
        result = installer.main()

    assert result == 1


def test_main_end_to_end_success_with_explicit_mod_dir(installer, tmp_path, monkeypatch) -> None:
    """Full happy path: explicit --source-dir and --mod-dir, target already
    exists, main() delegates to install_mod() and returns 0."""
    source = tmp_path / "source"
    mods_dir = tmp_path / "Mods"
    _make_tree(source, SOURCE_FILES)
    mods_dir.mkdir(parents=True)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "install_civ6_mod.py",
            "--source-dir",
            str(source),
            "--mod-dir",
            str(mods_dir),
        ],
    )

    result = installer.main()

    assert result == 0
    assert _tree_contents(mods_dir / installer.MOD_NAME) == _tree_contents(source)


def test_main_uses_get_default_mod_dir_when_mod_dir_not_passed(
    installer, tmp_path, monkeypatch
) -> None:
    """Without --mod-dir, main() resolves the target via get_default_mod_dir()
    rather than requiring the caller to know the Civ6 path."""
    source = tmp_path / "source"
    _make_tree(source, SOURCE_FILES)
    default_mods_dir = tmp_path / "auto-detected" / "Mods"
    default_mods_dir.mkdir(parents=True)

    monkeypatch.setattr(sys, "argv", ["install_civ6_mod.py", "--source-dir", str(source)])
    monkeypatch.setattr(installer, "get_default_mod_dir", lambda: default_mods_dir)

    result = installer.main()

    assert result == 0
    assert _tree_contents(default_mods_dir / installer.MOD_NAME) == _tree_contents(source)


def test_script_invoked_as_subprocess_missing_source_dir_exits_nonzero(tmp_path) -> None:
    """Smoke-tests the ``if __name__ == "__main__": sys.exit(main())`` wiring
    by actually invoking the script as a subprocess with a bad --source-dir."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--source-dir",
            str(tmp_path / "nonexistent"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1


# ---------------------------------------------------------------------------
# verify_mod_installation() and --check / --verbose CLI verification
# ---------------------------------------------------------------------------


def test_verify_mod_installation_matching_trees_pass(installer, tmp_path) -> None:
    """Identical source and installed mod trees pass SHA-256 verification."""
    src = tmp_path / "src"
    dst = tmp_path / "Mods" / installer.MOD_NAME
    _make_tree(src, SOURCE_FILES)
    _make_tree(dst, SOURCE_FILES)

    ok, problems = installer.verify_mod_installation(src, dst)
    assert ok is True
    assert problems == []


def test_verify_mod_installation_detects_missing_extra_and_mismatched_files(
    installer, tmp_path
) -> None:
    """Verification reports missing, extra, and SHA-256-mismatched files."""
    src = tmp_path / "src"
    dst = tmp_path / "Mods" / installer.MOD_NAME
    _make_tree(src, {"a.lua": "same", "b.lua": "missing_in_dst", "c.lua": "new_content"})
    _make_tree(dst, {"a.lua": "same", "c.lua": "old_content", "stale.lua": "extra"})

    ok, problems = installer.verify_mod_installation(src, dst)
    assert ok is False
    assert any("missing installed file: b.lua" in p for p in problems)
    assert any("unexpected extra file: stale.lua" in p for p in problems)
    assert any("SHA256 mismatch for c.lua" in p for p in problems)


def test_verify_mod_installation_missing_target_directory_fails_cleanly(
    installer, tmp_path
) -> None:
    """Missing target mod folder fails with an actionable error instead of raising."""
    src = tmp_path / "src"
    _make_tree(src, {"Main.lua": "print(1)\n"})

    ok, problems = installer.verify_mod_installation(src, tmp_path / "missing")
    assert ok is False
    assert any("missing installed mod directory" in p for p in problems)


def test_main_check_mode_exit_codes_and_verbose_logging(
    installer, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--check`` exits 0 on match (emitting debug hashes with ``--verbose``) and 1 on drift."""
    src = tmp_path / "repo_mod"
    mods_dir = tmp_path / "Mods"
    installed = mods_dir / installer.MOD_NAME
    files = {"AlphaZeroAI.modinfo": "<Mod/>", "Scripts/AlphaZeroMain.lua": "return 1\n"}
    _make_tree(src, files)
    _make_tree(installed, files)

    rc_ok = installer.main(
        ["--source-dir", str(src), "--mod-dir", str(mods_dir), "--check", "--verbose"]
    )
    out = capsys.readouterr().out
    assert rc_ok == 0
    assert "Verified AlphaZeroAI.modinfo:" in out

    (installed / "Scripts/AlphaZeroMain.lua").write_text("return 2\n", encoding="utf-8")
    rc_drift = installer.main(["--source-dir", str(src), "--mod-dir", str(mods_dir), "--check"])
    assert rc_drift == 1
