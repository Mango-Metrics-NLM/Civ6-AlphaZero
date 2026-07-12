"""Unit tests for ``scripts/install_civ6_mod.py``.

Focus: the in-place overwrite fallback used when ``shutil.rmtree`` fails
with ``WinError 5`` / ``PermissionError`` — OneDrive holds persistent locks
on synced Mods directories, so the remove-then-copy strategy cannot delete
the existing install even though individual files remain replaceable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Module loader - the script is not a package, so import it by file path.
# ---------------------------------------------------------------------------

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent.parent / "scripts" / "install_civ6_mod.py"
)


def _load_installer():
    spec = importlib.util.spec_from_file_location("install_civ6_mod", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def installer():
    return _load_installer()


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
