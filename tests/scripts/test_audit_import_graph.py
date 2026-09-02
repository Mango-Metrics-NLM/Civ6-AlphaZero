"""Tests for ``scripts/audit_import_graph.py``.

The audit is only useful if it (a) resolves re-exports, so a module reached via
``from pkg import Name`` is not reported dead, (b) discounts a package's own
``__init__`` re-export as a live importer, and (c) recognises entry points and
deploy references. Each is checked against a synthetic tree so the test does not
depend on the state of ``alpha_zero/``; one smoke test runs it on the real repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import audit_import_graph as aig  # noqa: E402


def _write(root: Path, rel: str, text: str = "") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def synthetic_repo(tmp_path: Path) -> Path:
    """A tiny ``alpha_zero`` package with one of every case the audit must handle."""
    pkg = "alpha_zero"
    _write(tmp_path, f"{pkg}/__init__.py", "")
    # Package whose __init__ re-exports from two submodules.
    _write(
        tmp_path,
        f"{pkg}/mp/__init__.py",
        "from .used_via_reexport import Thing\nfrom .only_reexported import Other\n",
    )
    _write(tmp_path, f"{pkg}/mp/used_via_reexport.py", "class Thing: ...\n")
    _write(tmp_path, f"{pkg}/mp/only_reexported.py", "class Other: ...\n")
    # Consumer reaches Thing only through the package re-export.
    _write(tmp_path, f"{pkg}/consumer.py", "from alpha_zero.mp import Thing\n")
    # Consumer of consumer, so consumer itself is live.
    _write(tmp_path, f"{pkg}/app.py", "import alpha_zero.consumer\n")
    # Attribute access through a package alias.
    _write(tmp_path, f"{pkg}/attr_user.py", "import alpha_zero.mp as mp\nx = mp.Other\n")
    _write(tmp_path, f"{pkg}/attr_root.py", "from alpha_zero import attr_user\n")
    # Genuinely dead module with only a test importer.
    _write(tmp_path, f"{pkg}/dead.py", "VALUE = 1\n")
    _write(tmp_path, "tests/test_dead.py", "from alpha_zero.dead import VALUE\n")
    # Entry point: no importer, but has a main guard.
    _write(tmp_path, f"{pkg}/cli.py", 'if __name__ == "__main__":\n    pass\n')
    # Deploy-referenced: no importer, named in a Dockerfile.
    _write(tmp_path, f"{pkg}/worker.py", "def run() -> None: ...\n")
    _write(tmp_path, "Dockerfile", 'CMD ["python", "-m", "alpha_zero.worker"]\n')
    # Dynamic import by literal string.
    _write(tmp_path, f"{pkg}/dyn_target.py", "")
    _write(
        tmp_path,
        f"{pkg}/dyn_user.py",
        'import importlib\nimportlib.import_module("alpha_zero.dyn_target")\n',
    )
    _write(tmp_path, f"{pkg}/dyn_root.py", "from alpha_zero import dyn_user\n")
    _write(tmp_path, f"{pkg}/app2.py", "from alpha_zero import app, attr_root, dyn_root\n")
    return tmp_path


def _by_module(reports: list[aig.ModuleReport]) -> dict[str, aig.ModuleReport]:
    return {r.module: r for r in reports}


def test_reexport_resolution_keeps_reexported_modules_alive(synthetic_repo: Path) -> None:
    reported = _by_module(aig.audit(synthetic_repo))
    assert "alpha_zero.mp.used_via_reexport" not in reported, "reached via `from pkg import Name`"
    assert "alpha_zero.mp.only_reexported" not in reported, "reached via attribute on package alias"
    assert "alpha_zero.consumer" not in reported


def test_own_package_init_does_not_count_as_live_importer(tmp_path: Path) -> None:
    _write(tmp_path, "alpha_zero/__init__.py", "")
    _write(tmp_path, "alpha_zero/p/__init__.py", "from .hidden import H\n")
    _write(tmp_path, "alpha_zero/p/hidden.py", "class H: ...\n")
    reported = _by_module(aig.audit(tmp_path))
    assert "alpha_zero.p.hidden" in reported, "a re-export nobody consumes is not a use"


def test_side_effect_import_from_own_package_init_is_live(tmp_path: Path) -> None:
    _write(tmp_path, "alpha_zero/__init__.py", "import alpha_zero._compat  # noqa: F401\n")
    _write(tmp_path, "alpha_zero/_compat.py", "import sys\n")
    reported = _by_module(aig.audit(tmp_path))
    assert "alpha_zero._compat" not in reported, "a plain import in __init__ executes the module"


def test_dead_module_is_reported_with_its_test_importer_count(synthetic_repo: Path) -> None:
    reported = _by_module(aig.audit(synthetic_repo))
    dead = reported["alpha_zero.dead"]
    assert dead.test_importers == 1
    assert dead.entry_point is False
    assert dead.deploy_refs == []


def test_entry_points_and_deploy_refs_are_classified_not_dead(synthetic_repo: Path) -> None:
    reported = _by_module(aig.audit(synthetic_repo))
    assert reported["alpha_zero.cli"].entry_point is True
    assert reported["alpha_zero.worker"].deploy_refs == ["Dockerfile"]


def test_dynamic_literal_import_counts(synthetic_repo: Path) -> None:
    reported = _by_module(aig.audit(synthetic_repo))
    assert "alpha_zero.dyn_target" not in reported


def test_package_filter(synthetic_repo: Path) -> None:
    reports = aig.audit(synthetic_repo, package_filter="alpha_zero.mp")
    assert all(r.module.startswith("alpha_zero.mp") for r in reports)


def test_cli_json_output_on_real_repo(capsys: pytest.CaptureFixture[str]) -> None:
    assert aig.main(["--json", "--root", str(REPO_ROOT)]) == 0
    out = capsys.readouterr().out
    assert out.lstrip().startswith("[")
    assert '"module"' in out


def test_cli_rejects_root_without_package(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        aig.main(["--root", str(tmp_path)])
    assert excinfo.value.code == 2
