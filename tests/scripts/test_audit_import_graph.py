"""Tests for ``scripts/audit_import_graph.py``.

The audit is only useful if it (a) resolves re-exports, so a module reached via
``from pkg import Name`` is not reported dead, (b) discounts a package's own
``__init__`` re-export as a live importer, and (c) recognises entry points and
deploy references. Each is checked against a synthetic tree so the test does not
depend on the state of ``alpha_zero/``; one smoke test runs it on the real repo.
"""

from __future__ import annotations

import logging
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


def test_from_dot_import_submodule_is_a_side_effect_use(tmp_path: Path) -> None:
    """``from . import registry`` executes the module, exactly like ``import pkg.registry``.

    The two spellings are semantically identical and the first is the usual
    shape of a registry-registration import, so treating only the second as a
    use reported live registry modules as dead.
    """
    _write(tmp_path, "alpha_zero/__init__.py", "")
    _write(
        tmp_path,
        "alpha_zero/pkg/__init__.py",
        "from . import registry_a\nimport alpha_zero.pkg.registry_c\n",
    )
    _write(tmp_path, "alpha_zero/pkg/registry_a.py", "REGISTERED = True\n")
    _write(tmp_path, "alpha_zero/pkg/registry_c.py", "REGISTERED = True\n")
    reported = _by_module(aig.audit(tmp_path))
    assert "alpha_zero.pkg.registry_a" not in reported, "`from . import x` is a use"
    assert "alpha_zero.pkg.registry_c" not in reported, "`import pkg.x` is a use"


def test_unparseable_file_warns_instead_of_silently_dropping(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A file whose imports are lost must say so: silence under-counts, not no-ops."""
    _write(tmp_path, "alpha_zero/__init__.py", "")
    _write(tmp_path, "alpha_zero/broken.py", "def f(:\n")
    _write(tmp_path, "alpha_zero/target.py", "VALUE = 1\n")
    with caplog.at_level(logging.WARNING, logger="audit_import_graph"):
        aig.audit(tmp_path)
    assert any("broken.py" in r.getMessage() for r in caplog.records), (
        "an unparseable file must warn -- its imports are missing from the graph"
    )


def test_bom_file_is_parsed(tmp_path: Path) -> None:
    """A UTF-8 BOM alone used to drop a file's imports (ast.parse rejects U+FEFF)."""
    _write(tmp_path, "alpha_zero/__init__.py", "")
    (tmp_path / "alpha_zero" / "importer.py").write_text(
        "\ufeffimport alpha_zero.target\n", encoding="utf-8"
    )
    _write(tmp_path, "alpha_zero/target.py", "VALUE = 1\n")
    _write(tmp_path, "alpha_zero/root.py", "from alpha_zero import importer\n")
    reported = _by_module(aig.audit(tmp_path))
    assert "alpha_zero.target" not in reported, "BOM file's imports must still count"


def test_workflow_reference_counts_as_a_deploy_ref(tmp_path: Path) -> None:
    """A module wired only from a CI workflow is an entry point, not dead code."""
    _write(tmp_path, "alpha_zero/__init__.py", "")
    _write(tmp_path, "alpha_zero/ci_only.py", "VALUE = 1\n")
    _write(
        tmp_path, ".github/workflows/ci.yml", "jobs:\n  x:\n    run: python -m alpha_zero.ci_only\n"
    )
    reported = _by_module(aig.audit(tmp_path))
    assert reported["alpha_zero.ci_only"].deploy_refs == [".github/workflows/ci.yml"]


# --- Build artifacts must not resurrect dead modules -------------------------
#
# A setuptools build or an editable install leaves `build/lib/alpha_zero/`: a
# second copy of every package `__init__.py`. Those copies import with ABSOLUTE
# names, but their own module name is `build.lib.alpha_zero.*`, so the audit
# credits them as EXTERNAL importers of the real modules. Because a package's
# own `__init__` re-export is deliberately not a live use
# (test_own_package_init_does_not_count_as_live_importer), the duplicate
# resurrects precisely the modules this audit exists to find. Measured on the
# real tree before the fix: 53 modules / 24,451 lines became 30 / 15,120, and
# the debt ratchet passed while the dead code was invisible -- a fail-open gate.
#
# The absolute import in the planted copy is the whole point: a relative import
# (`from .hidden import H`) resolves to `build.lib.alpha_zero.p.hidden` and this
# test would pass whether or not the fix is present.


@pytest.mark.parametrize("artifact_dir", ["build/lib", "dist", ".tox/py311/lib", ".eggs"])
def test_build_artifact_copy_does_not_hide_a_dead_module(tmp_path: Path, artifact_dir: str) -> None:
    """A copy of the package under a build directory must change nothing."""
    _write(tmp_path, "alpha_zero/__init__.py", "")
    _write(tmp_path, "alpha_zero/p/__init__.py", "from alpha_zero.p.hidden import H\n")
    _write(tmp_path, "alpha_zero/p/hidden.py", "class H: ...\n")

    before = _by_module(aig.audit(tmp_path))
    assert "alpha_zero.p.hidden" in before, "fixture is wrong: the module should start dead"

    # Exactly what a build leaves behind, absolute imports included.
    _write(tmp_path, f"{artifact_dir}/alpha_zero/__init__.py", "")
    _write(
        tmp_path,
        f"{artifact_dir}/alpha_zero/p/__init__.py",
        "from alpha_zero.p.hidden import H\n",
    )
    _write(tmp_path, f"{artifact_dir}/alpha_zero/p/hidden.py", "class H: ...\n")

    after = _by_module(aig.audit(tmp_path))
    assert "alpha_zero.p.hidden" in after, (
        f"a copy of the package under {artifact_dir}/ hid a dead module -- "
        "SKIP_PARTS must exclude build artifact directories"
    )
    assert set(before) == set(after), (
        f"the dead set changed when {artifact_dir}/ appeared: {sorted(set(before) ^ set(after))}"
    )


def test_skip_parts_covers_the_known_build_artifact_directories() -> None:
    """Pin the set itself, so removing an entry fails here rather than silently."""
    for required in ("build", "dist", ".tox", ".eggs"):
        assert required in aig.SKIP_PARTS, f"{required!r} missing from SKIP_PARTS"
