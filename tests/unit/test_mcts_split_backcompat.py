"""Backwards-compatibility guarantees for the ``civ6_mcts`` decomposition.

The MCTS family moved from the single module ``alpha_zero.core.civ6_mcts`` to
one module per variant under ``alpha_zero.core.mcts``. A re-export shim remains
at the historical path. These tests pin three separate guarantees:

1. **Identity** -- the old path resolves to the *same objects*, not equivalent
   ones (``test_shim_reexports_are_identical``).
2. **Patchability** -- ``mock.patch("alpha_zero.core.civ6_mcts.<Name>")`` still
   reaches the consumer. Identity alone does **not** imply this: a re-export
   shim trivially satisfies ``old is new`` while a patch on it becomes a silent
   no-op whenever the patched symbol's consumer also moved to a sibling module.
   That failure mode is invisible to an identity test, so it gets its own.
3. **Layering** -- ``gumbel`` and ``muzero`` must not depend on ``base``; both
   carry their own node and configuration types. Pinning this stops a future
   edit from quietly reintroducing the coupling the split removed.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
from unittest.mock import patch

import pytest

_SHIM = "alpha_zero.core.civ6_mcts"

# (new module path, names that must be re-exported identically by the shim)
_SHIM_MAP: list[tuple[str, tuple[str, ...]]] = [
    (
        "alpha_zero.core.mcts.mcts_protocols",
        ("ActionSpace", "MCTSEnvironment", "MCTSNetwork", "MuZeroNetworkProtocol"),
    ),
    ("alpha_zero.core.mcts.base", ("MCTSConfig", "MCTSNode", "_env_current_player")),
    ("alpha_zero.core.mcts.gumbel_ops", ("compute_completed_q", "gumbel_sample_actions")),
    ("alpha_zero.core.mcts.puct", ("Civ6MCTS",)),
    ("alpha_zero.core.mcts.simple", ("SimpleMCTS",)),
    ("alpha_zero.core.mcts.gumbel", ("GumbelMCTS", "GumbelMCTSConfig", "GumbelMCTSNode")),
    ("alpha_zero.core.mcts.muzero", ("MuZeroMCTS", "MuZeroMCTSConfig", "MuZeroNode")),
]

_FLAT = [(new, name) for new, names in _SHIM_MAP for name in names]


@pytest.mark.parametrize(("new_path", "name"), _FLAT)
def test_shim_reexports_are_identical(new_path: str, name: str) -> None:
    """The historical path and the new path resolve to the same object."""
    shim = importlib.import_module(_SHIM)
    new = importlib.import_module(new_path)
    assert hasattr(shim, name), f"{_SHIM} no longer exports {name}"
    assert getattr(shim, name) is getattr(new, name), (
        f"{_SHIM}.{name} is not the same object as {new_path}.{name} -- "
        "the shim is re-implementing rather than re-exporting"
    )


@pytest.mark.parametrize(("new_path", "name"), _FLAT)
def test_package_reexports_are_identical(new_path: str, name: str) -> None:
    """``alpha_zero.core.mcts`` re-exports the same objects as its submodules."""
    pkg = importlib.import_module("alpha_zero.core.mcts")
    new = importlib.import_module(new_path)
    assert getattr(pkg, name) is getattr(new, name)


def test_shim_all_is_complete_and_resolvable() -> None:
    """Every name in the shim's ``__all__`` resolves, and covers every moved name."""
    shim = importlib.import_module(_SHIM)
    for name in shim.__all__:
        assert hasattr(shim, name), f"{name} in __all__ but not resolvable"
    assert set(shim.__all__) == {name for _, name in _FLAT}


def test_core_package_still_exports_mcts_names() -> None:
    """``alpha_zero.core``'s historical re-exports are unchanged by the split."""
    core = importlib.import_module("alpha_zero.core")
    muzero = importlib.import_module("alpha_zero.core.mcts.muzero")
    for name in (
        "Civ6MCTS",
        "GumbelMCTS",
        "GumbelMCTSConfig",
        "MCTSConfig",
        "MCTSNode",
        "MuZeroMCTS",
        "MuZeroMCTSConfig",
        "SimpleMCTS",
    ):
        assert hasattr(core, name), f"alpha_zero.core stopped exporting {name}"
    assert core.MuZeroMCTS is muzero.MuZeroMCTS


def test_patching_the_shim_reaches_a_function_local_consumer() -> None:
    """A ``mock.patch`` on the shim is still seen by a call-time importer.

    This is the guarantee an identity test cannot give. ``MuZeroTrainingBridge``
    imports from the shim *inside* ``create_mcts`` (a deliberate circular-import
    deferral), so the patch must be observed at call time.
    """
    from alpha_zero.training.muzero_integration import MuZeroTrainingBridge

    with (
        patch(f"{_SHIM}.MuZeroMCTS") as mock_cls,
        patch(f"{_SHIM}.MuZeroMCTSConfig") as mock_cfg,
    ):
        bridge = MuZeroTrainingBridge()
        result = bridge.create_mcts(network=object())

    assert mock_cfg.called, "patched MuZeroMCTSConfig was never constructed"
    assert mock_cls.called, "patched MuZeroMCTS was never constructed"
    assert result is mock_cls.return_value


def _module_imports(rel_path: str) -> set[str]:
    """Absolute module names imported at module scope by ``rel_path``."""
    tree = ast.parse(pathlib.Path(rel_path).read_text())
    found: set[str] = set()
    for node in tree.body:  # module scope only -- deferred imports are exempt
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
    return found


@pytest.mark.parametrize("variant", ["gumbel", "muzero"])
def test_independent_variants_do_not_depend_on_base(variant: str) -> None:
    """``gumbel`` and ``muzero`` carry their own node/config types.

    They must not import ``alpha_zero.core.mcts.base``; the shared Gumbel
    helpers live in ``gumbel_ops`` precisely so ``muzero`` need not import
    ``gumbel`` either.
    """
    imports = _module_imports(f"alpha_zero/core/mcts/{variant}.py")
    assert "alpha_zero.core.mcts.base" not in imports


def test_muzero_does_not_import_gumbel_variant() -> None:
    """MuZero uses ``gumbel_ops``, never the Gumbel search module itself."""
    imports = _module_imports("alpha_zero/core/mcts/muzero.py")
    assert "alpha_zero.core.mcts.gumbel" not in imports
    assert "alpha_zero.core.mcts.gumbel_ops" in imports


def test_leaf_modules_import_no_sibling_variants() -> None:
    """``mcts_protocols``, ``gumbel_ops`` and ``base`` sit at the bottom."""
    for leaf in ("mcts_protocols", "gumbel_ops", "base"):
        imports = _module_imports(f"alpha_zero/core/mcts/{leaf}.py")
        siblings = {i for i in imports if i.startswith("alpha_zero.core.mcts.")}
        assert not siblings, f"{leaf} should be a leaf but imports {siblings}"
