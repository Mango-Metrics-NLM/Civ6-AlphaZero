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
    (
        "alpha_zero.core.mcts.gumbel",
        (
            "GumbelMCTS",
            "GumbelMCTSConfig",
            "GumbelMCTSNode",
            "POLICY_TARGET_ONE_HOT",
            "POLICY_TARGET_VISIT_COUNTS",
            "VALID_POLICY_TARGETS",
        ),
    ),
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


_PACKAGE = "alpha_zero.core.mcts"


def _module_imports(rel_path: str) -> set[str]:
    """Absolute module names imported at module scope by ``rel_path``.

    Two ways a forbidden edge could hide from an earlier version of this helper,
    both now closed:

    1. **Relative spelling.** Comparing ``ast.ImportFrom.module`` verbatim records
       ``from .base import X`` as ``"base"``, which never matches the absolute
       name. ``node.level`` is the number of leading dots; ``level == 1`` is the
       module's own package.
    2. **Deferral.** Walking only ``tree.body`` misses a function-local
       ``from .base import MCTSNode``. Function-local imports are a legitimate
       repo-wide pattern for breaking import cycles -- there are 45 of them -- but
       there is no legitimate reason to defer one of the two edges this split
       removed, and allowing it would let the dependency return invisibly. So the
       whole AST is walked.
    """
    tree = ast.parse(pathlib.Path(rel_path).read_text())
    found: set[str] = set()
    for node in ast.walk(tree):  # whole AST: a deferred import must not hide
        if isinstance(node, ast.ImportFrom):
            if node.level:
                # `from . import base` records the names; `from .base import X`
                # records the submodule. Both must resolve to absolute paths.
                parent = _PACKAGE.rsplit(".", node.level - 1)[0] if node.level > 1 else _PACKAGE
                base = f"{parent}.{node.module}" if node.module else parent
                found.add(base)
                if not node.module:
                    found.update(f"{parent}.{a.name}" for a in node.names)
            elif node.module:
                found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
    return found


@pytest.mark.parametrize("variant", ["gumbel", "muzero"])
def test_independent_variants_do_not_depend_on_base(variant: str) -> None:
    """``gumbel`` and ``muzero`` carry their own node/config types.

    They must not import ``alpha_zero.core.mcts.base`` at module scope *or*
    inside a function; the shared Gumbel helpers live in ``gumbel_ops`` precisely
    so ``muzero`` need not import ``gumbel`` either.
    """
    imports = _module_imports(f"alpha_zero/core/mcts/{variant}.py")
    assert "alpha_zero.core.mcts.base" not in imports


def test_muzero_does_not_import_gumbel_variant() -> None:
    """MuZero uses ``gumbel_ops``, never the Gumbel search module itself."""
    imports = _module_imports("alpha_zero/core/mcts/muzero.py")
    assert "alpha_zero.core.mcts.gumbel" not in imports
    assert "alpha_zero.core.mcts.gumbel_ops" in imports


# Layering inside alpha_zero/core/mcts/. A module may import only from a
# STRICTLY lower layer, never from its own layer or above. Expressed as data so
# a new module is one edit here rather than a new bespoke assertion.
#
# Layer 0 is the true bottom: `backup` owns the sign-correct value backup and
# the acting-player accessor, which every variant needs. It exists as its own
# layer precisely because `gumbel` and `muzero` must depend on neither `base`
# nor each other (CLAUDE.md 2026-08-30), so the shared rule cannot live in
# `base` without re-creating that dependency.
MCTS_LAYERS: tuple[tuple[str, ...], ...] = (
    ("backup", "mcts_protocols", "gumbel_ops"),
    # `muzero_tree` (MuZero's node, mover model and signed reward backup) sits
    # beside `base`: it needs `backup`'s sign rule and nothing above it.
    ("base", "muzero_tree"),
    ("puct", "simple", "gumbel", "muzero"),
)


def _layer_of(module: str) -> int | None:
    for index, layer in enumerate(MCTS_LAYERS):
        if module in layer:
            return index
    return None


def test_every_mcts_module_is_assigned_a_layer() -> None:
    """A new module must be placed in the layering, not silently ungated.

    Without this, adding a module to the package would exempt it from the
    layering rule below by omission -- the failure mode that let unwired suites
    accumulate (CLAUDE.md 2026-08-30).
    """
    on_disk = {
        path.stem
        for path in pathlib.Path("alpha_zero/core/mcts").glob("*.py")
        if path.stem != "__init__"
    }
    assigned = {module for layer in MCTS_LAYERS for module in layer}
    assert on_disk == assigned, (
        f"unassigned modules: {sorted(on_disk - assigned)}; "
        f"assigned but missing from disk: {sorted(assigned - on_disk)}"
    )


def test_mcts_modules_import_only_from_lower_layers() -> None:
    """No module imports a sibling at or above its own layer."""
    violations: list[str] = []
    for index, layer in enumerate(MCTS_LAYERS):
        for module in layer:
            imports = _module_imports(f"alpha_zero/core/mcts/{module}.py")
            for imported in imports:
                if not imported.startswith("alpha_zero.core.mcts."):
                    continue
                sibling = imported.removeprefix("alpha_zero.core.mcts.").split(".")[0]
                sibling_layer = _layer_of(sibling)
                if sibling_layer is None or sibling_layer >= index:
                    violations.append(
                        f"{module} (layer {index}) imports {sibling} (layer {sibling_layer})"
                    )
    assert not violations, "layering violations: " + "; ".join(violations)
