"""Regression gate: ``action_space_size`` has one source, not a scattered literal.

Why this exists (spec ``kb/specs/20260902-tech-debt-hardening.md`` WS3.1): the
environment's true size is ``MicroCivActionSpace.total_size`` == 376, yet the
live defaults in ``NetworkConfig``, ``MuZeroConfig``, ``NPlayerNetworkConfig``,
``MicroCivNetwork.__init__``, ``civ6_local._build_network_for_bridge`` and
``factories.create_simultaneous_mcts`` carried hand-copied ``381`` (five too
wide) or ``356`` (twenty too narrow). Every default now derives from
``alpha_zero.utils.constants.default_action_space_size``.

Two checks:

1. **AST walk** (not regex) over ``alpha_zero/``: any MicroCiv-width *integer
   literal* (``356``, ``376`` or ``381``) that is assigned to a name/attribute
   matching ``action_space[_size]``, passed as such a keyword, used as such a
   parameter's default, given as the fallback of
   ``getattr(x, "action_space_size", N)``, or used as the size in an
   ``np.ones(376)``-style mask constructor, is a violation -- outside the two
   source modules and an explicit burndown allowlist. Docstrings are string
   constants and are ignored by construction. ``376`` is flagged even though it
   is the *right* number: a hardcoded copy is exactly what goes stale the day
   ``max_units`` changes. The literal set is deliberately closed -- ``1000``
   (the full-Civ6 ``interfaces/`` space) and ``50000`` (``Civ6Network``) are
   different games, not copies of this one.
2. **Runtime equality**: the live defaults equal the environment's size.

The allowlist is a burndown list: an entry whose file still exists but no
longer has a violation fails the test, so entries are deleted as sites are
fixed and cannot rot.

Negative control (run 2026-09-02, recorded per CLAUDE.md 5.2 "a gate is not a
gate until something runs it")::

    $ sed -i 's/^    action_space_size: int = DEFAULT_ACTION_SPACE_SIZE$/    action_space_size: int = 381/' \\
          alpha_zero/core/civ6_network.py
    $ python -m pytest tests/regression/test_action_space_single_source.py -q -p no:cacheprovider
    E   AssertionError: Integer literals bound to action_space_size outside the source modules
        (use alpha_zero.utils.constants.DEFAULT_ACTION_SPACE_SIZE / default_action_space_size()):
            alpha_zero/core/civ6_network.py:70: 381 (annotated assignment to action_space_size)
    E   AssertionError: NetworkConfig() = 381, expected 376
    FAILED ...::test_no_action_space_literals_outside_source_modules
    FAILED ...::test_live_defaults_equal_environment_size
    2 failed, 2 passed, 1 warning in 4.29s

    The file was restored from a scratchpad copy (no git state was changed);
    ``grep -n "action_space_size: int = " alpha_zero/core/civ6_network.py``
    then showed ``DEFAULT_ACTION_SPACE_SIZE`` at lines 70 and 549 again.
"""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "alpha_zero"

#: The two modules permitted to *define* the size.
SOURCE_MODULES: frozenset[str] = frozenset(
    {
        "alpha_zero/utils/constants.py",
        "alpha_zero/core/civ6_actions.py",
    }
)

#: Burndown allowlist: file -> reason. Delete the entry when the file is fixed
#: or deleted; a stale entry (file exists, no violation) fails the test.
ALLOWLIST: dict[str, str] = {
    "alpha_zero/core/jax_network.py": (
        "JAX network keeps 4 literal 381 defaults; explicitly out of WS3.1 scope"
    ),
    "alpha_zero/hierarchical/network/hierarchical_network.py": (
        "hierarchical/* deferred (356 defaults); WS3.1 scope is the live MicroCiv paths"
    ),
    "alpha_zero/hierarchical/mcts/hierarchical_mcts.py": (
        "hierarchical/* deferred (action_space = 356, np.ones(356))"
    ),
    "alpha_zero/hierarchical/controllers/level_controller.py": (
        "hierarchical/* deferred (np.ones(356) mask fallback)"
    ),
    # Tier A dead modules scheduled for deletion in WS2. Entries are harmless
    # once the files are gone (missing files are skipped, not reported).
    "alpha_zero/training/curriculum_pipeline.py": "Tier A dead module, WS2 deletion",
    "alpha_zero/cloud/selfplay_worker.py": "legacy worker, WS2 deletion",
    "alpha_zero/cloud/evaluator.py": "legacy worker, WS2 deletion",
    "alpha_zero/cloud/trainer.py": "legacy worker, WS2 deletion",
    "alpha_zero/core/action_pruning.py": "Tier A dead module, WS2 deletion",
    # Correct value (376), wrong source: hand-copied rather than derived. These
    # are the deployed cloud path and the transformer family; WS3.1 fixed the
    # 381/356 sites first. Route each through DEFAULT_ACTION_SPACE_SIZE and
    # delete its entry.
    "alpha_zero/cloud/gcp/config.py": "hardcoded 376 (GCPConfig field)",
    "alpha_zero/cloud/gcp/selfplay_worker.py": "hardcoded 376 (load_model defaults)",
    "alpha_zero/cloud/gcp/tournament_worker.py": "hardcoded 376 (run_tournament default)",
    "alpha_zero/cloud/gcp/training_worker.py": "hardcoded 376 (three loader/factory defaults)",
    "alpha_zero/cloud/gcp/vertex_jobs.py": "hardcoded 376 (launch default)",
    "alpha_zero/core/transformer_hub_mixin.py": "hardcoded 376 (__init__ default)",
    "alpha_zero/core/transformer_network.py": (
        "hardcoded 376 (TransformerConfig field, getattr fallback, factory default)"
    ),
    "alpha_zero/evaluation/muzero_vs_alphazero_benchmark.py": "hardcoded 376 (config field)",
}

_NAME_PATTERN = re.compile(r"^action_space(_size)?$")
#: Every width the MicroCiv action space has ever been given in this repo. A
#: literal from this set bound to an action-space name -- or sizing an
#: ``np.ones(N)`` mask -- is a hardcoded copy of the environment's width.
_KNOWN_SIZE_LITERALS: frozenset[int] = frozenset({356, 376, 381})
_MASK_CONSTRUCTORS: frozenset[str] = frozenset({"ones", "zeros", "empty", "full"})


@dataclass(frozen=True)
class Violation:
    """One integer literal bound to an action-space name."""

    path: str
    line: int
    value: int
    context: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.value} ({self.context})"


def _int_literal(node: ast.AST | None) -> int | None:
    """Return the value of a bare MicroCiv-width integer literal node, else None."""
    if (
        isinstance(node, ast.Constant)
        and type(node.value) is int
        and node.value in _KNOWN_SIZE_LITERALS
    ):
        return node.value
    return None


class _ActionSpaceLiteralVisitor(ast.NodeVisitor):
    """Collect integer literals bound to action-space names."""

    def __init__(self, rel_path: str) -> None:
        self.rel_path = rel_path
        self.violations: list[Violation] = []

    def _record(self, node: ast.AST, value: int, context: str) -> None:
        self.violations.append(Violation(self.rel_path, node.lineno, value, context))

    @staticmethod
    def _target_name(target: ast.expr) -> str | None:
        if isinstance(target, ast.Name):
            return target.id
        if isinstance(target, ast.Attribute):
            return target.attr
        return None

    def visit_Assign(self, node: ast.Assign) -> None:
        value = _int_literal(node.value)
        if value is not None:
            for target in node.targets:
                name = self._target_name(target)
                if name and _NAME_PATTERN.match(name):
                    self._record(node, value, f"assignment to {name}")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        value = _int_literal(node.value)
        name = self._target_name(node.target)
        if value is not None and name and _NAME_PATTERN.match(name):
            self._record(node, value, f"annotated assignment to {name}")
        self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        args = node.args
        positional = args.posonlyargs + args.args
        # Defaults align with the *last* len(defaults) positional parameters.
        for param, default in zip(
            positional[len(positional) - len(args.defaults) :], args.defaults, strict=True
        ):
            value = _int_literal(default)
            if value is not None and _NAME_PATTERN.match(param.arg):
                self._record(default, value, f"default of parameter {param.arg} in {node.name}")
        for param, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
            value = _int_literal(default)
            if value is not None and _NAME_PATTERN.match(param.arg):
                self._record(default, value, f"default of parameter {param.arg} in {node.name}")
        self.generic_visit(node)

    visit_FunctionDef = _visit_function  # noqa: N815 - ast API
    visit_AsyncFunctionDef = _visit_function  # noqa: N815 - ast API

    def visit_Call(self, node: ast.Call) -> None:
        for keyword in node.keywords:
            value = _int_literal(keyword.value)
            if value is not None and keyword.arg and _NAME_PATTERN.match(keyword.arg):
                self._record(keyword.value, value, f"keyword {keyword.arg}=")

        func = node.func
        # getattr(cfg, "action_space_size", 381)
        if isinstance(func, ast.Name) and func.id == "getattr" and len(node.args) == 3:
            attr = node.args[1]
            value = _int_literal(node.args[2])
            if (
                isinstance(attr, ast.Constant)
                and isinstance(attr.value, str)
                and _NAME_PATTERN.match(attr.value)
                and value is not None
            ):
                self._record(node.args[2], value, f'getattr(..., "{attr.value}", N) fallback')

        # np.ones(376) / torch.zeros(381): a mask sized by a bare literal.
        if isinstance(func, ast.Attribute) and func.attr in _MASK_CONSTRUCTORS and node.args:
            value = _int_literal(node.args[0])
            if value in _KNOWN_SIZE_LITERALS:
                self._record(node.args[0], value, f"{func.attr}({value}) mask literal")

        self.generic_visit(node)


def _python_files() -> list[Path]:
    return sorted(p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _scan_repo() -> dict[str, list[Violation]]:
    """Return violations keyed by repo-relative path (only files with hits)."""
    found: dict[str, list[Violation]] = {}
    for path in _python_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in SOURCE_MODULES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        visitor = _ActionSpaceLiteralVisitor(rel)
        visitor.visit(tree)
        if visitor.violations:
            found[rel] = visitor.violations
    return found


def test_no_action_space_literals_outside_source_modules() -> None:
    """No integer literal may be bound to an action-space name outside the sources."""
    found = _scan_repo()
    unexpected = [v for rel, vs in found.items() if rel not in ALLOWLIST for v in vs]
    assert not unexpected, (
        "Integer literals bound to action_space_size outside the source modules "
        "(use alpha_zero.utils.constants.DEFAULT_ACTION_SPACE_SIZE / "
        "default_action_space_size()):\n  " + "\n  ".join(str(v) for v in unexpected)
    )


def test_allowlist_entries_are_not_stale() -> None:
    """Every allowlisted file that still exists must still have a violation.

    This is what makes the list a burndown rather than a permanent exemption.
    Files deleted by the WS2 sweep simply drop out.
    """
    found = _scan_repo()
    stale = [rel for rel in ALLOWLIST if (REPO_ROOT / rel).exists() and rel not in found]
    assert not stale, f"Allowlist entries with no remaining violation -- delete them: {stale}"


def test_live_defaults_equal_environment_size() -> None:
    """The dynamic source and every live default agree with the environment."""
    torch = pytest.importorskip("torch")
    del torch  # imported only to skip cleanly where torch is absent

    from alpha_zero.civ6_local import _build_network_for_bridge
    from alpha_zero.core.civ6_actions import MicroCivActionSpace
    from alpha_zero.core.civ6_network import MicroCivNetwork, NetworkConfig
    from alpha_zero.core.muzero_network import MuZeroConfig
    from alpha_zero.envs.microciv import MicroCivEnv
    from alpha_zero.multiplayer.network_heads import NPlayerNetworkConfig
    from alpha_zero.utils.constants import (
        ACTION_SPACE,
        DEFAULT_ACTION_SPACE_SIZE,
        default_action_space_size,
    )
    from alpha_zero.utils.factories import create_simultaneous_mcts

    expected = MicroCivActionSpace().total_size
    assert expected == int(MicroCivEnv(board_size=8).action_space.n), (
        "MicroCivActionSpace.total_size disagrees with the gym action space"
    )
    assert default_action_space_size() == expected
    assert DEFAULT_ACTION_SPACE_SIZE == expected
    assert ACTION_SPACE.compute_microciv_action_space_size() == expected

    observed = {
        "NetworkConfig()": NetworkConfig().action_space_size,
        "MuZeroConfig()": MuZeroConfig().action_space_size,
        "NPlayerNetworkConfig()": NPlayerNetworkConfig().action_space_size,
        "MicroCivNetwork()": MicroCivNetwork(board_size=4).config.action_space_size,
        "_build_network_for_bridge default": inspect.signature(_build_network_for_bridge)
        .parameters["action_space_size"]
        .default,
        "create_simultaneous_mcts default": inspect.signature(create_simultaneous_mcts)
        .parameters["action_space_size"]
        .default,
    }
    mismatched = {k: v for k, v in observed.items() if v != expected}
    assert not mismatched, ", ".join(
        f"{name} = {value}, expected {expected}" for name, value in mismatched.items()
    )


def test_explicit_action_space_size_still_wins() -> None:
    """Backward compatibility: a caller passing 381 gets a 381-wide head.

    Checkpoints trained with the old default must remain loadable when the
    caller states the width explicitly; the fix changes defaults, not
    validation.
    """
    pytest.importorskip("torch")
    from alpha_zero.core.civ6_network import MicroCivNetwork, NetworkConfig

    assert NetworkConfig(action_space_size=381).action_space_size == 381
    net = MicroCivNetwork(board_size=4, action_space_size=381)
    assert net.config.action_space_size == 381
