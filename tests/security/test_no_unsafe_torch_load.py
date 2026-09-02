"""Security gate: every ``torch.load`` in ``alpha_zero/`` is ``weights_only=True``.

``torch.load`` without ``weights_only=True`` unpickles arbitrary objects and so
executes arbitrary code from the checkpoint file. Spec
``kb/specs/20260902-tech-debt-hardening.md`` 2.5 counted seven such sites plus
an ``except Exception`` in ``core/checkpoint_utils.py`` that retried *every*
failure with ``weights_only=False`` (WS4.1/WS4.2).

Rules, enforced by an AST walk (multi-line calls included, since the parser --
not a regex -- sees the call):

* a ``torch.load(...)`` call **must** carry the keyword ``weights_only`` and it
  must be the literal ``True``;
* a missing keyword or a non-literal value is always a violation -- there is no
  allowlist for it;
* ``weights_only=False`` is permitted in exactly one place,
  ``alpha_zero/utils/safe_load.py``, exactly once, inside ``safe_torch_load``
  behind the ``allow_unsafe`` / ``CIV6AZ_ALLOW_UNSAFE_CHECKPOINT`` gate (a leaf
  module, re-exported by ``core/checkpoint_utils`` and used by
  ``utils/checkpoint.safe_load_checkpoint``); elsewhere only while the file is
  on the burndown allowlist below, at most the listed number of times.

Negative control (run 2026-09-02)::

    $ sed -i 's/torch.load(path, map_location=device, weights_only=True)/torch.load(path, map_location=device)/' \\
          alpha_zero/hierarchical/network/hierarchical_network.py
    $ python -m pytest tests/security/test_no_unsafe_torch_load.py -q -p no:cacheprovider
    FAILED ...::test_every_torch_load_is_weights_only_true - AssertionError:
      torch.load calls that are not weights_only=True:
        alpha_zero/hierarchical/network/hierarchical_network.py:581: weights_only keyword missing
    1 failed, 2 passed
    (keyword restored by re-editing the file; no git state change)
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "alpha_zero"

#: The only module allowed an unconditional ``weights_only=False`` call, and the
#: function that must contain it.
GATED_LOADER_MODULE = "alpha_zero/utils/safe_load.py"
GATED_LOADER_FUNCTION = "safe_torch_load"

#: Burndown allowlist for explicit ``weights_only=False``: file -> (max calls,
#: reason). Missing keywords are never allowlisted. Delete an entry when the
#: site is routed through ``safe_torch_load``; a stale entry fails the test.
UNSAFE_FALSE_ALLOWLIST: dict[str, tuple[int, str]] = {
    "alpha_zero/orchestrator/checkpointer.py": (1, "orchestrator checkpoint with optimizer/meta"),
    "alpha_zero/training/imitation_pretraining.py": (1, "pretraining checkpoint loader"),
    "alpha_zero/spacetimedb/serialization.py": (1, "explicit unsafe deserializer variant"),
    "alpha_zero/storage/gcs_backend.py": (1, "load_checkpoint with training metadata"),
    "alpha_zero/storage/local_backend.py": (1, "load_checkpoint with training metadata"),
    "alpha_zero/distributed/coordinator.py": (1, "coordinator resume path"),
    "alpha_zero/core/transformer_network.py": (1, "TransformerNetwork.load"),
    # Legacy module scheduled for deletion in WS2; harmless once gone.
    "alpha_zero/cloud/gcs_storage.py": (1, "legacy GCS manager, WS2 deletion"),
}


@dataclass(frozen=True)
class LoadCall:
    """One ``torch.load`` call site and how it spells ``weights_only``."""

    path: str
    line: int
    enclosing_function: str | None
    weights_only: object  # True / False / "missing" / "non-literal"

    def __str__(self) -> str:
        if self.weights_only == "missing":
            detail = "weights_only keyword missing"
        elif self.weights_only == "non-literal":
            detail = "weights_only is not a literal (cannot be verified statically)"
        else:
            detail = f"weights_only={self.weights_only}"
        return f"{self.path}:{self.line}: {detail}"


def _is_torch_load(func: ast.expr) -> bool:
    """True for ``torch.load`` and ``torch.serialization.load``."""
    if not (isinstance(func, ast.Attribute) and func.attr == "load"):
        return False
    base = func.value
    if isinstance(base, ast.Name):
        return base.id == "torch"
    if isinstance(base, ast.Attribute) and base.attr == "serialization":
        return isinstance(base.value, ast.Name) and base.value.id == "torch"
    return False


class _TorchLoadVisitor(ast.NodeVisitor):
    def __init__(self, rel_path: str) -> None:
        self.rel_path = rel_path
        self.calls: list[LoadCall] = []
        self._function_stack: list[str] = []

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    visit_FunctionDef = _visit_function  # noqa: N815 - ast API
    visit_AsyncFunctionDef = _visit_function  # noqa: N815 - ast API

    def visit_Call(self, node: ast.Call) -> None:
        if _is_torch_load(node.func):
            spelled: object = "missing"
            for keyword in node.keywords:
                if keyword.arg == "weights_only":
                    value = keyword.value
                    if isinstance(value, ast.Constant) and isinstance(value.value, bool):
                        spelled = value.value
                    else:
                        spelled = "non-literal"
            self.calls.append(
                LoadCall(
                    self.rel_path,
                    node.lineno,
                    self._function_stack[-1] if self._function_stack else None,
                    spelled,
                )
            )
        self.generic_visit(node)


def _scan() -> list[LoadCall]:
    calls: list[LoadCall] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        visitor = _TorchLoadVisitor(rel)
        visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=rel))
        calls.extend(visitor.calls)
    return calls


def test_every_torch_load_is_weights_only_true() -> None:
    """Missing / non-literal keywords are violations everywhere; False only where gated."""
    calls = _scan()
    assert calls, "scanner found no torch.load calls at all -- is the walk broken?"

    violations: list[LoadCall] = []
    false_counts: dict[str, int] = {}
    for call in calls:
        if call.weights_only is True:
            continue
        if call.weights_only is False:
            if call.path == GATED_LOADER_MODULE:
                continue  # checked separately below
            false_counts[call.path] = false_counts.get(call.path, 0) + 1
            allowed = UNSAFE_FALSE_ALLOWLIST.get(call.path)
            if allowed is None or false_counts[call.path] > allowed[0]:
                violations.append(call)
            continue
        violations.append(call)

    assert not violations, (
        "torch.load calls that are not weights_only=True (route them through "
        "alpha_zero.core.checkpoint_utils.safe_torch_load):\n  "
        + "\n  ".join(str(v) for v in violations)
    )


def test_gated_loader_has_exactly_one_unsafe_site() -> None:
    """``checkpoint_utils`` may spell ``weights_only=False`` once, inside the gate."""
    unsafe = [c for c in _scan() if c.path == GATED_LOADER_MODULE and c.weights_only is False]
    assert len(unsafe) == 1, f"expected one gated unsafe site, found: {unsafe}"
    assert unsafe[0].enclosing_function == GATED_LOADER_FUNCTION, (
        f"the unsafe torch.load must live in {GATED_LOADER_FUNCTION}, "
        f"found it in {unsafe[0].enclosing_function}"
    )


def test_unsafe_allowlist_is_not_stale() -> None:
    """Allowlisted files that still exist must still contain a ``weights_only=False``."""
    present = {c.path for c in _scan() if c.weights_only is False}
    stale = [
        rel for rel in UNSAFE_FALSE_ALLOWLIST if (REPO_ROOT / rel).exists() and rel not in present
    ]
    assert not stale, f"Allowlist entries with no remaining unsafe call -- delete them: {stale}"
