"""Every reported win rate must have a draw-aware companion (WS-E / E1).

Two invariants, both enforced by walking the AST rather than grepping, so that
``self.wins``, ``gen_a_wins`` and ``p1_wins`` are all one rule and a string
match on ``"wins /"`` is not:

1. **No hand-rolled win rate.** A division whose numerator's terminal
   identifier matches ``(^|_)wins?$`` is a win rate computed outside the SSOT
   (:mod:`alpha_zero.evaluation.scoring`). The only exception the rule itself
   makes is the *decisive* rate ``wins / (wins + losses)``, which is a
   different statistic and is not what the draw-blind sites were computing.
   Every other offender must be listed in ``draw_aware_allowlist.toml`` with a
   reason; the list is checked for staleness and for empty reasons, the same
   shape as ``NOT_GATED`` in the suite-wiring gate.

2. **Logged win rates travel with ``score_rate`` and ``draw_rate``.** A dict
   literal handed to ``log_metrics`` / ``log_metric`` / ``log_iteration`` /
   ``<something>_logger.log`` whose keys include a ``win_rate`` token must also
   carry a ``score_rate`` and a ``draw_rate`` key. Dict literals bound to a
   name in the same function and then passed are followed; dicts built
   dynamically are not (they are the extractor helpers, which are tested on
   their own).

Negative controls live in :class:`TestScannerNegativeControls` and run on
in-memory source, so they execute on every run rather than once at authoring
time: a win-rate division is flagged, a decisive rate is not, and a metrics
dict missing ``score_rate`` fails.
"""

from __future__ import annotations

import ast
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST_PATH = Path(__file__).with_name("draw_aware_allowlist.toml")
SCAN_ROOTS = ("alpha_zero", "scripts")
SKIP_PARTS = {"__pycache__", "build", "dist", ".tox", ".eggs", "tests"}

_WIN_NAME = re.compile(r"(^|_)wins?$")
_LOSS_NAME = re.compile(r"(^|_)loss(es)?$")
# A metric key carries a win rate when `win_rate` appears as a whole token
# anywhere in it: "eval/win_rate", "orchestrator/eval_win_rate",
# "validation/win_rate_vs_random" all count; "window_rate" does not.
_WIN_RATE_TOKEN = re.compile(r"(^|/|_)win_rate(_|$)")
_SCORE_RATE_TOKEN = re.compile(r"(^|/|_)score_rate(_|$)")
_DRAW_RATE_TOKEN = re.compile(r"(^|/|_)draw_rate(_|$)")
_METRIC_SINKS = {"log_metrics", "log_metric", "log_iteration"}
_MIN_REASON_CHARS = 12


@dataclass(frozen=True)
class Offender:
    path: str
    qualname: str
    lineno: int
    detail: str

    @property
    def key(self) -> str:
        return f"{self.path}::{self.qualname}"


def _terminal_identifier(node: ast.AST) -> str | None:
    """``wins`` for ``wins``, ``self.wins`` and ``stats.wins[0]``; None otherwise."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _terminal_identifier(node.value)
    return None


def _is_decisive_denominator(node: ast.AST) -> bool:
    """``wins + losses`` (either order): the decisive rate, deliberately not flagged."""
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)):
        return False
    names = [_terminal_identifier(node.left), _terminal_identifier(node.right)]
    if any(n is None for n in names):
        return False
    wins = [n for n in names if n is not None and _WIN_NAME.search(n)]
    losses = [n for n in names if n is not None and _LOSS_NAME.search(n)]
    return len(wins) == 1 and len(losses) == 1


class _QualnameVisitor(ast.NodeVisitor):
    """Base visitor that tracks ``Class.method`` qualnames while walking."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._stack: list[str] = []
        self.offenders: list[Offender] = []

    @property
    def qualname(self) -> str:
        return ".".join(self._stack) if self._stack else "<module>"

    def _enter(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    visit_ClassDef = _enter  # noqa: N815 - ast visitor protocol
    visit_FunctionDef = _enter  # noqa: N815
    visit_AsyncFunctionDef = _enter  # noqa: N815


class _WinDivisionVisitor(_QualnameVisitor):
    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Div):
            numerator = _terminal_identifier(node.left)
            if (
                numerator is not None
                and _WIN_NAME.search(numerator)
                and not _is_decisive_denominator(node.right)
            ):
                self.offenders.append(
                    Offender(self.path, self.qualname, node.lineno, ast.unparse(node))
                )
        self.generic_visit(node)


def find_win_divisions(source: str, path: str = "<string>") -> list[Offender]:
    """Divisions with a win-count numerator, excluding ``wins / (wins + losses)``."""
    visitor = _WinDivisionVisitor(path)
    visitor.visit(ast.parse(source))
    return visitor.offenders


def _key_text(node: ast.AST) -> str | None:
    """The string of a dict key: a constant, or an f-string's literal parts joined."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = [v.value for v in node.values if isinstance(v, ast.Constant)]
        return "".join(str(p) for p in parts) if parts else None
    return None


def _dict_keys(node: ast.Dict) -> list[str]:
    return [text for key in node.keys if key is not None and (text := _key_text(key))]


def _sink_name(func: ast.AST) -> str | None:
    """Return the sink's method name if ``func`` names a metrics sink."""
    if isinstance(func, ast.Attribute):
        if func.attr in _METRIC_SINKS:
            return func.attr
        if func.attr == "log":
            owner = _terminal_identifier(func.value)
            if owner is not None and owner.endswith("logger"):
                return "log"
        return None
    if isinstance(func, ast.Name) and func.id in _METRIC_SINKS:
        return func.id
    return None


class _MetricPairingVisitor(_QualnameVisitor):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        # name -> most recent dict literal bound to it in the current function.
        self._bindings: list[dict[str, ast.Dict]] = [{}]

    def _enter(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._bindings.append({})
        super()._enter(node)
        self._bindings.pop()

    visit_ClassDef = _enter  # noqa: N815
    visit_FunctionDef = _enter  # noqa: N815
    visit_AsyncFunctionDef = _enter  # noqa: N815

    def _bind(self, target: ast.AST, value: ast.AST | None) -> None:
        if isinstance(target, ast.Name) and isinstance(value, ast.Dict):
            self._bindings[-1][target.id] = value

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._bind(target, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._bind(node.target, node.value)
        self.generic_visit(node)

    def _resolve_dict(self, node: ast.AST) -> ast.Dict | None:
        if isinstance(node, ast.Dict):
            return node
        if isinstance(node, ast.Name):
            for scope in reversed(self._bindings):
                if node.id in scope:
                    return scope[node.id]
        return None

    def visit_Call(self, node: ast.Call) -> None:
        sink = _sink_name(node.func)
        if sink is not None:
            candidates = list(node.args) + [kw.value for kw in node.keywords]
            if sink == "log_metric":
                # Single-metric form: log_metric("name", value) cannot pair.
                for arg in node.args[:1]:
                    text = _key_text(arg)
                    if text is not None and _WIN_RATE_TOKEN.search(text):
                        self.offenders.append(
                            Offender(self.path, self.qualname, node.lineno, f"log_metric({text!r})")
                        )
            for arg in candidates:
                metrics = self._resolve_dict(arg)
                if metrics is None:
                    continue
                keys = _dict_keys(metrics)
                win_keys = [k for k in keys if _WIN_RATE_TOKEN.search(k)]
                if not win_keys:
                    continue
                has_score = any(_SCORE_RATE_TOKEN.search(k) for k in keys)
                has_draw = any(_DRAW_RATE_TOKEN.search(k) for k in keys)
                if not (has_score and has_draw):
                    missing = [
                        name
                        for name, ok in (("score_rate", has_score), ("draw_rate", has_draw))
                        if not ok
                    ]
                    self.offenders.append(
                        Offender(
                            self.path,
                            self.qualname,
                            node.lineno,
                            f"{sink} dict with {win_keys} lacks {missing}",
                        )
                    )
        self.generic_visit(node)


def find_unpaired_metric_dicts(source: str, path: str = "<string>") -> list[Offender]:
    """Metrics dicts that log a win rate without ``score_rate`` and ``draw_rate``."""
    visitor = _MetricPairingVisitor(path)
    visitor.visit(ast.parse(source))
    return visitor.offenders


def _python_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            if SKIP_PARTS & set(path.parts):
                continue
            files.append(path)
    return files


def _scan(finder: object) -> dict[str, list[Offender]]:
    found: dict[str, list[Offender]] = {}
    for path in _python_files():
        try:
            source = path.read_text(encoding="utf-8")
            offenders = finder(source, path.relative_to(REPO_ROOT).as_posix())  # type: ignore[operator]
        except (OSError, SyntaxError):
            continue
        for offender in offenders:
            found.setdefault(offender.key, []).append(offender)
    return found


def _load_allowlist() -> dict[str, dict[str, str]]:
    with ALLOWLIST_PATH.open("rb") as fh:
        data = tomllib.load(fh)
    return {
        "win_division": dict(data.get("win_division", {})),
        "metric_pairing": dict(data.get("metric_pairing", {})),
    }


def _format(offenders: list[Offender]) -> str:
    return "\n".join(f"  {o.path}:{o.lineno} [{o.qualname}] {o.detail}" for o in offenders)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


class TestNoHandRolledWinRate:
    def test_every_win_division_is_allowlisted(self) -> None:
        found = _scan(find_win_divisions)
        allowed = _load_allowlist()["win_division"]
        unlisted = [o for key, items in found.items() if key not in allowed for o in items]
        assert not unlisted, (
            "win rates computed outside alpha_zero.evaluation.scoring (route through "
            "scoring.win_rate/score_rate, or add to tests/regression/draw_aware_allowlist.toml "
            "with a reason):\n" + _format(unlisted)
        )

    def test_allowlist_is_not_stale(self) -> None:
        found = _scan(find_win_divisions)
        stale = sorted(set(_load_allowlist()["win_division"]) - set(found))
        assert not stale, f"allowlist entries no longer match a division: {stale}"


class TestLoggedWinRatesArePaired:
    def test_every_unpaired_metrics_dict_is_allowlisted(self) -> None:
        found = _scan(find_unpaired_metric_dicts)
        allowed = _load_allowlist()["metric_pairing"]
        unlisted = [o for key, items in found.items() if key not in allowed for o in items]
        assert not unlisted, (
            "a logged win_rate without score_rate and draw_rate beside it:\n" + _format(unlisted)
        )

    def test_allowlist_is_not_stale(self) -> None:
        found = _scan(find_unpaired_metric_dicts)
        stale = sorted(set(_load_allowlist()["metric_pairing"]) - set(found))
        assert not stale, f"allowlist entries no longer match an unpaired dict: {stale}"


class TestAllowlistHygiene:
    @pytest.mark.parametrize("section", ["win_division", "metric_pairing"])
    def test_reasons_are_real_sentences(self, section: str) -> None:
        entries = _load_allowlist()[section]
        assert entries, f"{section} should carry at least the SSOT's own entries"
        thin = {k: v for k, v in entries.items() if len(v.strip()) < _MIN_REASON_CHARS}
        assert not thin, f"{section} entries without a reason: {thin}"

    def test_keys_name_existing_files(self) -> None:
        for section in ("win_division", "metric_pairing"):
            for key in _load_allowlist()[section]:
                path, _, qualname = key.partition("::")
                assert (REPO_ROOT / path).is_file(), f"{section}: {path} does not exist"
                assert qualname, f"{section}: {key} lacks a qualname"


# ---------------------------------------------------------------------------
# Negative controls, on in-memory source so they run every time
# ---------------------------------------------------------------------------


class TestScannerNegativeControls:
    def test_flags_a_bare_win_rate(self) -> None:
        assert find_win_divisions("x = gen_a_wins / total_matches")

    def test_flags_an_attribute_win_rate(self) -> None:
        found = find_win_divisions(
            "class R:\n    def rate(self):\n        return self.wins / self.games"
        )
        assert [o.qualname for o in found] == ["R.rate"]

    def test_flags_len_denominator(self) -> None:
        assert find_win_divisions("r = p1_wins / len(matches)")

    def test_does_not_flag_the_decisive_rate(self) -> None:
        assert not find_win_divisions("r = wins / (wins + losses)")
        assert not find_win_divisions("r = self.wins / (self.losses + self.wins)")

    def test_does_not_flag_other_numerators(self) -> None:
        assert not find_win_divisions("r = losses / total\nd = draws / total\nw = window / 2")

    def test_flags_metrics_dict_missing_score_rate(self) -> None:
        source = 'logger.log_metrics({"eval/win_rate": w, "eval/draw_rate": d})'
        found = find_unpaired_metric_dicts(source)
        assert found and "score_rate" in found[0].detail

    def test_accepts_a_paired_metrics_dict(self) -> None:
        source = (
            'logger.log_metrics({"eval/win_rate": w, "eval/score_rate": s, "eval/draw_rate": d})'
        )
        assert not find_unpaired_metric_dicts(source)

    def test_follows_a_name_bound_dict(self) -> None:
        source = (
            "def f():\n"
            '    metrics = {"game/win_rate": 1.0}\n'
            "    self._mlflow_logger.log_iteration(iteration=1, game_metrics=metrics)\n"
        )
        found = find_unpaired_metric_dicts(source)
        assert [o.qualname for o in found] == ["f"]

    def test_fstring_keys_are_read(self) -> None:
        source = 'ml.log_metrics({f"{name}/win_rate": 1.0})'
        assert find_unpaired_metric_dicts(source)

    def test_single_metric_form_is_flagged(self) -> None:
        assert find_unpaired_metric_dicts('mlflow.log_metric("eval/win_rate", 0.5)')

    def test_logger_dot_log_counts_as_a_sink(self) -> None:
        assert find_unpaired_metric_dicts('wandb_logger.log({"eval/win_rate": 0.5})')
        assert not find_unpaired_metric_dicts('console.log({"eval/win_rate": 0.5})')
