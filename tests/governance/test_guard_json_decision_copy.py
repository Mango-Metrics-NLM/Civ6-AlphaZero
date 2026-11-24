"""The two hooks carry one copy each of ``json_decision``; they must stay byte-identical."""

from __future__ import annotations

from pathlib import Path

HOOKS = Path(__file__).resolve().parents[2] / ".claude" / "hooks"


def _function_body(path: Path, name: str) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line == f"{name}() {{")
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "}")
    return "\n".join(lines[start : end + 1])


def test_json_decision_is_identical_in_both_hooks() -> None:
    a = _function_body(HOOKS / "pretooluse_guard.sh", "json_decision")
    b = _function_body(HOOKS / "readonly_shell_guard.sh", "json_decision")
    assert a == b
    assert "jq -n" in a and "json.dumps" in a  # the extraction found the real builder
