"""Hook scripts must parse, be registered, and be gated by pre-commit.

A hook that fails ``bash -n`` exits non-zero at run time; a hook that is not
registered anywhere is dead; a guard edit that triggers no pre-commit hook is
unchecked. Each helper has a negative control further down.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path

import pytest

from tests.governance._bash import resolve_bash

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOKS_DIR = REPO_ROOT / ".claude" / "hooks"
SETTINGS = REPO_ROOT / ".claude" / "settings.json"
AGENTS_DIR = REPO_ROOT / ".claude" / "agents"
MAKEFILE = REPO_ROOT / "Makefile"
PRECOMMIT = REPO_ROOT / ".pre-commit-config.yaml"

_HOOK_REF_RE = re.compile(r"\.claude/hooks/([\w.-]+\.sh)")
GUARD_PATHS = (
    ".claude/hooks/pretooluse_guard.sh",
    ".claude/hooks/_guard_git.sh",
    ".claude/hooks/_guard_state.sh",
)


def bash_syntax_error(path: Path) -> str | None:
    """Return ``bash -n`` stderr when ``path`` does not parse, else ``None``."""
    proc = subprocess.run(
        [resolve_bash(), "-n", str(path)], capture_output=True, text=True, check=False
    )
    return None if proc.returncode == 0 else (proc.stderr or "bash -n failed")


def _settings_commands(settings: Path) -> list[str]:
    data = json.loads(settings.read_text(encoding="utf-8"))
    return [
        hook["command"]
        for groups in data.get("hooks", {}).values()
        for group in groups
        for hook in group.get("hooks", [])
        if "command" in hook
    ]


def missing_registered_files(settings: Path, hooks_dir: Path) -> list[str]:
    """Hook files named by settings.json commands that are not on disk."""
    return sorted(
        {
            name
            for command in _settings_commands(settings)
            for name in _HOOK_REF_RE.findall(command)
            if not (hooks_dir / name).is_file()
        }
    )


def orphan_hooks(settings: Path, hooks_dir: Path, agents_dir: Path) -> list[str]:
    """Hook files neither registered, sourced by another hook, nor named by an agent."""
    registered = {n for c in _settings_commands(settings) for n in _HOOK_REF_RE.findall(c)}
    agent_text = "\n".join(p.read_text(encoding="utf-8") for p in sorted(agents_dir.glob("*.md")))
    registered |= set(_HOOK_REF_RE.findall(agent_text))
    orphans = []
    for hook in sorted(hooks_dir.glob("*.sh")):
        if hook.name in registered:
            continue
        sourced = any(
            re.search(
                rf"^\s*(?:source|\.)\s+\S*{re.escape(hook.name)}", other.read_text("utf-8"), re.M
            )
            for other in hooks_dir.glob("*.sh")
            if other != hook
        )
        if not sourced:
            orphans.append(hook.name)
    return orphans


def _make_recipe_text(makefile: Path, target: str) -> str | None:
    """The target's own line plus its recipe lines, or ``None`` when undefined."""
    lines = makefile.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.startswith(f"{target}:"):
            body = [line]
            for follow in lines[i + 1 :]:
                if follow.startswith("\t"):
                    body.append(follow)
                else:
                    break
            return "\n".join(body)
    return None


_HOOKS = sorted(HOOKS_DIR.glob("*.sh"))


def test_hooks_directory_is_not_empty() -> None:
    assert len(_HOOKS) >= 10


@pytest.mark.parametrize("hook", _HOOKS, ids=lambda p: p.name)
def test_hook_passes_bash_n(hook: Path) -> None:
    assert bash_syntax_error(hook) is None


def test_settings_hook_commands_name_existing_files() -> None:
    assert missing_registered_files(SETTINGS, HOOKS_DIR) == []


def test_every_hook_is_reachable() -> None:
    assert orphan_hooks(SETTINGS, HOOKS_DIR, AGENTS_DIR) == []


def test_makefile_has_lint_hooks_target_run_by_validate() -> None:
    assert _make_recipe_text(MAKEFILE, "lint-hooks") is not None, "no lint-hooks: target"
    validate = _make_recipe_text(MAKEFILE, "validate")
    assert validate is not None
    assert "lint-hooks" in validate, "validate does not depend on or run lint-hooks"


_HOOK_ID_RE = re.compile(r"^(\s*)- id:\s*(\S+)\s*$")
_ANCHOR_RE = re.compile(r"^&(\w+)\s+")
_BLOCK_INDICATORS = {">", ">-", "|", "|-"}


def _yaml_scalar(raw: str, anchors: dict[str, str]) -> str:
    """One inline value: an alias, an anchored value or a (quoted) scalar."""
    value = raw.strip()
    if value.startswith("*"):
        return anchors.get(value[1:], "")
    anchor = _ANCHOR_RE.match(value)
    if anchor:
        value = value[anchor.end() :]
    if len(value) >= 2 and value[0] == value[-1] == "'":
        value = value[1:-1].replace("''", "'")
    elif len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1]
    if anchor:
        anchors[anchor.group(1)] = value
    return value


def parse_precommit_hooks(text: str) -> list[dict[str, str]]:
    """The hooks of a pre-commit config, each as ``key -> string``, read with the stdlib.

    The governance suite runs where only pytest and ruff are installed, so ``yaml`` is
    not available. This reads the shape the file uses: ``- id:`` items with ``key: value``
    lines one level deeper, anchors and aliases, quoted scalars and folded blocks (``>-``).
    """
    hooks: list[dict[str, str]] = []
    anchors: dict[str, str] = {}
    current: dict[str, str] | None = None
    key_indent = 0
    block_key: str | None = None
    block: list[str] = []
    for line in text.splitlines():
        lead = len(line) - len(line.lstrip())
        if block_key is not None and current is not None:
            if not line.strip() or lead > key_indent:
                block.append(line.strip())
                continue
            current[block_key] = " ".join(b for b in block if b)
            block_key = None
        match = _HOOK_ID_RE.match(line)
        if match:
            current = {"id": match.group(2)}
            key_indent = len(match.group(1)) + 2
            hooks.append(current)
            continue
        stripped = line.strip()
        if current is None or not stripped or stripped.startswith("#"):
            continue
        if lead < key_indent:
            current = None
            continue
        if lead == key_indent and ":" in stripped:
            key, _, value = stripped.partition(":")
            if value.strip() in _BLOCK_INDICATORS:
                block_key, block = key.strip(), []
            else:
                current[key.strip()] = _yaml_scalar(value, anchors)
    if block_key is not None and current is not None:
        current[block_key] = " ".join(b for b in block if b)
    return hooks


def _precommit_hooks() -> list[dict[str, str]]:
    return parse_precommit_hooks(PRECOMMIT.read_text(encoding="utf-8"))


def _guard_hooks() -> list[dict[str, str]]:
    return [
        h
        for h in _precommit_hooks()
        if h.get("files") and all(re.search(h["files"], p) for p in GUARD_PATHS)
    ]


GOVERNANCE_DIR = REPO_ROOT / "tests" / "governance"
# The guard and hook test files by name; a new one must be in the pre-commit entry.
_HOOK_TEST_NAME_RE = re.compile(r"^test_(hook_.*|.*guard.*|readonly_shell_guard|qa_shell_.*)\.py$")


def hook_test_files() -> set[str]:
    """Governance test files that must run on a hook edit.

    Every file named like a hook or guard test, and every file whose text names one of
    the real ``.claude/hooks/*.sh`` scripts.
    """
    names = [h.name for h in HOOKS_DIR.glob("*.sh")]
    required = set()
    for test in GOVERNANCE_DIR.glob("test_*.py"):
        text = test.read_text(encoding="utf-8")
        if _HOOK_TEST_NAME_RE.match(test.name) or any(n in text for n in names):
            required.add(test.name)
    return required


def selected_test_files(entry: str) -> set[str]:
    """The governance test files a pytest ``entry`` runs.

    Explicit ``.py`` paths are taken as they are. A directory is all of its test files,
    narrowed by ``-k`` at file level: a file counts only when its name contains one of
    the ``-k`` words. That is an approximation (``-k`` matches test names too), and it
    is why the entry names files instead of using ``-k``.
    """
    tokens = shlex.split(entry)
    k_expr = tokens[tokens.index("-k") + 1] if "-k" in tokens else None
    terms = [t for t in re.split(r"\s+|\(|\)", k_expr or "") if t and t not in {"or", "and", "not"}]
    selected: set[str] = set()
    for token in tokens:
        if token.endswith(".py"):
            selected.add(Path(token).name)
        elif token.startswith("tests/") and (REPO_ROOT / token).is_dir():
            for test in (REPO_ROOT / token).glob("test_*.py"):
                if k_expr is None or any(t in test.stem for t in terms):
                    selected.add(test.name)
    return selected


def test_precommit_fires_on_guard_edits() -> None:
    candidates = _guard_hooks()
    assert candidates, "no pre-commit hook whose files: matches all three guard files"
    entries = " ".join(str(h.get("entry", "")) for h in candidates)
    assert "test_hook_syntax" in entries, "entry does not run the hook-syntax test"
    assert "test_*guard*" in entries or "tests/governance" in entries, (
        "entry does not run the guard tests"
    )


def test_precommit_entry_selects_every_hook_test() -> None:
    selected: set[str] = set()
    for hook in _guard_hooks():
        selected |= selected_test_files(hook.get("entry", ""))
    missing = sorted(hook_test_files() - selected)
    assert not missing, f"hook tests a hook edit does not run: {missing}"


def test_precommit_entry_names_only_existing_files() -> None:
    for hook in _guard_hooks():
        for token in shlex.split(hook.get("entry", "")):
            if token.endswith(".py"):
                assert (REPO_ROOT / token).is_file(), token


# ---- negative controls -------------------------------------------------


def test_control_old_k_entry_misses_qa_shell_tests() -> None:
    old = (
        "python -m pytest -c tests/governance/pytest.ini --confcutdir=tests/governance -q "
        '-p no:cacheprovider -k "test_hook_syntax or guard" tests/governance'
    )
    missing = hook_test_files() - selected_test_files(old)
    assert "test_qa_shell_require_lua.py" in missing


def test_control_stdlib_parser_reads_aliases_quotes_and_blocks() -> None:
    text = (
        "repos:\n  - repo: local\n    hooks:\n"
        "      - id: a\n        files: &m '^(x/|y''s)'\n        entry: python a.py\n"
        "      - id: b\n        files: *m\n        entry: >-\n          python -m pytest\n"
        "          tests/one.py\n        language: system\n"
    )
    hooks = parse_precommit_hooks(text)
    assert hooks == [
        {"id": "a", "files": "^(x/|y's)", "entry": "python a.py"},
        {
            "id": "b",
            "files": "^(x/|y's)",
            "entry": "python -m pytest tests/one.py",
            "language": "system",
        },
    ]


# ---- negative controls (helpers) ----------------------------------------


def test_control_syntax_error_is_caught(tmp_path: Path) -> None:
    bad = tmp_path / "bad.sh"
    bad.write_text("#!/bin/bash\nif then fi (\n", encoding="utf-8")
    assert bash_syntax_error(bad) is not None
    good = tmp_path / "good.sh"
    good.write_text("#!/bin/bash\necho ok\n", encoding="utf-8")
    assert bash_syntax_error(good) is None


def test_control_missing_registered_file_is_caught(tmp_path: Path) -> None:
    (tmp_path / "present.sh").write_text("echo\n", encoding="utf-8")
    settings = tmp_path / "settings.json"
    cmd = 'bash "$CLAUDE_PROJECT_DIR"/.claude/hooks/%s'
    payload = {
        "hooks": {
            "Stop": [{"hooks": [{"command": cmd % "present.sh"}, {"command": cmd % "ghost.sh"}]}]
        }
    }
    settings.write_text(json.dumps(payload), encoding="utf-8")
    assert missing_registered_files(settings, tmp_path) == ["ghost.sh"]


def test_control_orphan_is_caught_and_sourcing_counts(tmp_path: Path) -> None:
    hooks = tmp_path / "hooks"
    agents = tmp_path / "agents"
    hooks.mkdir()
    agents.mkdir()
    (hooks / "main.sh").write_text('. "${D}/lib.sh"\n', encoding="utf-8")
    (hooks / "lib.sh").write_text("echo\n", encoding="utf-8")
    (hooks / "stray.sh").write_text("echo\n", encoding="utf-8")
    settings = tmp_path / "settings.json"
    cmd = 'bash "$CLAUDE_PROJECT_DIR"/.claude/hooks/main.sh'
    settings.write_text(
        json.dumps({"hooks": {"Stop": [{"hooks": [{"command": cmd}]}]}}), encoding="utf-8"
    )
    assert orphan_hooks(settings, hooks, agents) == ["stray.sh"]
