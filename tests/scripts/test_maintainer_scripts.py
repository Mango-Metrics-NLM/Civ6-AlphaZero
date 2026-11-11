"""Maintainer cut-over tooling: ``scripts/maintainer/{rulesets.py,branch_cutover.sh,merge_backmerge.sh}``.

Nothing here touches GitHub. The shell scripts take their ``gh`` from ``$GH``;
these tests point it at ``fake_gh`` (a shim around a small Python responder)
that logs every call and answers reads from a canned table. "A write ran" is
read from that log, never inferred from the scripts' own output.

Negative controls (run when written, then reverted):
* making ``mutate`` run its command without --execute failed
  ``test_dry_run_rulesets_performs_no_write``;
* deleting the CLAUDECODE refusal failed
  ``test_execute_refuses_inside_claude_code``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from scripts.maintainer import rulesets
from tests.scripts.git_fixture import GitRepo, git_env

REPO_ROOT = Path(__file__).resolve().parents[2]
CUTOVER = REPO_ROOT / "scripts" / "maintainer" / "branch_cutover.sh"
BACKMERGE = REPO_ROOT / "scripts" / "maintainer" / "merge_backmerge.sh"
BASH = shutil.which("bash")
REPO = "acme/widget"
OLD = "claude/civ6-alphazero-implementation-vMnun"

needs_bash = pytest.mark.skipif(BASH is None, reason="bash is not on PATH")


# ---------------------------------------------------------------------------
# rulesets.py
# ---------------------------------------------------------------------------


class TestRender:
    def test_sentinel_is_replaced_by_the_maintainer_id(self) -> None:
        body = rulesets.render(rulesets.load("release-tags"), maintainer_id=4242)
        assert body["bypass_actors"] == [
            {"actor_id": 4242, "actor_type": "User", "bypass_mode": "always"}
        ]
        assert rulesets.SENTINEL not in json.dumps(body)

    def test_branch_direction_can_be_withheld_for_step_4(self) -> None:
        full = rulesets.render(rulesets.load("qa"), maintainer_id=1)
        partial = rulesets.render(
            rulesets.load("qa"), maintainer_id=1, without_checks=["Branch Direction"]
        )

        def contexts(doc: dict[str, Any]) -> list[str]:
            rule = next(r for r in doc["rules"] if r["type"] == "required_status_checks")
            return [c["context"] for c in rule["parameters"]["required_status_checks"]]

        assert "Branch Direction" in contexts(full)
        assert "Branch Direction" not in contexts(partial)
        assert len(contexts(partial)) == len(contexts(full)) - 1

    def test_render_does_not_mutate_the_loaded_document(self) -> None:
        doc = rulesets.load("qa-main-merge")
        rulesets.render(doc, maintainer_id=7)
        assert doc["bypass_actors"][0]["actor_id"] == rulesets.SENTINEL

    @pytest.mark.parametrize("bad", [0, -1, True, "4242"])
    def test_invalid_maintainer_id_is_refused(self, bad: Any) -> None:
        with pytest.raises(rulesets.RulesetError):
            rulesets.render(rulesets.load("release-tags"), maintainer_id=bad)

    def test_a_surviving_sentinel_is_refused(self) -> None:
        doc = rulesets.load("release-tags")
        doc["conditions"]["ref_name"]["exclude"] = [rulesets.SENTINEL]
        with pytest.raises(rulesets.RulesetError, match="survived"):
            rulesets.render(doc, maintainer_id=5)

    def test_dropping_every_check_is_refused(self) -> None:
        doc = rulesets.load("dev")
        rule = next(r for r in doc["rules"] if r["type"] == "required_status_checks")
        every = [c["context"] for c in rule["parameters"]["required_status_checks"]]
        with pytest.raises(rulesets.RulesetError):
            rulesets.render(doc, maintainer_id=5, without_checks=every)

    def test_ruleset_id_needs_exactly_one_repository_match(self) -> None:
        listing = [
            {"id": 1, "name": "dev", "source_type": "Repository"},
            {"id": 2, "name": "dev", "source_type": "Organization"},
            {"id": 3, "name": "qa", "source_type": "Repository"},
        ]
        assert rulesets.ruleset_id(listing, "dev") == 1
        with pytest.raises(rulesets.RulesetError):
            rulesets.ruleset_id(listing, "main")
        with pytest.raises(rulesets.RulesetError):
            rulesets.ruleset_id([*listing, {"id": 4, "name": "qa"}], "qa")

    def test_cli_render_and_id_of(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert (
            rulesets.main(
                ["render", "main", "--maintainer-id", "9", "--without-check", "Branch Direction"]
            )
            == 0
        )
        body = json.loads(capsys.readouterr().out)
        assert body["name"] == "main"
        monkeypatch.setattr(
            sys, "stdin", __import__("io").StringIO(json.dumps([{"id": 11, "name": "main"}]))
        )
        assert rulesets.main(["id-of", "main"]) == 0
        assert capsys.readouterr().out.strip() == "11"

    def test_cutover_order_covers_every_file(self) -> None:
        on_disk = {p.stem for p in rulesets.RULESET_DIR.glob("*.json")}
        assert set(rulesets.CUTOVER_ORDER) == on_disk


# ---------------------------------------------------------------------------
# A fake gh
# ---------------------------------------------------------------------------

_RESPONDER = textwrap.dedent(
    """
    import json, os, sys
    args = sys.argv[1:]
    with open(os.environ["FAKE_GH_LOG"], "a", encoding="utf-8") as fh:
        fh.write(json.dumps(args) + "\\n")
    table = json.load(open(os.environ["FAKE_GH_TABLE"], encoding="utf-8"))
    for entry in table:
        prefix = entry["prefix"]
        if args[: len(prefix)] == prefix:
            sys.stdout.write(entry.get("stdout", ""))
            sys.exit(entry.get("exit", 0))
    # Unlisted writes succeed silently; unlisted reads fail loudly.
    if "-X" in args or args[:2] in (["pr", "close"], ["pr", "comment"], ["pr", "merge"], ["pr", "update-branch"]):
        sys.exit(0)
    sys.stderr.write("fake gh: no canned answer for " + " ".join(args) + "\\n")
    sys.exit(1)
    """
)


class FakeGh:
    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.log = root / "gh.log"
        self.table_path = root / "gh_table.json"
        responder = root / "fake_gh.py"
        responder.write_text(_RESPONDER, encoding="utf-8")
        self.shim = root / "gh"
        python = Path(sys.executable).as_posix()
        self.shim.write_text(
            f'#!/usr/bin/env bash\nexec "{python}" "{responder.as_posix()}" "$@"\n',
            encoding="utf-8",
        )
        self.shim.chmod(0o755)
        self.table: list[dict[str, Any]] = []
        self.log.write_text("", encoding="utf-8")

    def answer(self, prefix: list[str], stdout: str = "", exit: int = 0) -> None:
        self.table.append({"prefix": prefix, "stdout": stdout, "exit": exit})

    def calls(self) -> list[list[str]]:
        return [
            json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines() if line
        ]

    def writes(self) -> list[list[str]]:
        return [
            c
            for c in self.calls()
            if "-X" in c
            or c[:2]
            in (["pr", "close"], ["pr", "comment"], ["pr", "merge"], ["pr", "update-branch"])
        ]

    def env(self, **extra: str) -> dict[str, str]:
        self.table_path.write_text(json.dumps(self.table), encoding="utf-8")
        env = {**git_env(), "GH": self.shim.as_posix(), "PYTHON": Path(sys.executable).as_posix()}
        env.update(
            {
                "FAKE_GH_LOG": str(self.log),
                "FAKE_GH_TABLE": str(self.table_path),
                "CIV6AZ_REPO": REPO,
            }
        )
        env.pop("CLAUDECODE", None)
        env.update(extra)
        return env


def _run(
    script: Path, args: list[str], env: dict[str, str], cwd: Path
) -> subprocess.CompletedProcess[str]:
    assert BASH is not None
    return subprocess.run(
        [BASH, script.as_posix(), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.fixture
def gh(tmp_path: Path) -> FakeGh:
    fake = FakeGh(tmp_path / "fakegh")
    fake.answer(["api", "user", "--jq", ".id"], "4242\n")
    fake.answer(["api", "user", "--jq", ".login"], "maintainer\n")
    return fake


# ---------------------------------------------------------------------------
# branch_cutover.sh
# ---------------------------------------------------------------------------


@needs_bash
class TestCutover:
    def test_dry_run_rulesets_performs_no_write(self, gh: FakeGh, tmp_path: Path) -> None:
        out = tmp_path / "out"
        proc = _run(CUTOVER, ["rulesets"], gh.env(CIV6AZ_CUTOVER_DIR=out.as_posix()), tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert gh.writes() == []
        assert proc.stdout.count("-X POST repos/acme/widget/rulesets --input") == 6
        for name in rulesets.CUTOVER_ORDER:
            body = (out / f"ruleset-{name}.json").read_text(encoding="utf-8")
            assert "Branch Direction" not in body
            assert rulesets.SENTINEL not in body

    def test_execute_rulesets_posts_all_six_in_order(self, gh: FakeGh, tmp_path: Path) -> None:
        out = tmp_path / "out"
        proc = _run(
            CUTOVER, ["--execute", "rulesets"], gh.env(CIV6AZ_CUTOVER_DIR=out.as_posix()), tmp_path
        )
        assert proc.returncode == 0, proc.stderr
        posts = gh.writes()
        assert [c[:4] for c in posts] == [["api", "-X", "POST", f"repos/{REPO}/rulesets"]] * 6
        names = [json.loads(Path(c[-1]).read_text(encoding="utf-8"))["name"] for c in posts]
        assert names == list(rulesets.CUTOVER_ORDER)
        tags = json.loads((out / "ruleset-release-tags.json").read_text(encoding="utf-8"))
        assert tags["bypass_actors"][0]["actor_id"] == 4242

    def test_execute_refuses_inside_claude_code(self, gh: FakeGh, tmp_path: Path) -> None:
        proc = _run(CUTOVER, ["--execute", "rulesets"], gh.env(CLAUDECODE="1"), tmp_path)
        assert proc.returncode == 2
        assert "Claude Code" in proc.stderr
        assert gh.writes() == []

    def test_execute_refuses_the_agent_machine_user(self, tmp_path: Path) -> None:
        fake = FakeGh(tmp_path / "fakegh")
        fake.answer(["api", "user", "--jq", ".login"], "civ6az-agent\n")
        proc = _run(CUTOVER, ["--execute", "settings"], fake.env(), tmp_path)
        assert proc.returncode == 2
        assert fake.writes() == []

    def test_create_branches_at_the_release_commit(self, gh: FakeGh, tmp_path: Path) -> None:
        repo, _bare = GitRepo.with_bare_remote(tmp_path / "git")
        release = repo.commit("release 2.0.0")
        repo.tag("v2.0.0")
        repo.push("main", "--tags")
        gh.answer(["api", f"repos/{REPO}/branches/"], exit=1)  # neither qa nor main exists yet
        env = gh.env(CIV6AZ_RELEASE_COMMIT=release[:7])
        proc = _run(CUTOVER, ["--execute", "create-branches"], env, repo.path)
        assert proc.returncode == 0, proc.stderr
        refs = [c for c in gh.writes() if c[3] == f"repos/{REPO}/git/refs"]
        assert [c[-2:] for c in refs] == [["-f", f"sha={release}"], ["-f", f"sha={release}"]]
        assert [c[5] for c in refs] == ["ref=refs/heads/qa", "ref=refs/heads/main"]

    def test_create_branches_refuses_when_the_tag_is_elsewhere(
        self, gh: FakeGh, tmp_path: Path
    ) -> None:
        repo, _bare = GitRepo.with_bare_remote(tmp_path / "git")
        release = repo.commit("release 2.0.0")
        repo.commit("later")
        repo.tag("v2.0.0")  # on the wrong commit
        repo.push("main", "--tags")
        proc = _run(
            CUTOVER,
            ["--execute", "create-branches"],
            gh.env(CIV6AZ_RELEASE_COMMIT=release),
            repo.path,
        )
        assert proc.returncode == 2
        assert "is not v2.0.0^{commit}" in proc.stderr
        assert gh.writes() == []

    def test_rename_refuses_an_unexpected_default(self, gh: FakeGh, tmp_path: Path) -> None:
        gh.answer(["api", f"repos/{REPO}", "--jq", ".default_branch"], "dev\n")
        proc = _run(CUTOVER, ["--execute", "rename"], gh.env(), tmp_path)
        assert proc.returncode == 2
        assert gh.writes() == []

    def test_rename_renames_the_old_default_to_dev(self, gh: FakeGh, tmp_path: Path) -> None:
        gh.answer(["api", f"repos/{REPO}", "--jq", ".default_branch"], OLD + "\n")
        proc = _run(CUTOVER, ["--execute", "rename"], gh.env(), tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert gh.writes() == [
            ["api", "-X", "POST", f"repos/{REPO}/branches/{OLD}/rename", "-f", "new_name=dev"]
        ]

    def _dev_with_t6(self, tmp_path: Path, merged: bool) -> GitRepo:
        repo, _bare = GitRepo.with_bare_remote(tmp_path / "git")
        files = {"README.md": "x\n"}
        if merged:
            files |= {
                "scripts/check_branch_direction.py": "#\n",
                ".github/workflows/branch-policy.yml": "x\n",
            }
        repo.commit("dev tip", files)
        repo.push("main:dev")
        return repo

    def test_add_branch_direction_waits_for_t6_on_dev(self, gh: FakeGh, tmp_path: Path) -> None:
        repo = self._dev_with_t6(tmp_path, merged=False)
        proc = _run(CUTOVER, ["--execute", "add-branch-direction"], gh.env(), repo.path)
        assert proc.returncode == 2
        assert "not merged into dev" in proc.stderr
        assert gh.writes() == []

    def test_add_branch_direction_puts_the_full_rulesets(self, gh: FakeGh, tmp_path: Path) -> None:
        repo = self._dev_with_t6(tmp_path, merged=True)
        listing = [
            {"id": 10 + i, "name": n, "source_type": "Repository"}
            for i, n in enumerate(rulesets.CUTOVER_ORDER)
        ]
        gh.answer(["api", f"repos/{REPO}/rulesets"], json.dumps(listing))
        out = tmp_path / "out"
        proc = _run(
            CUTOVER,
            ["--execute", "add-branch-direction"],
            gh.env(CIV6AZ_CUTOVER_DIR=out.as_posix()),
            repo.path,
        )
        assert proc.returncode == 0, proc.stderr
        puts = gh.writes()
        assert [c[3] for c in puts] == [f"repos/{REPO}/rulesets/{i}" for i in (10, 11, 12)]
        for call in puts:
            assert call[2] == "PUT"
            assert "Branch Direction" in Path(call[-1]).read_text(encoding="utf-8")

    def test_settings(self, gh: FakeGh, tmp_path: Path) -> None:
        proc = _run(CUTOVER, ["--execute", "settings"], gh.env(), tmp_path)
        assert proc.returncode == 0, proc.stderr
        (call,) = gh.writes()
        assert call[:4] == ["api", "-X", "PATCH", f"repos/{REPO}"]
        assert "allow_rebase_merge=false" in call and "delete_branch_on_merge=true" in call
        assert "allow_update_branch=true" in call and "allow_auto_merge=true" in call

    def test_housekeeping(self, gh: FakeGh, tmp_path: Path) -> None:
        gh.answer(["pr", "list"], "95\n130\n131\n")
        proc = _run(CUTOVER, ["--execute", "housekeeping"], gh.env(), tmp_path)
        assert proc.returncode == 0, proc.stderr
        writes = gh.writes()
        assert [c[:3] for c in writes if c[1] == "close"] == [
            ["pr", "close", "95"],
            ["pr", "close", "120"],
        ]
        assert [c[2] for c in writes if c[1] == "comment"] == ["130", "131"]
        assert [c[2] for c in writes if c[1] == "update-branch"] == ["106", "69", "49", "32"]

    def test_verify_fails_on_a_pr_not_based_on_dev(self, gh: FakeGh, tmp_path: Path) -> None:
        gh.answer(["api", f"repos/{REPO}", "--jq", ".default_branch"], "dev\n")
        gh.answer(["pr", "list"], "#140 -> qa\n")
        gh.answer(
            ["api", f"repos/{REPO}/rulesets", "--jq"], "\n".join(rulesets.CUTOVER_ORDER) + "\n"
        )
        proc = _run(CUTOVER, ["verify"], gh.env(), tmp_path)
        assert proc.returncode == 1
        assert "#140 -> qa" in proc.stdout

    def test_verify_passes_and_prints_the_manual_checklist(
        self, gh: FakeGh, tmp_path: Path
    ) -> None:
        gh.answer(["api", f"repos/{REPO}", "--jq", ".default_branch"], "dev\n")
        gh.answer(["pr", "list"], "")
        gh.answer(
            ["api", f"repos/{REPO}/rulesets", "--jq"], "\n".join(rulesets.CUTOVER_ORDER) + "\n"
        )
        proc = _run(CUTOVER, ["verify"], gh.env(), tmp_path)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "civ6az-agent cannot create a v* tag" in proc.stdout

    def test_rollback_restores_the_snapshot(self, gh: FakeGh, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        (out / "before.json").write_text(
            json.dumps({"delete_branch_on_merge": False, "allow_update_branch": False,
                        "allow_auto_merge": False, "allow_rebase_merge": True}),
            encoding="utf-8",
        )  # fmt: skip
        listing = [{"id": 31, "name": "dev"}, {"id": 32, "name": "release-tags"}]
        gh.answer(["api", f"repos/{REPO}/rulesets"], json.dumps(listing))
        proc = _run(
            CUTOVER, ["--execute", "rollback"], gh.env(CIV6AZ_CUTOVER_DIR=out.as_posix()), tmp_path
        )
        assert proc.returncode == 0, proc.stderr
        writes = gh.writes()
        deletes = [c[3] for c in writes if c[2] == "DELETE"]
        assert deletes == [
            f"repos/{REPO}/rulesets/31",
            f"repos/{REPO}/rulesets/32",
            f"repos/{REPO}/git/refs/heads/qa",
            f"repos/{REPO}/git/refs/heads/main",
        ]
        patch = next(c for c in writes if c[2] == "PATCH")
        assert "allow_rebase_merge=true" in patch and "delete_branch_on_merge=false" in patch

    def test_local_retargets_the_clone(self, gh: FakeGh, tmp_path: Path) -> None:
        repo, _bare = GitRepo.with_bare_remote(tmp_path / "git")
        repo.commit("tip")
        repo.git("branch", "-m", "main", OLD)
        repo.push(f"{OLD}:dev")
        repo.git(
            "--git-dir", _bare.as_posix(), "symbolic-ref", "HEAD", "refs/heads/dev"
        )  # as after the rename
        proc = _run(CUTOVER, ["--execute", "local"], gh.env(), repo.path)
        assert proc.returncode == 0, proc.stderr
        assert repo.git("rev-parse", "--abbrev-ref", "HEAD") == "dev"
        assert repo.git("rev-parse", "--abbrev-ref", "dev@{upstream}") == "origin/dev"

    def test_unknown_step_is_refused(self, gh: FakeGh, tmp_path: Path) -> None:
        assert _run(CUTOVER, ["nuke"], gh.env(), tmp_path).returncode == 2


# ---------------------------------------------------------------------------
# merge_backmerge.sh
# ---------------------------------------------------------------------------

SHA = "a" * 40


@needs_bash
class TestMergeBackmerge:
    def _pr(
        self, gh: FakeGh, head: str, base: str = "dev", cross: str = "false", state: str = "OPEN"
    ) -> None:
        gh.answer(["pr", "view", "12"], f"{head}\t{base}\t{cross}\t{state}\t{SHA}\n")

    def test_backmerge_is_merged_with_a_merge_commit(self, gh: FakeGh, tmp_path: Path) -> None:
        self._pr(gh, "backmerge/v2.1.0-to-dev")
        proc = _run(BACKMERGE, ["--execute", "12"], gh.env(), tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert gh.writes() == [
            ["pr", "merge", "12", "--repo", REPO, "--merge", "--match-head-commit", SHA]
        ]

    def test_dry_run_does_not_merge(self, gh: FakeGh, tmp_path: Path) -> None:
        self._pr(gh, "backmerge/v2.1.0-to-dev")
        proc = _run(BACKMERGE, ["12"], gh.env(), tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert gh.writes() == []
        assert "--merge" in proc.stdout

    @pytest.mark.parametrize(
        ("head", "base", "cross", "state"),
        [
            ("feature/x", "dev", "false", "OPEN"),
            ("backmerge/v2.1.0-to-dev", "main", "false", "OPEN"),
            ("backmerge/v2.1.0-to-dev", "dev", "true", "OPEN"),
            ("backmerge/v2.1.0-to-dev", "dev", "false", "MERGED"),
        ],
    )
    def test_refusals(
        self, gh: FakeGh, tmp_path: Path, head: str, base: str, cross: str, state: str
    ) -> None:
        self._pr(gh, head, base, cross, state)
        proc = _run(BACKMERGE, ["--execute", "12"], gh.env(), tmp_path)
        assert proc.returncode == 2
        assert gh.writes() == []

    @pytest.mark.parametrize("arg", ["12x", "", "-1"])
    def test_bad_pr_number(self, gh: FakeGh, tmp_path: Path, arg: str) -> None:
        assert _run(BACKMERGE, [arg], gh.env(), tmp_path).returncode == 2

    def test_execute_refuses_inside_claude_code(self, gh: FakeGh, tmp_path: Path) -> None:
        self._pr(gh, "backmerge/v2.1.0-to-dev")
        proc = _run(BACKMERGE, ["--execute", "12"], gh.env(CLAUDECODE="1"), tmp_path)
        assert proc.returncode == 2
        assert gh.writes() == []


def test_scripts_are_not_in_the_guard_allowlist_by_accident() -> None:
    """The maintainer scripts must never be pre-approved for agents."""
    settings = json.loads((REPO_ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
    allowed = " ".join(settings.get("permissions", {}).get("allow", []))
    assert "maintainer" not in allowed
    assert os.path.basename(str(CUTOVER)) not in allowed
