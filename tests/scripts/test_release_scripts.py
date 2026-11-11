"""Release mechanics (plan S3.5): ``release_version``, ``release_checks``, ``tag_release``.

Git is real (temporary repositories, a bare local remote for ``tag_release``);
the GitHub API is a fake keyed by request path. Nothing reaches GitHub.

Negative controls (run when written, then reverted):
* dropping the ``offending_final_paths`` check from ``verify_tag`` failed
  ``test_final_whose_diff_touches_code_is_refused``;
* resolving PR checks on the tagged commit instead of the PR head failed
  ``test_pr_scoped_checks_are_read_from_the_pr_head``;
* removing the lineage check from ``tag_release`` failed
  ``test_tag_release_refuses_a_pre_r2_commit``.
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

from scripts import release_checks as rc
from scripts import release_version as rv
from scripts.check_branch_direction import ApiError, Git
from scripts.maintainer import tag_release as tr
from tests.scripts.git_fixture import GitRepo, git_env

REPO = "acme/widget"

R2_RELEASE_YML = """\
on:
  push:
    tags: ["v*.*.*"]
permissions:
  contents: read
jobs:
  verify-tag:
    runs-on: ubuntu-latest
  wait-for-ci:
    runs-on: ubuntu-latest
  release:
    needs: [verify-tag, wait-for-ci]
    permissions:
      contents: write
    runs-on: ubuntu-latest
"""
PRE_R2_RELEASE_YML = """\
on:
  push:
    tags: ["v*.*.*"]
jobs:
  release:
    permissions:
      contents: write
    runs-on: ubuntu-latest
"""


def pyproject(version: str) -> str:
    return f'[project]\nname = "x"\nversion = "{version}"\n\n[tool.other]\nversion = "9.9.9"\n'


# ---------------------------------------------------------------------------
# release_version
# ---------------------------------------------------------------------------


class TestReleaseVersion:
    @pytest.mark.parametrize(
        ("tag", "base", "pep440", "is_rc"),
        [
            ("v2.1.0-rc.1", "2.1.0", "2.1.0rc1", True),
            ("v2.1.0-rc.12", "2.1.0", "2.1.0rc12", True),
            ("v2.1.0", "2.1.0", "2.1.0", False),
            ("refs/tags/v10.0.3", "10.0.3", "10.0.3", False),
        ],
    )
    def test_parse(self, tag: str, base: str, pep440: str, is_rc: bool) -> None:
        parsed = rv.parse_tag(tag)
        assert (parsed.base, parsed.pep440, parsed.is_rc) == (base, pep440, is_rc)
        assert parsed.name == tag.removeprefix("refs/tags/")

    @pytest.mark.parametrize(
        "tag",
        [
            "v2.1",
            "2.1.0",
            "v2.1.0-rc.0",
            "v2.1.0-beta.1",
            "v02.1.0",
            "v2.1.0-rc.1x",
            "v2.1.0rc1",
            "",
        ],
    )
    def test_invalid_tags_are_refused(self, tag: str) -> None:
        with pytest.raises(rv.ReleaseError):
            rv.parse_tag(tag)

    def test_the_base_version_must_match_pyproject(self) -> None:
        rv.check_matches(rv.parse_tag("v2.1.0-rc.3"), pyproject("2.1.0"))
        with pytest.raises(rv.ReleaseError, match=r"2.1.1"):
            rv.check_matches(rv.parse_tag("v2.1.1"), pyproject("2.1.0"))

    def test_stamp_touches_only_the_project_version(self) -> None:
        out = rv.stamp_pyproject(pyproject("2.1.0"), rv.parse_tag("v2.1.0-rc.1"))
        assert 'version = "2.1.0rc1"' in out
        assert 'version = "9.9.9"' in out  # [tool.other] untouched
        assert rv.pyproject_version(out) == "2.1.0rc1"

    def test_stamp_init(self) -> None:
        text = '"""doc"""\n__version__ = "2.1.0"\nx = "2.1.0"\n'
        out = rv.stamp_init(text, rv.parse_tag("v2.1.0-rc.4"))
        assert '__version__ = "2.1.0rc4"' in out and 'x = "2.1.0"' in out

    def test_stamp_refuses_a_mismatch(self) -> None:
        with pytest.raises(rv.ReleaseError):
            rv.stamp_pyproject(pyproject("2.0.0"), rv.parse_tag("v2.1.0-rc.1"))

    def test_notes_come_from_the_base_section(self) -> None:
        changelog = "# C\n\n## [Unreleased]\n\n## [2.1.0] — 2026-12-01 — T\n\n- a\n\n## [2.0.0] — x\n\n- b\n"
        notes = rv.release_notes(changelog, rv.parse_tag("v2.1.0-rc.2"), repo=REPO)
        assert notes is not None
        assert "- a" in notes and "- b" not in notes
        assert "v2.1.0-rc.2" in notes
        assert rv.release_notes(changelog, rv.parse_tag("v3.0.0"), repo=REPO) is None

    def test_cli_stamp_check_and_notes(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(pyproject("2.1.0"), encoding="utf-8")
        (tmp_path / "init.py").write_text('__version__ = "2.1.0"\n', encoding="utf-8")
        (tmp_path / "CHANGELOG.md").write_text("## [2.1.0] — d\n\n- z\n", encoding="utf-8")
        tag = "v2.1.0-rc.1"
        assert (
            rv.main(["check", "--tag", tag, "--pyproject", str(tmp_path / "pyproject.toml")]) == 0
        )
        args = [
            "notes",
            "--tag",
            tag,
            "--changelog",
            str(tmp_path / "CHANGELOG.md"),
            "--repo",
            REPO,
        ]
        assert rv.main([*args, "--out", str(tmp_path / "notes.md")]) == 0
        assert "- z" in (tmp_path / "notes.md").read_text(encoding="utf-8")
        stamp = ["stamp", "--tag", tag, "--pyproject", str(tmp_path / "pyproject.toml")]
        assert rv.main([*stamp, "--init", str(tmp_path / "init.py")]) == 0
        assert (
            rv.pyproject_version((tmp_path / "pyproject.toml").read_text(encoding="utf-8"))
            == "2.1.0rc1"
        )
        assert "2.1.0rc1" in (tmp_path / "init.py").read_text(encoding="utf-8")
        assert (
            rv.main(["check", "--tag", "v9.0.0", "--pyproject", str(tmp_path / "pyproject.toml")])
            == 1
        )


# ---------------------------------------------------------------------------
# A release-line repository
# ---------------------------------------------------------------------------


class Line:
    """main/qa/dev with release.yml, pyproject and evidence, in one repo."""

    def __init__(
        self, repo: GitRepo, version: str = "2.1.0", release_yml: str = R2_RELEASE_YML
    ) -> None:
        self.repo = repo
        self.root = repo.commit(
            "root",
            {
                "pyproject.toml": pyproject(version),
                ".github/workflows/release.yml": release_yml,
                "scripts/release_checks.py": "# marker\n",
                "alpha_zero/core.py": "x = 1\n",
            },
        )
        for name in ("main", "qa", "dev"):
            repo.set_remote(name, self.root)

    def promote(self, message: str = "promote dev") -> str:
        """A new commit on qa (an rc candidate)."""
        self.repo.checkout("qa", self.repo.sha("refs/remotes/origin/qa"))
        sha = self.repo.commit(message)
        self.repo.set_remote("qa", sha)
        return sha

    def release(self, files: dict[str, str] | None = None) -> str:
        """Merge qa into main (optionally with extra changes on qa first)."""
        if files:
            self.repo.checkout("qa", self.repo.sha("refs/remotes/origin/qa"))
            self.repo.set_remote("qa", self.repo.commit("qa change", files))
        self.repo.checkout("main", self.repo.sha("refs/remotes/origin/main"))
        sha = self.repo.merge("refs/remotes/origin/qa", "Release")
        self.repo.set_remote("main", sha)
        return sha


@pytest.fixture
def line(tmp_path: Path) -> Line:
    return Line(GitRepo(tmp_path / "repo"))


def _git(r: GitRepo) -> Git:
    return Git(r.path, env=git_env())


class FakeApi:
    def __init__(self) -> None:
        self.responses: dict[str, Any] = {}
        self.calls: list[str] = []

    def pulls_for(self, sha: str, prs: list[dict[str, Any]]) -> None:
        self.responses[f"repos/{REPO}/commits/{sha}/pulls?per_page=100"] = prs

    def runs(self, sha: str, name: str, runs: list[dict[str, Any]] | Any) -> None:
        key = f"repos/{REPO}/commits/{sha}/check-runs?check_name={quote(name)}&filter=latest&per_page=100"
        self.responses[key] = runs

    def __call__(self, path: str) -> Any:
        self.calls.append(path)
        value = self.responses.get(path, {"check_runs": []} if "/check-runs?" in path else None)
        if callable(value):
            return value()
        if value is None:
            raise ApiError(f"no fake answer for {path}")
        return value


def merged_pr(
    merge_sha: str, head_ref: str, head_sha: str = "f" * 40, **extra: Any
) -> dict[str, Any]:
    return {
        "number": 3,
        "merged_at": "2026-12-01T00:00:00Z",
        "merge_commit_sha": merge_sha,
        "head": {"ref": head_ref, "sha": head_sha},
        **extra,
    }


def run(
    conclusion: str | None, *, status: str = "completed", app: str = "github-actions", id_: int = 1
) -> dict:
    return {"id": id_, "status": status, "conclusion": conclusion, "app": {"slug": app}}


# ---------------------------------------------------------------------------
# verify_tag
# ---------------------------------------------------------------------------


class TestVerifyTag:
    def test_rc_on_qa_with_main_ancestor_passes(self, line: Line) -> None:
        sha = line.promote()
        assert (
            rc.verify_tag(
                _git(line.repo), rv.parse_tag("v2.1.0-rc.1"), sha, api=FakeApi(), repo=REPO
            )
            == []
        )

    def test_rc_version_must_match_pyproject(self, line: Line) -> None:
        sha = line.promote()
        problems = rc.verify_tag(
            _git(line.repo), rv.parse_tag("v2.2.0-rc.1"), sha, api=FakeApi(), repo=REPO
        )
        assert any("pyproject" in p for p in problems)

    def test_rc_must_be_on_qa(self, line: Line) -> None:
        line.repo.checkout("elsewhere", line.root)
        sha = line.repo.commit("not on qa")
        problems = rc.verify_tag(
            _git(line.repo), rv.parse_tag("v2.1.0-rc.1"), sha, api=FakeApi(), repo=REPO
        )
        assert any("origin/qa" in p for p in problems)

    def test_rc_requires_main_as_an_ancestor(self, line: Line) -> None:
        line.repo.checkout("main", line.root)
        line.repo.set_remote("main", line.repo.commit("hotfix not back-merged"))
        sha = line.promote()
        problems = rc.verify_tag(
            _git(line.repo), rv.parse_tag("v2.1.0-rc.1"), sha, api=FakeApi(), repo=REPO
        )
        assert any("origin/main is not an ancestor" in p for p in problems)

    def test_final_after_evidence_only_changes_passes(self, line: Line) -> None:
        line.repo.tag("v2.1.0-rc.1", line.promote())
        final = line.release({"docs/evidence/release_v2_1_0/report.md": "pass\n"})
        assert (
            rc.verify_tag(_git(line.repo), rv.parse_tag("v2.1.0"), final, api=FakeApi(), repo=REPO)
            == []
        )

    def test_final_whose_diff_touches_code_is_refused(self, line: Line) -> None:
        line.repo.tag("v2.1.0-rc.1", line.promote())
        final = line.release({"alpha_zero/core.py": "x = 2\n"})
        problems = rc.verify_tag(
            _git(line.repo), rv.parse_tag("v2.1.0"), final, api=FakeApi(), repo=REPO
        )
        assert any("alpha_zero/core.py" in p for p in problems)

    def test_the_highest_rc_is_numeric_not_lexical(self, line: Line) -> None:
        line.repo.tag("v2.1.0-rc.9", line.promote("rc9"))
        line.repo.tag("v2.1.0-rc.10", line.promote("rc10"))
        assert rc.highest_rc(_git(line.repo), "2.1.0").name == "v2.1.0-rc.10"  # type: ignore[union-attr]

    def test_final_must_be_on_main(self, line: Line) -> None:
        line.repo.tag("v2.1.0-rc.1", line.promote())
        sha = line.repo.sha("refs/remotes/origin/qa")  # tagging qa, not main
        problems = rc.verify_tag(
            _git(line.repo), rv.parse_tag("v2.1.0"), sha, api=FakeApi(), repo=REPO
        )
        assert any("origin/main" in p for p in problems)

    def test_hotfix_final_without_rc_needs_a_hotfix_pr(self, line: Line) -> None:
        line.repo.checkout("main", line.root)
        line.repo.checkout("hotfix/crash", line.root)
        line.repo.commit("fix", {"alpha_zero/core.py": "x = 3\n"})
        line.repo.checkout("main")
        final = line.repo.merge("hotfix/crash", "Merge hotfix")
        line.repo.set_remote("main", final)
        api = FakeApi()
        api.pulls_for(final, [merged_pr(final, "hotfix/crash")])
        assert (
            rc.verify_tag(_git(line.repo), rv.parse_tag("v2.1.0"), final, api=api, repo=REPO) == []
        )
        api.pulls_for(final, [merged_pr(final, "feature/x")])
        problems = rc.verify_tag(_git(line.repo), rv.parse_tag("v2.1.0"), final, api=api, repo=REPO)
        assert any("hotfix/*" in p for p in problems)


class TestMergedPr:
    def test_exactly_one_merged_pr_with_this_merge_commit(self) -> None:
        sha = "a" * 40
        api = FakeApi()
        api.pulls_for(
            sha, [merged_pr(sha, "dev"), {**merged_pr("b" * 40, "x")}, {"merged_at": None}]
        )
        assert rc.merged_pr_for_commit(api, REPO, sha)["head"]["ref"] == "dev"

    @pytest.mark.parametrize("count", [0, 2])
    def test_zero_or_two_matches_fail(self, count: int) -> None:
        sha = "a" * 40
        api = FakeApi()
        api.pulls_for(sha, [merged_pr(sha, f"b{i}") for i in range(count)])
        with pytest.raises(ApiError, match="exactly one"):
            rc.merged_pr_for_commit(api, REPO, sha)

    def test_missing_merge_commit_sha_fails_closed(self) -> None:
        sha = "a" * 40
        api = FakeApi()
        api.pulls_for(sha, [{"merged_at": "t", "head": {"ref": "dev", "sha": "f" * 40}}])
        with pytest.raises(ApiError, match="merge_commit_sha"):
            rc.merged_pr_for_commit(api, REPO, sha)


# ---------------------------------------------------------------------------
# wait_for_ci
# ---------------------------------------------------------------------------

TAGGED = "c" * 40
HEAD = "d" * 40


def _green_push(api: FakeApi) -> None:
    for name in rc.PUSH_CHECKS:
        api.runs(TAGGED, name, {"check_runs": [run("success")]})


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps = 0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        self.now += seconds


def _wait(api: FakeApi, tag: str, clock: Clock | None = None, timeout: float = 600) -> list[str]:
    clock = clock or Clock()
    return rc.wait_for_ci(
        api,
        REPO,
        TAGGED,
        rv.parse_tag(tag),
        timeout_s=timeout,
        interval_s=30,
        sleep=clock.sleep,
        clock=clock,
    )


class TestWaitForCi:
    def test_pr_scoped_checks_are_read_from_the_pr_head(self) -> None:
        api = FakeApi()
        _green_push(api)
        api.pulls_for(TAGGED, [merged_pr(TAGGED, "dev", HEAD)])
        api.runs(HEAD, "Branch Direction", {"check_runs": [run("success")]})
        assert _wait(api, "v2.1.0-rc.1") == []
        assert any(
            f"commits/{HEAD}/check-runs?check_name=Branch%20Direction" in c for c in api.calls
        )
        assert not any(
            f"commits/{TAGGED}/check-runs?check_name=Branch%20Direction" in c for c in api.calls
        )

    def test_a_failed_push_check_fails(self) -> None:
        api = FakeApi()
        _green_push(api)
        api.runs(TAGGED, "Security Scan", {"check_runs": [run("failure")]})
        api.pulls_for(TAGGED, [merged_pr(TAGGED, "dev", HEAD)])
        api.runs(HEAD, "Branch Direction", {"check_runs": [run("success")]})
        problems = _wait(api, "v2.1.0-rc.1")
        assert problems and "Security Scan" in problems[0]

    def test_pending_checks_are_polled_until_green(self) -> None:
        api = FakeApi()
        _green_push(api)
        api.pulls_for(TAGGED, [merged_pr(TAGGED, "dev", HEAD)])
        answers = iter(
            [{"check_runs": [run(None, status="in_progress")]}] * 2
            + [{"check_runs": [run("success")]}] * 9
        )
        api.runs(TAGGED, "Coverage Report", lambda: next(answers))
        api.runs(HEAD, "Branch Direction", {"check_runs": [run("success")]})
        clock = Clock()
        assert _wait(api, "v2.1.0-rc.1", clock) == []
        assert clock.sleeps == 2

    def test_a_check_that_never_arrives_times_out(self) -> None:
        api = FakeApi()
        _green_push(api)
        api.runs(TAGGED, "Coverage Report", {"check_runs": []})
        api.pulls_for(TAGGED, [merged_pr(TAGGED, "dev", HEAD)])
        api.runs(HEAD, "Branch Direction", {"check_runs": [run("success")]})
        problems = _wait(api, "v2.1.0-rc.1", timeout=120)
        assert problems and "timed out" in problems[0] and "Coverage Report" in problems[0]

    def test_a_check_from_another_app_does_not_count(self) -> None:
        api = FakeApi()
        _green_push(api)
        api.pulls_for(TAGGED, [merged_pr(TAGGED, "dev", HEAD)])
        api.runs(HEAD, "Branch Direction", {"check_runs": [run("success", app="impostor")]})
        assert _wait(api, "v2.1.0-rc.1", timeout=60)

    def test_the_newest_run_wins(self) -> None:
        api = FakeApi()
        _green_push(api)
        api.pulls_for(TAGGED, [merged_pr(TAGGED, "dev", HEAD)])
        api.runs(
            HEAD, "Branch Direction", {"check_runs": [run("success", id_=2), run("failure", id_=1)]}
        )
        assert _wait(api, "v2.1.0-rc.1") == []

    def test_a_non_hotfix_final_also_needs_release_evidence(self) -> None:
        api = FakeApi()
        _green_push(api)
        api.pulls_for(TAGGED, [merged_pr(TAGGED, "qa", HEAD)])
        api.runs(HEAD, "Branch Direction", {"check_runs": [run("success")]})
        assert _wait(api, "v2.1.0", timeout=60)  # Release Evidence missing
        api.runs(HEAD, "Release Evidence", {"check_runs": [run("success")]})
        assert _wait(api, "v2.1.0") == []

    def test_a_hotfix_final_needs_only_branch_direction(self) -> None:
        api = FakeApi()
        _green_push(api)
        api.pulls_for(TAGGED, [merged_pr(TAGGED, "hotfix/crash", HEAD)])
        api.runs(HEAD, "Branch Direction", {"check_runs": [run("success")]})
        assert _wait(api, "v2.1.1") == []

    def test_no_producing_pr_fails(self) -> None:
        api = FakeApi()
        _green_push(api)
        api.pulls_for(TAGGED, [])
        assert _wait(api, "v2.1.0-rc.1")


def test_push_checks_are_the_ruleset_checks_minus_branch_direction() -> None:
    main = json.loads(Path(".github/rulesets/main.json").read_text(encoding="utf-8"))
    rule = next(r for r in main["rules"] if r["type"] == "required_status_checks")
    contexts = {c["context"] for c in rule["parameters"]["required_status_checks"]}
    assert set(rc.PUSH_CHECKS) == contexts - {"Branch Direction", "Release Evidence"}
    assert len(rc.PUSH_CHECKS) == 8


# ---------------------------------------------------------------------------
# R2 lineage and tag_release
# ---------------------------------------------------------------------------


class TestLineage:
    def test_r2_release_workflow_is_recognised(self, line: Line) -> None:
        assert rc.is_r2_lineage(_git(line.repo), line.root)

    def test_pre_r2_release_workflow_is_not(self, tmp_path: Path) -> None:
        old = Line(GitRepo(tmp_path / "old"), release_yml=PRE_R2_RELEASE_YML)
        assert not rc.is_r2_lineage(_git(old.repo), old.root)


@pytest.fixture
def remote_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Line:
    """A Line whose main/qa/dev live on a bare local remote."""
    monkeypatch.delenv("CLAUDECODE", raising=False)
    monkeypatch.delenv("CIV6AZ_R2_MERGE_SHA", raising=False)
    repo, _bare = GitRepo.with_bare_remote(tmp_path / "git")
    line = Line(repo)
    repo.push(
        f"{line.root}:refs/heads/main", f"{line.root}:refs/heads/qa", f"{line.root}:refs/heads/dev"
    )
    repo.git("fetch", "-q", "origin")
    return line


def _push_qa(line: Line, files: dict[str, str] | None = None, start: str | None = None) -> str:
    line.repo.checkout("qa", start or line.repo.sha("refs/remotes/origin/qa"))
    sha = line.repo.commit("qa work", files)
    line.repo.push("-f", "qa:qa")
    return sha


def _tag(line: Line, *args: str, api: FakeApi | None = None) -> int:
    return tr.main([*args, "--repo", REPO], api=api or FakeApi(), git=_git(line.repo))


def _refusal(line: Line, *args: str) -> str:
    """Run tag_release expecting a refusal (exit 1); return its single reason."""
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert _tag(line, *args) == 1
    reasons = [r for r in err.getvalue().splitlines() if r.startswith("  - ")]
    assert len(reasons) == 1, err.getvalue()  # refused for one reason, the one under test
    return reasons[0]


def _remote_tags(line: Line) -> str:
    return line.repo.git("ls-remote", "--tags", "origin")


class TestTagRelease:
    def test_dry_run_creates_no_tag(
        self, remote_line: Line, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _push_qa(remote_line)
        assert _tag(remote_line, "v2.1.0-rc.1") == 0
        assert "git push origin refs/tags/v2.1.0-rc.1" in capsys.readouterr().out
        assert _remote_tags(remote_line) == ""

    def test_execute_tags_qa_and_pushes(self, remote_line: Line) -> None:
        sha = _push_qa(remote_line)
        assert _tag(remote_line, "v2.1.0-rc.1", "--execute") == 0
        assert f"{sha}\trefs/tags/v2.1.0-rc.1^{{}}" in remote_line.repo.git(
            "ls-remote", "--tags", "origin"
        )

    def test_execute_refuses_inside_claude_code(
        self, remote_line: Line, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _push_qa(remote_line)
        monkeypatch.setenv("CLAUDECODE", "1")
        assert _tag(remote_line, "v2.1.0-rc.1", "--execute") == 2
        assert _remote_tags(remote_line) == ""

    def test_tag_release_refuses_a_pre_r2_commit(self, remote_line: Line) -> None:
        _push_qa(remote_line, {".github/workflows/release.yml": PRE_R2_RELEASE_YML})
        assert "does not descend from the LF-R2 merge" in _refusal(
            remote_line, "v2.1.0-rc.1", "--execute"
        )
        assert _remote_tags(remote_line) == ""

    def test_tag_release_refuses_a_commit_before_the_recorded_r2_merge(
        self, remote_line: Line, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        old = remote_line.root
        r2 = _push_qa(remote_line)
        _push_qa(remote_line, start=old)  # a qa tip that does not contain r2
        monkeypatch.setenv("CIV6AZ_R2_MERGE_SHA", r2)
        assert "does not contain the recorded LF-R2 merge" in _refusal(remote_line, "v2.1.0-rc.1")

    def test_tag_release_refuses_a_final_whose_diff_touches_code(self, remote_line: Line) -> None:
        line = remote_line
        rc1 = _push_qa(line)
        line.repo.tag("v2.1.0-rc.1", rc1)
        line.repo.push("refs/tags/v2.1.0-rc.1")
        _push_qa(line, {"alpha_zero/core.py": "x = 5\n"})
        line.repo.checkout("main", line.root)
        line.repo.merge("qa", "Release v2.1.0")
        line.repo.push("main:main")
        assert "alpha_zero/core.py changed after v2.1.0-rc.1" in _refusal(
            line, "v2.1.0", "--execute"
        )
        assert "refs/tags/v2.1.0\n" not in _remote_tags(line) + "\n"
        assert "v2.1.0^{}" not in _remote_tags(line).replace("v2.1.0-rc.1^{}", "")

    def test_tag_release_refuses_an_rc_when_main_is_not_an_ancestor(
        self, remote_line: Line
    ) -> None:
        line = remote_line
        line.repo.checkout("main", line.root)
        line.repo.commit("hotfix on main")
        line.repo.push("main:main")
        _push_qa(line)
        assert "origin/main is not an ancestor" in _refusal(line, "v2.1.0-rc.1")

    def test_tag_release_refuses_an_existing_tag(self, remote_line: Line) -> None:
        sha = _push_qa(remote_line)
        remote_line.repo.tag("v2.1.0-rc.1", sha)
        remote_line.repo.push("refs/tags/v2.1.0-rc.1")
        assert "already exists" in _refusal(remote_line, "v2.1.0-rc.1")

    def test_tag_release_refuses_an_rc_below_the_highest(self, remote_line: Line) -> None:
        sha = _push_qa(remote_line)
        remote_line.repo.tag("v2.1.0-rc.3", sha)
        remote_line.repo.push("refs/tags/v2.1.0-rc.3")
        _push_qa(remote_line)
        assert "not above the highest existing rc" in _refusal(remote_line, "v2.1.0-rc.2")

    def test_bad_tag_is_a_usage_error(self, remote_line: Line) -> None:
        assert _tag(remote_line, "2.1.0") == 2


def test_tag_release_module_runs_as_a_script() -> None:
    """``python -m scripts.maintainer.tag_release --help`` works from the repo root."""
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.maintainer.tag_release", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--execute" in proc.stdout
