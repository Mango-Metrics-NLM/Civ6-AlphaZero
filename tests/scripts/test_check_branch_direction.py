"""Branch Direction (plan S3.4): ``scripts/check_branch_direction.py``.

Every rule is exercised on a real temporary git repository
(``tests/scripts/git_fixture.py``); only the GitHub API is faked. The remote
state each test needs (``origin/main``, ``origin/qa``, ``origin/dev``) is set with
``update-ref``, so a test reads as the topology it checks.

Negative controls (run when written, then reverted):
* making ``_promotion`` return allowed unconditionally failed
  ``test_promotion_denied_when_main_is_not_an_ancestor``;
* dropping the ``in_range`` clause from the merge-parent check failed
  ``test_backmerge_with_three_update_merges_is_allowed`` (the update merges'
  first parents are the branch's own earlier commits);
* accepting ``+`` lines from ``git cherry`` failed
  ``test_hand_edited_rcfix_is_denied``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import check_branch_direction as cbd
from tests.scripts.git_fixture import GitRepo, git_env

REPO_NAME = "acme/widget"


@pytest.fixture
def repo(tmp_path: Path) -> GitRepo:
    """main/qa/dev all at one root commit ``A``."""
    r = GitRepo(tmp_path / "repo")
    r.commit("A: root", {"README.md": "root\n", "alpha_zero/core.py": "x = 1\n"})
    for name in ("main", "qa", "dev"):
        r.set_remote(name, "HEAD")
    return r


def _git(r: GitRepo) -> cbd.Git:
    return cbd.Git(r.path, env=git_env())


def _eval(r: GitRepo, base: str, head_ref: str, head: str = "HEAD") -> cbd.Decision:
    return cbd.evaluate(_git(r), base=base, head_ref=head_ref, head_sha=r.sha(head))


# ---------------------------------------------------------------------------
# Which heads each base accepts (S3.1)
# ---------------------------------------------------------------------------


class TestDirections:
    @pytest.mark.parametrize("head", ["feature/x", "fix/y", "chore/z", "docs/a", "claude/b"])
    def test_feature_branches_into_dev_are_allowed(self, repo: GitRepo, head: str) -> None:
        repo.checkout(head, "HEAD")
        repo.commit("work")
        assert _eval(repo, "dev", head).allowed

    @pytest.mark.parametrize("head", ["main", "qa", "hotfix/x", "evidence/v1.0.0-rc.1", "rcfix/x"])
    def test_refused_heads_into_dev(self, repo: GitRepo, head: str) -> None:
        repo.checkout("work", "HEAD")
        repo.commit("work")
        decision = _eval(repo, "dev", head)
        assert not decision.allowed
        assert any(head in reason for reason in decision.reasons)

    @pytest.mark.parametrize("head", ["chore/x", "feature/y", "main", "hotfix/z"])
    def test_sideways_heads_into_qa_are_denied(self, repo: GitRepo, head: str) -> None:
        repo.checkout("work", "HEAD")
        repo.commit("work")
        assert not _eval(repo, "qa", head).allowed

    @pytest.mark.parametrize("head", ["dev", "feature/y", "backmerge/x", "rcfix/x"])
    def test_only_qa_and_hotfix_into_main(self, repo: GitRepo, head: str) -> None:
        repo.checkout("work", "HEAD")
        repo.commit("work")
        assert not _eval(repo, "main", head).allowed

    def test_release_pr_qa_into_main_is_allowed(self, repo: GitRepo) -> None:
        repo.checkout("qa", "HEAD")
        repo.commit("promoted")
        assert _eval(repo, "main", "qa").allowed

    def test_release_pr_denied_when_qa_lacks_main(self, repo: GitRepo) -> None:
        repo.checkout("main", "HEAD")
        repo.commit("hotfix landed on main")
        repo.set_remote("main", "HEAD")
        repo.checkout("qa", "HEAD~1")
        repo.commit("rc work")
        assert not _eval(repo, "main", "qa").allowed

    def test_unprotected_base_has_no_direction_policy(self, repo: GitRepo) -> None:
        repo.checkout("feature/child", "HEAD")
        repo.commit("work")
        decision = _eval(repo, "feature/parent", "feature/child")
        assert decision.allowed
        assert decision.rule == "unprotected-base"


# ---------------------------------------------------------------------------
# Promotion dev -> qa
# ---------------------------------------------------------------------------


class TestPromotion:
    def test_promotion_allowed_when_main_is_an_ancestor(self, repo: GitRepo) -> None:
        repo.checkout("dev", "HEAD")
        repo.commit("feature F")
        repo.set_remote("dev", "HEAD")
        decision = _eval(repo, "qa", "dev")
        assert decision.allowed
        assert decision.rule == "promotion"

    def test_promotion_denied_when_main_is_not_an_ancestor(self, repo: GitRepo) -> None:
        root = repo.sha()
        repo.checkout("main", root)
        repo.commit("hotfix H")
        repo.set_remote("main", "HEAD")
        repo.checkout("dev", root)
        repo.commit("feature F")
        repo.set_remote("dev", "HEAD")
        decision = _eval(repo, "qa", "dev")
        assert not decision.allowed
        text = " ".join(decision.reasons)
        assert "back-merge" in text and "squashed" in text


# ---------------------------------------------------------------------------
# backmerge/*
# ---------------------------------------------------------------------------


def _hotfixed_main(repo: GitRepo) -> tuple[str, str]:
    """Root A on every branch, then a hotfix H on main. Returns (A, H)."""
    root = repo.sha()
    repo.checkout("main", root)
    hotfix = repo.commit("hotfix H", {"alpha_zero/core.py": "x = 2\n"})
    repo.set_remote("main", hotfix)
    return root, hotfix


class TestBackmerge:
    def test_backmerge_with_three_update_merges_is_allowed(self, repo: GitRepo) -> None:
        root, hotfix = _hotfixed_main(repo)
        repo.checkout("dev", root)
        dev_tips = []
        repo.checkout("backmerge/v1.0.1-to-dev", hotfix)
        for i in range(3):
            # dev advances by a squash merge, then "Update branch -> merge commit".
            repo.checkout("dev")
            dev_tips.append(repo.commit(f"feature F{i}", {f"feat{i}.txt": f"{i}\n"}))
            repo.checkout("backmerge/v1.0.1-to-dev")
            repo.merge("dev", f"Merge branch 'dev' into backmerge (update {i})")
        repo.set_remote("dev", dev_tips[-1])
        decision = _eval(repo, "dev", "backmerge/v1.0.1-to-dev")
        assert decision.allowed, decision.reasons
        assert decision.rule == "backmerge"
        # each update merge's --remerge-diff goes to the job summary
        assert sum("remerge-diff" in d for d in decision.details) == 3

    def test_backmerge_with_a_commit_authored_on_the_branch_is_denied(self, repo: GitRepo) -> None:
        _root, hotfix = _hotfixed_main(repo)
        repo.checkout("backmerge/v1.0.1-to-dev", hotfix)
        stray = repo.commit("sneaked in on the back-merge")
        decision = _eval(repo, "dev", "backmerge/v1.0.1-to-dev")
        assert not decision.allowed
        assert any(stray[:12] in reason for reason in decision.reasons)

    def test_backmerge_into_qa_containing_a_dev_only_commit_is_denied(self, repo: GitRepo) -> None:
        root, hotfix = _hotfixed_main(repo)
        repo.checkout("dev", root)
        dev_only = repo.commit("unreleased dev work")
        repo.set_remote("dev", dev_only)
        repo.checkout("backmerge/v1.0.1-to-qa", hotfix)
        repo.merge("dev", "merge dev into the qa back-merge")
        decision = _eval(repo, "qa", "backmerge/v1.0.1-to-qa")
        assert not decision.allowed
        assert any(dev_only[:12] in reason for reason in decision.reasons)

    def test_backmerge_into_qa_from_main_is_allowed(self, repo: GitRepo) -> None:
        _root, hotfix = _hotfixed_main(repo)
        repo.checkout("backmerge/v1.0.1-to-qa", hotfix)
        assert _eval(repo, "qa", "backmerge/v1.0.1-to-qa").allowed

    def test_backmerge_into_dev_may_carry_qa_commits(self, repo: GitRepo) -> None:
        root = repo.sha()
        repo.checkout("qa", root)
        qa_only = repo.commit("evidence merged into qa")
        repo.set_remote("qa", qa_only)
        repo.checkout("backmerge/v1.0.0-to-dev", qa_only)
        assert _eval(repo, "dev", "backmerge/v1.0.0-to-dev").allowed

    def test_empty_backmerge_is_denied(self, repo: GitRepo) -> None:
        repo.checkout("backmerge/nothing", "HEAD")
        assert not _eval(repo, "dev", "backmerge/nothing").allowed


# ---------------------------------------------------------------------------
# hotfix/*
# ---------------------------------------------------------------------------


class TestHotfix:
    def test_hotfix_from_main_is_allowed(self, repo: GitRepo) -> None:
        repo.checkout("hotfix/crash", "HEAD")
        repo.commit("fix the crash")
        decision = _eval(repo, "main", "hotfix/crash")
        assert decision.allowed
        assert decision.rule == "hotfix"

    def test_hotfix_not_based_on_main_is_denied(self, repo: GitRepo) -> None:
        root = repo.sha()
        repo.checkout("main", root)
        repo.set_remote("main", repo.commit("main moved on"))
        repo.checkout("hotfix/stale", root)
        repo.commit("fix")
        assert not _eval(repo, "main", "hotfix/stale").allowed

    def test_hotfix_check_fails_closed_without_the_dev_ref(self, repo: GitRepo) -> None:
        repo.git("update-ref", "-d", "refs/remotes/origin/dev")
        repo.checkout("hotfix/crash", "HEAD")
        repo.commit("fix")
        with pytest.raises(cbd.GitError, match="origin/dev"):
            _eval(repo, "main", "hotfix/crash")

    def test_hotfix_carrying_a_dev_commit_is_denied(self, repo: GitRepo) -> None:
        root = repo.sha()
        repo.checkout("dev", root)
        dev_only = repo.commit("unreleased")
        repo.set_remote("dev", dev_only)
        repo.checkout("hotfix/leaky", root)
        repo.commit("fix")
        repo.merge("dev", "pull dev in")
        decision = _eval(repo, "main", "hotfix/leaky")
        assert not decision.allowed
        assert any(dev_only[:12] in reason for reason in decision.reasons)


# ---------------------------------------------------------------------------
# evidence/*
# ---------------------------------------------------------------------------


def _rc_tagged_qa(repo: GitRepo, tag: str = "v1.1.0-rc.1") -> str:
    repo.checkout("qa", "HEAD")
    rc = repo.commit("promote dev")
    repo.set_remote("qa", rc)
    repo.tag(tag, rc)
    return rc


class TestEvidence:
    def test_evidence_only_change_is_allowed(self, repo: GitRepo) -> None:
        rc = _rc_tagged_qa(repo)
        repo.checkout("evidence/v1.1.0-rc.1", rc)
        repo.commit("cards", {"docs/evidence/release_v1_1_0/report.md": "verdict\n"})
        decision = _eval(repo, "qa", "evidence/v1.1.0-rc.1")
        assert decision.allowed, decision.reasons
        assert decision.rule == "evidence"

    def test_evidence_touching_alpha_zero_is_denied(self, repo: GitRepo) -> None:
        rc = _rc_tagged_qa(repo)
        repo.checkout("evidence/v1.1.0-rc.1", rc)
        repo.commit(
            "cards and a sneaky fix",
            {"docs/evidence/release_v1_1_0/report.md": "verdict\n", "alpha_zero/core.py": "x=9\n"},
        )
        decision = _eval(repo, "qa", "evidence/v1.1.0-rc.1")
        assert not decision.allowed
        assert any("alpha_zero/core.py" in reason for reason in decision.reasons)

    def test_evidence_renaming_code_into_evidence_is_denied(self, repo: GitRepo) -> None:
        """I2: with rename detection the diff names only docs/evidence/core.py."""
        rc = _rc_tagged_qa(repo)
        repo.checkout("evidence/v1.1.0-rc.1", rc)
        (repo.path / "docs" / "evidence").mkdir(parents=True, exist_ok=True)
        repo.git("mv", "alpha_zero/core.py", "docs/evidence/core.py")
        repo.git("commit", "-q", "-m", "move the code into evidence")
        decision = _eval(repo, "qa", "evidence/v1.1.0-rc.1")
        assert not decision.allowed
        assert any("alpha_zero/core.py" in reason for reason in decision.reasons)

    def test_evidence_without_its_rc_tag_as_ancestor_is_denied(self, repo: GitRepo) -> None:
        rc = _rc_tagged_qa(repo)
        repo.checkout("qa", rc)
        repo.set_remote("qa", repo.commit("qa moved on"))
        repo.tag("v1.1.0-rc.2", "HEAD")
        repo.checkout("evidence/v1.1.0-rc.2", rc)  # cut from rc.1, named rc.2
        repo.commit("cards", {"docs/evidence/x/report.md": "v\n"})
        assert not _eval(repo, "qa", "evidence/v1.1.0-rc.2").allowed

    def test_evidence_branch_not_naming_an_rc_tag_is_denied(self, repo: GitRepo) -> None:
        rc = _rc_tagged_qa(repo)
        repo.checkout("evidence/latest", rc)
        repo.commit("cards", {"docs/evidence/x/report.md": "v\n"})
        assert not _eval(repo, "qa", "evidence/latest").allowed

    def test_evidence_with_missing_tag_is_denied(self, repo: GitRepo) -> None:
        rc = _rc_tagged_qa(repo)
        repo.checkout("evidence/v9.9.9-rc.1", rc)
        repo.commit("cards", {"docs/evidence/x/report.md": "v\n"})
        assert not _eval(repo, "qa", "evidence/v9.9.9-rc.1").allowed


# ---------------------------------------------------------------------------
# rcfix/*
# ---------------------------------------------------------------------------


def _dev_fix(repo: GitRepo) -> tuple[str, str]:
    """qa at root; dev carries a squash-merged fix D. Returns (root, D)."""
    root = repo.sha()
    repo.checkout("dev", root)
    repo.commit("unrelated dev work", {"other.txt": "o\n"})
    fix = repo.commit("fix: the bug", {"alpha_zero/core.py": "x = 1  # fixed\n"})
    repo.set_remote("dev", fix)
    return root, fix


class TestRcfix:
    def test_clean_cherry_pick_x_is_allowed(self, repo: GitRepo) -> None:
        root, fix = _dev_fix(repo)
        repo.checkout("rcfix/bug", root)
        repo.git("cherry-pick", "-x", fix)
        decision = _eval(repo, "qa", "rcfix/bug")
        assert decision.allowed, decision.reasons
        assert decision.rule == "rcfix"

    def test_hand_edited_rcfix_is_denied(self, repo: GitRepo) -> None:
        root, fix = _dev_fix(repo)
        repo.checkout("rcfix/bug", root)
        repo.git("cherry-pick", "-n", fix)
        (repo.path / "alpha_zero/core.py").write_text("x = 1  # fixed differently\n")
        repo.git("add", "-A")
        repo.git("commit", "-q", "-m", "fix: the bug (edited)")
        decision = _eval(repo, "qa", "rcfix/bug")
        assert not decision.allowed
        assert any("patch-equivalent" in reason for reason in decision.reasons)

    def test_rcfix_with_a_merge_commit_is_denied(self, repo: GitRepo) -> None:
        root, fix = _dev_fix(repo)
        repo.checkout("rcfix/bug", root)
        repo.git("cherry-pick", "-x", fix)
        repo.checkout("side", root)
        repo.commit("side")
        repo.checkout("rcfix/bug")
        repo.merge("side", "merge side")
        assert not _eval(repo, "qa", "rcfix/bug").allowed

    def test_empty_rcfix_is_denied(self, repo: GitRepo) -> None:
        repo.checkout("rcfix/nothing", "HEAD")
        assert not _eval(repo, "qa", "rcfix/nothing").allowed


# ---------------------------------------------------------------------------
# The API seam, same-repo heads, the CLI
# ---------------------------------------------------------------------------


def _pr(base: str, head: str, sha: str, head_repo: str | None = REPO_NAME) -> dict[str, Any]:
    return {
        "number": 7,
        "base": {"ref": base, "repo": {"full_name": REPO_NAME}},
        "head": {
            "ref": head,
            "sha": sha,
            "repo": None if head_repo is None else {"full_name": head_repo},
        },
    }


class FakeApi:
    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def __call__(self, path: str) -> Any:
        self.calls.append(path)
        return self.responses[path]


class TestRun:
    def test_base_and_head_are_read_from_the_api(self, repo: GitRepo) -> None:
        repo.checkout("feature/x", "HEAD")
        sha = repo.commit("work")
        api = FakeApi({f"repos/{REPO_NAME}/pulls/7": _pr("dev", "feature/x", sha)})
        decision = cbd.run(pr_number=7, repo=REPO_NAME, api=api, git=_git(repo))
        assert decision.allowed
        assert api.calls == [f"repos/{REPO_NAME}/pulls/7"]

    @pytest.mark.parametrize("head_repo", ["someone/fork", None])
    def test_head_must_be_in_the_same_repository(
        self, repo: GitRepo, head_repo: str | None
    ) -> None:
        repo.checkout("feature/x", "HEAD")
        sha = repo.commit("work")
        api = FakeApi({f"repos/{REPO_NAME}/pulls/7": _pr("dev", "feature/x", sha, head_repo)})
        decision = cbd.run(pr_number=7, repo=REPO_NAME, api=api, git=_git(repo))
        assert not decision.allowed
        assert decision.rule == "same-repo"

    def test_main_writes_summary_and_exits_nonzero_on_denial(
        self, repo: GitRepo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo.checkout("chore/x", "HEAD")
        sha = repo.commit("work")
        api = FakeApi({f"repos/{REPO_NAME}/pulls/7": _pr("qa", "chore/x", sha)})
        summary = tmp_path / "summary.md"
        monkeypatch.setenv("PR_NUMBER", "7")
        monkeypatch.setenv("REPO", REPO_NAME)
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
        code = cbd.main(api=api, git=_git(repo))
        assert code == 1
        text = summary.read_text(encoding="utf-8")
        assert "Branch Direction" in text and "chore/x" in text and "DENIED" in text

    def test_main_exits_zero_when_allowed(
        self, repo: GitRepo, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo.checkout("feature/x", "HEAD")
        sha = repo.commit("work")
        api = FakeApi({f"repos/{REPO_NAME}/pulls/7": _pr("dev", "feature/x", sha)})
        monkeypatch.setenv("PR_NUMBER", "7")
        monkeypatch.setenv("REPO", REPO_NAME)
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        assert cbd.main(api=api, git=_git(repo)) == 0
        assert "ALLOWED" in capsys.readouterr().out

    @pytest.mark.parametrize("pr", ["", "7; rm -rf /", "-1"])
    def test_main_refuses_a_malformed_pr_number(
        self, repo: GitRepo, monkeypatch: pytest.MonkeyPatch, pr: str
    ) -> None:
        monkeypatch.setenv("PR_NUMBER", pr)
        monkeypatch.setenv("REPO", REPO_NAME)
        assert cbd.main(api=FakeApi({}), git=_git(repo)) == 2

    def test_api_failure_fails_closed(self, repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken(_path: str) -> Any:
            raise cbd.ApiError("rate limited")

        monkeypatch.setenv("PR_NUMBER", "7")
        monkeypatch.setenv("REPO", REPO_NAME)
        assert cbd.main(api=broken, git=_git(repo)) == 2

    def test_missing_head_commit_is_fetched_from_the_pull_ref(self, repo: GitRepo) -> None:
        calls: list[list[str]] = []

        class RecordingGit(cbd.Git):
            def run(self, *args: str, check: bool = True) -> cbd.GitResult:
                calls.append(list(args))
                return super().run(*args, check=check)

        git = RecordingGit(repo.path, env=git_env())
        missing = "0" * 40
        with pytest.raises(cbd.GitError):
            git.ensure_commit(missing, pr_number=7)
        assert ["fetch", "--no-tags", "origin", "+refs/pull/7/head"] in calls

    def test_gh_api_pins_the_api_version_for_merge_commit_sha(self) -> None:
        """2026-03-10 removed ``merge_commit_sha``; the default caller pins 2022-11-28."""
        argv = cbd.gh_api_argv("repos/a/b/pulls/1")
        assert "X-GitHub-Api-Version: 2022-11-28" in argv
        assert argv[:2] == ["gh", "api"]
        assert not any(a in ("-X", "--method", "-f", "-F", "--input") for a in argv)


def test_decision_serialises_to_json() -> None:
    d = cbd.Decision(allowed=False, rule="x", base="qa", head_ref="chore/x", reasons=["r"])
    assert json.loads(d.to_json())["allowed"] is False
