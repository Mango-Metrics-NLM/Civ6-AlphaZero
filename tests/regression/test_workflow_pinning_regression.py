"""Workflow supply-chain invariants: pinned actions, and a PR trigger that fires.

Two defects this gates against, both found by the 2026-09-04 standards audit:

1. **Mutable action tags.** 26 of 27 ``uses:`` references were on a moving major
   tag (``actions/checkout@v4``). A tag is repointable by its owner, so a
   compromised or simply changed upstream silently alters what CI executes with
   the repository's token. Pinning to a full commit SHA makes the executed code
   immutable; the trailing ``# vX.Y.Z`` comment keeps it readable and gives
   Dependabot something to bump.

2. **A ``pull_request`` trigger naming a branch that is not the default.**
   ``ci.yml`` filtered on ``branches: [main]`` while this repository's default
   branch is ``claude/civ6-alphazero-implementation-vMnun``. Pull requests into
   the real default branch therefore never produced a ``pull_request``-event
   run. Anything ``pull_request``-only would have been silently dead.

Parsed with ``yaml``, never regex: CLAUDE.md (2026-08-30) records that a hand
parser over YAML is how ``validate_skills`` came to validate a folded block
scalar vacuously.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.regression.branch_policy import protected_branches

WORKFLOW_DIR = Path(".github/workflows")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
# A local action (`./.github/actions/foo`) has no upstream to pin; a reusable
# workflow reference is covered by the same rule as an action.
LOCAL_PREFIXES = ("./", "../")


def _workflow_files() -> list[Path]:
    return sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml"))


def _iter_uses(node: Any) -> list[str]:
    """Every ``uses:`` value anywhere in a parsed workflow, at any nesting depth."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "uses" and isinstance(value, str):
                found.append(value)
            else:
                found.extend(_iter_uses(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_iter_uses(item))
    return found


def _triggers(doc: dict) -> dict:
    """Return the ``on:`` block.

    PyYAML resolves the bare key ``on`` to the boolean ``True`` (YAML 1.1), so a
    lookup for the string ``"on"`` silently misses and every assertion built on
    it would pass vacuously.
    """
    if True in doc:
        return doc[True]
    return doc.get("on", {})


def test_workflow_directory_is_not_empty() -> None:
    """Guard against the whole suite passing because it found no files."""
    assert _workflow_files(), "no workflow files found; every other test here would be vacuous"


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_every_action_is_pinned_to_a_sha(path: Path) -> None:
    """No ``uses:`` reference may sit on a mutable tag or branch."""
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    unpinned: list[str] = []
    for ref in _iter_uses(doc):
        if ref.startswith(LOCAL_PREFIXES):
            continue
        _, _, version = ref.partition("@")
        if not SHA_RE.match(version):
            unpinned.append(ref)
    assert not unpinned, (
        f"{path.name} references actions by mutable ref: {unpinned}. "
        "Pin to a full 40-character commit SHA with a trailing '# vX.Y.Z' comment."
    )


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_pinned_actions_carry_a_version_comment(path: Path) -> None:
    """A bare SHA is unreadable and unmaintainable; require the version comment."""
    missing = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if re.search(r"uses:\s*[^\s]+@[0-9a-f]{40}", line) and "#" not in line
    ]
    assert not missing, f"{path.name} has SHA-pinned actions with no '# vX.Y.Z' comment: {missing}"


def test_ci_pull_request_trigger_is_not_filtered_to_a_stale_branch() -> None:
    """A ``branches`` filter on ``pull_request`` must not exclude the default branch.

    The safest form is no filter at all, so every pull request runs CI. If a
    filter is reintroduced it must name the repository's real default branch,
    which is deliberately read from git rather than hardcoded here -- hardcoding
    it is what produced the original defect.
    """
    doc = yaml.safe_load((WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8"))
    triggers = _triggers(doc)
    assert "pull_request" in triggers, "ci.yml no longer runs on pull_request at all"

    pr = triggers["pull_request"]
    if not pr or "branches" not in pr:
        return  # unfiltered: every PR runs CI, which is what we want

    import subprocess

    head = subprocess.run(
        ["git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    default_branch = head.removeprefix("origin/") if head else ""
    if not default_branch:
        pytest.skip("origin/HEAD is not resolvable in this checkout")

    patterns = pr["branches"]
    matched = any(
        default_branch == pattern
        or pattern in ("**", "*")
        or default_branch.startswith(pattern.rstrip("*/"))
        for pattern in patterns
    )
    assert matched, (
        f"ci.yml pull_request filters on {patterns}, which does not include the "
        f"repository default branch {default_branch!r}: PRs into it would run no CI"
    )


def test_the_pinning_check_rejects_a_mutable_tag(tmp_path: Path) -> None:
    """Negative control: the check must fail on a planted mutable reference."""
    planted = tmp_path / "planted.yml"
    planted.write_text(
        "name: X\non:\n  push:\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - uses: actions/checkout@v4\n"
    )
    doc = yaml.safe_load(planted.read_text(encoding="utf-8"))
    refs = _iter_uses(doc)
    assert refs == ["actions/checkout@v4"], "the uses-extractor missed a reference"
    assert not SHA_RE.match(refs[0].partition("@")[2]), "a mutable tag was accepted as a SHA"


def test_the_uses_extractor_finds_nested_references() -> None:
    """Negative control for the walker itself: it must not miss deep nesting."""
    doc = yaml.safe_load(
        "jobs:\n  a:\n    steps:\n      - uses: x/y@" + "0" * 40 + "\n"
        "  b:\n    steps:\n      - uses: p/q@" + "1" * 40 + "\n"
    )
    assert len(_iter_uses(doc)) == 2


# ---------------------------------------------------------------------------
# Release-readiness invariants added 2026-11-10 (P5). Each has an in-memory
# negative control below, so the predicate is proven to discriminate before the
# parametrised tests over the real workflows are trusted.
# ---------------------------------------------------------------------------

#: Packages a workflow may `pip install` without `==`, each with the reason.
#: Anything else without an exact pin is a gate whose definition changes under
#: you. Keep this list short; a new entry needs a sentence, not a name.
UNPINNED_INSTALL_ALLOWED: dict[str, str] = {
    "pip": (
        "`python -m pip install --upgrade pip` tracks the runner's pip; pinning it "
        "would add a fifth pin to keep in step for no supply-chain gain"
    ),
    "torch": (
        "the version floor lives in pyproject.toml (`torch>=2.0.0`); the CI step "
        "exists only to choose the CPU wheel index before the dev extra resolves, "
        "and an exact pin here would be a second torch pin that CI alone obeys"
    ),
}

_PIP_FLAGS_WITH_VALUE = frozenset(
    {
        "-r",
        "--requirement",
        "-c",
        "--constraint",
        "-e",
        "--editable",
        "-i",
        "--index-url",
        "--extra-index-url",
        "-f",
        "--find-links",
        "-t",
        "--target",
        "--prefix",
        "--root",
        "--platform",
        "--python-version",
        "--implementation",
        "--abi",
    }
)


def _default_branch() -> str:
    """The repository's default branch, resolved rather than hardcoded.

    Hardcoding it is what produced the original ``branches: [main]`` defect.
    Order: an explicit override, the clone's ``origin/HEAD`` (not set by
    actions/checkout), then the harness settings file, which the PreToolUse
    push guard reads for the same purpose.
    """
    override = os.environ.get("CIV6AZ_DEFAULT_BRANCH", "").strip()
    if override:
        return override
    head = subprocess.run(
        ["git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if head:
        return head.removeprefix("origin/")
    settings = Path(".claude/settings.json")
    if settings.exists():
        try:
            declared = (
                json.loads(settings.read_text(encoding="utf-8"))
                .get("env", {})
                .get("CIV6AZ_DEFAULT_BRANCH")
            )
        except json.JSONDecodeError:
            declared = None
        if declared:
            return str(declared)
    pytest.fail(
        "cannot determine the default branch: origin/HEAD is unset and "
        ".claude/settings.json declares no CIV6AZ_DEFAULT_BRANCH"
    )


def _is_glob(pattern: str) -> bool:
    return any(ch in pattern for ch in "*?[")


def _push_branches(doc: dict) -> list[str]:
    push = _triggers(doc).get("push")
    if not isinstance(push, dict):
        return []
    branches = push.get("branches") or []
    if isinstance(branches, str):
        branches = [branches]
    return [str(b) for b in branches]


def stale_push_branches(
    doc: dict, default_branch: str, protected: tuple[str, ...] = ()
) -> list[str]:
    """``push.branches`` entries that are not protected, the default, ``main`` or a glob.

    The protected list (``dev qa main`` plus ``CIV6AZ_PROTECTED_BRANCHES``,
    see ``branch_policy``) is allowed because those are the branches whose
    pushes must run CI. While the old default branch is still on that list the
    rename cannot open a window with no CI on the default branch.
    """
    allowed = {default_branch, "main", *protected}
    return [b for b in _push_branches(doc) if b not in allowed and not _is_glob(b)]


# The workflows that gate merges and releases, and so must run on every
# protected branch (plan S3.1 / LF-R1).
PROTECTED_PUSH_WORKFLOWS = ("ci.yml", "codeql.yml", "security-scan.yml", "scorecard.yml")


def missing_protected_push_branches(doc: dict, protected: tuple[str, ...]) -> list[str]:
    """Protected branches a workflow's ``push`` trigger does not name."""
    named = set(_push_branches(doc))
    return [b for b in protected if b not in named]


def cancel_in_progress_offenders(doc: dict, protected: tuple[str, ...]) -> list[str]:
    """Protected branches whose push runs this workflow could cancel.

    A run on a protected branch records a state the promotion and release gates
    read, so a newer push must not cancel it. Accepted: no concurrency block,
    ``cancel-in-progress: false``, or an expression that names
    ``refs/heads/<branch>`` for every protected branch the push trigger lists
    (``${{ !contains(fromJSON('[...]'), github.ref) }}``).
    """
    pushed = [b for b in _push_branches(doc) if b in protected]
    if not pushed:
        return []
    concurrency = doc.get("concurrency")
    if not isinstance(concurrency, dict):
        return []
    value = concurrency.get("cancel-in-progress", False)
    if value is False or (isinstance(value, str) and value.strip().lower() == "false"):
        return []
    if value is True or not isinstance(value, str) or "${{" not in value:
        return pushed
    return [b for b in pushed if f'"refs/heads/{b}"' not in value]


def _pip_install_args(command: str) -> list[str] | None:
    """The arguments of a ``pip install`` in one shell command, or None."""
    try:
        tokens = shlex.split(command, comments=True)
    except ValueError:
        return None
    for index, token in enumerate(tokens):
        if token not in ("pip", "pip3"):
            continue
        rest = tokens[index + 1 :]
        while rest and rest[0].startswith("-"):
            rest.pop(0)
        if rest and rest[0] == "install":
            return rest[1:]
        return None
    return None


def unpinned_installs(script: str) -> list[str]:
    """Requirement tokens in ``pip install`` lines that carry no ``==``.

    A local project (``.``, ``.[dev]``, ``-e .``), a requirements or constraints
    file, a path or URL, and the packages in :data:`UNPINNED_INSTALL_ALLOWED`
    are not offences; everything else must be an exact pin.
    """
    joined = re.sub(r"\\\n", " ", script)
    offenders: list[str] = []
    for command in re.split(r"&&|\|\||;|\n", joined):
        args = _pip_install_args(command)
        if not args:
            continue
        i = 0
        while i < len(args):
            token = args[i]
            if token in _PIP_FLAGS_WITH_VALUE:
                i += 2
                continue
            if token.startswith("-"):
                i += 1
                continue
            if token.startswith(".") or "/" in token or token.startswith("git+") or "@" in token:
                i += 1
                continue
            name = re.split(r"[\[<>=!~ ]", token, maxsplit=1)[0].lower()
            if name in UNPINNED_INSTALL_ALLOWED:
                i += 1
                continue
            if "==" not in token:
                offenders.append(token)
            i += 1
    return offenders


def _steps(doc: dict) -> list[tuple[str, dict, dict]]:
    """``(job name, job, step)`` for every step in a parsed workflow."""
    out = []
    for job_name, job in (doc.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            if isinstance(step, dict):
                out.append((job_name, job, step))
    return out


def secret_steps_with_unpinned_installs(doc: dict) -> list[str]:
    """Steps that can see a secret AND install something unpinned.

    An unpinned install is an arbitrary-code hole; next to a secret it is an
    exfiltration channel. Checked at step level and through job-level ``env``.
    """
    offenders = []
    for job_name, job, step in _steps(doc):
        sees_secret = "secrets." in yaml.safe_dump(step) or "secrets." in yaml.safe_dump(
            job.get("env") or {}
        )
        if not sees_secret:
            continue
        run = step.get("run")
        if isinstance(run, str) and (bad := unpinned_installs(run)):
            offenders.append(f"{job_name}/{step.get('name', '<unnamed>')}: {bad}")
    return offenders


def checkouts_persisting_credentials(doc: dict) -> list[str]:
    """``actions/checkout`` steps that do not set ``persist-credentials: false``."""
    offenders = []
    for job_name, _job, step in _steps(doc):
        uses = step.get("uses", "")
        if not isinstance(uses, str) or not uses.startswith("actions/checkout@"):
            continue
        value = (step.get("with") or {}).get("persist-credentials")
        if value is False or (isinstance(value, str) and value.lower() == "false"):
            continue
        offenders.append(f"{job_name}: {uses}")
    return offenders


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_push_triggers_name_only_the_default_branch_main_or_a_glob(path: Path) -> None:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    protected = protected_branches()
    stale = stale_push_branches(doc, _default_branch(), protected)
    assert not stale, (
        f"{path.name} push trigger names {stale}: not a protected branch "
        f"({' '.join(protected)}), not the default branch ({_default_branch()!r}), "
        "not a glob. A concrete branch name here goes stale silently when the "
        "branch is renamed or deleted."
    )


@pytest.mark.parametrize("name", PROTECTED_PUSH_WORKFLOWS)
def test_gating_workflows_run_on_every_protected_branch(name: str) -> None:
    doc = yaml.safe_load((WORKFLOW_DIR / name).read_text(encoding="utf-8"))
    missing = missing_protected_push_branches(doc, protected_branches())
    assert not missing, (
        f"{name} does not run on push to {missing}. Every protected branch "
        "(.claude/settings.json CIV6AZ_PROTECTED_BRANCHES) needs a CI record of "
        "each state it reaches."
    )


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_runs_on_protected_branches_are_never_cancelled(path: Path) -> None:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    offenders = cancel_in_progress_offenders(doc, protected_branches())
    assert not offenders, (
        f"{path.name}: cancel-in-progress can cancel push runs on {offenders}. Use "
        "`${{ !contains(fromJSON('[\"refs/heads/<branch>\", ...]'), github.ref) }}` "
        "naming every protected branch the push trigger lists."
    )


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_every_pip_install_is_pinned(path: Path) -> None:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    offenders = []
    for job_name, _job, step in _steps(doc):
        run = step.get("run")
        if isinstance(run, str) and (bad := unpinned_installs(run)):
            offenders.append(f"{job_name}/{step.get('name', '<unnamed>')}: {bad}")
    assert not offenders, (
        f"{path.name} installs packages without an exact `==` pin: {offenders}. "
        "Pin them, or install them through the project's extras where pyproject "
        "already constrains them."
    )


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_no_secret_bearing_step_runs_an_unpinned_install(path: Path) -> None:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    offenders = secret_steps_with_unpinned_installs(doc)
    assert not offenders, (
        f"{path.name}: steps that see a secret also run an unpinned install: {offenders}"
    )


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_every_checkout_disables_credential_persistence(path: Path) -> None:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    offenders = checkouts_persisting_credentials(doc)
    assert not offenders, (
        f"{path.name}: actions/checkout without `persist-credentials: false`: {offenders}. "
        "No step in this repository pushes with the checkout's token, so it has no "
        "business staying in .git/config for later steps to read."
    )


def test_unpinned_install_allowlist_entries_have_reasons() -> None:
    empty = [name for name, why in UNPINNED_INSTALL_ALLOWED.items() if len(why.strip()) < 20]
    assert not empty, f"allowlist entries without a real reason: {empty}"


class TestTheNewChecksDiscriminate:
    """Negative controls on in-memory YAML, independent of the real workflows."""

    _ONE_JOB = (
        "name: X\non:\n  push:\n    branches: [{branches}]\njobs:\n  a:\n"
        "    runs-on: ubuntu-latest\n    steps:\n{steps}"
    )

    def test_a_stale_push_branch_is_reported(self) -> None:
        doc = yaml.safe_load(
            self._ONE_JOB.format(branches="main, feature/old, 'claude/**'", steps="      []")
        )
        assert stale_push_branches(doc, "trunk") == ["feature/old"]

    def test_default_main_and_globs_are_accepted(self) -> None:
        doc = yaml.safe_load(
            self._ONE_JOB.format(branches="main, trunk, 'release/*'", steps="      []")
        )
        assert stale_push_branches(doc, "trunk") == []

    def test_a_push_trigger_without_a_filter_is_fine(self) -> None:
        doc = yaml.safe_load("on:\n  push:\n  pull_request:\njobs: {}\n")
        assert stale_push_branches(doc, "trunk") == []

    def test_protected_branches_are_accepted_and_others_still_reported(self) -> None:
        doc = yaml.safe_load(
            self._ONE_JOB.format(branches="dev, qa, main, legacy, feature/old", steps="      []")
        )
        assert stale_push_branches(doc, "dev", ("dev", "qa", "main", "legacy")) == ["feature/old"]

    def test_a_missing_protected_branch_is_reported(self) -> None:
        doc = yaml.safe_load(self._ONE_JOB.format(branches="dev, main", steps="      []"))
        assert missing_protected_push_branches(doc, ("dev", "qa", "main")) == ["qa"]

    _CANCEL = (
        "on:\n  push:\n    branches: [dev, qa]\nconcurrency:\n  group: g\n{cancel}jobs: {{}}\n"
    )

    def test_literal_cancel_in_progress_on_a_protected_push_is_reported(self) -> None:
        doc = yaml.safe_load(self._CANCEL.format(cancel="  cancel-in-progress: true\n"))
        assert cancel_in_progress_offenders(doc, ("dev", "qa", "main")) == ["dev", "qa"]

    def test_an_expression_missing_a_branch_is_reported(self) -> None:
        expr = "${{ !contains(fromJSON('[\"refs/heads/dev\"]'), github.ref) }}"
        doc = yaml.safe_load(self._CANCEL.format(cancel=f"  cancel-in-progress: {expr}\n"))
        assert cancel_in_progress_offenders(doc, ("dev", "qa", "main")) == ["qa"]

    def test_a_complete_expression_or_false_is_accepted(self) -> None:
        expr = '${{ !contains(fromJSON(\'["refs/heads/dev","refs/heads/qa"]\'), github.ref) }}'
        doc = yaml.safe_load(self._CANCEL.format(cancel=f"  cancel-in-progress: {expr}\n"))
        assert cancel_in_progress_offenders(doc, ("dev", "qa", "main")) == []
        doc = yaml.safe_load(self._CANCEL.format(cancel="  cancel-in-progress: false\n"))
        assert cancel_in_progress_offenders(doc, ("dev", "qa", "main")) == []

    def test_cancelling_feature_branch_runs_is_fine(self) -> None:
        doc = yaml.safe_load(
            "on:\n  push:\n    branches: ['claude/**']\n"
            "concurrency:\n  group: g\n  cancel-in-progress: true\njobs: {}\n"
        )
        assert cancel_in_progress_offenders(doc, ("dev", "qa", "main")) == []

    def test_an_unpinned_install_is_reported(self) -> None:
        script = (
            "python -m pip install --upgrade pip\n"
            'pip install -e ".[dev]" pytest-cov\n'
            "pip install ruff==0.15.8 pytest\n"
            "pip install -r requirements.txt build==1.6.1 && pip install huggingface_hub"
        )
        assert unpinned_installs(script) == ["pytest-cov", "pytest", "huggingface_hub"]

    def test_pinned_and_project_installs_are_accepted(self) -> None:
        script = (
            "python -m pip install --upgrade pip\n"
            'pip install -e ".[dev,hub]"\n'
            "pip install zizmor==1.32.0 \\\n  pip-audit==2.10.1\n"
            'python -m pip install "torch>=2.0.0" --index-url https://download.pytorch.org/whl/cpu\n'
            "pip install git+https://github.com/x/y@abc\n"
            "pip-audit -r requirements.txt --format json"
        )
        assert unpinned_installs(script) == []

    def test_a_secret_step_with_an_unpinned_install_is_reported(self) -> None:
        steps = (
            "      - name: deploy\n        env:\n          TOKEN: ${{{{ secrets.T }}}}\n"
            "        run: pip install huggingface_hub\n"
            "      - name: clean\n        run: pip install huggingface_hub\n"
        )
        doc = yaml.safe_load(self._ONE_JOB.format(branches="main", steps=steps))
        assert secret_steps_with_unpinned_installs(doc) == ["a/deploy: ['huggingface_hub']"]

    def test_a_secret_step_with_a_pinned_install_is_accepted(self) -> None:
        steps = (
            "      - name: deploy\n        env:\n          TOKEN: ${{{{ secrets.T }}}}\n"
            "        run: pip install huggingface_hub==0.36.2\n"
        )
        doc = yaml.safe_load(self._ONE_JOB.format(branches="main", steps=steps))
        assert secret_steps_with_unpinned_installs(doc) == []

    def test_a_checkout_persisting_credentials_is_reported(self) -> None:
        sha = "0" * 40
        steps = (
            f"      - uses: actions/checkout@{sha}\n"
            f"      - uses: actions/checkout@{sha}\n        with:\n          fetch-depth: 0\n"
            f"      - uses: actions/checkout@{sha}\n        with:\n"
            "          persist-credentials: false\n"
            f"      - uses: actions/setup-python@{sha}\n"
        )
        doc = yaml.safe_load(self._ONE_JOB.format(branches="main", steps=steps))
        assert checkouts_persisting_credentials(doc) == [
            f"a: actions/checkout@{sha}",
            f"a: actions/checkout@{sha}",
        ]
