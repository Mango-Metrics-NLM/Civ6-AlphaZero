"""Regression tests for repository-hygiene invariants.

These are cheap, deterministic, dependency-free checks on the repository's own
configuration. They exist because each one pins a defect that was actually
found in this repo rather than a hypothetical one:

1. **No tracked file may be gitignored.** Five were: ``docker-compose.yml`` and
   the four ``kb/sessions/*.state.json`` role-state files. Git only applies
   ignore rules to *untracked* paths, so the contradiction is silent -- until
   someone adds a new file of that shape and it vanishes from ``git add``.
   CLAUDE.md 14.5 rule 3 calls the session-state files "the ONLY mutable state
   per role", so ignoring them contradicted the documented workflow.
2. **No duplicate ignore entries.** ``.gitignore`` had accumulated 83, which
   makes the file hard to audit and hides which rule actually matches.
3. **A secret-scanning config must exist and stay scoped.** An allowlist that
   matches everything would turn the gate into a no-op while still reporting
   success.

They run in the default unit lane (no marker, no heavy imports) so they gate
every PR.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Ignore rules whose whole purpose is to match broadly; a duplicate-free check
# still applies, but these are expected to be globs.
_GITIGNORE = REPO_ROOT / ".gitignore"
_GITLEAKS_CONFIG = REPO_ROOT / ".gitleaks.toml"


def _git(*args: str) -> str:
    """Run git in the repo and return stdout, or skip if git is unavailable."""
    if shutil.which("git") is None:  # pragma: no cover - environment guard
        pytest.skip("git is not available")
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


def _allowlist_paths_from(text: str) -> list[str]:
    """Every ``paths`` regex in the config's allowlists, parsed as real TOML.

    The first version of this was a regex that recognised only triple-single-
    quoted strings. A catch-all spelled ``"^.*$"`` -- an ordinary, valid TOML
    basic string -- was therefore invisible to it, while the other entries kept
    ``assert patterns`` satisfied. So the behavioural check above it could be
    bypassed by writing the exemption in a different quoting style, which is the
    same defect one layer down: a gate reporting success on input it never saw.

    ``tomllib`` is stdlib from Python 3.11 and this project requires 3.10+ with
    CI on 3.11/3.12, so the "stay stdlib-only" constraint the hand parser was
    written for is satisfied without hand-parsing anything. Both a top-level
    ``[allowlist]`` and the ``[[rules.allowlists]]`` form are covered.
    """
    data = tomllib.loads(text)
    found: list[str] = []

    def _collect(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "paths" and isinstance(value, list):
                    found.extend(str(item) for item in value)
                else:
                    _collect(value)
        elif isinstance(node, list):
            for item in node:
                _collect(item)

    _collect(data)
    return found


def _gitleaks_allowlist_paths() -> list[str]:
    """The live config's allowlist path regexes."""
    return _allowlist_paths_from(_GITLEAKS_CONFIG.read_text(encoding="utf-8"))


def _ignore_entries() -> list[str]:
    lines = _GITIGNORE.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


class TestGitignoreInvariants:
    """`.gitignore` must not contradict what the repository actually tracks."""

    def test_no_tracked_file_is_ignored(self) -> None:
        """A tracked-but-ignored file is a silent trap for the next contributor.

        Git applies ignore rules only to untracked paths, so such a file keeps
        working until someone adds a sibling -- which then silently fails to
        stage. Fix by negating the specific path (``!path``), not by deleting
        the broad rule.
        """
        tracked = [p for p in _git("ls-files").splitlines() if p]
        if not tracked:  # pragma: no cover - not a git checkout
            pytest.skip("not a git working tree")

        # --no-index so the answer reflects the rules, not the index state.
        proc = subprocess.run(
            ["git", "check-ignore", "--stdin", "--no-index"],
            cwd=REPO_ROOT,
            input="\n".join(tracked),
            capture_output=True,
            text=True,
            check=False,
        )
        offenders = sorted(p for p in proc.stdout.splitlines() if p)
        assert not offenders, (
            "these files are tracked by git but matched by .gitignore, so a new "
            f"file of the same shape could not be added: {offenders}"
        )

    def test_no_duplicate_ignore_entries(self) -> None:
        """Duplicates make it ambiguous which rule is load-bearing."""
        entries = _ignore_entries()
        seen: set[str] = set()
        dupes: list[str] = []
        for entry in entries:
            if entry in seen:
                dupes.append(entry)
            seen.add(entry)
        assert not dupes, f".gitignore has duplicate entries: {sorted(set(dupes))}"

    def test_locally_fetched_tooling_is_ignored(self) -> None:
        """`make gitleaks-install` drops a binary in `.tools/`; never commit it."""
        assert any(e.rstrip("/") == ".tools" for e in _ignore_entries()), (
            ".tools/ must be gitignored -- `make gitleaks-install` writes a binary there"
        )


class TestSecretScanningConfig:
    """The gitleaks gate must exist and must not allowlist itself into a no-op."""

    def test_config_exists(self) -> None:
        assert _GITLEAKS_CONFIG.is_file(), (
            ".gitleaks.toml is missing -- `make security-secrets` and the CI "
            "secret-scan step both depend on it"
        )

    def test_config_extends_default_ruleset(self) -> None:
        """Inheriting upstream rules means a version bump adds detections."""
        text = _GITLEAKS_CONFIG.read_text(encoding="utf-8")
        assert "useDefault = true" in text, (
            "the config must extend the default gitleaks ruleset rather than "
            "restating a frozen subset of it"
        )

    def test_allowlist_does_not_match_ordinary_source_paths(self) -> None:
        """Validate the allowlist's *behaviour*, not four exact spellings.

        The first version of this test looked for the literal strings ``.*``
        wrapped in quotes. An equivalent catch-all spelled ``^.*$`` contains none
        of them, matches every path, and left the test green -- a gate that
        reports success while detecting nothing.

        This compiles the real allowlist patterns and asserts that none of them
        matches ordinary tracked source.
        """
        patterns = _gitleaks_allowlist_paths()
        assert patterns, "no allowlist paths parsed from .gitleaks.toml -- parser drift?"

        probes = [
            "alpha_zero/core/civ6_network.py",
            "alpha_zero/core/mcts/gumbel_ops.py",
            "scripts/validate_agents.py",
            "tests/unit/test_mcts_split_backcompat.py",
            "README.md",
            "Makefile",
        ]
        for raw in patterns:
            compiled = re.compile(raw)
            hits = [p for p in probes if compiled.search(p)]
            assert not hits, (
                f"gitleaks allowlist pattern {raw!r} matches ordinary source {hits}; "
                "an over-broad exemption hides real secrets"
            )

    @pytest.mark.parametrize(
        "quoting",
        [
            pytest.param("'''^.*$'''", id="triple-single"),
            pytest.param('"^.*$"', id="basic-string"),
            pytest.param("'^.*$'", id="literal-string"),
            pytest.param('"""^.*$"""', id="triple-double"),
        ],
    )
    def test_a_catch_all_is_detected_however_it_is_quoted(self, quoting: str) -> None:
        """The control must run through the real parser, not around it.

        The earlier version of this control compiled ``^.*$`` directly, so it
        proved the *predicate* discriminates while saying nothing about the
        *extraction*. The parser only understood triple-single-quoted strings,
        so a catch-all written ``"^.*$"`` would have been dropped before the
        predicate ever saw it -- and the suite stayed green.
        """
        config = f"[allowlist]\npaths = [\n    {quoting},\n]\n"
        patterns = _allowlist_paths_from(config)
        assert patterns, f"parser dropped a {quoting} entry entirely"
        assert any(re.compile(raw).search("alpha_zero/core/civ6_network.py") for raw in patterns), (
            "a catch-all written this way slips past the allowlist check"
        )

    def test_parser_reads_rule_scoped_allowlists_too(self) -> None:
        """``[[rules.allowlists]]`` is the other place a path exemption can hide."""
        config = '[[rules]]\nid = "x"\nregex = "y"\n[[rules.allowlists]]\npaths = ["^.*$"]\n'
        assert _allowlist_paths_from(config) == ["^.*$"]

    def test_pinned_version_is_single_sourced(self) -> None:
        """The tool version lives in the Makefile, not scattered across configs."""
        makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        assert "GITLEAKS_VERSION ?=" in makefile, (
            "GITLEAKS_VERSION must be an overridable Makefile variable so local "
            "runs and CI resolve the same tool version"
        )
