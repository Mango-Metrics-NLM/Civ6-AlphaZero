"""Every file path a Makefile recipe names must exist in the tree.

The 2026-09-04 standards audit found ``make setup-bucket`` running ``gsutil
lifecycle set infra/gcs_lifecycle.json`` against a file that has never been in
the repository, next to four ``run-*`` targets wrapping a launcher whose submit
path imports a function that does not exist. A target that fails on its first
file reference is documentation pretending to be automation, and nothing short
of running every target would have shown it -- so this test reads the Makefile
instead.

Scope is deliberately literal: tokens in recipe lines (and simple ``VAR ?= value``
defaults) that look like repository paths -- anything containing a ``/`` or
ending in a source/config extension -- are resolved against the tree. Anything
built from a make variable (``$(CURDIR)/.tools``), a URL, an absolute system
path or a shell operator is not a repository path and is skipped. Paths the
Makefile *creates* (coverage reports, the test-storage scratch dir, build
output) live in :data:`GENERATED`, each with the target that produces it, the
same reasoned-allowlist shape as ``NOT_GATED`` in the suite-wiring gate.

Negative controls: the in-tree one was ``infra/gcs_lifecycle.json`` itself --
this file failed on the unmodified Makefile before that reference was removed
(recorded in the PR) -- and :func:`test_a_missing_path_in_a_planted_makefile_is_reported`
plants a fresh defect so the control survives the fix.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / "Makefile"

#: Paths a recipe writes rather than reads, keyed to the target that produces
#: them. An entry needs the producing target so a stale exemption is visible.
GENERATED: dict[str, str] = {
    "./test_storage": "local-smoke writes it; clean-test-storage removes it",
    "coverage-cloud.json": "test-cloud-coverage writes it; clean-all removes it",
}

_EXTENSIONS = (
    "py",
    "yaml",
    "yml",
    "toml",
    "json",
    "txt",
    "cfg",
    "ini",
    "md",
    "sh",
)
# A slash-separated relative path, a bare filename with a source/config
# extension, or an rc-style dotfile such as ``.coveragerc-cloud``. The dotfile
# arm is deliberately narrow: ``rm -rf .coverage`` names a product, and a
# ``python -c '...open(f).read()...'`` one-liner yields ``.read`` as a token.
_PATH_TOKEN = re.compile(
    r"^[\w.*-]+(?:/[\w.*-]+)+/?$"
    r"|^\.?[\w*-]+\.(?:" + "|".join(_EXTENSIONS) + r")$"
    r"|^\.[\w-]*rc[\w-]*$"
)
_SYSTEM_PREFIXES = ("/", "~", "http://", "https://", "gs://", "git+")


def _candidate_lines(text: str) -> list[str]:
    """Recipe lines (tab-indented) plus simple variable defaults."""
    lines = []
    for raw in text.splitlines():
        if raw.startswith("\t"):
            lines.append(raw)
        elif re.match(r"^[A-Z_][A-Z0-9_]*\s*\??=", raw):
            lines.append(raw.split("=", 1)[1])
    return lines


def _tokens(line: str) -> list[str]:
    """Shell-ish words with flag prefixes (``--cov-config=X``) and quotes stripped."""
    out = []
    for word in re.split(r"[\s;|&()]+", line):
        word = word.strip("'\"`,")
        if not word or "$" in word or word.startswith(_SYSTEM_PREFIXES):
            continue
        if word.startswith("-") and "=" in word:
            word = word.split("=", 1)[1]
        if word.startswith("-"):
            continue
        out.append(word.strip("'\""))
    return out


def referenced_paths(text: str) -> set[str]:
    """Every repository-path-looking token in the Makefile's recipes."""
    found: set[str] = set()
    for line in _candidate_lines(text):
        for token in _tokens(line):
            if _PATH_TOKEN.match(token) and token not in (".", ".."):
                found.add(token.rstrip("/"))
    return found


def missing_paths(text: str, root: Path, generated: dict[str, str]) -> list[str]:
    """Referenced paths with no file, directory or glob match under ``root``."""
    missing = []
    for ref in sorted(referenced_paths(text)):
        if ref in generated:
            continue
        if any(ch in ref for ch in "*?["):
            if not glob.glob(str(root / ref)):
                missing.append(ref)
        elif not (root / ref).exists():
            missing.append(ref)
    return missing


def test_the_extractor_sees_the_paths_the_makefile_visibly_uses() -> None:
    """Guard against a parser that finds nothing and therefore proves nothing."""
    refs = referenced_paths(MAKEFILE.read_text(encoding="utf-8"))
    for expected in (
        "dockerfiles/Dockerfile.vertex",
        ".coveragerc-cloud",
        "scripts/check_coverage_floors.py",
    ):
        assert expected in refs, f"{expected} is named by a recipe but the extractor missed it"


def test_every_path_a_recipe_names_exists() -> None:
    missing = missing_paths(MAKEFILE.read_text(encoding="utf-8"), REPO_ROOT, GENERATED)
    assert not missing, (
        f"Makefile recipes reference paths that do not exist: {missing}. Either the "
        "target is dead (delete it), the path moved (fix it), or the recipe creates "
        "it (add it to GENERATED with the producing target)."
    )


def test_generated_exemptions_are_still_referenced() -> None:
    """A GENERATED entry no recipe names any more is stale; drop it."""
    refs = referenced_paths(MAKEFILE.read_text(encoding="utf-8"))
    stale = sorted(p for p in GENERATED if p not in refs)
    assert not stale, f"GENERATED lists paths no recipe references: {stale}"


def test_a_missing_path_in_a_planted_makefile_is_reported(tmp_path: Path) -> None:
    """Negative control: a recipe naming a file that is not there must fail."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "real.yaml").write_text("a: 1\n")
    planted = (
        "CONFIG ?= configs/real.yaml\n"
        ".PHONY: deploy\n"
        "deploy:\n"
        "\tgsutil lifecycle set infra/gcs_lifecycle.json gs://$(BUCKET)\n"
        "\tpython scripts/run.py --config $(CONFIG)\n"
        "\tcurl https://example.com/x/y.json -o out.json\n"
    )
    missing = missing_paths(planted, tmp_path, {"out.json": "curl writes it"})
    assert missing == ["infra/gcs_lifecycle.json", "scripts/run.py"], missing


def test_variable_built_and_system_paths_are_not_checked() -> None:
    """The extractor must not flag what it cannot resolve statically."""
    line = (
        "\tcurl -sSfL $(URL) | tar -xz -C $(CURDIR)/.tools gitleaks; "
        "cat /dev/null; echo gs://bucket/x.json https://a/b.py"
    )
    assert referenced_paths(line) == set()
