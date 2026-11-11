"""Every ``CLAUDE.md §N`` / ``§N.M`` citation must resolve to a heading that exists.

CLAUDE.md is cited by section number from ~75 places (specs, handoffs, scripts,
the PR template). Its eventual split is deferred (CONFIRM-FIRST), but this gate
lands now so that a split, a renumbering or a deleted section cannot silently
orphan a citation. Two shapes are checked:

* in every tracked text file outside ``docs/archive/``: ``CLAUDE.md §N`` or
  ``CLAUDE.md §N.M`` (an optional closing backtick between the name and the
  section sign is accepted, as in ```CLAUDE.md` §5.3``);
* inside CLAUDE.md itself: every bare ``§N`` / ``§N.M``.

A citation resolves when CLAUDE.md has ``## SECTION N:`` (for ``§N``) or
``### N.M`` (for ``§N.M``). The pure functions are tested on in-memory strings so
the negative control does not depend on the repository's state.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"

# A section sign followed by a number; the "CLAUDE.md" prefix is matched in a
# second pattern so the same core serves both shapes.
_SECTION_RE = r"§\s*(\d+)(?:\.(\d+))?"
CITATION_RE = re.compile(r"CLAUDE\.md`?\s*" + _SECTION_RE)
BARE_SECTION_RE = re.compile(_SECTION_RE)

_H2_RE = re.compile(r"^## SECTION (\d+):", re.MULTILINE)
_H3_RE = re.compile(r"^### (\d+)\.(\d+)\b", re.MULTILINE)

TRACKED_GLOBS = ("*.md", "*.py", "*.sh", "*.json", "*.yaml", "*.yml")
EXCLUDED_PREFIXES = ("docs/archive/",)


def parse_headings(text: str) -> tuple[set[str], set[str]]:
    """Return (``{"1", "2", ...}``, ``{"1.1", "5.3", ...}``) from CLAUDE.md text."""
    sections = {m.group(1) for m in _H2_RE.finditer(text)}
    subsections = {f"{m.group(1)}.{m.group(2)}" for m in _H3_RE.finditer(text)}
    return sections, subsections


def find_citations(text: str, pattern: re.Pattern[str] = CITATION_RE) -> list[str]:
    """Every cited section id in ``text``, as ``"N"`` or ``"N.M"`` strings."""
    out: list[str] = []
    for m in pattern.finditer(text):
        out.append(m.group(1) if m.group(2) is None else f"{m.group(1)}.{m.group(2)}")
    return out


def dangling(citations: list[str], sections: set[str], subsections: set[str]) -> list[str]:
    return [c for c in citations if c not in (sections if "." not in c else subsections)]


def tracked_text_files(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--", *TRACKED_GLOBS],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [
        root / line
        for line in out.splitlines()
        if line and not line.startswith(EXCLUDED_PREFIXES) and (root / line).is_file()
    ]


# --- negative controls on in-memory strings ---------------------------------

FAKE_CLAUDE = "## SECTION 1: A\n\n### 1.1 B\n\n## SECTION 2: C\n\n### 2.1 D\n"


def test_parse_headings_from_string():
    assert parse_headings(FAKE_CLAUDE) == ({"1", "2"}, {"1.1", "2.1"})


def test_citation_shapes_are_recognised():
    # Built by concatenation so this file does not cite the sections it tests.
    text = " ".join(
        [
            "CLAUDE.md " + "§" + "2",
            "(`CLAUDE.md` " + "§" + "1.1)",
            "CLAUDE.md" + "§" + "2.1",
        ]
    )
    assert find_citations(text) == ["2", "1.1", "2.1"]


def test_dangling_citation_is_detected():
    sections, subsections = parse_headings(FAKE_CLAUDE)
    text = "see CLAUDE.md " + "§" + "99.9 and CLAUDE.md " + "§" + "7"
    assert dangling(find_citations(text), sections, subsections) == ["99.9", "7"]


def test_resolving_citation_is_accepted():
    sections, subsections = parse_headings(FAKE_CLAUDE)
    text = "see CLAUDE.md " + "§" + "1.1 and CLAUDE.md " + "§" + "2"
    assert dangling(find_citations(text), sections, subsections) == []


def test_unrelated_section_signs_are_not_citations():
    # A charter citation is not a CLAUDE.md citation.
    assert find_citations("amendment protocol in " + "§" + "7") == []


# --- the gate over the real tree ---------------------------------------------


def test_every_claude_md_citation_resolves():
    sections, subsections = parse_headings(CLAUDE_MD.read_text(encoding="utf-8"))
    assert sections and subsections, "CLAUDE.md heading parse returned nothing"
    problems: list[str] = []
    total = 0
    for path in tracked_text_files(REPO_ROOT):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # a binary that matched a text glob
            continue
        pattern = BARE_SECTION_RE if path == CLAUDE_MD else CITATION_RE
        cites = find_citations(text, pattern)
        total += len(cites)
        for c in dangling(cites, sections, subsections):
            problems.append(f"{path.relative_to(REPO_ROOT)}: CLAUDE.md §{c}")
    assert total > 0, "no CLAUDE.md citations found at all — the scan is broken"
    assert not problems, "dangling CLAUDE.md citations:\n" + "\n".join(problems)
