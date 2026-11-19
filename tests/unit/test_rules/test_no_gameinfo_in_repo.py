"""No tracked file is a real GameInfo export: the tables are the user's game data (GM-K2).

The importer refuses to write into the checkout; this holds the other half, that
nothing of the kind was committed by hand. A GameInfo export is a JSON object
with a ``Technologies`` key. The synthetic fixture is the only one allowed.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ALLOWED = frozenset({"tests/fixtures/gameinfo/synthetic_export.json"})
MARKER = "Technologies"


def looks_like_gameinfo_export(path: Path) -> bool:
    """A JSON object with a ``Technologies`` key; unparsable text is judged by its bytes."""
    try:
        raw = path.read_bytes()
    except OSError:
        return False
    try:
        document = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return f'"{MARKER}"'.encode() in raw
    return isinstance(document, dict) and MARKER in document


def offenders(root: Path, relative_paths: Iterable[str], allowed: frozenset[str]) -> list[str]:
    return sorted(
        rel
        for rel in relative_paths
        if rel not in allowed and looks_like_gameinfo_export(root / rel)
    )


def tracked_json_files() -> list[str]:
    # No `skip` here: without git the check cannot run, and a silent skip would read as a pass.
    done = subprocess.run(
        ["git", "ls-files", "-z", "*.json"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        timeout=60,
    )
    return [item for item in done.stdout.decode("utf-8").split("\0") if item]


def test_git_lists_tracked_json_files() -> None:
    tracked = tracked_json_files()
    assert tracked, "git ls-files returned nothing: the hygiene check would be vacuous"
    assert ".claude/settings.json" in tracked


def test_the_allowlisted_fixture_is_itself_a_detectable_export() -> None:
    fixture = REPO_ROOT / next(iter(ALLOWED))
    assert looks_like_gameinfo_export(fixture)
    assert offenders(REPO_ROOT, [next(iter(ALLOWED))], frozenset()) == [next(iter(ALLOWED))]


def test_no_tracked_json_file_is_a_gameinfo_export() -> None:
    found = offenders(REPO_ROOT, tracked_json_files(), ALLOWED)
    assert not found, f"GameInfo exports are game data and must not be committed: {found}"


def test_the_detector_flags_a_planted_export(tmp_path: Path) -> None:
    (tmp_path / "real.json").write_text('{"Technologies": [], "Units": []}', encoding="utf-8")
    (tmp_path / "other.json").write_text('{"Units": []}', encoding="utf-8")
    (tmp_path / "list.json").write_text('[{"Technologies": 1}]', encoding="utf-8")
    (tmp_path / "jsonc.json").write_text('// c\n{"Technologies": []}', encoding="utf-8")
    names = ["real.json", "other.json", "list.json", "jsonc.json"]
    assert offenders(tmp_path, names, frozenset()) == ["jsonc.json", "real.json"]
    assert offenders(tmp_path, names, frozenset({"real.json"})) == ["jsonc.json"]
