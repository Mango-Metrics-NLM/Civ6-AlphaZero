"""No tracked file is a real GameInfo export: the tables are the user's game data (GM-K2).

The importer refuses to write into the checkout; this holds the other half, that
nothing of the kind was committed by hand. Two things are looked for, by
content and in every tracked data file whatever its name (``.json``, ``.JSON``,
``.txt``, ``.yaml`` ...): a raw export (a JSON object with a ``Technologies``
key) and the importer's own output (it starts with a fixed header line). The
synthetic fixture is the only file allowed.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ALLOWED = frozenset({"tests/fixtures/gameinfo/synthetic_export.json"})
MARKER = "Technologies"
#: First line the importer writes; a committed file starting with it is imported game data.
OUTPUT_HEADER = b"# Imported from a GameInfo export"
#: Suffixes (lower-cased) of files that can hold an export or the importer's output.
DATA_SUFFIXES = frozenset({".json", ".jsonl", ".txt", ".yaml", ".yml", ".dat", ".export", ""})


def looks_like_gameinfo_export(path: Path) -> bool:
    """A JSON object with a ``Technologies`` key; unparsable text is judged by its bytes."""
    try:
        raw = path.read_bytes()
    except OSError:
        return False
    if raw.lstrip().startswith(OUTPUT_HEADER):
        return True
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


def tracked_data_files() -> list[str]:
    """Every tracked file whose suffix could hold game data, compared without case."""
    done = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        timeout=60,
    )
    names = [item for item in done.stdout.decode("utf-8").split("\0") if item]
    return [name for name in names if Path(name).suffix.lower() in DATA_SUFFIXES]


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


def test_no_tracked_data_file_is_game_data_whatever_its_name() -> None:
    """Review finding on PR 147: a ``.json``-only check misses ``snapshot.txt``,
    ``snapshot.JSON`` and the importer's YAML output copied into the checkout."""
    tracked = tracked_data_files()
    assert len(tracked) > len(tracked_json_files()), "the wider listing found nothing extra"
    found = offenders(REPO_ROOT, tracked, ALLOWED)
    assert not found, f"game data must not be committed: {found}"


def test_the_detector_flags_exports_under_other_names_and_the_importers_output(
    tmp_path: Path,
) -> None:
    export = '{"Technologies": [], "TechnologyPrereqs": []}'
    (tmp_path / "snapshot.txt").write_text(export, encoding="utf-8")
    (tmp_path / "snapshot.JSON").write_text(export, encoding="utf-8")
    (tmp_path / "noext").write_text(export, encoding="utf-8")
    generated = OUTPUT_HEADER.decode() + " by `python -m alpha_zero.rules import-gameinfo`.\n"
    (tmp_path / "real.yaml").write_text(generated + "schema_version: 1\n", encoding="utf-8")
    (tmp_path / "plain.yaml").write_text("schema_version: 1\nid: x\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("nothing here\n", encoding="utf-8")
    names = ["snapshot.txt", "snapshot.JSON", "noext", "real.yaml", "plain.yaml", "notes.txt"]
    assert offenders(tmp_path, names, frozenset()) == [
        "noext",
        "real.yaml",
        "snapshot.JSON",
        "snapshot.txt",
    ]


def test_the_header_constant_is_what_the_importer_writes() -> None:
    """If the importer's first line changes, this detector must change with it."""
    from alpha_zero.rules.gameinfo import convert

    text = convert({"Technologies": [], "TechnologyPrereqs": []}).text
    assert text.encode("utf-8").startswith(OUTPUT_HEADER)
