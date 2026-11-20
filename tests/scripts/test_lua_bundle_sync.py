"""Tests for ``scripts/build_lua_bundle.py`` and the committed Lua bundle (GM-B0).

The Civ6 mod loads one file, ``AlphaZeroMain.lua``, generated from the module
files in modinfo order. These tests hold the committed bundle to the generator,
and hold the generator to its own contract on copies of the mod directory.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "build_lua_bundle.py"

_MARKER = re.compile(r"^-- ========== (BEGIN|END) (\S+) ==========$")
_FIX = "Edit the module files in mod/AlphaZeroAI/Scripts/ (never the bundle) and run: "
_FIX += "python scripts/build_lua_bundle.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_lua_bundle", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bb = _load()
MOD_DIR: Path = bb.DEFAULT_MOD_DIR
SCRIPTS: Path = MOD_DIR / "Scripts"
BUNDLE: Path = SCRIPTS / "AlphaZeroMain.lua"


def _norm(text: str) -> str:
    return text.replace("\r\n", "\n")


def _listed() -> list[str]:
    modinfo = next(MOD_DIR.glob("*.modinfo")).read_text(encoding="utf-8")
    return re.findall(r"<File>\s*Scripts/([^<\s]+\.lua)\s*</File>", modinfo)


def _listed_modules() -> list[str]:
    return [n for n in _listed() if n != "AlphaZeroMain.lua"]


def _digest() -> str:
    return hashlib.sha256(BUNDLE.read_bytes()).hexdigest()


@pytest.fixture(scope="module", autouse=True)
def real_bundle_untouched() -> Iterator[None]:
    """The tests mutate copies only; fail if the real bundle moved."""
    before = _digest()
    yield
    assert _digest() == before, "the test run modified the real committed bundle"


@pytest.fixture
def mod_copy(tmp_path: Path) -> Path:
    dest = tmp_path / "AlphaZeroAI"
    shutil.copytree(MOD_DIR, dest)
    return dest


def _cbundle(mod: Path) -> Path:
    return mod / "Scripts" / "AlphaZeroMain.lua"


# --------------------------------------------------------------------------
# 1. The gate
# --------------------------------------------------------------------------


def test_committed_bundle_equals_generated() -> None:
    generated = bb.build_bundle(MOD_DIR)
    committed = _norm(BUNDLE.read_bytes().decode("utf-8"))
    assert generated == committed, (
        f"AlphaZeroMain.lua is out of date ({bb.first_difference(generated, committed)}). {_FIX}"
    )


def test_check_exits_zero_on_repo(capsys: pytest.CaptureFixture[str]) -> None:
    code = bb.main(["--check"])
    err = capsys.readouterr().err
    assert code == 0, f"{err}\n{_FIX}"


# --------------------------------------------------------------------------
# 2. Structure of the committed bundle
# --------------------------------------------------------------------------


def _markers() -> list[tuple[int, str, str]]:
    lines = _norm(BUNDLE.read_bytes().decode("utf-8")).split("\n")
    return [
        (i, m.group(1), m.group(2)) for i, line in enumerate(lines) if (m := _MARKER.match(line))
    ]


def test_every_listed_module_appears_once_in_modinfo_order() -> None:
    listed = _listed_modules()
    assert listed, "modinfo lists no Lua module"
    markers = _markers()
    begins = [name for _, kind, name in markers if kind == "BEGIN"]
    ends = [name for _, kind, name in markers if kind == "END"]
    assert begins == listed
    assert ends == listed
    # Strictly BEGIN X, END X, BEGIN Y, END Y ... : matched and not nested.
    sequence = [(kind, name) for _, kind, name in markers]
    expected = [(k, n) for n in listed for k in ("BEGIN", "END")]
    assert sequence == expected


def test_only_header_before_first_begin_and_nothing_after_last_end() -> None:
    text = _norm(BUNDLE.read_bytes().decode("utf-8"))
    first = text.index("-- ========== BEGIN ")
    assert text[:first] == bb.HEADER
    last_end = text.rindex("-- ========== END ")
    assert text[last_end:].count("\n") == 1 and text.endswith("==========\n")


def test_bundle_is_not_bundled() -> None:
    assert "AlphaZeroMain.lua" not in _listed_modules()
    assert all(name != "AlphaZeroMain.lua" for _, _, name in _markers())


@pytest.mark.parametrize("name", _listed_modules())
def test_module_file_has_no_marker_line(name: str) -> None:
    text = _norm((SCRIPTS / name).read_bytes().decode("utf-8"))
    offenders = [line for line in text.split("\n") if _MARKER.match(line)]
    assert not offenders, f"{name} contains bundle marker lines: {offenders}"


# --------------------------------------------------------------------------
# 3. modinfo <-> Scripts/
# --------------------------------------------------------------------------


def test_every_lua_file_is_bundle_or_listed() -> None:
    on_disk = {p.name for p in SCRIPTS.glob("*.lua")}
    allowed = set(_listed_modules()) | {"AlphaZeroMain.lua"}
    assert on_disk <= allowed, (
        f"unlisted modules would silently not ship: {sorted(on_disk - allowed)}"
    )


def test_every_listed_file_exists() -> None:
    missing = [n for n in _listed() if not (SCRIPTS / n).is_file()]
    assert not missing, f"modinfo lists files that do not exist: {missing}"


# --------------------------------------------------------------------------
# 4. Negative controls on a copy
# --------------------------------------------------------------------------


def test_edited_module_fails_check_then_rewrite_fixes(
    mod_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = mod_copy / "Scripts" / "Config.lua"
    original = module.read_bytes()
    lines = original.decode("utf-8").split("\n")
    lines[3] = lines[3] + " -- negative control edit"
    module.write_bytes("\n".join(lines).encode("utf-8"))
    bundle = _cbundle(mod_copy)
    before = bundle.read_bytes()

    assert bb.main(["--check", "--mod-dir", str(mod_copy)]) == 1
    err = capsys.readouterr().err
    assert "out of date" in err
    # Header is 7 lines, BEGIN marker is line 8, so Config.lua line 4 is bundle line 12.
    assert re.search(r"line 12:", err), err
    assert "negative control edit" in err
    assert bundle.read_bytes() == before, "--check must not write"

    assert bb.main(["--mod-dir", str(mod_copy)]) == 0
    assert bundle.read_bytes() != before
    assert b"negative control edit" in bundle.read_bytes()
    capsys.readouterr()
    assert bb.main(["--check", "--mod-dir", str(mod_copy)]) == 0


def test_removed_module_exits_2(mod_copy: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (mod_copy / "Scripts" / "Logger.lua").unlink()
    before = _cbundle(mod_copy).read_bytes()
    for argv in (["--check"], []):
        assert bb.main([*argv, "--mod-dir", str(mod_copy)]) == 2
        assert "Logger.lua" in capsys.readouterr().err
    assert _cbundle(mod_copy).read_bytes() == before


def _modinfo(mod: Path) -> Path:
    return next(mod.glob("*.modinfo"))


def test_module_listed_twice_exits_2(mod_copy: Path, capsys: pytest.CaptureFixture[str]) -> None:
    info = _modinfo(mod_copy)
    text = info.read_text(encoding="utf-8")
    text = text.replace(
        "<File>Scripts/Logger.lua</File>",
        "<File>Scripts/Logger.lua</File>\n        <File>Scripts/Logger.lua</File>",
    )
    info.write_text(text, encoding="utf-8")
    assert bb.main(["--check", "--mod-dir", str(mod_copy)]) == 2
    assert "more than once" in capsys.readouterr().err


def test_no_modinfo_exits_2(mod_copy: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _modinfo(mod_copy).unlink()
    assert bb.main(["--check", "--mod-dir", str(mod_copy)]) == 2
    assert "modinfo" in capsys.readouterr().err


def test_two_modinfos_exit_2(mod_copy: Path, capsys: pytest.CaptureFixture[str]) -> None:
    shutil.copy(_modinfo(mod_copy), mod_copy / "Other.modinfo")
    assert bb.main(["--check", "--mod-dir", str(mod_copy)]) == 2
    assert "modinfo" in capsys.readouterr().err


def test_modinfo_listing_no_lua_module_exits_2(
    mod_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    info = _modinfo(mod_copy)
    text = info.read_text(encoding="utf-8")
    text = re.sub(r"<File>Scripts/(?!AlphaZeroMain)[^<]*\.lua</File>", "", text)
    info.write_text(text, encoding="utf-8")
    assert bb.main(["--check", "--mod-dir", str(mod_copy)]) == 2
    assert "no Lua module" in capsys.readouterr().err


def _reference_bundle_lf() -> bytes:
    """The committed bundle with LF line endings.

    A checkout with ``core.autocrlf=true`` holds it with CRLF, while the generator
    always writes LF; only the reference is normalised, so the generated bytes are
    still compared exactly (review finding on PR 148).
    """
    return BUNDLE.read_bytes().replace(b"\r\n", b"\n")


def test_missing_bundle_check_fails_then_is_created(
    mod_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _cbundle(mod_copy)
    bundle.unlink()
    assert bb.main(["--check", "--mod-dir", str(mod_copy)]) == 1
    assert "missing" in capsys.readouterr().err
    assert not bundle.exists(), "--check must not create the bundle"
    assert bb.main(["--mod-dir", str(mod_copy)]) == 0
    assert bundle.read_bytes() == _reference_bundle_lf()


# --------------------------------------------------------------------------
# 5. Line endings
# --------------------------------------------------------------------------


def _to_crlf(path: Path) -> None:
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))


def test_crlf_checkout_is_up_to_date_and_rewrite_is_lf(mod_copy: Path) -> None:
    scripts = mod_copy / "Scripts"
    for path in scripts.glob("*.lua"):
        _to_crlf(path)
    _to_crlf(_modinfo(mod_copy))
    assert b"\r\n" in _cbundle(mod_copy).read_bytes()
    assert bb.main(["--check", "--mod-dir", str(mod_copy)]) == 0

    # Force a rewrite (stale bundle), then it must be LF only.
    _cbundle(mod_copy).write_bytes(b"stale\r\n")
    assert bb.main(["--mod-dir", str(mod_copy)]) == 0
    data = _cbundle(mod_copy).read_bytes()
    assert b"\r" not in data
    assert data == _reference_bundle_lf()


# --------------------------------------------------------------------------
# 6. write_bundle atomicity
# --------------------------------------------------------------------------


def test_failed_replace_leaves_old_bundle_and_no_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "AlphaZeroMain.lua"
    target.write_bytes(b"old contents\n")

    def boom(src: object, dst: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(bb.os, "replace", boom)
    with pytest.raises(OSError, match="simulated"):
        bb.write_bundle(target, "new contents\n")
    assert target.read_bytes() == b"old contents\n"
    assert not list(tmp_path.glob("*.tmp"))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["AlphaZeroMain.lua"]


def test_write_bundle_writes_lf_bytes(tmp_path: Path) -> None:
    target = tmp_path / "b.lua"
    bb.write_bundle(target, "a\nb\n")
    assert target.read_bytes() == b"a\nb\n"
    assert not list(tmp_path.glob("*.tmp"))


# --------------------------------------------------------------------------
# 7. first_difference
# --------------------------------------------------------------------------


def test_first_difference_reports_one_based_line() -> None:
    msg = bb.first_difference("a\nb\nc", "a\nX\nc")
    assert msg.startswith("line 2:")
    assert "'b'" in msg and "'X'" in msg
    assert bb.first_difference("same\nx", "diff\nx").startswith("line 1:")


def test_first_difference_prefix_reports_line_counts() -> None:
    msg = bb.first_difference("a\nb\nc", "a\nb")
    assert "line count" in msg
    assert "generated 3" in msg and "committed 2" in msg
    msg = bb.first_difference("a", "a\nb\nc")
    assert "generated 1" in msg and "committed 3" in msg


# --------------------------------------------------------------------------
# 8. One real subprocess
# --------------------------------------------------------------------------


def test_script_check_from_repo_root_exits_zero() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/build_lua_bundle.py", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"{proc.stderr}\n{_FIX}"
    assert "up to date" in proc.stdout


# --------------------------------------------------------------------------
# Probes of input handling (each asserts the behaviour a reader would expect)
# --------------------------------------------------------------------------


def test_modinfo_entry_spacing_is_tolerated(mod_copy: Path) -> None:
    info = _modinfo(mod_copy)
    text = info.read_text(encoding="utf-8").replace(
        "<File>Scripts/Logger.lua</File>", "<File>  Scripts/Logger.lua\n   </File>"
    )
    info.write_text(text, encoding="utf-8")
    assert bb.main(["--check", "--mod-dir", str(mod_copy)]) == 0


def test_modinfo_backslash_entry_is_not_silently_dropped(
    mod_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A Windows-style entry for a real module must not make the module vanish."""
    info = _modinfo(mod_copy)
    text = info.read_text(encoding="utf-8").replace(
        "<File>Scripts/Logger.lua</File>", "<File>Scripts\\Logger.lua</File>"
    )
    info.write_text(text, encoding="utf-8")
    code = bb.main(["--check", "--mod-dir", str(mod_copy)])
    capsys.readouterr()
    # Either it is understood (bundle still complete -> 0) or it is refused (2);
    # silently generating a bundle without Logger.lua (1) means the module is dropped.
    assert code in (0, 2)


def test_modinfo_subdirectory_entry_bundles_the_file(mod_copy: Path) -> None:
    sub = mod_copy / "Scripts" / "sub"
    sub.mkdir()
    (sub / "Extra.lua").write_text("-- extra\n", encoding="utf-8")
    info = _modinfo(mod_copy)
    text = info.read_text(encoding="utf-8").replace(
        "<File>Scripts/Main.lua</File>",
        "<File>Scripts/Main.lua</File>\n        <File>Scripts/sub/Extra.lua</File>",
    )
    info.write_text(text, encoding="utf-8")
    assert bb.main(["--mod-dir", str(mod_copy)]) == 0
    assert "-- ========== BEGIN sub/Extra.lua ==========" in _cbundle(mod_copy).read_text("utf-8")


def test_module_without_trailing_newline_keeps_markers_on_own_lines(mod_copy: Path) -> None:
    module = mod_copy / "Scripts" / "Main.lua"
    module.write_bytes(module.read_bytes().rstrip(b"\n"))
    assert bb.main(["--mod-dir", str(mod_copy)]) == 0
    lines = _cbundle(mod_copy).read_text(encoding="utf-8").split("\n")
    assert "-- ========== END Main.lua ==========" in lines


def test_module_with_bom_is_not_embedded_mid_bundle(mod_copy: Path) -> None:
    module = mod_copy / "Scripts" / "Logger.lua"
    module.write_bytes(b"\xef\xbb\xbf" + module.read_bytes())
    code = bb.main(["--mod-dir", str(mod_copy)])
    if code == 0:
        # Lua 5.1 does not accept U+FEFF in the middle of a chunk.
        assert b"\xef\xbb\xbf" not in _cbundle(mod_copy).read_bytes()
    else:
        assert code == 2


def test_non_utf8_module_is_a_layout_error_not_a_traceback(
    mod_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (mod_copy / "Scripts" / "Logger.lua").write_bytes(b"-- caf\xe9\n")
    assert bb.main(["--check", "--mod-dir", str(mod_copy)]) == 2
    assert "Logger.lua" in capsys.readouterr().err


def test_non_utf8_committed_bundle_is_a_layout_error_not_a_traceback(
    mod_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _cbundle(mod_copy).write_bytes(b"\xff\xfe broken")
    assert bb.main(["--check", "--mod-dir", str(mod_copy)]) in (1, 2)
    capsys.readouterr()


def test_commented_out_modinfo_entry_is_not_bundled(mod_copy: Path) -> None:
    info = _modinfo(mod_copy)
    text = info.read_text(encoding="utf-8").replace(
        "<Files>", "<Files>\n<!-- <File>Scripts/Retired.lua</File> -->", 1
    )
    info.write_text(text, encoding="utf-8")
    code = bb.main(["--check", "--mod-dir", str(mod_copy)])
    assert code == 0, "a commented-out modinfo entry must not change the bundle"


# --------------------------------------------------------------------------
# Modinfo parsing (<Files> block, comments, entry spelling)
# --------------------------------------------------------------------------


def _edit_modinfo(mod: Path, old: str, new: str) -> None:
    info = _modinfo(mod)
    text = info.read_text(encoding="utf-8")
    assert old in text, f"fixture drift: {old!r} not in the modinfo"
    info.write_text(text.replace(old, new, 1), encoding="utf-8")


_LOGGER = "<File>Scripts/Logger.lua</File>"


def _refused(mod: Path, capsys: pytest.CaptureFixture[str], needle: str) -> None:
    before = _cbundle(mod).read_bytes()
    for argv in (["--check"], []):
        assert bb.main([*argv, "--mod-dir", str(mod)]) == 2
        assert needle in capsys.readouterr().err
    assert _cbundle(mod).read_bytes() == before, "a refused layout must not rewrite the bundle"


def test_no_files_block_exits_2(mod_copy: Path, capsys: pytest.CaptureFixture[str]) -> None:
    info = _modinfo(mod_copy)
    text = info.read_text(encoding="utf-8")
    text = re.sub(r"</?Files>", "", text)
    info.write_text(text, encoding="utf-8")
    _refused(mod_copy, capsys, "<Files>")


def test_two_files_blocks_exit_2(mod_copy: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit_modinfo(
        mod_copy, "</Files>", "</Files>\n<Files>\n<File>Scripts/Extra.lua</File>\n</Files>"
    )
    _refused(mod_copy, capsys, "<Files>")


def test_self_closing_files_block_exits_2(
    mod_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    info = _modinfo(mod_copy)
    text = re.sub(r"<Files>.*</Files>", "<Files/>", info.read_text(encoding="utf-8"), flags=re.S)
    info.write_text(text, encoding="utf-8")
    assert bb.main(["--check", "--mod-dir", str(mod_copy)]) == 2
    capsys.readouterr()


@pytest.mark.parametrize(
    "spelling",
    [
        "Scripts/Logger.LUA",
        "scripts/Logger.lua",
        "SCRIPTS/Logger.lua",
        "Scripts\\Logger.lua",
        "./Scripts/Logger.lua",
        "Logger.lua",
        "Scripts/../Scripts/Logger.lua",
        "Scripts/../x.lua",
        "Scripts/sub/../Logger.lua",
    ],
)
def test_misspelled_lua_entry_exits_2_naming_it(
    mod_copy: Path, capsys: pytest.CaptureFixture[str], spelling: str
) -> None:
    _edit_modinfo(mod_copy, _LOGGER, f"<File>{spelling}</File>")
    _refused(mod_copy, capsys, repr(spelling))


def test_non_lua_entries_are_skipped(mod_copy: Path) -> None:
    _edit_modinfo(mod_copy, _LOGGER, _LOGGER + "\n<File>Textures/icon.dds</File>")
    assert bb.main(["--check", "--mod-dir", str(mod_copy)]) == 0


def test_multiline_comment_hides_several_file_lines(mod_copy: Path) -> None:
    comment = (
        "<!-- retired:\n"
        "     <File>Scripts/Ghost.lua</File>\n"
        "     <File>Scripts\\Bad.LUA</File>\n"
        "     <File>Scripts/../x.lua</File>\n"
        "-->"
    )
    _edit_modinfo(mod_copy, "<Files>", "<Files>\n" + comment)
    assert bb.main(["--check", "--mod-dir", str(mod_copy)]) == 0


def test_comment_hiding_the_files_block_counts_as_no_block(
    mod_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _edit_modinfo(mod_copy, "<Files>", "<!-- <Files>")
    _edit_modinfo(mod_copy, "</Files>", "</Files> -->")
    assert bb.main(["--check", "--mod-dir", str(mod_copy)]) == 2
    capsys.readouterr()


def test_lua_file_outside_files_block_is_not_bundled(mod_copy: Path) -> None:
    extra = mod_copy / "Scripts" / "Outside.lua"
    extra.write_text("-- outside\n", encoding="utf-8")
    _edit_modinfo(mod_copy, "<InGameActions>", "<InGameActions>\n<File>Scripts/Outside.lua</File>")
    assert bb.main(["--check", "--mod-dir", str(mod_copy)]) == 0
    assert "Outside.lua" not in _cbundle(mod_copy).read_text(encoding="utf-8")


def test_real_modinfo_comment_does_not_claim_a_hand_maintained_bundle() -> None:
    text = next(MOD_DIR.glob("*.modinfo")).read_text(encoding="utf-8")
    assert "hand-maintained" not in text.lower()
    assert "build_lua_bundle.py" in text


def test_every_non_lua_modinfo_entry_exists() -> None:
    modinfo = next(MOD_DIR.glob("*.modinfo")).read_text(encoding="utf-8")
    entries = re.findall(
        r"<File>\s*([^<]+?)\s*</File>", re.sub(r"<!--.*?-->", "", modinfo, flags=re.S)
    )
    assert "Scripts/AlphaZeroMain.xml" in entries
    missing = [e for e in entries if not (MOD_DIR / e).is_file()]
    assert not missing, f"modinfo lists files that do not exist: {missing}"


def test_xml_wrapper_loads_the_bundle() -> None:
    xml = (SCRIPTS / "AlphaZeroMain.xml").read_text(encoding="utf-8")
    assert 'FileName="AlphaZeroMain"' in xml


def test_bom_in_modinfo_or_bundle_is_refused(
    mod_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _cbundle(mod_copy).write_bytes(b"\xef\xbb\xbf" + _cbundle(mod_copy).read_bytes())
    assert bb.main(["--check", "--mod-dir", str(mod_copy)]) in (1, 2)
    capsys.readouterr()


# Probes: attributes and oddities in the modinfo (desired behaviour: loud or correct,
# never a silently dropped module).


def test_file_entry_with_attributes_is_not_silently_dropped(
    mod_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _edit_modinfo(mod_copy, _LOGGER, '<File Platform="x">Scripts/Logger.lua</File>')
    code = bb.main(["--check", "--mod-dir", str(mod_copy)])
    capsys.readouterr()
    assert code in (0, 2), "Logger.lua vanished from the generated bundle without an error"


def test_files_block_with_attributes_is_not_silently_ignored(
    mod_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _edit_modinfo(mod_copy, "<Files>", '<Files Foo="1">')
    code = bb.main(["--check", "--mod-dir", str(mod_copy)])
    capsys.readouterr()
    assert code in (0, 2)


def test_comment_with_stray_terminator_does_not_leak_entries(mod_copy: Path) -> None:
    _edit_modinfo(
        mod_copy, "<Files>", "<Files>\n<!-- a --> <!-- b <File>Scripts/Ghost.lua</File> -->"
    )
    assert bb.main(["--check", "--mod-dir", str(mod_copy)]) == 0


def test_cdata_wrapped_entry_is_not_silently_dropped(
    mod_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _edit_modinfo(mod_copy, _LOGGER, "<File><![CDATA[Scripts/Logger.lua]]></File>")
    code = bb.main(["--check", "--mod-dir", str(mod_copy)])
    capsys.readouterr()
    assert code in (0, 2)
