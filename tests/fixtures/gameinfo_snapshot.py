"""Shared helpers for the GM-B4 GameInfo snapshot tests. Not collected as tests.

``STUB_DOCUMENT`` is what the invented game (``tests/fixtures/lua/civ6_stub.lua``)
must export, written out by hand so the Lua tests compare against an independent
expectation rather than against the exporter's own output. Nothing here is real
game data: every name and number is invented. It is Python source, never a data
file, so the repository's "no GameInfo-shaped file is tracked" hygiene test has
nothing to object to.
"""

from __future__ import annotations

import json
import re
import zlib
from typing import Any

from tests.fixtures import lua_harness as h

MARKER = "AZ_GAMEINFO"
MAX_PIECE = 1800
TABLES = (
    "Technologies",
    "TechnologyPrereqs",
    "Civics",
    "CivicPrereqs",
    "Buildings",
    "Units",
)

STUB_DOCUMENT: dict[str, Any] = {
    "Technologies": [
        {"TechnologyType": "TECH_POTTERY", "Cost": 25, "EraType": "ERA_ANCIENT"},
        {"TechnologyType": "TECH_MINING", "Cost": 50, "EraType": "ERA_ANCIENT"},
        {"TechnologyType": "TECH_WRITING", "Cost": 60, "EraType": "ERA_ANCIENT"},
        {"TechnologyType": "TECH_BRONZE_WORKING", "Cost": 120, "EraType": "ERA_CLASSICAL"},
    ],
    "TechnologyPrereqs": [{"Technology": "TECH_BRONZE_WORKING", "PrereqTech": "TECH_MINING"}],
    "Civics": [
        {"CivicType": "CIVIC_CODE_OF_LAWS", "Cost": 20, "EraType": "ERA_ANCIENT"},
        {"CivicType": "CIVIC_CRAFTSMANSHIP", "Cost": 40, "EraType": "ERA_ANCIENT"},
        {"CivicType": "CIVIC_FOREIGN_TRADE", "Cost": 40, "EraType": "ERA_ANCIENT"},
        {"CivicType": "CIVIC_EARLY_EMPIRE", "Cost": 70, "EraType": "ERA_ANCIENT"},
    ],
    "CivicPrereqs": [
        {"Civic": "CIVIC_CRAFTSMANSHIP", "PrereqCivic": "CIVIC_CODE_OF_LAWS"},
        {"Civic": "CIVIC_FOREIGN_TRADE", "PrereqCivic": "CIVIC_CODE_OF_LAWS"},
        {"Civic": "CIVIC_EARLY_EMPIRE", "PrereqCivic": "CIVIC_FOREIGN_TRADE"},
    ],
    "Buildings": [
        {"BuildingType": "BUILDING_MONUMENT", "Cost": 60},
        {"BuildingType": "BUILDING_GRANARY", "Cost": 65, "PrereqTech": "TECH_POTTERY"},
    ],
    "Units": [
        {"UnitType": "UNIT_SETTLER", "Cost": 80, "Combat": 0, "BaseMoves": 2},
        {"UnitType": "UNIT_WARRIOR", "Cost": 40, "Combat": 20, "BaseMoves": 2},
        {
            "UnitType": "UNIT_ARCHER",
            "Cost": 60,
            "Combat": 15,
            "BaseMoves": 2,
            "PrereqTech": "TECH_BRONZE_WORKING",
        },
    ],
}

#: Lua run BEFORE the bundle loads: ``print`` records into ``Civ6Stub.printed``.
PRINT_CAPTURE = (
    "Civ6Stub.printed = {}\n"
    "print = function(...)\n"
    "  local parts = {}\n"
    "  for i = 1, select('#', ...) do parts[#parts + 1] = tostring((select(i, ...))) end\n"
    "  table.insert(Civ6Stub.printed, table.concat(parts, ' '))\n"
    "end\n"
)

#: Replace ``GameInfo.<table>`` with an iterator over ``rows`` (a Lua list expression).
_REPLACE = (
    "GameInfo.{table} = function() local i = 0 return function() i = i + 1 return list[i] end end"
)


def replace_table(table: str, rows_lua: str) -> str:
    """Lua setup: ``GameInfo.<table>`` iterates the Lua list expression ``rows_lua``."""
    return f"do local list = {rows_lua}\n{_REPLACE.format(table=table)} end\n"


def extra_units(count: int) -> str:
    """Lua setup: ``count`` more valid unit rows after the stub's three."""
    return (
        "do local list = {}\n"
        "for row in GameInfo.Units() do list[#list + 1] = row end\n"
        f"for n = 1, {count} do\n"
        "  list[#list + 1] = { UnitType = 'UNIT_EXTRA_' .. n, Cost = 50, Combat = 10, BaseMoves = 2 }\n"
        "end\n"
        f"{_REPLACE.format(table='Units')} end\n"
    )


def padded_units(pad: int) -> str:
    """Lua setup: one more unit row whose type name is ``pad`` characters (an identifier).

    The padding is ``z``: the mod's encoder escapes every character from ``0`` to ``x``
    as ``\\u00XX`` (a Lua 5.1 pattern written with ``\\x`` escapes), so ``z`` is one of
    the few characters that adds exactly one character of JSON.
    """
    return (
        "do local list = {}\n"
        "for row in GameInfo.Units() do list[#list + 1] = row end\n"
        f"list[#list + 1] = {{ UnitType = 'U' .. string.rep('z', {pad - 1}) }}\n"
        f"{_REPLACE.format(table='Units')} end\n"
    )


#: A game small enough that its snapshot fits one 1800-character line even through the
#: mod's escaping encoder (the stub's full snapshot does not).
SMALL_SETUP = replace_table(
    "Technologies",
    "{ { TechnologyType = 'TECH_A', Cost = 1, EraType = 'ERA_ANCIENT' } }",
) + "".join(
    replace_table(t, "{}")
    for t in ("TechnologyPrereqs", "Civics", "CivicPrereqs", "Buildings", "Units")
)
SMALL_DOCUMENT: dict[str, Any] = {
    "Technologies": [{"TechnologyType": "TECH_A", "Cost": 1, "EraType": "ERA_ANCIENT"}],
    "TechnologyPrereqs": [],
    "Civics": [],
    "CivicPrereqs": [],
    "Buildings": [],
    "Units": [],
}


def extra_unit_documents(count: int) -> list[dict[str, Any]]:
    return [
        {"UnitType": f"UNIT_EXTRA_{n}", "Cost": 50, "Combat": 10, "BaseMoves": 2}
        for n in range(1, count + 1)
    ]


def load_printing(setup: str = "") -> Any:
    """The mod loaded on the stub with ``print`` captured from before the bundle runs."""
    return h.load_mod(setup=PRINT_CAPTURE + setup)


def clear_printed(runtime: Any) -> None:
    runtime.execute("Civ6Stub.printed = {}")


def printed(runtime: Any) -> list[str]:
    """Everything ``print`` was called with since the capture started, one entry per call."""
    count = runtime.eval("#Civ6Stub.printed")
    return [runtime.eval(f"Civ6Stub.printed[{i}]") for i in range(1, count + 1)]


def lua_export(runtime: Any) -> tuple[Any, list[str], str]:
    """``AlphaZeroGameInfo.Export()`` through the mod's encoder: ``(document, failed, text)``."""
    text, failed = runtime.execute(
        "local d, f = AlphaZeroGameInfo.Export() "
        "return AlphaZeroJson.Encode(d), table.concat(f, ',')"
    )
    assert isinstance(text, str), "the mod's encoder could not encode the snapshot"
    return json.loads(text), [n for n in str(failed).split(",") if n], text


def messages(lines: list[str], marker: str = MARKER) -> list[tuple[str, list[int]]]:
    """The complete messages among ``lines``, in order: ``(json_text, piece_lengths)``.

    The line format is checked as it is read: chunk numbers run 1..n in order and the
    END line carries n. An incomplete trailing chunk set is not returned.
    """
    found: list[tuple[str, list[int]]] = []
    pieces: list[tuple[int, int, str]] = []
    for line in lines:
        if line.startswith(f"{marker}:"):
            text = line[len(marker) + 1 :]
            found.append((text, [len(text)]))
        elif line.startswith(f"{marker}_CHUNK:"):
            match = re.fullmatch(rf"{marker}_CHUNK:(\d+)/(\d+):(.*)", line, re.DOTALL)
            assert match, f"malformed chunk line {line[:80]!r}"
            pieces.append((int(match[1]), int(match[2]), match[3]))
        elif line.startswith(f"{marker}_END:"):
            # GameInfo: ``<marker>_END:<n>:<adler32>`` (a plain decimal count and checksum).
            # Other markers (states) keep ``<marker>_END:<n>``.
            checked = marker == MARKER
            pattern = rf"{marker}_END:(\d+):(\d+)" if checked else rf"{marker}_END:(\d+)"
            end = re.fullmatch(pattern, line)
            assert end, f"malformed END line {line[:80]!r}"
            count = int(end[1])
            assert [p[0] for p in pieces] == list(range(1, count + 1)), "chunks out of order"
            assert {p[1] for p in pieces} == {count}, "chunk totals disagree with END"
            joined = "".join(p[2] for p in pieces)
            if checked:
                assert int(end[2]) == adler32(joined), (
                    "END checksum is not the Adler-32 of the text"
                )
            found.append((joined, [len(p[2]) for p in pieces]))
            pieces = []
    return found


def log_text(lines: list[str], prefix: str = "") -> str:
    """The lines as a ``Lua.log`` text, each carrying the game's context prefix."""
    return "".join(f"{prefix}{line}\n" for line in lines)


#: The smallest document the capture accepts: all six tables present (some empty).
MINIMAL_DOCUMENT: dict[str, Any] = {
    "Technologies": [
        {"TechnologyType": "TECH_A", "Cost": 10, "EraType": "ERA_ANCIENT"},
        {"TechnologyType": "TECH_B", "Cost": 20, "EraType": "ERA_ANCIENT"},
    ],
    "TechnologyPrereqs": [{"Technology": "TECH_B", "PrereqTech": "TECH_A"}],
    "Civics": [{"CivicType": "CIVIC_X", "Cost": 10, "EraType": "ERA_ANCIENT"}],
    "CivicPrereqs": [],
    "Buildings": [],
    "Units": [],
}


def single_line(document: dict[str, Any], prefix: str = "") -> str:
    return f"{prefix}AZ_GAMEINFO:{json.dumps(document, separators=(',', ':'))}\n"


def adler32(text: str) -> int:
    """The END line's checksum: Adler-32 of the JSON text's UTF-8 bytes."""
    return zlib.adler32(text.encode("utf-8"))


def compact_text(document: dict[str, Any]) -> str:
    """The JSON text ``chunk_lines`` / ``single_line`` cut and print."""
    return json.dumps(document, separators=(",", ":"))


def end_line(total: int, checksum: int | str | None, prefix: str = "") -> str:
    """``AZ_GAMEINFO_END:<n>:<checksum>``; ``checksum=None`` leaves the checksum off."""
    tail = "" if checksum is None else f":{checksum}"
    return f"{prefix}AZ_GAMEINFO_END:{total}{tail}\n"


def begin_line(prefix: str = "") -> str:
    """The line the mod prints when an export attempt starts."""
    return f"{prefix}AZ_GAMEINFO_BEGIN\n"


def chunk_lines(
    document: dict[str, Any],
    pieces: int = 3,
    *,
    header_total: int | None = None,
    end_total: int | None = None,
    checksum: int | str | None = "auto",
) -> list[str]:
    """A chunk set of ``pieces`` pieces; the totals in the headers / END line can be wrong.

    The END line carries the Adler-32 of the whole text unless ``checksum`` says otherwise
    (``None`` omits it, anything else is printed as given).
    """
    text = compact_text(document)
    size = -(-len(text) // pieces)
    parts = [text[i : i + size] for i in range(0, len(text), size)]
    shown = header_total if header_total is not None else len(parts)
    lines = [f"AZ_GAMEINFO_CHUNK:{i}/{shown}:{p}\n" for i, p in enumerate(parts, 1)]
    total = end_total if end_total is not None else len(parts)
    return [*lines, end_line(total, adler32(text) if checksum == "auto" else checksum)]
