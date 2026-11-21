"""The mod's JSON encoder writes strings a JSON reader gets back unchanged.

Its control-character pass used the pattern ``[\\x00-\\x1f]``. Lua 5.1 has no
``\\x`` escape, so that was the class ``x``, ``0`` to ``x``, ``1``, ``f``: every
digit, upper-case letter and underscore, and most lower-case letters, were
written as ``\\u00XX`` (valid JSON, six times as long), and the backslash that
the pass before it had just written for a quote or a newline was escaped
again, which turned ``\\n`` into the two characters backslash and n and a
quote into invalid JSON.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.fixtures import lua_harness as h


@pytest.fixture(scope="module")
def runtime() -> Any:
    return h.load_mod()


def encode_bytes(runtime: Any, text: str) -> str:
    """The mod's encoding of ``text``, handed to Lua byte by byte."""
    data = text.encode("utf-8")
    source = "string.char(" + ", ".join(str(b) for b in data) + ")" if data else '""'
    result = runtime.eval(f"AlphaZeroJson.Encode({source})")
    encoded = result[0] if isinstance(result, tuple) else result
    assert isinstance(encoded, str)
    return encoded


PLAIN = "TECH_Mining_12 xyz ABC-def.ghi:[1,2]{k}"


class TestPlainText:
    def test_plain_ascii_is_written_as_it_is(self, runtime: Any) -> None:
        assert encode_bytes(runtime, PLAIN) == f'"{PLAIN}"'

    @pytest.mark.parametrize("code", [c for c in range(32, 127) if chr(c) not in '"\\'])
    def test_every_printable_ascii_character_is_literal(self, runtime: Any, code: int) -> None:
        assert encode_bytes(runtime, chr(code)) == f'"{chr(code)}"'

    def test_an_exported_state_has_no_unicode_escapes(self, runtime: Any) -> None:
        encoded = runtime.eval("AlphaZeroJson.Encode(AlphaZeroStateExporter.ExportState(0))")
        text = encoded[0] if isinstance(encoded, tuple) else encoded
        assert "\\u00" not in text
        assert '"TECH_POTTERY"' in text


class TestRoundTrip:
    @pytest.mark.parametrize(
        "text",
        [
            'say "hello"',
            "back\\slash",
            "line one\nline two",
            "tab\there",
            "carriage\rreturn",
            "bell\x07 and escape\x1b",
            "nul\x00inside",
            "delete\x7f",
            'all of it: "\\\n\r\t\b\f\x01\x1f',
            "",
            PLAIN,
            "café 中",
        ],
    )
    def test_a_json_reader_gets_the_string_back(self, runtime: Any, text: str) -> None:
        assert json.loads(encode_bytes(runtime, text)) == text

    @pytest.mark.parametrize("text", ['say "hello"', "line one\nline two", "back\\slash", PLAIN])
    def test_the_mods_own_decoder_gets_it_back(self, runtime: Any, text: str) -> None:
        data = text.encode("utf-8")
        source = "string.char(" + ", ".join(str(b) for b in data) + ")"
        assert runtime.eval(f"AlphaZeroJson.Decode(AlphaZeroJson.Encode({source})) == {source}")

    @pytest.mark.parametrize("code", list(range(0, 32)))
    def test_every_control_character_is_escaped(self, runtime: Any, code: int) -> None:
        encoded = encode_bytes(runtime, chr(code))
        assert chr(code) not in encoded
        assert json.loads(encoded) == chr(code)

    def test_a_message_with_a_quote_does_not_break_the_state(self, runtime: Any) -> None:
        # The end-turn blocker's message is free text from the game.
        runtime.execute(
            "Civ6Stub.notifications[0] = {{ id = 1, blocking = 3,"
            " message = 'Choose \"Mining\"\\nor another', dismissed = false }}"
        )
        try:
            state = h.export_state(runtime)
        finally:
            runtime.execute("Civ6Stub.notifications[0] = {}")
        assert state["end_turn_blocking_message"] == 'Choose "Mining"\nor another'


class TestControlCharactersAreSpelledOut:
    """The escaped set is the byte range JSON names, not the C runtime's idea of "control".

    Lua's ``%c`` class asks the C library's ``iscntrl``. Under the Windows C runtime that is
    true for some bytes at and above 0x80, so the continuation bytes of a UTF-8 character
    were rewritten as ``\\u00XX`` and a non-ASCII string did not come back (seen on
    Windows with ``"café 中"``; on Linux the same test passed). The range is now written
    out, so the result cannot depend on the platform.
    """

    def test_the_source_does_not_use_the_c_librarys_control_class(self) -> None:
        source = (h.REPO_ROOT / "mod/AlphaZeroAI/Scripts/JsonEncoder.lua").read_text("utf-8")
        gsub_lines = [line for line in source.splitlines() if "string.gsub(str" in line]
        assert gsub_lines, "premise: the encoder escapes with one string.gsub call"
        assert all("%c" not in line for line in gsub_lines), gsub_lines

    @pytest.mark.parametrize("byte", [*range(0x80, 0x100)])
    def test_every_byte_from_0x80_up_passes_through_unchanged(
        self, runtime: Any, byte: int
    ) -> None:
        # Compared inside Lua: a lone high byte is not valid UTF-8 and cannot cross to Python.
        same = runtime.eval(
            f"(AlphaZeroJson.Encode(string.char({byte}))) == string.char(34, {byte}, 34)"
        )
        assert same is True

    @pytest.mark.parametrize("byte", [*range(0, 0x20), 0x7F])
    def test_every_control_byte_is_escaped(self, runtime: Any, byte: int) -> None:
        result = runtime.eval(f"AlphaZeroJson.Encode(string.char({byte}))")
        encoded = result[0] if isinstance(result, tuple) else result
        assert isinstance(encoded, str) and encoded.startswith('"\\')
        assert json.loads(encoded) == chr(byte)
