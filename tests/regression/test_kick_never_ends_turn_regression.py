"""The kick's Lua can never end the turn (contract PR 1). Pure Python, no lupa."""

from __future__ import annotations

from alpha_zero.bridge import tuner_socket_bridge as bridge_module

FORBIDDEN = ("ACTION_ENDTURN", "RequestAction", "EndTurn", "ENDTURN")


def test_the_rearm_lua_constant_exists_and_writes_the_flag() -> None:
    lua = bridge_module.REARM_LUA
    assert isinstance(lua, str)
    assert "ExposedMembers.AlphaZeroRearm" in lua
    assert "{nonce}" in lua


def test_the_rearm_lua_cannot_end_the_turn() -> None:
    lua = bridge_module.REARM_LUA
    for word in FORBIDDEN:
        assert word not in lua, word


def test_no_lua_constant_in_the_module_ends_the_turn() -> None:
    kicks = {
        name: value
        for name, value in vars(bridge_module).items()
        if isinstance(value, str) and "AlphaZeroRearm" in value
    }
    assert kicks, "no constant carries the re-arm flag"
    for name, value in kicks.items():
        for word in FORBIDDEN:
            assert word not in value, f"{name} contains {word}"


def test_the_mods_rearm_line_is_not_read_as_a_state_ack_or_heartbeat() -> None:
    # Review L6: the mod prints ``AZ_REARM:<nonce>:accepted|ignored:<reason>``; no prefix the
    # log tailer acts on may start it, and it may start none of them.
    from alpha_zero.bridge import log_tailer

    rearm = "AZ_REARM:"
    prefixes = [
        value
        for name, value in vars(log_tailer).items()
        if name.endswith("_PREFIX") and isinstance(value, str)
    ]
    assert prefixes, "the tailer's prefixes moved"
    for prefix in prefixes:
        assert not rearm.startswith(prefix), prefix
        assert not prefix.startswith(rearm), prefix
