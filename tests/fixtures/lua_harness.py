"""Run the AlphaZeroAI mod's Lua outside the game (GM-B1).

``load_mod()`` gives a Lua 5.1 runtime (the version Civ6 scripts target) with a
hand-written stand-in for the game's globals (``tests/fixtures/lua/civ6_stub.lua``)
and the mod's generated bundle loaded on top, exactly as the game loads it.

The runtime is locked down before the mod runs. The mod uses ``io.open``,
``os.execute``, ``os.remove``, ``os.rename`` and ``dofile`` when it finds them,
and with them it would create directories and files in the working directory.
In the game the Lua sandbox has no ``io`` library, so removing them here is the
game's own situation, not a test convenience. Whether the game also lacks the
other names removed here was not verified against a running game. lupa's
``python`` table is removed too, so Lua cannot reach Python.

``lupa`` is an optional extra (``pip install -e '.[lua]'``). Tests skip without
it, except where ``CIV6AZ_REQUIRE_LUA=1`` is set (CI's test job sets it): there a
missing ``lupa`` is a failure, because a skip is not a pass.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = REPO_ROOT / "mod" / "AlphaZeroAI" / "Scripts" / "AlphaZeroMain.lua"
STUB = Path(__file__).resolve().parent / "lua" / "civ6_stub.lua"
#: Unit, city and combat requests for the stand-in game, loaded right after it.
STUB_COMMANDS = Path(__file__).resolve().parent / "lua" / "civ6_stub_commands.lua"
REQUIRE_ENV = "CIV6AZ_REQUIRE_LUA"

#: Run before anything else: take away what the game's sandbox does not have, and
#: everything that could touch the file system or load code. Lua 5.1's ``module``
#: goes too: ``module('os')`` hands back the original library from the
#: loaded-module registry even after the global is gone, so the registry entries
#: are cleared first and then ``module`` itself.
_LOCKDOWN = """
if package and package.loaded then
    for _, name in ipairs({"io", "os", "debug", "package"}) do
        package.loaded[name] = nil
    end
end
module = nil
io = nil
dofile = nil
loadfile = nil
require = nil
package = nil
debug = nil
python = nil
os = { time = os.time, clock = os.clock, date = os.date }
"""


def lua_required() -> bool:
    """Whether a missing ``lupa`` must fail instead of skip (CI sets the flag)."""
    return os.environ.get(REQUIRE_ENV, "").strip() == "1"


def import_lua51() -> Any:
    """``lupa.lua51``, or skip; under ``CIV6AZ_REQUIRE_LUA=1`` a missing ``lupa`` fails."""
    try:
        from lupa import lua51
    except ImportError as exc:
        if lua_required():
            pytest.fail(
                f"{REQUIRE_ENV}=1 but lupa is not importable ({exc}); "
                "install the 'lua' extra: pip install -e '.[lua]'"
            )
        pytest.skip("lupa is not installed (pip install -e '.[lua]')")
    return lua51


def new_runtime() -> Any:
    """A locked-down Lua 5.1 runtime with no access to Python or the file system."""
    lua51 = import_lua51()
    runtime = lua51.LuaRuntime(
        unpack_returned_tuples=True, register_eval=False, register_builtins=False
    )
    runtime.execute(_LOCKDOWN)
    return runtime


def load_mod(*, stub: bool = True, setup: str | None = None) -> Any:
    """A runtime with the game stub (optional), ``setup`` Lua, then the mod's bundle.

    ``setup`` runs after the stub and before the bundle: use it to change the
    invented game for one test (remove a global, replace a method).
    """
    runtime = new_runtime()
    if stub:
        runtime.execute(STUB.read_text(encoding="utf-8"))
        runtime.execute(STUB_COMMANDS.read_text(encoding="utf-8"))
    if setup:
        runtime.execute(setup)
    runtime.execute(BUNDLE.read_text(encoding="utf-8"))
    return runtime


def export_state(runtime: Any, player_id: int = 0) -> dict[str, Any]:
    """The mod's exported state for ``player_id``, as Python data.

    The state goes through the mod's own JSON encoder, as it does in the game,
    so what comes back is what the Python side would actually receive.
    """
    result = runtime.eval(
        f"AlphaZeroJson.Encode(AlphaZeroStateExporter.ExportState({int(player_id)}))"
    )
    # Encode returns the text, or nil and an error message.
    encoded, error = result if isinstance(result, tuple) else (result, None)
    if not isinstance(encoded, str):
        raise AssertionError(f"the mod's JSON encoder failed on its own state: {error}")
    decoded: dict[str, Any] = json.loads(encoded)
    return decoded


def execute(runtime: Any, command: dict[str, Any], player_id: int = 0) -> tuple[Any, Any, Any]:
    """Run one command through the mod's executor: ``(success, error_code, message)``."""
    decode = runtime.eval("AlphaZeroJson.Decode")
    run = runtime.eval("AlphaZeroActionExecutor.Execute")
    decoded = decode(json.dumps(command))
    table = decoded[0] if isinstance(decoded, tuple) else decoded
    result = run(int(player_id), table)
    success, error_code, message = (*result, None, None)[:3]
    return success, error_code, message


def stub_calls(runtime: Any) -> list[str]:
    """Names of the game-changing calls the mod made on the stub, in order."""
    count = runtime.eval("#Civ6Stub.calls")
    return [runtime.eval(f"Civ6Stub.calls[{i}].name") for i in range(1, count + 1)]
