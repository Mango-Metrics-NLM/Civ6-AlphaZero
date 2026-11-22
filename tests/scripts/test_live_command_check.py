"""GM-B7c-1: ``scripts/live_command_check.py`` (build_command, run_check, usage errors).

``run_check`` is driven against a small fake bridge that answers the way a
``LiveBridge`` does (``wait_for_state(require_new=...)``, ``backlog_state()``,
``next_msg_id()``, ``send_and_wait``); the states it serves are real
``GameStateData`` objects. Nothing here talks to a game, plays one, or prices one.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alpha_zero.bridge.log_tailer import AckTimeoutError, StateTimeoutError  # noqa: E402
from alpha_zero.bridge.protocol import (  # noqa: E402
    ActionType,
    CommandMessage,
    GameStateData,
    LuaCommand,
)
from scripts import live_command_check as lcc  # noqa: E402

TIMEOUT = 0.1


# --- fixtures ---


def make_state(
    *,
    turn: int = 5,
    units: list[tuple[int, int, int, int]] | None = None,
    research: str | None = "TECH_MINING",
    blocking: str | None = None,
) -> GameStateData:
    """Own units as (id, x, y, moves)."""
    unit_rows = units if units is not None else [(1, 2, 3, 2)]
    return GameStateData.from_dict(
        {
            "turn_id": turn,
            "player_id": 0,
            "map": {"width": 0, "height": 0, "tiles": []},
            "units": [
                {
                    "id": uid,
                    "unit_type": "UNIT_WARRIOR",
                    "owner": 0,
                    "x": x,
                    "y": y,
                    "movement": moves,
                    "max_movement": 2,
                }
                for uid, x, y, moves in unit_rows
            ],
            "cities": [],
            "player": {"id": 0, "civilization": "CIVILIZATION_ROME", "current_research": research},
            "end_turn_blocking": blocking,
        }
    )


OK_ACK: dict[str, Any] = {"msg_id": 77, "success": True}
REFUSED_ACK: dict[str, Any] = {
    "msg_id": 77,
    "success": False,
    "error_code": "WOULD_START_WAR",
    "error_message": "that tile holds a unit of a player we are at peace with",
}
MOVE = {"action_type": "unit_move", "unit_id": 1, "target": {"x": 3, "y": 3}}


class FakeBridge:
    """Serves one current state, then one next state; scripts the ack."""

    def __init__(
        self,
        *,
        current: GameStateData | None,
        after: GameStateData | None = None,
        ack: dict[str, Any] | None = OK_ACK,
        backlog: GameStateData | None = None,
    ) -> None:
        self._current = current
        self._after = after
        self._ack = ack
        self._backlog = backlog
        self.sent: list[CommandMessage] = []

    async def wait_for_state(
        self, timeout: float | None = None, require_new: bool = True
    ) -> GameStateData:
        state = self._after if require_new else self._current
        if state is None:
            raise StateTimeoutError(timeout or 0.0)
        return state

    def backlog_state(self) -> GameStateData | None:
        return self._backlog

    def next_msg_id(self) -> int:
        return 77

    async def send_and_wait(
        self, command: CommandMessage, timeout: float | None = None
    ) -> dict[str, Any]:
        self.sent.append(command)
        if self._ack is None:
            raise AckTimeoutError(command.msg_id, timeout or 0.0)
        return self._ack


async def _run(bridge: FakeBridge, command: dict[str, Any], **kwargs: Any) -> tuple[int, list[str]]:
    lines: list[str] = []
    code = await lcc.run_check(bridge, command, timeout=TIMEOUT, out=lines.append, **kwargs)
    return code, lines


def _text(lines: list[str]) -> str:
    return "\n".join(lines).lower()


# --- build_command ---


def _args(**overrides: Any) -> argparse.Namespace:
    """A parsed-flags namespace; both spellings of the --json attribute are set."""
    base: dict[str, Any] = {
        "action": None,
        "json": None,
        "json_command": None,
        "unit_id": None,
        "city_id": None,
        "x": None,
        "y": None,
        "item": None,
        "item_type": None,
    }
    base.update(overrides)
    if "json" in overrides:
        base["json_command"] = overrides["json"]
    return argparse.Namespace(**base)


class TestBuildCommand:
    def test_end_turn_has_only_an_action_type(self) -> None:
        assert lcc.build_command(_args(action="end_turn")) == {"action_type": "end_turn"}

    def test_unit_move(self) -> None:
        cmd = lcc.build_command(_args(action="unit_move", unit_id=5, x=3, y=4))
        assert cmd == {"action_type": "unit_move", "unit_id": 5, "target": {"x": 3, "y": 4}}

    def test_a_target_at_the_origin_is_kept(self) -> None:
        cmd = lcc.build_command(_args(action="unit_move", unit_id=5, x=0, y=0))
        assert cmd["target"] == {"x": 0, "y": 0}

    def test_city_produce_with_numeric_item(self) -> None:
        cmd = lcc.build_command(
            _args(action="city_produce", city_id=10, item="7", item_type="unit")
        )
        assert cmd == {
            "action_type": "city_produce",
            "city_id": 10,
            "item_id": 7,
            "item_type": "unit",
        }

    def test_research_by_type_name(self) -> None:
        cmd = lcc.build_command(_args(action="research_tech", item="TECH_MINING"))
        assert cmd == {"action_type": "research_tech", "item_id": "TECH_MINING"}

    def test_json_command_is_taken_as_given(self) -> None:
        raw = '{"action_type": "unit_fortify", "unit_id": 9, "metadata": {"why": "check"}}'
        assert lcc.build_command(_args(json=raw)) == {
            "action_type": "unit_fortify",
            "unit_id": 9,
            "metadata": {"why": "check"},
        }

    @pytest.mark.parametrize("action", [a.value for a in ActionType])
    def test_every_action_type_builds_a_command_the_protocol_reads(self, action: str) -> None:
        cmd = lcc.build_command(_args(action=action))
        assert LuaCommand.from_dict(cmd).action_type is ActionType(action)

    def test_the_built_command_round_trips_through_the_protocol(self) -> None:
        cmd = lcc.build_command(_args(action="unit_move", unit_id=5, x=3, y=4))
        assert LuaCommand.from_dict(cmd).to_dict() == cmd


class TestUsageErrors:
    """Usage errors are argparse errors: SystemExit(2), before any bridge is built."""

    @staticmethod
    def _exit_code(argv: list[str]) -> Any:
        with pytest.raises(SystemExit) as info:
            lcc.main(argv)
        return info.value.code

    def test_neither_action_nor_json(self) -> None:
        assert self._exit_code([]) == 2

    def test_both_action_and_json(self) -> None:
        assert (
            self._exit_code(["--action", "end_turn", "--json", '{"action_type":"end_turn"}']) == 2
        )

    def test_unknown_action(self) -> None:
        assert self._exit_code(["--action", "unit_teleport"]) == 2

    def test_json_with_an_unknown_action_type(self) -> None:
        assert self._exit_code(["--json", '{"action_type": "unit_teleport"}']) == 2

    def test_json_that_is_not_json(self) -> None:
        assert self._exit_code(["--json", "{not json"]) == 2

    def test_json_that_is_not_an_object(self) -> None:
        assert self._exit_code(["--json", "[1, 2]"]) == 2

    def test_no_game_count_flag(self) -> None:
        assert self._exit_code(["--action", "end_turn", "--num_games", "10"]) == 2


# --- run_check: exit codes ---


class TestRunCheckExitCodes:
    async def test_accepted_and_effect_seen_exits_0(self) -> None:
        bridge = FakeBridge(current=make_state(), after=make_state(units=[(1, 3, 3, 1)]))
        code, lines = await _run(bridge, MOVE)
        assert code == 0
        assert lines  # it says something

    async def test_command_is_sent_as_given(self) -> None:
        bridge = FakeBridge(current=make_state(), after=make_state(units=[(1, 3, 3, 1)]))
        await _run(bridge, MOVE)
        assert len(bridge.sent) == 1
        assert bridge.sent[0].msg_id == 77
        assert [c.to_dict() for c in bridge.sent[0].commands] == [MOVE]

    async def test_the_ack_is_printed(self) -> None:
        bridge = FakeBridge(current=make_state(), after=make_state(), ack=REFUSED_ACK)
        _, lines = await _run(bridge, MOVE)
        text = _text(lines)
        assert "would_start_war" in text
        assert "peace" in text

    async def test_the_changes_are_printed(self) -> None:
        bridge = FakeBridge(current=make_state(), after=make_state(units=[(1, 3, 3, 1)]))
        _, lines = await _run(bridge, MOVE)
        assert any("1" in line for line in lines)
        assert len(lines) >= 3  # state, answer, changes/verdict

    async def test_refused_exits_1_even_when_the_state_changed(self) -> None:
        bridge = FakeBridge(
            current=make_state(), after=make_state(units=[(1, 3, 3, 1)]), ack=REFUSED_ACK
        )
        code, _ = await _run(bridge, MOVE)
        assert code == 1

    async def test_accepted_but_nothing_changed_exits_1(self) -> None:
        bridge = FakeBridge(current=make_state(), after=make_state())
        code, _ = await _run(bridge, MOVE)
        assert code == 1

    async def test_expected_effect_false_exits_1_though_something_else_changed(self) -> None:
        bridge = FakeBridge(current=make_state(), after=make_state(research="TECH_POTTERY"))
        code, _ = await _run(bridge, MOVE)
        assert code == 1

    async def test_not_judged_command_with_a_change_exits_0(self) -> None:
        cmd = {"action_type": "unit_build", "unit_id": 1}
        bridge = FakeBridge(current=make_state(), after=make_state(units=[(1, 2, 3, 0)]))
        code, _ = await _run(bridge, cmd)
        assert code == 0

    async def test_not_judged_command_with_no_change_exits_1(self) -> None:
        cmd = {"action_type": "unit_build", "unit_id": 1}
        bridge = FakeBridge(current=make_state(), after=make_state())
        code, _ = await _run(bridge, cmd)
        assert code == 1

    async def test_end_turn_that_advances_the_turn_exits_0(self) -> None:
        bridge = FakeBridge(current=make_state(turn=5), after=make_state(turn=6))
        code, _ = await _run(bridge, {"action_type": "end_turn"})
        assert code == 0

    async def test_end_turn_that_does_not_advance_exits_1(self) -> None:
        bridge = FakeBridge(
            current=make_state(turn=5),
            after=make_state(turn=5, blocking="ENDTURN_BLOCKING_UNITS"),
        )
        code, _ = await _run(bridge, {"action_type": "end_turn"})
        assert code == 1

    async def test_no_state_exits_2_and_sends_nothing(self) -> None:
        bridge = FakeBridge(current=None)
        code, lines = await _run(bridge, MOVE)
        assert code == 2
        assert bridge.sent == []
        assert lines  # says why

    async def test_no_ack_exits_2(self) -> None:
        bridge = FakeBridge(current=make_state(), after=make_state(), ack=None)
        code, _ = await _run(bridge, MOVE)
        assert code == 2

    async def test_no_state_after_the_command_exits_2(self) -> None:
        bridge = FakeBridge(current=make_state(), after=None)
        code, _ = await _run(bridge, MOVE)
        assert code == 2


# --- run_check: the backlog fallback ---


class TestRunCheckBacklog:
    async def test_backlog_state_is_used_when_asked_and_a_warning_is_printed(self) -> None:
        bridge = FakeBridge(
            current=None, backlog=make_state(), after=make_state(units=[(1, 3, 3, 1)])
        )
        code, lines = await _run(bridge, MOVE, use_backlog=True)
        assert code == 0
        assert len(bridge.sent) == 1
        assert "backlog" in _text(lines) or "already in" in _text(lines)

    async def test_the_changes_are_measured_from_the_backlog_state(self) -> None:
        # the next state equals the backlog state: nothing changed, so a move is not seen
        bridge = FakeBridge(current=None, backlog=make_state(), after=make_state())
        code, _ = await _run(bridge, MOVE, use_backlog=True)
        assert code == 1

    async def test_backlog_is_not_used_unless_asked(self) -> None:
        bridge = FakeBridge(current=None, backlog=make_state(), after=make_state())
        code, _ = await _run(bridge, MOVE, use_backlog=False)
        assert code == 2
        assert bridge.sent == []

    async def test_use_backlog_with_no_backlog_exits_2(self) -> None:
        bridge = FakeBridge(current=None, backlog=None, after=make_state())
        code, _ = await _run(bridge, MOVE, use_backlog=True)
        assert code == 2
        assert bridge.sent == []

    async def test_a_fresh_state_wins_over_the_backlog_without_a_warning(self) -> None:
        bridge = FakeBridge(
            current=make_state(),
            backlog=make_state(turn=1),
            after=make_state(units=[(1, 3, 3, 1)]),
        )
        code, lines = await _run(bridge, MOVE, use_backlog=True)
        assert code == 0
        assert "backlog" not in _text(lines)


class TestLightweightImportsAndPaths:
    def test_build_bridge_uses_env_and_cli_overrides(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_ipc = tmp_path / "env_ipc"
        env_log = tmp_path / "env_Lua.log"
        cli_ipc = tmp_path / "cli_ipc"
        cli_log = tmp_path / "cli_Lua.log"
        monkeypatch.setenv("CIV6AZ_LOCAL_BRIDGE_IPC_DIR", str(env_ipc))
        monkeypatch.setenv("CIV6AZ_LOCAL_LOG_PATH", str(env_log))
        args = argparse.Namespace(ipc_dir=None, log_path=None)
        bridge_env = lcc._build_bridge(args)
        assert bridge_env._config.comm_dir == env_ipc
        assert bridge_env._tailer._log_path == env_log

        args_cli = argparse.Namespace(ipc_dir=str(cli_ipc), log_path=str(cli_log))
        bridge_cli = lcc._build_bridge(args_cli)
        assert bridge_cli._config.comm_dir == cli_ipc
        assert bridge_cli._tailer._log_path == cli_log

    def test_build_bridge_does_not_import_torch_or_numpy(self, tmp_path: Path) -> None:
        import subprocess

        code = (
            "import argparse, sys\n"
            "sys.path.insert(0, '.')\n"
            "import scripts.live_command_check as lcc\n"
            "args = argparse.Namespace(ipc_dir=sys.argv[1], log_path=sys.argv[2])\n"
            "lcc._build_bridge(args)\n"
            "heavy = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'numpy'))\n"
            "print(heavy)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code, str(tmp_path / "ipc"), str(tmp_path / "Lua.log")],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "[]"

    def test_build_bridge_uses_single_source_transport_env_vars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CIV6_BRIDGE_TRANSPORT", "socket")
        monkeypatch.setenv("CIV6_BRIDGE_TUNER_HOST", "127.0.0.5")
        monkeypatch.setenv("CIV6_BRIDGE_TUNER_PORT", "4321")
        args = argparse.Namespace(ipc_dir=str(tmp_path / "ipc"), log_path=str(tmp_path / "Lua.log"))
        bridge = lcc._build_bridge(args)
        assert bridge._config.command_transport == "socket"
        assert bridge._config.tuner_host == "127.0.0.5"
        assert bridge._config.tuner_port == 4321

    def test_verbose_flag_parses_and_emits_stderr_debug_logs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        parser = lcc.create_parser()
        assert parser.parse_args(["--action", "end_turn", "--verbose"]).verbose is True
        log_file = tmp_path / "Lua.log"
        log_file.write_text("", encoding="utf-8")
        monkeypatch.setenv("CIV6AZ_LOCAL_BRIDGE_IPC_DIR", str(tmp_path / "ipc"))
        monkeypatch.setenv("CIV6AZ_LOCAL_LOG_PATH", str(log_file))
        with caplog.at_level(logging.DEBUG):
            logging.getLogger().setLevel(logging.WARNING)
            rc = lcc.main(["--action", "end_turn", "--timeout", "0.05", "--verbose"])
        captured = capsys.readouterr()
        assert rc == lcc.EXIT_NO_CONTACT
        assert any("Configured LiveBridge:" in rec.message for rec in caplog.records)
        assert any("Waiting for initial game state" in rec.message for rec in caplog.records)
        assert "no state from the game within" in captured.out
