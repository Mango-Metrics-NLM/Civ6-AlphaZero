#!/usr/bin/env python
"""Send ONE command to a running Civilization VI game and say what changed.

For the maintainer's live checks (``docs/LIVE_CIV6.md``): each checklist item is
one command and a few lines of output. The tool connects the way
``python -m alpha_zero.civ6_local --mode civ6`` does, takes the state the mod
last exported, sends the command, waits for the next state and prints the
mod's answer and every difference between the two states.

    python scripts/live_command_check.py --action unit_move --unit-id 65536 --x 10 --y 7
    python scripts/live_command_check.py --action city_produce --city-id 0 \\
        --item UNIT_WARRIOR --item-type unit
    python scripts/live_command_check.py --json '{"action_type": "end_turn"}'

Exit codes: 0 the command was accepted and the next state shows its effect;
1 it was refused, or accepted with nothing to show for it; 2 no state or no
answer arrived (or the arguments were wrong).

It plays no game and chooses nothing by itself.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from alpha_zero.bridge.protocol import ActionType, CommandMessage, LuaCommand  # noqa: E402
from alpha_zero.bridge.state_diff import changes, expected_effect  # noqa: E402

logger = logging.getLogger("live_command_check")

EXIT_EFFECT = 0
EXIT_NO_EFFECT = 1
EXIT_NO_CONTACT = 2

_ACTIONS = sorted(action.value for action in ActionType)


def create_parser() -> argparse.ArgumentParser:
    """The command-line parser."""
    parser = argparse.ArgumentParser(
        description="Send one command to a running Civ6 game and print what changed.",
    )
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--action", choices=_ACTIONS, help="the command's action type")
    what.add_argument("--json", dest="json_command", help="the whole command as JSON")
    parser.add_argument("--unit-id", type=int)
    parser.add_argument("--city-id", type=int)
    parser.add_argument("--x", type=int)
    parser.add_argument("--y", type=int)
    parser.add_argument("--item", help="item name (TECH_MINING, UNIT_WARRIOR) or number")
    parser.add_argument("--item-type", help="unit, building, wonder or district")
    parser.add_argument("--timeout", type=float, default=30.0, help="seconds to wait (default 30)")
    parser.add_argument("--ipc-dir", help="bridge IPC directory (default: as civ6_local)")
    parser.add_argument("--log-path", help="path to Lua.log (default: as civ6_local)")
    parser.add_argument(
        "--use-backlog",
        action="store_true",
        help="if no fresh state arrives, act on the last state already in Lua.log",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="emit diagnostic debug logs to stderr",
    )
    return parser


def build_command(args: argparse.Namespace) -> dict[str, Any]:
    """The command to send, as a dictionary.

    Raises:
        ValueError: If the command is not one the bridge can express.
    """
    if args.json_command is not None:
        try:
            command = json.loads(args.json_command)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--json is not valid JSON: {exc}") from None
        if not isinstance(command, dict):
            raise ValueError("--json must be a JSON object")
        if command.get("action_type") not in _ACTIONS:
            raise ValueError(f"unknown action_type {command.get('action_type')!r}")
        return command

    command = {"action_type": args.action}
    if args.unit_id is not None:
        command["unit_id"] = args.unit_id
    if args.city_id is not None:
        command["city_id"] = args.city_id
    if (args.x is None) != (args.y is None):
        raise ValueError("--x and --y go together")
    if args.x is not None:
        command["target"] = {"x": args.x, "y": args.y}
    if args.item is not None:
        command["item_id"] = int(args.item) if args.item.lstrip("-").isdigit() else args.item
    if args.item_type is not None:
        command["item_type"] = args.item_type
    return command


def _ack_fields(ack: Any) -> tuple[bool, str | None, str | None]:
    """(success, code, message) of an ack given as a dict or an object."""
    get = ack.get if isinstance(ack, dict) else lambda key, default=None: getattr(ack, key, default)
    code, message = get("error_code"), get("error_message")
    executed = get("executed") or []
    for entry in executed if isinstance(executed, list) else []:
        if isinstance(entry, dict) and not entry.get("success", True):
            code = entry.get("error_code") or code
            message = entry.get("message") or message
    return bool(get("success", False)), code, message


async def run_check(
    bridge: Any,
    command: dict[str, Any],
    *,
    timeout: float = 30.0,
    use_backlog: bool = False,
    out: Callable[[str], None] = print,
) -> int:
    """Send ``command`` through ``bridge`` and report what the next state shows.

    Returns:
        ``EXIT_EFFECT``, ``EXIT_NO_EFFECT`` or ``EXIT_NO_CONTACT``.
    """
    started = time.monotonic()
    logger.debug(
        "Waiting for initial game state (timeout=%.1fs, use_backlog=%s)", timeout, use_backlog
    )
    try:
        before = await bridge.wait_for_state(timeout=timeout, require_new=False)
    except Exception as exc:
        before = bridge.backlog_state() if use_backlog else None
        if before is None:
            out(f"no state from the game within {timeout:g}s ({exc}); is it the agent's turn?")
            return EXIT_NO_CONTACT
        out("WARNING: no fresh state; using the last state already in Lua.log")
    out(f"state: turn {before.turn_id}, blocker {before.end_turn_blocking}")

    lua_command = LuaCommand.from_dict(command)
    message = CommandMessage.create_single(msg_id=bridge.next_msg_id(), command=lua_command)
    # Judge what is actually sent: the protocol turns an id written as text
    # into a number, and the states carry numbers.
    command = lua_command.to_dict()
    out(f"sending: {json.dumps(command, sort_keys=True)}")
    try:
        ack = await bridge.send_and_wait(message, timeout=timeout)
    except Exception as exc:
        out(f"no answer from the mod within {timeout:g}s ({exc})")
        return EXIT_NO_CONTACT
    logger.debug("Raw ack received in %.3fs: %r", time.monotonic() - started, ack)
    success, code, text = _ack_fields(ack)
    out(f"answer: {'accepted' if success else 'REFUSED'}" + (f" [{code}] {text}" if code else ""))
    if not success:
        return EXIT_NO_EFFECT

    try:
        after = await bridge.wait_for_state(timeout=timeout, require_new=True)
    except Exception as exc:
        if command.get("action_type") == "end_turn":
            out(
                f"end of turn accepted, NOT confirmed: no new state within {timeout:g}s "
                "(the other players may still be moving; a longer --timeout may show it)"
            )
            return EXIT_NO_EFFECT
        out(f"no state after the command within {timeout:g}s ({exc})")
        return EXIT_NO_CONTACT

    lines = changes(before, after)
    logger.debug(
        "Post-command state turn=%s received in %.3fs", after.turn_id, time.monotonic() - started
    )
    for line in lines:
        out(f"changed: {line}")
    if not lines:
        out("changed: nothing")
    verdict = expected_effect(command, before, after)
    if verdict is None:
        out("expected effect: not judged for this command")
        return EXIT_EFFECT if lines else EXIT_NO_EFFECT
    out(f"expected effect: {'seen' if verdict else 'NOT seen'}")
    return EXIT_EFFECT if verdict else EXIT_NO_EFFECT


def _build_bridge(args: argparse.Namespace) -> Any:
    """A ``LiveBridge`` configured as ``civ6_local.run_civ6_bridge`` configures it.

    Shares live IPC and ``Lua.log`` precedence with
    :mod:`alpha_zero.local.gameplay_config` (``--ipc-dir`` ->
    ``CIV6AZ_LOCAL_BRIDGE_IPC_DIR`` -> default comm dir) so this lightweight
    CLI always targets the same IPC directory as ``civ6_local`` without
    importing ``torch`` or ``numpy``.
    """
    from alpha_zero.bridge.config import BridgeConfig
    from alpha_zero.bridge.live_bridge import LiveBridge
    from alpha_zero.local.gameplay_config import (
        resolve_live_bridge_ipc_dir,
        resolve_live_civ6_log_path,
    )

    ipc_dir = resolve_live_bridge_ipc_dir(args.ipc_dir)
    log_path = resolve_live_civ6_log_path(args.log_path)
    bridge_cfg = BridgeConfig.from_live_env(comm_dir=ipc_dir)
    bridge_cfg.ensure_comm_dir()
    logger.debug(
        "Configured LiveBridge: comm_dir=%s log_path=%s transport=%s endpoint=%s:%d",
        bridge_cfg.comm_dir,
        log_path,
        bridge_cfg.command_transport,
        bridge_cfg.tuner_host,
        bridge_cfg.tuner_port,
    )
    return LiveBridge(bridge_cfg, log_path=log_path)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)
    if getattr(args, "verbose", False):
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stderr,
        )
        logging.getLogger().setLevel(logging.DEBUG)
    try:
        command = build_command(args)
        LuaCommand.from_dict(command)  # a command the protocol cannot express is a usage error
    except Exception as exc:
        parser.error(str(exc))

    async def _session() -> int:
        bridge = _build_bridge(args)
        await bridge.start()
        try:
            return await run_check(
                bridge, command, timeout=args.timeout, use_backlog=args.use_backlog
            )
        finally:
            await bridge.stop()

    try:
        return asyncio.run(_session())
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"could not reach the game: {exc}", file=sys.stderr)
        return EXIT_NO_CONTACT


if __name__ == "__main__":
    sys.exit(main())
