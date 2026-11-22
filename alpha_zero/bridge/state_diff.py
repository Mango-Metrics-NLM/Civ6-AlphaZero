"""What changed between two exported game states, and whether a command did
what it was sent to do.

Pure: reads :class:`~alpha_zero.bridge.protocol.GameStateData` and nothing
else. Used by ``scripts/live_command_check.py`` to say, for one command sent
to a running game, what the next state shows.

The judgement is only as good as the export: a change the mod does not write
(a unit's fortified state, for one) cannot be seen here, and the command then
reads as having had no effect.
"""

from __future__ import annotations

from typing import Any

from alpha_zero.bridge.protocol import GameStateData

__all__ = ["changes", "expected_effect", "summarize"]

Summary = dict[str, Any]


def summarize(state: GameStateData) -> Summary:
    """The parts of ``state`` a command can change, as integers and strings.

    Only the acting player's own units and cities are listed.
    """
    me = state.player_id
    units = {
        int(unit.id): {"x": int(unit.x), "y": int(unit.y), "moves": float(unit.moves_remaining)}
        for unit in state.units
        if unit.owner_id == me
    }
    cities = {
        int(city.id): {"producing": city.current_production}
        for city in state.cities
        if city.owner_id == me
    }
    research: str | None = None
    civic: str | None = None
    for player in state.players:
        if getattr(player, "id", None) == me:
            research = player.current_research
            civic = player.current_civic
            break
    return {
        "turn_id": int(state.turn_id),
        "units": units,
        "cities": cities,
        "research": research,
        "civic": civic,
        "end_turn_blocking": state.end_turn_blocking,
    }


def _summary(value: GameStateData | Summary) -> Summary:
    return value if isinstance(value, dict) else summarize(value)


def changes(before: GameStateData | Summary, after: GameStateData | Summary) -> list[str]:
    """One line per difference between two states; empty when nothing differs."""
    old, new = _summary(before), _summary(after)
    lines: list[str] = []

    for key, label in (
        ("end_turn_blocking", "end-turn blocker"),
        ("civic", "civic"),
        ("research", "research"),
        ("turn_id", "turn"),
    ):
        if old[key] != new[key]:
            lines.append(f"{label}: {old[key]} -> {new[key]}")

    for city_id in sorted(set(old["cities"]) | set(new["cities"])):
        was, now = old["cities"].get(city_id), new["cities"].get(city_id)
        if was is None:
            lines.append(f"city {city_id}: new")
        elif now is None:
            lines.append(f"city {city_id}: gone")
        elif was["producing"] != now["producing"]:
            lines.append(f"city {city_id}: producing {was['producing']} -> {now['producing']}")

    for unit_id in sorted(set(old["units"]) | set(new["units"])):
        was, now = old["units"].get(unit_id), new["units"].get(unit_id)
        if was is None:
            lines.append(f"unit {unit_id}: new")
        elif now is None:
            lines.append(f"unit {unit_id}: gone")
        else:
            if (was["x"], was["y"]) != (now["x"], now["y"]):
                lines.append(
                    f"unit {unit_id}: moved ({was['x']},{was['y']}) -> ({now['x']},{now['y']})"
                )
            if was["moves"] != now["moves"]:
                lines.append(f"unit {unit_id}: moves {was['moves']:g} -> {now['moves']:g}")
    return lines


def expected_effect(
    command: dict[str, Any],
    before: GameStateData | Summary,
    after: GameStateData | Summary,
) -> bool | None:
    """Whether ``after`` shows what ``command`` was sent to do.

    Args:
        command: The command as sent (the shape of ``LuaCommand.to_dict()``).
        before: The state the command was chosen on.
        after: The next state the mod exported.

    Returns:
        True or False for the command types judged here; None for any other.
    """
    old, new = _summary(before), _summary(after)
    action = command.get("action_type")
    unit_id = command.get("unit_id")
    city_id = command.get("city_id")
    blocker_changed = bool(old["end_turn_blocking"] != new["end_turn_blocking"])

    was_unit = old["units"].get(unit_id)
    now_unit = new["units"].get(unit_id)
    gone = was_unit is not None and now_unit is None
    moves_changed = bool(was_unit and now_unit and was_unit["moves"] != now_unit["moves"])
    tile_changed = bool(
        was_unit and now_unit and (was_unit["x"], was_unit["y"]) != (now_unit["x"], now_unit["y"])
    )

    if action == "unit_move":
        return gone or tile_changed or moves_changed
    if action in ("unit_attack", "unit_ranged_attack"):
        return gone or moves_changed
    if action in ("unit_skip", "unit_fortify", "unit_sleep"):
        return moves_changed or blocker_changed
    if action == "unit_found_city":
        return gone or bool(set(new["cities"]) - set(old["cities"]))
    if action == "city_produce":
        was_city, now_city = old["cities"].get(city_id), new["cities"].get(city_id)
        return bool(was_city and now_city and was_city["producing"] != now_city["producing"])
    if action == "research_tech":
        return bool(old["research"] != new["research"] or blocker_changed)
    if action == "research_civic":
        return bool(old["civic"] != new["civic"] or blocker_changed)
    if action == "end_turn":
        return bool(old["turn_id"] != new["turn_id"])
    return None
