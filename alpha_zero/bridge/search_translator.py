"""One simulator action to one live command (GM-B6a).

The live search (``Civ6MctsPolicy``) plays in the MicroCiv simulator seeded
from the live state, so its answer is one of the simulator's actions: "own
unit 2 moves north-east", "own city 0 builds a warrior". Nothing turned that
into something the game can be asked to do. :class:`SearchTranslator` does,
for one live state:

* units and cities are found by their live id, never by their place in a list
  (the simulator leaves units out, which shifts every place after them);
* a move goes toward the centre of the destination board cell: the offered
  tile that gets nearest it, and only if it gets nearer than standing still.
  On a map larger than the board a cell covers many tiles, so "one cell
  north-east" has no single tile;
* an attack goes only into the destination cell, and a move never becomes an
  attack;
* production picks an offered unit of the simulator's class;
* every command returned is one the game offered in this state.

What cannot be translated gets a reason, not an exception, and
:meth:`SearchTranslator.root_mask` leaves it out of the search's root, so
every candidate the search weighs is playable.

Pure: no asyncio, no torch. The distances are Euclidean on the offset grid
with the x axis wrapping, an approximation of hex distance that is enough to
say which offered neighbour lies toward a cell.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeGuard

import numpy as np

from alpha_zero.bridge.microciv_proxy import (
    SIM_ARCHER,
    SIM_SETTLER,
    SIM_WARRIOR,
    simulator_unit_class,
)

logger = logging.getLogger(__name__)

__all__ = ["SearchTranslator", "Translation"]

#: Slack for comparing a tile with the edge of a cell.
_EPSILON = 1e-9

#: The name preferred for each simulator class when it is among the offers.
_PREFERRED_ITEM: dict[str, str] = {
    SIM_WARRIOR: "UNIT_WARRIOR",
    SIM_SETTLER: "UNIT_SETTLER",
    SIM_ARCHER: "UNIT_ARCHER",
}


@dataclass(frozen=True)
class Translation:
    """A live command, or the reason there is none."""

    command: dict[str, Any] | None
    reason: str | None = None


def _no(reason: str) -> Translation:
    return Translation(None, reason)


def _is_int(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _xy(tile: tuple[int, int]) -> dict[str, int]:
    return {"x": tile[0], "y": tile[1]}


def _tile(value: Any) -> tuple[int, int] | None:
    """A target as an (x, y) pair of ints, or None."""
    if isinstance(value, Mapping):
        value = (value.get("x"), value.get("y"))
    if isinstance(value, (tuple, list)) and len(value) == 2 and all(_is_int(v) for v in value):
        return int(value[0]), int(value[1])
    return None


class SearchTranslator:
    """Translate the simulator's actions for one live state.

    Args:
        converted_state: Output of ``StateConverter.convert`` (it carries the
            raw state as ``_raw_state``).
        own_unit_ids: Live ids of the simulator's current-player units, in
            simulator order (``BridgeMCTSEnv.own_unit_ids()``).
        own_city_ids: The same for cities.
        allowed: Optional veto: a command for which it returns False is not
            translatable (the turn's ledger, in GM-B6b).
    """

    def __init__(
        self,
        converted_state: Any,
        own_unit_ids: Sequence[int],
        own_city_ids: Sequence[int],
        *,
        allowed: Callable[[dict[str, Any]], bool] | None = None,
    ) -> None:
        self._unit_ids = list(own_unit_ids)
        self._city_ids = list(own_city_ids)
        self._allowed = allowed
        state = converted_state if isinstance(converted_state, Mapping) else {}
        raw = state.get("_raw_state")

        self._board = state.get("_board_size")
        map_size = state.get("_map_size")
        self._map: tuple[int, int] | None = None
        if (
            isinstance(map_size, (tuple, list))
            and len(map_size) == 2
            and all(_is_int(v) and v > 0 for v in map_size)
        ):
            self._map = (int(map_size[0]), int(map_size[1]))
        if not (_is_int(self._board) and self._board > 0):
            self._board = None

        # Board cell and live tile of every converted unit, by live id.
        self._unit_cell: dict[int, tuple[int, int]] = {}
        self._unit_tile: dict[int, tuple[int, int]] = {}
        # Only the acting player's own units: the game's unit ids are per
        # owner, so another player's unit can carry the same id.
        me = getattr(raw, "player_id", None)
        for unit in state.get("units", []) if isinstance(state.get("units"), list) else []:
            if not isinstance(unit, Mapping) or not _is_int(unit.get("_id")):
                continue
            if _is_int(me) and "_owner_id" in unit and unit["_owner_id"] != me:
                continue
            cell, tile = _tile(unit.get("position")), _tile(unit.get("_original_pos"))
            if cell is not None:
                self._unit_cell.setdefault(unit["_id"], cell)
            if tile is not None:
                self._unit_tile.setdefault(unit["_id"], tile)

        # What the game offered, by kind and actor.
        self._end_turn = False
        self._unit_offers: dict[tuple[str, int], list[tuple[int, int]]] = {}
        self._untargeted: set[tuple[str, int]] = set()
        self._produce: dict[int, list[str]] = {}
        for offer in self._offers(raw):
            kind = getattr(offer, "action_type", None)
            unit_id, city_id = getattr(offer, "unit_id", None), getattr(offer, "city_id", None)
            if kind == "end_turn":
                self._end_turn = True
            elif kind in ("move", "attack", "ranged_attack") and _is_int(unit_id):
                tiles = [t for t in map(_tile, getattr(offer, "targets", None) or []) if t]
                self._unit_offers.setdefault((kind, unit_id), []).extend(tiles)
            elif kind in ("skip", "found_city") and _is_int(unit_id):
                self._untargeted.add((kind, unit_id))
            elif kind == "produce" and _is_int(city_id):
                self._produce.setdefault(city_id, []).extend(self._unit_names(offer))

    @staticmethod
    def _offers(raw: Any) -> list[Any]:
        try:
            return list(raw.legal_actions)
        except Exception:  # no raw state, or one that cannot be read: nothing was offered
            return []

    @staticmethod
    def _unit_names(offer: Any) -> list[str]:
        """The offered item names whose kind is a unit (or not stated)."""
        kinds = getattr(offer, "item_types", None)
        kinds = kinds if isinstance(kinds, Mapping) else {}
        names: list[str] = []
        for item in getattr(offer, "items", None) or []:
            if isinstance(item, str) and item and kinds.get(item, "unit") == "unit":
                names.append(item)
        return names

    # ------------------------------------------------------------------
    def translate(self, action: Any) -> Translation:
        """The live command for one decoded simulator action, or why there is none."""
        try:
            result = self._translate(action)
        except Exception as exc:  # a malformed action or state is a reason, not a crash
            return _no(f"could not be read ({type(exc).__name__})")
        if result.command is not None and self._allowed is not None:
            try:
                permitted = bool(self._allowed(dict(result.command)))
            except Exception:
                permitted = False
            if not permitted:
                return _no("ruled out for this turn")
        return result

    def _translate(self, action: Any) -> Translation:
        if not isinstance(action, Mapping):
            return _no("not an action")
        kind = action.get("type")
        if kind == "end_turn":
            if not self._end_turn:
                return _no("end_turn was not offered")
            return Translation({"action_type": "end_turn"})
        if kind == "produce":
            return self._produce_command(action)
        if kind not in ("move", "attack", "skip", "found_city"):
            return _no(f"unknown simulator action {kind!r}")

        index = action.get("unit_idx")
        if not _is_int(index) or not 0 <= index < len(self._unit_ids):
            return _no("no such unit in the simulator")
        unit_id = self._unit_ids[index]
        if kind == "skip":
            if ("skip", unit_id) not in self._untargeted:
                return _no("skip was not offered for the unit")
            return Translation({"action_type": "unit_skip", "unit_id": unit_id})
        if kind == "found_city":
            if ("found_city", unit_id) not in self._untargeted:
                return _no("found_city was not offered for the unit")
            return Translation({"action_type": "unit_found_city", "unit_id": unit_id})
        return self._targeted(kind, unit_id, action.get("direction"))

    def _targeted(self, kind: str, unit_id: int, direction: Any) -> Translation:
        delta = _tile(direction)
        cell, tile = self._unit_cell.get(unit_id), self._unit_tile.get(unit_id)
        if delta is None or cell is None or tile is None:
            return _no("the unit's place or the direction is unknown")
        if self._board is None or self._map is None:
            return _no("the board or the map size is unknown")
        destination = (cell[0] + delta[0], cell[1] + delta[1])
        if not (0 <= destination[0] < self._board and 0 <= destination[1] < self._board):
            return _no("the destination is off the board")
        centre = self._destination_point(cell, delta)

        def nearest(tiles: list[tuple[int, int]]) -> tuple[int, int]:
            return min(tiles, key=lambda t: (self._distance(t, centre), t[1], t[0]))

        if kind == "move":
            offered = self._unit_offers.get(("move", unit_id), [])
            if not offered:
                return _no("no move was offered for the unit")
            best = nearest(offered)
            if not self._distance(best, centre) < self._distance(tile, centre):
                return _no("no offered tile is nearer the destination")
            return Translation(
                {"action_type": "unit_move", "unit_id": unit_id, "target": _xy(best)}
            )

        # An attack: a melee attack offered into the destination cell, else a
        # ranged one (the game offers an archer `ranged_attack`, and the
        # simulator has one attack action for both).
        for offer, command in (("attack", "unit_attack"), ("ranged_attack", "unit_ranged_attack")):
            inside = [
                t
                for t in self._unit_offers.get((offer, unit_id), [])
                if self._in_cell(t, destination, centre)
            ]
            if inside:
                return Translation(
                    {"action_type": command, "unit_id": unit_id, "target": _xy(nearest(inside))}
                )
        if not any(self._unit_offers.get((o, unit_id)) for o in ("attack", "ranged_attack")):
            return _no("no attack was offered for the unit")
        return _no("no offered attack into that cell")

    def _in_cell(
        self, tile: tuple[int, int], cell: tuple[int, int], point: tuple[float, float]
    ) -> bool:
        """Whether an attack on ``tile`` is an attack into ``cell``.

        On a map at least as large as the board the converter's own scaling
        decides (round, then clamp at the board's edge), so the answer agrees
        with the cell the simulator saw the target in. On a smaller map a
        cell is a fraction of a tile and the box around the destination point
        decides.
        """
        width, height = self._size()
        if width >= self._cells() and height >= self._cells():
            return self._cell_of(tile) == cell
        return self._inside_step(tile, point)

    def _produce_command(self, action: Mapping[str, Any]) -> Translation:
        index, wanted = action.get("city_idx"), action.get("unit_type")
        if not _is_int(index) or not 0 <= index < len(self._city_ids):
            return _no("no such city in the simulator")
        city_id = self._city_ids[index]
        names = sorted({n for n in self._produce.get(city_id, []) if _class_of(n) == wanted})
        if not names:
            return _no("no offered unit of that class")
        preferred = _PREFERRED_ITEM.get(wanted) if isinstance(wanted, str) else None
        name = preferred if preferred in names else names[0]
        return Translation(
            {
                "action_type": "city_produce",
                "city_id": city_id,
                "item_id": name,
                "item_type": "unit",
            }
        )

    # ------------------------------------------------------------------
    def _size(self) -> tuple[int, int]:
        """The map's (width, height); the callers have checked it is known."""
        if self._map is None:
            raise ValueError("the map size is unknown")
        return self._map

    def _cells(self) -> int:
        """The board's side; the callers have checked it is known."""
        if self._board is None:
            raise ValueError("the board size is unknown")
        return int(self._board)

    def _steps(self) -> tuple[float, float]:
        """Game tiles per board cell on each axis, never less than one tile.

        On a map larger than the board a cell covers several tiles. On one
        smaller than the board a cell is a fraction of a tile, and one cell
        over would not be a different tile at all; a step is then one tile.
        """
        return max(1.0, self._size()[0] / self._cells()), max(1.0, self._size()[1] / self._cells())

    def _destination_point(
        self, cell: tuple[int, int], delta: tuple[int, int]
    ) -> tuple[float, float]:
        """Where one step from ``cell`` in ``delta`` lies, in game coordinates.

        The converter scales tile x onto cell ``round(x * B / W)``, so cell
        ``cx`` is centred on ``cx * W / B``. On a map at least as large as the
        board this is the centre of the destination cell.
        """
        step_x, step_y = self._steps()
        return (
            cell[0] * self._size()[0] / self._cells() + delta[0] * step_x,
            cell[1] * self._size()[1] / self._cells() + delta[1] * step_y,
        )

    def _cell_of(self, tile: tuple[int, int]) -> tuple[int, int]:
        """The board cell of a game tile, by the converter's own scaling."""
        width, height = self._size()
        top = self._cells() - 1
        x = round(tile[0] * self._cells() / width)
        y = round(tile[1] * self._cells() / height)
        return max(0, min(top, x)), max(0, min(top, y))

    def _inside_step(self, tile: tuple[int, int], point: tuple[float, float]) -> bool:
        """Whether ``tile`` lies in the cell-sized box around ``point``."""
        step_x, step_y = self._steps()
        dx = abs(tile[0] - point[0]) % self._size()[0]
        dx = min(dx, self._size()[0] - dx)
        return dx <= step_x / 2 + _EPSILON and abs(tile[1] - point[1]) <= step_y / 2 + _EPSILON

    def _distance(self, tile: tuple[int, int], point: tuple[float, float]) -> float:
        dx = abs(tile[0] - point[0]) % self._size()[0]
        dx = min(dx, self._size()[0] - dx)  # the map wraps east-west
        return math.hypot(dx, tile[1] - point[1])

    # ------------------------------------------------------------------
    def root_mask(
        self, valid_mask: Any, decode: Callable[[int], Mapping[str, Any]]
    ) -> tuple[np.ndarray, dict[int, dict[str, Any]], dict[str, int]]:
        """Which of the simulator's legal actions can be played live.

        Args:
            valid_mask: The simulator's own legal-action mask.
            decode: Action index to decoded action
                (``MicroCivActionSpace.decode_action``).

        Returns:
            ``(mask, commands, reasons)``: the mask with only translatable
            indexes set, the command of each, and how often each reason kept
            an action out.
        """
        source = np.asarray(valid_mask)
        mask = np.zeros_like(source)
        commands: dict[int, dict[str, Any]] = {}
        reasons: dict[str, int] = {}
        for index in (int(i) for i in np.flatnonzero(source)):
            try:
                result = self.translate(decode(index))
            except Exception as exc:
                result = _no(f"could not be decoded ({type(exc).__name__})")
            if result.command is not None:
                mask[index] = source[index]
                commands[index] = result.command
            else:
                reason = result.reason or "unknown"
                reasons[reason] = reasons.get(reason, 0) + 1
        return mask, commands, reasons


_UNIT_TABLE: dict[str, str] | None = None


def _unit_table() -> Mapping[str, str]:
    """The converter's unit table (name -> category), read once."""
    global _UNIT_TABLE
    if _UNIT_TABLE is None:
        from alpha_zero.bridge.state_converter import UnitMapping

        _UNIT_TABLE = dict(UnitMapping().UNITS)
    return _UNIT_TABLE


def _class_of(name: str) -> str | None:
    """The simulator class of a unit the converter's table lists, else None."""
    category = _unit_table().get(name) if isinstance(name, str) else None
    if category is None:
        return None
    return simulator_unit_class(category, name)
