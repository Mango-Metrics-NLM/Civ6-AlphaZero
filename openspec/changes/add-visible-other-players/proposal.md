# Export what the player can see of other players, and nothing it cannot

## Why

The mod's export of other players' units and cities exists but is switched off
(`OBSERVE_ALL_PLAYERS = false`), so the live agent never sees anyone else: the
encoder's enemy planes are empty and the search's simulator has no opponent
(gap GM-B8). Switching the flag on as it stood would have been wrong:

- players were found with `for i = 0, GetWasEverAliveCount() - 1`, a count used as
  ids (barbarians and city-states have high ids), through a call the game's own UI
  code does not use;
- a unit counted as seen when its tile was visible, so a stealth unit would leak;
- a city counted as seen when its tile was ever revealed, with live numbers;
- every other owner became "the enemy" in Python, a neutral scout included.

The export also leaked with the flag off: a tile that is not visible now carried its
current owner, improvement and resource; with no visibility table every tile was
exported as visible; and an attack was offered on a plot whose occupant the player
cannot see.

This is slice GM-B8a. The search-side use of an opponent (the simulator's
elimination rule) is GM-B8b, on top of the GM-B6 branches.

## What Changes

- `mod/AlphaZeroAI/Scripts/StateExporter.lua`: other players from
  `Game.GetPlayers{Alive = true}` (fallback: ids below `GameDefines.MAX_PLAYERS`);
  a foreign unit only when the game says the player sees its tile and the unit
  (`IsVisible` and `IsUnitVisible`), with only the facts the game shows; a foreign
  city only while its tile is visible now; `other_players` (`id`, `kind`, `at_war`,
  `met`) and `observes_others` in the state. Tiles not visible now carry no owner,
  improvement, resource or yields; with no visibility table nothing is visible.
- `mod/AlphaZeroAI/Scripts/ActionExecutor.lua`: an attack's target has to be an
  occupant the player sees.
- `OBSERVE_ALL_PLAYERS` stays off by default until a live sitting has seen the
  export be right.
- `alpha_zero/bridge/protocol.py`: `observes_others`, `other_players` (additive).
- New `alpha_zero/bridge/other_players.py`; `StateConverter` puts only hostile
  owners (at war, barbarian, free city) on the enemy side and counts the rest.

## Impact

- Capability: `civ6-bridge` (three requirements added).
- Live mod behaviour changes for both modes: tiles in fog lose their live facts and
  attack offers respect visibility, always; other players are exported only with
  the setting on.
- Nothing on the release play path. Not verified in a real game: the calls are
  taken from the game's UI code, and the stand-in game models that reading.
