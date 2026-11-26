# Pre-registration: expressivity census of search-in-the-simulator (zero games)

Status: registered before any measurement. Author: sdlc-ml-scientist. Plays no game,
loads no network, runs no search. games_total: 0. Gap register row GM-B11.

## Question

When the bridge's search plays in the MicroCiv stand-in seeded from a live state, how much
of what the live game offers can it say at all? If little, further work on
search-in-the-simulator is not worth doing until the simulator grows or the network stops
being tied to its board.

## What is computed, per state and per board size B in {8, 16}

Fixed for every state, with no randomness:

    cs  = StateConverter(board_size=B, num_players=2, research_registry=False).convert(state)
    env = BridgeMCTSEnv.from_bridge_state(cs, seed=0)          # max_turns default
    tr  = SearchTranslator(cs, env.own_unit_ids(), env.own_city_ids(), allowed=None)
    mask, commands, reasons = tr.root_mask(env.env.get_valid_actions(),
                                           env.env.action_space_handler.decode_action)

`allowed=None`: no turn ledger. The seed does not affect the root mask, because nothing at
the root draws a random number.

**Offered atomic commands.** These come from `cs["_raw_state"].legal_actions`. Each
`LegalAction` other than `end_turn` is split into one command per target or item:

- `move` -> (`unit_move`, unit_id, tile)
- `attack` -> (`unit_attack`, unit_id, tile)
- `ranged_attack` -> (`unit_ranged_attack`, unit_id, tile)
- `skip` -> (`unit_skip`, unit_id)
- `found_city` -> (`unit_found_city`, unit_id)
- `produce` -> (`city_produce`, city_id, item name), one per item, units and buildings alike
- any other kind (fortify, research_tech, research_civic, ...) -> (kind, actor, item or
  target), one per item or target

Duplicates are removed. The same key, built from the values of `commands`, is the
translated set T. By construction every element of T is an offered command
(tests/fixtures/bridge_search_translation.is_offered is the independent check). The census
asserts this, and a failed assertion voids the state.

**Own units.** Units in `cs["_raw_state"].units` with `owner_id == player_id`.

### E1: offered commands the search can express
- Numerator: offered atomic commands (not end_turn) whose key is in T.
- Denominator: all offered atomic commands (not end_turn).
- Denominator zero: the state has no E1 value. It is excluded from the E1 median and the
  number excluded is reported.
- Also reported, but not used by the stop rule: E1 by kind, and the upper bound U = the
  share of offered atomic commands whose kind the translator handles (unit_move,
  unit_attack, unit_ranged_attack, unit_skip, unit_found_city, and city_produce of a unit
  item). E1 <= U by construction. Research and building items count against E1. If median
  U < 0.3, the E1 condition of the stop rule fires for a reason known from the offer mix
  alone, and the memo must say so.

### E2: own units the search can actually move or use
- Denominator: own units with at least one offered move, attack, ranged_attack or
  found_city command.
- Numerator: those units with at least one command in T of action_type unit_move,
  unit_attack, unit_ranged_attack or unit_found_city carrying their unit_id. Skip does not
  count.
- Units the stand-in leaves out are in the denominator and never in the numerator. So are
  units past the simulator's cap of 20 units.
- Denominator zero: no E2 value, excluded and counted.

### E3: land the stand-in marks impassable
- Cells: board cells of `cs["terrain"]` that received at least one exported game tile (see
  "Computability"). A cell is water when its resolved `"type"` is `coast` or `ocean`.
- Numerator: non-water cells with `simulator_terrain(cell) == "mountain"`. This is the
  function and input `BridgeMCTSEnv.__init__` uses.
- Denominator: non-water cells with at least one tile.
- Denominator zero: no E3 value, excluded and counted.
- Note: `proxy_stats["impassable_tiles"]` counts water too, so it is not E3.

### E4: own units the stand-in leaves out
- Numerator: `env.proxy_stats["own_units_not_represented"]`.
- Denominator: units in `cs["units"]` with `player == cs["current_player"]`.
- Denominator zero: no E4 value, excluded and counted.

**Aggregation.** One value per state per B. Report, per B: the median, the interquartile
range (25th and 75th percentiles, numpy's default linear interpolation), the
distribution-free 95% interval for the median from order statistics (for N = 40, ranks 14
and 27), the number of states with a value, and the same figures for states after turn 30.
If states come from more than one game, also report per-game medians. B = 8 and B = 16 use
the same states.

## Stop rule (fixed now)

At B = 8: if median E1 < 0.3 OR median E2 < 0.5, no further work on search-in-the-simulator
(GM-B6 onward, search-mode wiring, translator refinements) until either the simulator gains
what the census shows missing or the network is no longer tied to its board size. B = 16
results are reported and neither trigger nor lift the stop. The point median decides. If the
95% interval for the median straddles the threshold, the memo says "borderline", but the
rule still applies as written. Not stopping is not a go: it means only that expressivity
does not rule the work out.

## Input required for a real answer

- States from a real game's `Lua.log` (complete `AZ_STATE` records, reassembled from pieces
  the way the socket transport reassembles them), produced by the current mod (state schema
  2 or later, with `observes_others` set).
- Sampling, fixed: the first complete state of each game turn. No hand selection. If more
  qualifying states exist, use them all.
- At least N = 40 states, at least 10 of them after turn 30. Why 40: at N = 20 the 95%
  order-statistic interval for the median runs from rank 6 to rank 15, about the whole
  IQR, so it cannot separate a median of 0.25 from 0.35. At N = 40 it runs from rank 14 to
  rank 27, roughly the 34th to 66th percentiles, which is narrower than the IQR.
  Consecutive turns of one game are correlated, so 40 states from one game are worth fewer
  than 40 independent ones. Two or more games are preferred, and the memo states how many
  games there were.
- Recorded with the result: the mod version, the map size, the civilization, who played
  the turns (human, greedy policy), the commit sha, and an evidence block per
  `docs/evidence/SCHEMA.md` (`search_epoch: post-2026-09-04`, `games_total: 0`,
  `seed: 0`).

## What a run on synthetic stand-in states does and does not show

A run on the fixtures (`tests/fixtures/bridge_search_states.make_state`,
`tests/fixtures/bridge_search_translation.rich_state`) shows that the four computations run,
handle the zero-denominator cases, and match hand-counted values on small states; that is
the census's unit test. It does not estimate any E: the fixtures' maps, unit mixes and
offer lists were written by hand by the authors of the translator, and their offers do not
come from the game's own check (`AlphaZeroActionExecutor.Check`). No number from synthetic
states is quoted as a census result.

## Threats to validity

1. One game, or a few, from one map type and era. Late-game units (ships, support units,
   great people) raise E4 and lower E2; early states overstate expressivity.
2. Weighting: E1 counts atomic commands, so a city offering 30 items outweighs a unit with
   6 moves. The by-kind breakdown and U are reported for this reason.
3. Who played the turns decides which situations appear (war, peace, settling).
4. The translator measures moves with an approximate distance (Euclidean, wrapping east to
   west), and the converter's scaling is lossy: at B = 8 on an 84x54 map one cell is 10.5 x
   6.75 tiles. Both are part of what is measured, not errors in the measurement.
5. Caps: 20 units and 5 cities per player in the simulator's action space.
6. Exported tiles: whether unrevealed tiles are exported is not known. The share of
   exported tiles that are revealed is reported beside E3.
7. Expressivity is necessary, not sufficient: high E1 and E2 say nothing about whether the
   search chooses well. A B = 16 pass does not help A1's networks, which are board 8.
8. Code drift: the census is bound to the sha it runs at. A later change to the converter,
   proxy or translator needs a re-run.

## Computability against the code (checked by reading, not by running)

- E1, E2: computable now from `SearchTranslator.root_mask` (its `commands`) and the raw
  `legal_actions`. No change needed.
- E4: computable now from `BridgeMCTSEnv.proxy_stats["own_units_not_represented"]` and
  `cs["units"]`. No change needed.
- E3: **not fully computable from what the functions return.** `StateConverter._convert_terrain`
  fills every cell with a default `plains` entry before writing real tiles, so a cell no
  tile reached cannot be told from real plains. Workaround with no code change: after
  `convert`, map each raw tile through the same converter instance's `_scale_coords(x, y)`
  (a private method) to get the set of cells that received a tile. Smallest proper change:
  have `_convert_terrain` record the tile count per cell (an additive key in the cell dict,
  e.g. `"_tiles": n`), or expose a public `cell_of(x, y)`.
- Reading states: there is no public function that yields every state in a `Lua.log`.
  `LogTailer` keeps only the latest or backlog state (`_scan_backlog`, `_handle_chunk*`).
  Smallest change: a public `iter_log_states(path)` in `alpha_zero/bridge/log_tailer.py`
  that feeds each line through the existing parser (chunk reassembly included) and yields
  each completed state. Until then the census can subclass `LogTailer` and collect from
  `_publish_state`, which is a private hook.

## Implementation notes (appended after implementation, before any real log was read)

Author: sdlc-ml-scientist, 2026-11-23. The registered text above is unchanged. These record
choices the registration left open. None changes a definition or the stop rule.

1. The check that every translated command was offered uses the census's own command key on
   both sides: raw `legal_actions` on one side, the translator's commands on the other.
   `bridge_search_translation.is_offered` is not used, because it rejects ranged attacks. It
   checks the translator, not the key function; the key function is checked against hand
   counts in the tests.
2. One input log is treated as one game. Within a log, a turn is sampled once: its first
   complete state. If the turn number goes back (a reload, or a second game in the same log),
   later states of turns already sampled are not used, and the number of times this happened
   is reported as `turn_went_back`. When it is above zero, the memo says the log may hold more
   than one game or a reload, and the game count it states is a lower bound.
3. A state is "complete" only if its pieces are exactly 1..n of its end line and its JSON
   parses. States dropped for a missing piece or unreadable JSON are counted and reported. The
   next complete state of that turn is then the turn's first complete state.
4. A sampled state that cannot be parsed or measured, or whose translated commands are not
   all offered, is void. Its reason is recorded and it is never replaced by another state of
   the same turn. A state void on either board is void on both, so both boards use the same
   states. Void states do not count toward the minimum of 40 (10 after turn 30). If more than
   10% of sampled states are void, the memo lists the reasons as a threat to validity.
5. In U and in E1, a production item with no `item_types` entry counts as a unit, as the
   translator reads it.
6. Units past the simulator's 20-unit cap count in E2's denominator but not in E4's
   numerator, because E4 is `own_units_not_represented` as registered. E4 therefore counts
   only units with no simulator stand-in.
7. E1 by kind is pooled counts per kind over all states, not a median over states. It is
   descriptive only.
8. "Straddles the threshold" means lower bound < threshold <= upper bound, matching the strict
   "<" that fires the rule. With fewer than 6 values there is no 95% interval. "After turn 30"
   means turn > 30. If neither condition fires and a median is missing, the verdict is
   "undetermined".
9. States from a mod without `observes_others` or below schema 2 are measured, but their
   presence means the output is not a census result.
10. The output is a census result only if every source is a `Lua.log`, the minimum is met by
    non-void states, no state is from an older mod, both boards 8 and 16 were computed, and
    the mod version and who played the turns are recorded (not null). On any other input the
    stop-rule field reads "not decided", and the computed answer is shown only under a name
    that says it is unregistered. The script cannot tell a real log from a synthetic one; the
    recorded fields are the attestation that it is real.
11. The script refuses to run if the commit sha cannot be resolved.
12. (Added 2026-11-23 after the automated review of PR #169, by the executor; it applies note 4
    to a case nobody had listed.) The live state parser is tolerant: it drops an invalid tile,
    unit, city, player, offered action, or a target or item inside an offered action, and
    returns the rest. A sampled state from which the parser dropped anything is void on both
    boards, with the kinds and counts recorded, and the number of such states is reported.

## Exploratory addendum (designed 2026-11-23, after the asymmetry probe came out vacuous)

Author: sdlc-ml-scientist. Not part of the registered census and not used by its stop rule.
For each sampled census state at board 8, record whether the merged rule's losing ending can
fire at all: the player has no city and `own_units_not_represented == 0`. Report the count and
share in three turn bands: up to turn 10, turns 11-30, after turn 30. There is no verdict. If
the share after turn 10 is zero, the memo says the losing-only ending cannot affect play after
the first city, and the asymmetry question is closed for live play.
