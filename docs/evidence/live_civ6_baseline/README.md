# Live Civilization VI Bridge Baseline & GM-B11 Expressivity Census (2026-11-26)

> **Scope & non-claims:** Post-`2026-09-04` wiring and expressivity evidence only
> (`git_sha` `a3905895a0b9014a98fcdb07e1f36adb786c7596`, `mod_version` `1.1.0-a3905895`).
> Zero games were played in the simulator and no trained Civ6 policy exists.
> Nothing here is evidence of playing strength or learning.

## Environment

- **Host:** Windows 11 (`AMD64 Family 26 Model 68 Stepping 0`, `24` logical cores), Python `3.11.9`, NumPy `1.26.4`, PyTorch `2.11.0+cu128`.
- **Game & mod:** Live Steam Civilization VI DX12 (`CivilizationVI_DX12`), single-player standard-speed `74×46` match (`CIVILIZATION_JAPAN`, AI Player `0`), `AlphaZeroAI` installed with `13/13` SHA-256 parity (`scripts/install_civ6_mod.py --check`), headless FireTuner socket transport (`CIV6_BRIDGE_TRANSPORT=socket`, `127.0.0.1:4318`, `100` Lua states).
- **Network / policy:** Untrained random-weights `AlphaZeroNetwork` at demo bridge dimensions (`max_units=20`, `max_cities=5`, `num_production_items=20`).

## Live Command Probes (`AZ_ACK` + `state_diff`)

Measured via `scripts/live_command_check.py --use-backlog` over FireTuner socket transport on live Turns `2–37`:

| Checklist / Command | Probe & Outcome | Post-Command State Diff |
|---|---|---|
| **Item 13** (`unit_move`, multi-turn destination) | `unit 131073` `(43,16) -> (41,15)`: `accepted` (`ack` `0.860s`, state `1.079s`, exit `0`) | `ENDTURN_BLOCKING_UNITS -> None`; moved `(43,16)->(42,15)`, moves `2->1` |
| **Step move** (`unit_move` adjacent) | `unit 131073` `(42,14) -> (42,13)`: `accepted` (`ack` `0.078s`, state `0.313s`, exit `0`) | moved `(42,14)->(42,13)`, moves `2->0` |
| **`research_civic`** | `CIVIC_FOREIGN_TRADE`: `accepted` (`ack` `0.203s`, state `0.406s`, exit `0`) | `civic: CIVIC_CRAFTSMANSHIP -> CIVIC_FOREIGN_TRADE` |
| **Item 11** (`unit_attack` vs peaceful `p10`) | `unit 131073 @ (42,13) -> (43,12)` (`war=false`): `REFUSED [WOULD_START_WAR]` | `Refused: (43,12) would start a war with player 10`; verified `WAR_WITH_10=false` |
| **Item 12a** (`unit_move` onto peaceful occupied tile) | `unit 131073 @ (42,13) -> (43,12)` (`mv=2`, Turn 37): `REFUSED [WOULD_START_WAR]` (`mv=0` Turn 36: `REFUSED [NOT_ENOUGH_MOVEMENT]`) | War guard blocked occupied-tile move; still at peace (`hostile combat` against an active war target & `unit_ranged_attack` remain open — no war/barbarian unit reached `d=1`) |
| **`unit_fortify` / `unit_skip`** | Warrior `196608` `unit_fortify` (Turn 36): `accepted` (exit `0`); Builder `262146` `unit_skip` (Turn 35): `accepted` (exit `0`); `unit_skip` when `0` moves/no blocker: `REFUSED [OPERATION_REFUSED]` | Cleared `ENDTURN_BLOCKING_UNITS -> None` when active; `TurnLedger` throttled `unit_skip` on Turn 36 because `movement` remained `2` (`changed: nothing`) |
| **Secondary commands (honest refusals / unverified UI paths)** | Civilian `unit_sleep` (Builder `262146`): `REFUSED [OPERATION_REFUSED] Sleep refused by the game`; `city_buy` (`UNIT_SCOUT` str -> `ProtocolValidationError`, int `--item 1 --item-type unit` at Kyoto `65536` `gold=204.69` -> `REFUSED [EXECUTION_FAILED]`); `unit_build` (`Builder 327683`, `--item 0`): `accepted` (`Build started`, no `CanStartOperation` check), `changed: nothing`; `city_citizen` (no `--item` -> `REFUSED [INVALID_COMMAND]`; adjacent plot `1301` -> `REFUSED [EXECUTION_FAILED] Citizen lock failed`) | Matches `SetCitizenLocked` void-under-`pcall` and "written from memory" notes in [`../../LIVE_CIV6.md`](../../LIVE_CIV6.md) |

## Multi-Turn Autonomous `--mode civ6` Soaks (Turns `1 → 41`)

- **Item 29 (10-turn autonomous soak — PASS, exit `0`):** Starting Turn 9 after dismissing a modal Firaxis `AdvisorPopup` on Turn 8, `python -m alpha_zero.civ6_local --mode civ6` with `CIV6AZ_AGENT_MAX_TURNS=0 CIV6AZ_AGENT_MAX_GAME_TURNS=10` completed **10 full turns (`Turns 9 → 19`)** autonomously (`234` turns/steps, `234` states, `234` acks, `100` heartbeats, `0` ack failures, `0` reconnects, `0` errors), stopping cleanly at `Reached max_game_turns=10; stopping` (including Turn 15 `ENDTURN_BLOCKING_FILL_CIVIC_SLOT` up to `199` actions).
- **Follow-up autonomous soak (`Turns 19 → 35`, exit `3`):** Ran **16 additional turns** (`118` commands, `117` states, `118` acks, `67` heartbeats, `0` errors), met city-state `pid=11` (`MET=1`), accumulated `gold=204.6875`, grew Kyoto to pop `3–4`, and exited cleanly with `3` (`on_stall=exit`) when stacked Builders left `ENDTURN_BLOCKING_UNITS` active.
- **Socket-driven Turn `37 → 41` advance:** Advanced cleanly through Turn `41` (`80` commands), stopping on `ENDTURN_BLOCKING_PANTHEON` (known unsupported blocker).
- **Root cause of AI turn pauses (`Turn 8` and `Turn 36`):** First-contact / tutorial Firaxis UI overlays (`AdvisorPopup`, `TutorialUIRoot`, `hidden=false` in their own Lua contexts without `NotificationManager` entries) pause AI turn processing (`Players[1]:IsTurnActive() == true`, `Players[0]:IsTurnActive() == false`) until dismissed by clicking through in the game UI; zero `LF-H1` turn-cycle deactivations occurred on the `end_turn` path.

## `GM-B11` Search-in-the-Simulator Expressivity Census

Machine-generated summary artifact: [`census_summary.json`](census_summary.json) (preregistration: [`../../../kb/specs/20261123-expressivity-census-preregistration.md`](../../../kb/specs/20261123-expressivity-census-preregistration.md)).

- **Command:** `python scripts/census_search_expressivity.py --log japan_lua.log --out <out> --boards 8 16 --mod-version 1.1.0-a3905895 --played-by ianshank`
- **Sample quality:** `states_read: 561`, `states_sampled: 41` (Turns `1 → 41`), `states_measured: 41`, `states_after_turn_30: 11`, `states_void: 0`, `states_unreadable: 0`, `states_dropped_missing_piece: 0`, `turn_went_back: 0`, `tiles_revealed_share` median `1.0`. Meets registered minimum (`40` states, `10` after turn `30`).
- **Registered stop rule (`board=8`):**
  - **`verdict: "stop"`** (`decided_by_this_input: true`, `borderline: false`, `e1_fires_from_the_offer_mix_alone: false`).
  - **`E1` (expressible fraction of offered legal commands):** median **`0.2667`** (95% CI `[0.2500, 0.2692]`, `after_turn_30`: `0.2424`) < threshold `0.30` -> **`fires: true`**. Pooled `E1` by kind (`board=8`): `unit_move` `220/426` (`51.64%`), `unit_skip` `8/43` (`18.60%`), `unit_found_city` `1/1` (`100.0%`), `city_produce` `0/240`, `research_tech` `0/160`, `research_civic` `0/26`, `fortify` `0/18`.
  - **`E2` (usable fraction of units with offers):** median **`0.6667`** (95% CI `[0.6667, 1.0000]`, `after_turn_30`: `0.5000`) >= threshold `0.50` -> **`fires: false`**.
  - **Ceiling `U`:** median **`0.7143`** (`[0.6667, 0.7500]`); exploratory `loss_can_fire`: `0.10` (`≤T10`), `0.00` (`T11–30`), `0.00` (`>T30`).
  - At `board=16`, `E1` median remained **`0.2667`** (`after_turn_30`: `0.1538`; `unit_move`: `209/426`, `49.06%`) and `E2` median **`0.6667`**.

**Consequence per preregistration:** Search in the MicroCiv stand-in simulator stops under the registered `GM-B11` criterion (`E1 = 0.2667 < 0.30`) until the simulator expands or policy targets decouple from the MicroCiv board.
