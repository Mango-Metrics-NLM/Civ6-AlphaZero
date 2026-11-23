# Game-Mechanics Gap Register (companion to the program plan)

Companion to [`20261119-game-mechanics-plan.md`](20261119-game-mechanics-plan.md). One row per open gap. Style follows [`20261111-local-first-gap-register.md`](20261111-local-first-gap-register.md).

## Id namespace

Ids are always written with the `GM-` prefix. Blocks: GM-D docs, GM-K kernel, GM-B bridge, GM-S simulator.

The prefix keeps them apart from every other numbering in the repository:
- `LF-` ids (the local-first register) carry their own prefix, and the LF register's own release marker "S0" means sprint 0. A rung such as GM-S0 is never written without `GM-`.
- The multiplayer analysis's G1-G14 and the 2026-09-04 audit's G1-G10 are bare `G<number>` ids. A `GM-` id never drops its prefix.
- T0..T18 are v2.1 task ids. Rungs are `GM-S0..S7` for that reason.

Auth: A = autonomous · CF = confirm first · M = maintainer only · OS = needs OpenSpec. "(approved)" means covered by the maintainer's approval of the plan on 2026-11-19 (plan section 7); every later rung is confirmed again in its own OpenSpec change.

## Gap register (23 rows)

Phase 1 rows are off the release play path (`scripts/release_play_path.txt`) and carry release "now (pre-v2.1.0, off the play path)". Phase 1 PR order, at most two open at a time: GM-D1, GM-S0a, GM-K1, GM-K2, GM-B0, GM-B1, GM-B2, GM-B3, GM-B4, GM-B5. S-row releases are the plan's proposed slots; see "Capacity note".

| GM | Gap (source) | Rel | Auth | Owner | Acceptance |
|---|---|---|---|---|---|
| D1 | Docs claim systems that do not exist (tech tree in MODULE_REFERENCE, CLAUDE.md s7, CHANGELOG 0.14.1); no register for this program (plan 0.1a, 2) | now (pre-v2.1.0, off the play path) | A + OS | product-owner | `check_doc_sync`, `validate_kb`, `validate_openspec` exit 0 |
| S0a | No fingerprint of the default env, so "default stays byte-identical" is unchecked (plan 2) | now (pre-v2.1.0, off the play path) | A | qa-sdet | digest cannot change without an `ENV_VERSION` bump; negative control recorded |
| K1 | No data-driven rule definitions: new leaf package `alpha_zero/rules/` (plan 2.1) | now (pre-v2.1.0, off the play path) | CF + OS (approved) | dev | each validator rule has a failing fixture; hash stable under CRLF and key order; debt ratchet flat; coverage at or above 85% |
| K2 | No kernel CLI, GameInfo importer or pacing model (plan 2, 3.3) | now (pre-v2.1.0, off the play path) | A | dev | CLI exit codes tested; importer refuses to write inside the repository; pacing bands asserted |
| B0 | Loaded Lua bundle drifted from its module sources by 27 lines; no generator (plan 0.1c, 2) | now (pre-v2.1.0, off the play path) | CF (approved) | dev | first generated `AlphaZeroMain.lua` byte-identical to the committed one; the gate has a negative control |
| B1 | Lua is only text-parsed in tests (plan 2) | now (pre-v2.1.0, off the play path) | A (`lupa` test extra approved) | dev | exporter runs against stubbed `Players` / `GameInfo` |
| B2 | Exporter never fills researched techs, buildings, districts; research never offered; END_TURN never reports a block (plan 2.2) | now (pre-v2.1.0, off the play path) | CF (approved for the exporter fields and research actions; END_TURN reporting a blocked turn is **not** approved and is confirmed at its PR) + M (live check) | dev | plan 2.2 items tested on the harness; live checklist passed by the maintainer |
| B3 | Encoder tech and civic slots always zero; no tech registry (plan 0.1c, 2) | now (pre-v2.1.0, off the play path) | CF (additive wire fields approved; the change spans four bridge modules, which is **not** approved and is confirmed at its PR) | dev | encoder slots non-zero on a fixture; a missing registry warns and counts |
| B4 | No way to capture the real game's GameInfo tables for the kernel importer (plan 2) | now (pre-v2.1.0, off the play path) | A | dev | file lands only under `CIV6AZ_DATA_ROOT`; hygiene test keeps snapshots out of the repository |
| B5 | `--mcts` on the live bridge cannot work as wired: 84-channel network on 13-channel search, index decoded in the wrong space, errors become END_TURN, test mocks the search (plan 2.3; to be confirmed by a test first) | now (pre-v2.1.0, off the play path) | CF (approved) | dev | un-mocked test shows the failure, then the mode refuses with its gap id |
| B7a | The mod's commands cannot be carried out by the game: move sends an operation type that does not exist, skip / fortify / sleep are sent as commands under names the game's UI does not use, production sends a raw id; the next state is exported in the request's frame; the command window is 30 s from turn start (task plan `20261120-live-command-layer-plan.md`, E1-E3, E7, E8) | now (pre-v2.1.0, off the play path) | CF + OS (approved 2026-11-20) + M (live sitting) | dev | reverting any of E1, E2, E3, E7, E8 fails a named test; a refused operation answers with its reason; tests assert on the stand-in's game state, not on recorded calls |
| B7b | The exporter offers actions the game refuses: hand neighbour tables wrong for both row parities, hand legality rules, peace-time attacks, `CanProduce` by index (E4-E6); greedy production uses a per-process hash (E10) | now (pre-v2.1.0, off the play path) | CF + OS (approved 2026-11-20) + M | dev | every offered move target is one step away for both parities against an independent Python check; a production command carries the row's hash, the same in two processes |
| B7c | The loop can resend a refused command and counts commands as turns (E9); no tool sends one command and shows what changed | now (pre-v2.1.0, off the play path) | CF (approved 2026-11-20) | dev | a refused command is not sent twice in a turn; `--max-turns 3` plays three game turns; `scripts/live_command_check.py` prints the state difference |
| B6a | Search-mode translator: simulator action to live command (replaces the single B6 row; depends on B7a-c and the live sitting) | after the live sitting | CF + OS (approved 2026-11-20) | architect | every returned command is one the game offered; ids survive a reordered unit list; an 84x54 map at board 8; a root with no visible enemy has non-terminal children |
| B6b | Search-mode seam: a policy protocol that returns a command, the simulator network built from a checkpoint's provenance, an un-mocked end-to-end test; the `--mcts` refusal stays (lifting it is parked, unscheduled) | after B6a | CF + OS (approved 2026-11-20) | architect | end-to-end and spin tests pass; no training data is written (NG-1); play-path diff empty; debt ratchet passes |
| B8a | The mod exports nothing of other players (the export is off, and as written would miss barbarians and city-states, show stealth units, and call every other owner the enemy); with the flag off it still gave live facts about tiles in fog and offered attacks on hidden units | now (pre-v2.1.0, off the play path) | CF + OS (approved 2026-11-22) | architect | a foreign unit is exported only when the game reports its tile and the unit visible; fogged tiles carry no owner, improvement or resource; hostile owners only on the enemy side; `OBSERVE_ALL_PLAYERS` off by default until seen live; play-path diff empty |
| B8b | With an opponent seeded, the search's simulator reads killing the last visible enemy as winning the game | after B8a and B6b | CF + OS (approved 2026-11-22) | architect, ml-scientist | the acting player's win by elimination is not an end of the simulated game; own elimination still is; decision record counts what was seeded and left out |
| B9 | Four places where the mod goes ahead when it should refuse, in the default setup: the end-of-turn check answers "nothing blocks" when the game cannot be asked; a war-check answer that is not a list counts as "no war"; the declare-war command has no gate; attacks on barbarians are offered to an agent that is not shown them. Also `findAIPlayer` still uses a call the game's UI code never uses (five-role review, 2026-11-23) | before the next live session | CF (approved 2026-11-22 as "PR 2") | architect | each refuses, with a test on game state through the mod's Lua and a planted defect |
| B10 | The stand-in's defaults still model what the live session disproved (`UI.CanEndTurn()` false while blocked, one-step `CanProduce`, `CurrentlyBuilding()` returning the item, long `YieldTypes` keys); only the tests written for PR #166 switch the live behaviour on | with B9 | A (tests only) | qa-sdet | the live behaviour is the default and the old one a named override; the whole Lua suite passes |
| B11 | Nobody has measured how much of a real position the search's simulator can express (a board-8 network on a standard map: about 70 game tiles per cell) | zero games; needs the maintainer's `Lua.log` for a real answer | A (scripts and tests) | ml-scientist | census over exported states at board 8 and 16; stop rule fixed beforehand: median share of reachable offered commands under 0.3, or of own units with a translatable move under 0.5, at board 8 stops work on search-in-the-simulator |
| B8c | A city seen earlier and now in fog is forgotten (no last-seen memory); resources the player's technology does not reveal are exported on visible tiles; a parsed foreign unit defaults `can_attack` to true | not scheduled | CF | architect | registered |
| S0 | Preset seam and env hardening, no new rules (plan 3.1) | v2.2 (proposed) | CF + OS | architect | default golden fingerprint and determinism gates green; preset id and hash in provenance and refused on mismatch at the 5 load sites; defects in plan 3.1 fixed; clone-completeness test |
| S1 | `growth-v1`: cities work tiles, grow from yields (plan 3.2) | v2.2 (proposed) | CF + OS | dev | zero executor rejections over 200 seeded games; clone at most 1.25x and seconds per game at most 1.5x of default |
| S2 | `science-v1`: research, techs, science victory (plan 3.3) | v2.3 (proposed) | CF + OS | ml-sci | calibration bands met and hash pinned; forced-win oracle "one tech from victory" solved by every search variant; clone at most 1.5x, seconds per game at most 2x |
| S3 | `warfare-v1`: ranged attacks, city hit points and capture, walls (plan 4) | v3.0 | CF + OS | architect | OpenSpec archived; maintainer decision on `default` versus preset-only recorded |
| S4 | `civics-v1`: culture, civic tree, passive governments (plan 5) | later (M5 path) | CF + OS | architect | entry: `science-v1` probe verdict recorded |
| S5 | `districts-v1`: placement and adjacency (plan 5) | later (M5 path) | CF + OS | architect | entry: throughput budget holds |
| S6 | `world-v1`: resources, improvements, barbarians (plan 5) | later (M5 path) | CF + OS | architect | entry: search decision on chance recorded |
| S7 | `society-v1`: religion, great people, trade, diplomacy, eras (plan 5) | later (M5 path) | CF + OS | architect | entry: LF-B6, C1, C2, C3 done |

## State of the bridge rows on 2026-11-23

All of B0 to B8b are merged on the default branch, and their OpenSpec changes are archived into the `civ6-bridge` baseline. "Seen live" follows `docs/LIVE_CIV6.md`, "Status at a glance", which is the place to update.

| GM | Merged with | Seen live |
|---|---|---|
| B0, B1 | #148, #149 | n/a (build and test tooling) |
| B2 | #150, changed by #166 | export yes (2026-11-21); the blocked end of turn as changed by #166: no |
| B3 | #151 | `research_tech` answer seen (2026-11-21); the registry mode: no |
| B4 | #152, #154 | yes (2026-11-21) |
| B5 | #153 | n/a (a refusal) |
| B7a | #158 | answer seen for move, skip, fortify, found city, warrior build (2026-11-21); attacks, the war check, sleep: no |
| B7b | #159 | no |
| B7c | #160, #161 | no |
| B6a, B6b | #162, #163 | no (refused at start-up) |
| B8a, B8b | #164, #165 | no (off by default; refused at start-up) |

**Bridge features are frozen until A1 has run.** Open bridge work is B9 and B10 (safety and test truth before the next live session) and B11 (a measurement). B8c, lifting the `--mcts` refusal and turning `OBSERVE_ALL_PLAYERS` on stay unscheduled.

## Cross-reference to LF rows

The LF rows keep their own release and authority. This table records only which GM row carries or depends on them.

| LF row | GM relation |
|---|---|
| LF-B4 tie-break | GM-S0 adds the pure `score_outcome`; GM-S3 carries LF-B4 (v3.0); whether it changes `default` is the maintainer's decision at v3.0 planning |
| LF-B5 city capture | GM-S3 carries it |
| LF-B6 fog mask leak | Every GM-S rung depends on it (fog stays off until LF-B6 and LF-C3 land); GM-S0 notes the "fog" defect; GM-S7 entry |
| LF-C1 seed NPlayerMCTS | GM-S7 entry criterion |
| LF-C2 3-player run | GM-S7 entry criterion |
| LF-C3 wire DeterminizedMCTS | Every GM-S rung depends on it (fog off until it lands); GM-S7 entry |
| LF-H1 turn-cycle stall | GM-B2 tests the END_TURN blocker as a lead (a hypothesis, to confirm in the live checklist); GM-B7c-1 fixes a second lead, states discarded by the socket transport when they arrive with their ack (also a hypothesis until seen live); the live session of 2026-11-21 (PR #166) saw the leading cause once: an `end_turn` sent while a blocker was active, let through because `UI.CanEndTurn()` answers true, deactivates the turn. Fixed in the mod; the fix has not had a multi-turn run, so LF-H1 stays open |
| LF-H2 live session with a trained checkpoint | Consumes GM-B2 and GM-B3 (state parity); search mode has its translator and seam (GM-B6a, b) and stays refused until a trained checkpoint, a live check and the registered comparison exist |
| LF-H3 atomic IPC salvage | No GM row carries it; GM-B2 and GM-B3 touch the same bridge files, so coordinate at v2.3 |
| LF-H4 Unciv stub | No GM row |
| LF-H5 84-channel encoder | GM-B3 fills its tech and civic slots; the later full-scale path in plan section 5 depends on it |

## Capacity note

The spec slots GM-S0 and GM-S1 at v2.2, GM-S2 at v2.3 and GM-S3 at v3.0. The scrum-master review recommends GM-S0 after L0-L4 and before the A1-L evidence run, GM-S1 at v2.3, and GM-S2 at a v2.4 (or v2.3 if GM-S1 is dropped), because v2.2 and v2.3 are already full and review time is the binding constraint. Not applied here; the maintainer decides.

## Blocked on

- Maintainer: T0b, A1 and A1-R runs (GPU machine); the GM-S rungs wait for A1 and A1-R to be recorded and for v2.1.0 to ship.
- Maintainer: every PR merge, merges into qa and main, and tags.
- Maintainer: the live checklist in `docs/LIVE_CIV6.md` (its status table says what the session of 2026-11-21 covered). Attacks and the war check come first, after B9.
- Maintainer decisions (plan section 10): horizon for the `science-v1` probe; `enhanced` alias or removal; LF-B4 / LF-B5 on `default` or preset-only; Phase 1 merges before the v2.1.0 tag.
- GM-K1 -> GM-K2 and GM-B3; GM-B0 -> GM-B1 -> GM-B2 -> GM-B3, GM-B4. GM-S0a before any play-path change.
- Done and merged: GM-D1, S0a, K1, K2, B0 to B8b. Every GM-S rung needs its own OpenSpec change and task plan.
- Not blocked: B9, B10, B11 (each off the play path).
