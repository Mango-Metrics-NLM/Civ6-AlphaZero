# Game-Mechanics Gap Closure: Program Plan (science, technology, full gameplay)

**Date:** 2026-11-19 · **Session:** session_01CXusP9Ct9R7eg25YLcY7R7 · **Register:** [`20261119-game-mechanics-gap-register.md`](20261119-game-mechanics-gap-register.md)
**Status:** approved by the maintainer in the planning session of 2026-11-19 (four scoping decisions, then the plan). Role reviews are recorded in section 12.

**Goal:** close the gap between what the agent trains on and what decides a Civilization VI game: add the missing
game systems (science and technology first) to the training environment as opt-in presets, bring the real-game
bridge to parity with them, and correct the documentation that claims systems which do not exist.

**Architecture (one simulator, two uses).** A MicroCiv preset is the self-play training environment and also the
forward model that search uses at inference, when the bridge seeds a simulator state from the live game
(`BridgeMCTSEnv`, `alpha_zero/bridge/mcts_policy.py:58`). The working hypothesis is that simulator fidelity is what transfers to live
play (untested), and the bridge has two jobs: export the state those mechanics need, and accept the actions they produce.

**Scope rule:** this is a program plan. Only Phase 1 is cut into slices here. Every later rung gets its own
OpenSpec change and task plan, and is confirmed again there.

## 0. Context

### 0.1 What the documentation and the code say

| Kind of gap | Where it is recorded | State in code |
|---|---|---|
| (a) Systems missing from the training env | Claimed as existing: `docs/MODULE_REFERENCE.md` (`gameplay` row: "tech tree"), CLAUDE.md section 7, `CHANGELOG.md` 0.14.1 (`envs/micro_civ_tech.py`). Promised: `alpha_zero/mechanics/__init__.py:25-31`. In no register. | No tech tree, research, science yield or civics exists anywhere in Python. `alpha_zero/gameplay/` (growth, economy, combat, victory) is real but reachable only through `EnhancedMicroCivEnv`, which the local training path never builds and which is defective (section 3.1). `alpha_zero/mechanics/` (11 managers) is standalone, mostly dead and unfit for search (no serialisation in five managers, own `random.Random` instances, a `DeprecationWarning` on import, coverage-omitted): a content reference only. |
| (b) Registered rule gaps | `kb/specs/20261111-local-first-gap-register.md`: LF-B4 tie-break, LF-B5 city capture, LF-B6 fog mask leak, LF-C1..C3 | Open, slotted at v2.3 / v3.0, CONFIRM-FIRST + OpenSpec |
| (c) Real-Civ6 bridge parity | LF-H1..H5; `docs/LIVE_CIV6.md` "Known limitations" | The Lua exporter never fills `techs_researched`, city `buildings` or `districts`; research is executable but never offered; the 84-channel encoder's tech and civic slots are always zero; `--mcts` on the live bridge cannot work as wired; the loaded Lua bundle has drifted from its source files (27 lines) |

### 0.2 How this plan was made

Three read-only surveys (documented gaps; mechanics code inventory; constraints on environment changes), external
research (section 11), four scoping questions answered by the maintainer, then an independent design review that
verified the claims below against the code (138 tool calls). Two claims come from reading, not from execution,
and their slices start by confirming them with a test: the growth stall (section 3.1) and the `--mcts` failure
(GM-B5). The A1 import closure (61 modules) and the Lua bundle comparison were computed.

### 0.3 Decisions (maintainer, 2026-11-19)

1. **Scope:** all three kinds of gap, as one staged program, plus doc corrections.
2. **Approach:** opt-in MicroCiv presets. Fix and extend `EnhancedMicroCivEnv` and `alpha_zero/gameplay/`. The
   `default` preset stays byte-identical. No external simulator and no charter carve-out (NG-1 stands).
3. **Fidelity:** a small, original, Civ6-inspired rule set in the repository, with a data-driven engine so the
   real tree can be loaded from a local export of the maintainer's own game's GameInfo tables (never committed).
4. **Timing:** nothing on the release play path changes until v2.1.0 ships. Until then: planning artifacts, doc
   fixes, bridge parity and a rules kernel outside the play path.

### 0.4 Constraints

- **Release play path.** `scripts/release_play_path.txt` covers `alpha_zero/envs/**`, `alpha_zero/gameplay/**`,
  `alpha_zero/evaluation/**`, `core/civ6_actions.py`, `core/civ6_network.py`, `training_civ6.py` and more. The
  release verifier requires an empty `git diff --name-only` over those pathspecs between an evidence run's
  commit and the release candidate, so adding any file under those globs (even an `Agent.md`) makes evidence
  stale. `alpha_zero/bridge/**`, `mod/**` and `tests/**` are not on it, and the bridge is outside the A1 import
  closure. T0b, A1 and A1-R have not run.
- **Debt gates.** `alpha_zero/envs/microciv.py` is 988 lines against a 1,000-line ratchet. A new module is dead
  until a non-test module other than its own package `__init__` imports it (entry points excepted).
- **Shapes.** 13 channels / 64 vector / 376 actions are literals in about 90 test files and in the frozen cloud
  sites; action totals of 356 and 381 are policed literals and must be avoided in any preset.
- **Determinism.** The gates compare run against run: every draw uses the env RNG, `clone()` carries all state,
  terrain stays immutable (it is shared by reference between clones).
- **Budget and governance.** 1,000 games per job. Rule changes are CONFIRM-FIRST and need an OpenSpec change.

## 1. Naming and tracks

| Track | What | Starts |
|---|---|---|
| GM-D docs | Correct the false claims; register this program's gaps | Now |
| GM-K kernel | Data-driven rule definitions, validator, registry, GameInfo importer, outside the play path | Now |
| GM-B bridge | Lua export of techs, civics, buildings; research actions; converter and search fixes (LF-H1..H5) | Now |
| GM-S simulator | Rungs GM-S0..S7 on `EnhancedMicroCivEnv` + `gameplay/` (folds in LF-B4, B5, B6, C1..C3) | After v2.1.0 |

- Ids are always written with the `GM-` prefix. Rungs are `GM-S0..S7`, because T0..T18 are v2.1 task ids.
- "Ruleset" already means GitHub branch rulesets in this repository. Prose says "game rule set", and the one
  selector stays `gameplay_preset`, the term `docs/EVALUATION.md` section 7.1 registers.

## 2. Phase 1: now, with an empty play-path diff

| Id | Change | Main files | Done when |
|---|---|---|---|
| GM-D1 | Truth pass and tracking | `docs/MODULE_REFERENCE.md`; CLAUDE.md sections 1.3 and 7; `CHANGELOG.md` (dated erratum, history intact); `docs/LIVE_CIV6.md`; `ROADMAP.md`; new `docs/GAME_RULES.md`; this spec and its register; two OpenSpec changes | `check_doc_sync`, `validate_kb`, `validate_openspec` exit 0 |
| GM-S0a | Golden fingerprint of the default env (tests only) | `tests/regression/test_default_env_golden_regression.py`, fixture under `tests/fixtures/golden/` | Seeded legal play at board 8, 30 and 60 turns, 2 and 3 players; integer game state only (no float tensors); the digest cannot change without an `ENV_VERSION` bump; negative control recorded |
| GM-K1 | Kernel core: new leaf package `alpha_zero/rules/` | `spec.py`, `loader.py`, `validate.py`, `queries.py`, `registry.py`, `builtin.py`, `__main__.py` (`validate`, `hash`; the entry point is the package's live importer, so it ships here), `data/science_v1.yaml`, `Agent.md`; `pyproject.toml` package-data; `docs/MODULE_REFERENCE.md` | Every validator rule has a failing fixture; hash stable under CRLF and key order; debt ratchet flat; coverage at or above 85% |
| GM-K2 | Kernel tooling | `gameinfo.py`, `pacing.py`; `pace` and `import-gameinfo` added to `__main__.py` | CLI exit codes tested; the importer refuses to write inside the repository; pacing bands asserted |
| GM-B0 | Generated Lua bundle | new `scripts/build_lua_bundle.py`, `tests/scripts/test_lua_bundle_sync.py`; module sources reconciled to the bundle | The first generated `AlphaZeroMain.lua` is byte-identical to the committed one, so live behaviour does not move; the 27 differing source lines are listed in the PR as candidates for GM-B2; the gate has a negative control |
| GM-B1 | Lua test harness | `lupa` in its own test extra installed by CI; `tests/fixtures/lua/civ6_stub.lua` | The exporter runs against stubbed `Players` / `GameInfo` (today Lua is only text-parsed) |
| GM-B2 | Exporter parity | `mod/AlphaZeroAI/Scripts/StateExporter.lua`, `ActionExecutor.lua`, bundle | Section 2.2 items tested on the harness (wiring only); live behaviour confirmed by the maintainer's checklist |
| GM-B3 | Python wiring | `alpha_zero/bridge/protocol.py` (additive fields only), new `bridge/tech_registry.py`, `bridge/state_converter.py`, `bridge/action_converter.py` | Encoder tech and civic slots are non-zero on a fixture; a missing registry warns and counts |
| GM-B4 | GameInfo snapshot | Lua `export_gameinfo`; new `alpha_zero/bridge/gameinfo_capture.py` | The file lands only under `CIV6AZ_DATA_ROOT`; a hygiene test keeps snapshots out of the repository |
| GM-B5 | Search proxy and `--mcts` | new `alpha_zero/bridge/microciv_proxy.py`; `bridge/mcts_policy.py`; `alpha_zero/civ6_local.py` | An un-mocked search test shows the failure, then the mode refuses with its gap id until an action translator exists |

PR order: GM-D1, GM-S0a, GM-K1, GM-K2, GM-B0, GM-B1, GM-B2, GM-B3, GM-B4, GM-B5, in two lanes (D1, K1, K2) and
(B0, B1, B2, B3, B4), with S0a and B5 as fillers. At most two open, counted inside the existing cap of three open
agent PRs; Phase 1 yields to the v2.1 tasks when review time is short. Dependencies: K1 before K2 and B3; B0, B1,
B2, B3 in order; B2 before B4; K2's importer and B4's snapshot share one file format, fixed by whichever lands
first. Negative controls are recorded in each PR body.

### 2.1 Kernel (`alpha_zero/rules/`)

- **Leaf package.** Standard library plus `yaml`; no `alpha_zero` imports, no torch, no per-game state.
- **Modules.** `spec.py`: frozen `TechSpec`, `UnitSpec`, `BuildingSpec`, `ModifierSpec`, `VictorySpec`,
  `RulesetSpec`, integers only. `loader.py`: strict `safe_load`, unknown key is an error, `canonical_json()`,
  `content_sha256()`. `validate.py`: unique ids, prerequisites resolve, acyclic, capstone reachable, costs
  positive, unlock references resolve. `queries.py`: prerequisite closure, `available(spec, researched)`,
  `research_cost(spec, tech, boosted)` in integer percent. `registry.py`: name-to-index maps in declared order,
  with a fingerprint. `builtin.py`: id to resource plus pinned hash; loading raises on a mismatch.
- **Data.** The committed rule set is package data (`alpha_zero/rules/data/*.yaml` in
  `[tool.setuptools.package-data]`), because `configs/` is not installed into the pinned wheel that run cards use.
- **Identity.** `(preset id, sha256 of the canonical parsed form)`, so CRLF on the Windows workstation cannot
  change it. A rule change is a new id (`science-v2`). An append-only log in `docs/GAME_RULES.md` is held equal
  to the pinned hashes by a test, as the A1 registration digest is.
- **Gates.** Live importers now are its own CLI and the bridge registry, so `dead_modules` stays flat and the A1
  closure is untouched. It needs an `Agent.md` and a `docs/MODULE_REFERENCE.md` row (submodule count 30 to 31).
- **After v2.1.0.** Add per-file play-path pathspecs for the kernel modules `gameplay/science.py` imports and for
  `alpha_zero/rules/data/**`, not the whole directory, so a CLI edit cannot stale evidence.
- **Rejected homes.** `mechanics/` (omitted, warns, eager imports); `gameplay/rules/` (on the play path today);
  `bridge/` (wrong dependency direction later); `configs/` (not installed; not seen by the closure test).

### 2.2 GM-B2 exporter changes

Every new call is `pcall`-guarded. Reference for calls proven over FireTuner: civ6-mcp `src/civ_mcp/lua/tech.py`
(MIT). The mod loads `Scripts/AlphaZeroMain.lua`, a bundle of the module files in the InGame UI context
(`mod/AlphaZeroAI/AlphaZeroAI.modinfo`), so every Lua change lands in the module file and, through GM-B0, in the bundle.

- Researched techs and civics as type names (`GetTechs():HasTech(idx)`, `GetCulture():HasCivic(idx)` over
  `GameInfo.Technologies()` / `GameInfo.Civics()`), for the local player. Today `techs_researched` is always
  empty and civics are filled only if `GetResearchedCivics` exists.
- Current research with cost, progress and boost flag, as additive fields; per-city buildings and districts
  (hard-coded empty today); a `schema_version`.
- Research offered as legal actions from `CanResearch`. Execution exists (`executeResearchTech`,
  `executeResearchCivic`) and Python can convert it (`bridge/action_converter.py`), but the handlers call only
  the GameCore setters; add `UI.RequestPlayerOperation` with the setter as fallback.
- END_TURN reports a blocked turn instead of returning success unconditionally, and the state carries the
  end-turn blocking type. The bundle has no reference to `EndTurnBlocking`, `NotificationManager` or
  `PlayerOperations`, and an unanswered research choice blocks end-turn in the real game. This is a lead for
  the turn-cycle stall (LF-H1), to confirm in the maintainer's live checklist; it is a hypothesis, not a finding.

### 2.3 GM-B5 detail

With `--mcts`, `alpha_zero/civ6_local.py` hands `Civ6MctsPolicy` the 84-channel network with the hierarchical
action head. The search evaluates it on the simulator's 13-channel observations; the returned index is in the
simulator's 376-action space but is decoded as a hierarchical action; any exception becomes END_TURN
(`bridge/agent_loop.py`); and the unit test mocks the search (`tests/unit/test_bridge_mcts_policy.py`). The
proxy fix maps unit classes and impassable terrain, so that settlers can found cities and mountains block
movement inside the search. Until an action translator exists (registered as a later GM-B row), the mode
refuses at start-up with its gap id.

## 3. Phase 2: after v2.1.0 ships (play path; one OpenSpec change per rung)

Rules for every rung: fog off until LF-B6 and LF-C3 land; integer expected-damage combat with no dice; settler
auto-founding off (the base env draws for it in `_process_settlers`); yields resolved for all players at round
end, which removes one source of seat bias (move order still differs; the probe measures the rest); exact score
ties are draws.

### 3.1 GM-S0. Preset seam and hardening: no new rules (v2.2)

- **One factory.** `create_game_env` (`alpha_zero/envs/env_factory.py`) gains `num_players`, `fog_of_war` and a
  preset registry. Callers must declare they are preset-aware, so the dormant Vertex workers, web and inference
  refuse new presets without being edited. The three `MicroCivEnv(...)` sites in `alpha_zero/training_civ6.py`,
  `alpha_zero/evaluation/benchmark.py`, `alpha_zero/cloud/gcp/arena.py` (the local judge) and
  `scripts/calibrate_throughput.py` move onto it. `--gameplay_preset` on the trainer and the benchmark.
- **Provenance and refusal.** `build_provenance` (`alpha_zero/utils/checkpoint.py`) records the preset id and
  hash; a mismatch is refused at the five load sites beside `HorizonMismatch`. It ships with the seam, because
  two presets with equal shapes would otherwise cross-load silently.
- **Shapes.** Indices 0-375 and channels 0-12 keep their meaning in every preset. New action blocks and planes
  are appended, so default weights transfer as a prefix. Free vector slots are 26-63 (12-25 are used with more
  than two players). Fix `MicroCivNetwork`'s fixed `vector_dim=64` and the heuristic agent's "produce is the
  last block" assumption (`alpha_zero/evaluation/baseline_agents.py`).
- **Stop the copy-paste overrides.** Move the dataclasses, rendering and map generation out of `microciv.py`
  and add small hooks whose defaults are today's code. Base `clone()` owns the base copy, RNG included. A
  reflective test requires every env attribute to be declared shared-immutable or copied. Proof of no change:
  the GM-S0a golden fingerprint and the determinism gates stay green.
- **Defects to fix in `alpha_zero/envs/microciv_enhanced.py`:**
  - no `num_players` argument; `step`, attack and production copied from the base with literals and
    `1 - current_player`; a two-player turn-limit fallback and render;
  - growth works the centre tile only, so by arithmetic from the defaults a plains city stops at population 2
    (confirm with a test first);
  - `GameplayConfig.victory` is ignored (`VictorySystem(config=None)`);
  - `clone()` drops the RNG and the player count, deep-copies terrain and round-trips state through dicts;
  - `TerrainSystem` is built and never called; two config families are translated by hand;
  - `alpha_zero/gameplay/combat.py` creates an unseeded RNG per call;
  - `check_score_victory` (`alpha_zero/gameplay/victory.py`) gives exact ties to the first maximum; replace it
    with one pure `score_outcome(scores) -> winner | None`;
  - "fog" hides nothing (every unit is still encoded) and duplicates `alpha_zero/multiplayer/fog_of_war.py`.
- `enhanced` stays as a deprecated alias; removing it is a breaking change for a later decision.

### 3.2 GM-S1. Growth: preset `growth-v1` (v2.2, experimental)

Cities work the centre tile plus one ring-1 tile per population in a fixed priority order (no citizen actions),
grow from food and produce from tile yields; score decides at the turn limit. Actions stay 376; four planes are
added (tile food, tile production, own and enemy city population). Gold is deferred until it has a sink
(rush-buying): `EconomySystem.purchase` is never called today.

### 3.3 GM-S2. Science and technology: preset `science-v1` (v2.3)

- New `alpha_zero/gameplay/science.py`: a `ScienceSystem` over the kernel, with an explicit `copy()` over
  integer and tuple state (not a dict round-trip, which is the clone-cost defect).
- Research is an appended action block, legal only when no research is in progress; building is legal only on
  an empty queue; END_TURN stays legal, so a turn always terminates. Mask and executor call the same predicates.
- City science = 2 + population // 2 + buildings; overflow carries. Techs gate units, buildings and modifiers.
- A single capstone finisher wins; several in one round are decided by score; otherwise score at the turn limit
  (`VictoryConfig.score_tech_value` already exists).

| Era (first-pass cost) | Tech (prerequisites) and what it unlocks |
|---|---|
| I (20) | agriculture: Granary +2 food · mining: Workshop +2 production · writing: Library +2 science |
| II (45) | irrigation (agriculture): +1 food per worked plains · bronze_working (mining): Swordsman · mathematics (writing): +1 science per Library · construction (mining): +1 production per worked hills |
| III (90) | education (mathematics): University +4 science · engineering (mathematics, construction): population cap +2 · steel (bronze_working, engineering): Knight · astronomy (education): Observatory +3 science |
| Capstone (180) | scientific_theory (astronomy, engineering): science victory |

- Shortest path to the capstone: 580; full tree: 780. Names are generic and the structure is original.
- Archery joins with ranged attacks in GM-S3; until then the archer is strictly weaker than the warrior.
- Boosts (40% of cost on a trigger, as in Civ6) stay in the schema, tested, and are switched on as `science-v2`:
  they need event hooks in combat, founding and growth, and would confound the first learnability probe.

**Calibration, not guesswork.** The protocol is the pre-registration template in `docs/GAME_RULES.md` section 3
(authored by the ML-scientist review, which found the first draft's bands passable by construction):
(1) `python -m alpha_zero.rules pace`, a zero-game timeline model; (2) exploratory sweeps on calibration seeds
only, tuning the era and capstone costs one factor at a time; (3) freeze the hash and the scripted agent and log
the rule set; (4) one confirmation on held-out seeds, every band read on an interval bound, not a point estimate.
Bands: capstone timing at 0.60-0.70 of `max_turns`; scripted agent against random at least 0.90; random against
random draws at most 0.10 and capstone finishes at most 5%; seat balance by an equivalence test.

### 3.4 Acceptance and probes for every rung

- Zero executor rejections over 200 seeded games for each new action block; run-versus-run determinism.
- Clone at most 1.25x (GM-S1) or 1.5x (GM-S2) of the default's cost, and seconds per game at most 1.5x or 2x at
  an equal horizon, measured with `scripts/calibrate_throughput.py` and recorded under `docs/evidence/`. These
  are ratios until T0b gives absolute numbers.
- For a rule set that declares a capstone: a "one tech from victory" oracle that every search variant
  solves, with the vacuity guards of `tests/unit/test_forced_win_oracle.py`. A rule set without a capstone
  (`growth-v1`) names its own known-answer position in its OpenSpec change.
- Probes, each its own run card under 1,000 games on existing runners (no new cost model): random against
  random (600 games, 300 pairs: draws and seat balance; the seat-0 score needs a small tool over the per-game
  records, since the paired design cancels seat bias out of the headline score); a scripted positive control (3 x 200) from a new, measured
  `alpha_zero/evaluation/ruleset_agents.py`; an A1-shaped training card, pre-registered in `docs/GAME_RULES.md`
  and never inside the A1 markers of `docs/EVALUATION.md`.
- Training cards run only after A1 and A1-R are recorded, and are confirmatory only if both PASS. A scripted
  control shows the rules reward a strategy, not that self-play finds it. Running another rule set as a remedy
  after an A1 FAIL is a separate question that must be pre-registered before A1's verdict is seen; it cannot
  change that verdict.

## 4. Phase 3: v3.0

**GM-S3. Warfare: preset `warfare-v1`.** Range-2 attacks and Archery, city hit points and capture (LF-B5),
walls. LF-B4 is `score_outcome`. Whether LF-B4 and LF-B5 also change `default` (an `ENV_VERSION` bump, as the
local-first program plan assumes) or stay preset-only is the maintainer's decision at v3.0 planning. Preset-only
keeps the default game and every A1-era checkpoint valid, and the v3.0 registration then names its preset.

## 5. Later, toward M5 ("full gameplay"): outline

| Rung | Adds | Entry criteria |
|---|---|---|
| GM-S4 `civics-v1` | Culture and a civic tree on the same kernel; governments as passive modifiers | `science-v1` probe verdict recorded |
| GM-S5 `districts-v1` | Placement in per-game state, adjacency | Throughput budget holds |
| GM-S6 `world-v1` | Resources, improvements, barbarians: the first stochastic events | A search decision on chance (chance nodes or determinisation), because clones copy the RNG and the search would otherwise foresee the dice |
| GM-S7 `society-v1` | Religion, great people, trade, diplomacy, eras | LF-B6, C1, C2, C3 done |

**End state (a ROADMAP-level aim; the charter's M5 text is unchanged and no amendment is proposed).** Every Civ6 system class has a MicroCiv counterpart at an agreed abstraction, bridge parity for its
state and actions, and a passed probe. Live play comes first by search inside the abstraction (with an action
translator), later through the full-scale encoder and action space (LF-H5). It is not a rules-exact clone.
Named research problems, not promised here: map scale and the 20-unit / 5-city caps, 300-turn horizons, partial
observability, general-sum diplomacy.

## 6. Reuse

| Need | Existing code |
|---|---|
| Preset seam | `create_game_env`, `GAMEPLAY_PRESETS` (`alpha_zero/envs/env_factory.py`); `GameplayConfig` flags (`alpha_zero/gameplay/config.py`) |
| Growth, yields, victory | `CityGrowthSystem`, `TerrainSystem`, `VictorySystem` (`alpha_zero/gameplay/`), after the GM-S0 fixes |
| Research on the real game | `executeResearchTech` / `executeResearchCivic` (`mod/AlphaZeroAI/Scripts/ActionExecutor.lua`); `_convert_tech_action` (`alpha_zero/bridge/action_converter.py`) |
| Refusal pattern | `HorizonMismatch`, `build_provenance` (`alpha_zero/utils/checkpoint.py`) |
| Digest-pinned registration | `alpha_zero/evaluation/a1_protocol.py` and its regression test, as the model for `docs/GAME_RULES.md` |
| Test patterns | `tests/unit/test_action_mask_collision.py`, `tests/unit/test_microciv_clone.py`, `tests/unit/test_forced_win_oracle.py`; `.claude/skills/negative-control` |
| Measurement | `scripts/calibrate_throughput.py`; `docs/evidence/draw_rate_sweep/draw_rate_sweep.py`; `scripts/run_card.py CARD --price_only` |

## 7. Confirm-first items (CLAUDE.md section 3.2)

**Covered by the maintainer's approval of this plan (Phase 1):** the new top-level package and package-data
entry; generating the loaded Lua bundle (byte-identical first), then the GM-B2 changes to live mod behaviour;
additive bridge wire fields; `--mcts` on the live bridge refusing until a translator exists (a documented
command); the CHANGELOG erratum; `lupa` as a test-only extra.

**Confirmed again in their own OpenSpec change (later):** the seam (more than three modules); deleting the
copied Enhanced code (over 100 lines); deprecating `enhanced`; every new rule; preset-derived network widths;
the v3.0 default-rule question; stochastic events in search.

**Found by the architect review and not covered by the plan approval; each is confirmed by the maintainer at its
PR:** END_TURN reporting a blocked turn (GM-B2, a behaviour change of an existing acknowledgement); GM-B3 spanning
four bridge modules. For GM-S0: fixing `MicroCivNetwork`'s fixed `vector_dim` (network architecture), the
provenance refusal at five load sites (breaks existing behaviour), and editing `alpha_zero/cloud/gcp/arena.py`
(the frozen Vertex path and the promotion judge). For GM-S2: editing `scripts/release_play_path.txt` (a release gate).

**No charter amendment.** MicroCiv stays the sole training environment (NG-1) and the mod stays read-only on
official surfaces (NG-3). No distributed-coordination change, no new external service, no run over 1,000 games.

## 8. Verification

- Every Phase 1 PR has an empty play-path diff:
  `git diff --name-only $(git merge-base origin/<default-branch> HEAD) HEAD -- $(grep -v '^#' scripts/release_play_path.txt | sed '/^$/d; s/^/:(glob)/')`
- Gates: `make verify`; `pytest tests/scripts/test_release_play_path_closure.py tests/regression/test_debt_ratchet.py tests/regression/test_suite_wiring_regression.py`; `make validate`.
- Docs: `python scripts/check_doc_sync.py`, `python scripts/validate_kb.py`, `python scripts/validate_openspec.py --change <id>`.
- Kernel: `python -m alpha_zero.rules validate science-v1`, `hash`, `pace`.
- Bridge: `python scripts/build_lua_bundle.py --check`, the Lua-harness and fixture tests,
  `python -m alpha_zero.civ6_local --mode civ6 --dry-run`.
- Live (maintainer checklist): reinstall the mod, issue one research choice, see one reported end-turn blocker,
  take one GameInfo snapshot.
- After v2.1.0: the golden fingerprint, `pytest tests/regression/test_training_determinism_regression.py`, the
  200-game mask/executor test per preset, the calibration sweeps, `--price_only` on each probe card.
- Negative controls are recorded for the golden test, the bundle gate, the clone-completeness test and the tie rule.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Richer rules are not learnable within 1,000 games | One mechanic family per rung; training cards gated on A1; the scripted control only shows a signal exists, so this risk stays open until a training card runs |
| Evidence goes stale | Empty play-path diff in Phase 1, checked per PR; later, per-file pathspecs for the kernel modules the env imports |
| Clone cost grows | Integer and bitmask state, shared immutables, explicit `copy()`, ratio budgets |
| Shape changes ripple through tests | Default stays 13 / 64 / 376 and the golden test proves it; indices and channels only ever append |
| Mask and executor disagree | Shared predicates; the 200-game no-rejection test parametrised over presets |
| Game-data licensing | Only an original rule set is committed; snapshots hold identifiers and numbers, live under the data root and are kept out of the repository by a test |
| Simulator differs from real Civ6 | Data-driven engine plus the GameInfo import; the abstraction and scale gaps are registered, not hidden |
| Lua bundle drift | Generated bundle plus a sync gate |
| `science-v1` without city capture is mostly a race | Accepted for that rung; GM-S3 adds the military counter |

## 10. Open decisions (none blocks Phase 1)

| Decision | Default until decided | Decided in |
|---|---|---|
| Horizon the `science-v1` probe registers (60 as A1, or 80 as the first pacing pass assumed) | Calibrate at 60; move only if the bands cannot be met | The probe's pre-registration |
| `enhanced`: deprecated alias, or removed | Deprecated alias | GM-S0 OpenSpec change |
| LF-B4 / LF-B5 change `default`, or stay preset-only | Preset-only is recommended | v3.0 planning |
| Bridge action translator for search mode | `--mcts` refuses until it exists | A later GM-B row |
| A recorded live payload as a committed fixture | Synthetic fixtures only; no real tables or names lists committed | GM-B3 |
| Phase 1 PRs merging before the v2.1.0 tag | Allowed: they do not touch the play path | The maintainer, at merge |

## 11. Sources

- CivRealm, ICLR 2024 (Freeciv, Gymnasium API): technology, government and diplomacy as separate observation and
  action groups; mini-games with full-game I/O for curriculum. https://arxiv.org/abs/2401.10568
- Terra Nova (MIT, JAX, Civ V-inspired): data-driven tech rows `{id, name, cost, era, prereq}`; four victory
  types. https://arxiv.org/abs/2511.15378, https://github.com/trevormcinroe/terra_nova
- Tribes, AIIDE 2020 (a Polytopia re-implementation built for MCTS research): a 24-tech tree as a scale
  reference. https://ojs.aaai.org/index.php/AIIDE/article/view/7438
- CivBench / civ6-mcp (MIT): real Civ6 over the same FireTuner protocol, with working Lua for researched techs,
  costs, progress, boosts, prerequisites, setting research and end-turn blockers.
  https://arxiv.org/abs/2609.02459, https://github.com/lmwilki/civ6-mcp
- Sampled MuZero (large action spaces): https://arxiv.org/abs/2104.06303
- Civ6 boost rule (40% since Rise and Fall): https://forums.civfanatics.com/threads/rise-and-fall-eurkas-nerfed-to-40.627427/page-2
- Gymnasium registry documentation (variants by id and kwargs).

Not reachable from the planning session: `civilization.fandom.com`, `yairm210.github.io`. Exact Civ6 formulas
were therefore not taken from the wiki; real numbers come from the game's own GameInfo tables through GM-B4.

## 12. Review record

Planning handoff of 2026-11-19. Each role reviewed this spec within its remit and authored the text named; the
findings below are folded in above.

| Role | Authored | Findings folded in |
|---|---|---|
| sdlc-product-owner | Appendix A; the doc corrections; ROADMAP entries | In scope, no non-goal or invariant touched. The end state is a ROADMAP-level aim, not a charter change. Acceptance criteria name a command or test; negative controls are recorded in the PR body. |
| sdlc-scrum-master | The gap register | Review time is the binding constraint: Phase 1 counts inside the three-PR cap. GM-K1 would merge dead without its entry point, so `__main__.py` moved into GM-K1. v2.2 and v2.3 are already full: the rung slots here are proposals and the register records the alternative (GM-S1 at v2.3, GM-S2 later) for the maintainer. GM-B2 needs the live checklist. |
| sdlc-ml-scientist | `docs/GAME_RULES.md` | Bands must be read on interval bounds; calibration and confirmation seeds must be disjoint and the hash frozen first; the seat band is an equivalence test on 600 games; a remedy rule set after an A1 FAIL is a forking path unless pre-registered; three claims softened (seat bias, learnability, fidelity transfer). |
| sdlc-architect | The two OpenSpec changes | GM-K1 needed its entry point to stay live (done). GM-S0a gets its own OpenSpec change (one requirement on `training-reproducibility`), opened with its PR, and its fixture records the numpy version. The blocked END_TURN is a protocol behaviour change, not additive: it ships with its Python handling in one PR. Section 7 gained the items it missed. GM-B0 pins Lua line endings; the Lua harness uses a 5.1 runtime and fails, not skips, in its CI job; the GM-B5 proxy uses existing MicroCiv rules only. |

## Appendix A. User stories and acceptance criteria (Phase 1)

Each criterion is decided by a named command or test. Where this spec does not name a test file, the PR names it, and the criterion then reads "the test named in the PR". "Play-path diff" below means the command in section 8.

**GM-D1 (truth pass and tracking).** As a contributor, I want the documents to stop claiming systems that are not in the tree, and the program tracked, so that I can trust what I read.
- `python scripts/check_doc_sync.py`, `python scripts/validate_kb.py` and `python scripts/validate_openspec.py --change add-game-rules-kernel` (and `--change add-bridge-research-parity`) exit 0.
- The play-path diff (section 8) prints nothing.
- `docs/GAME_RULES.md` exists and is linked exactly once from `docs/README.md` (check_doc_sync D6 and D8).
- Each of the two ROADMAP "Now" bullets names its `openspec/changes/<id>` (D9).
- `git diff` of `CHANGELOG.md` shows additions only, below the 0.14.1 entry and under `[Unreleased]`; no existing line is changed or removed.
- No file under `alpha_zero/` is modified.

**GM-S0a (golden fingerprint).** As a maintainer, I want a fingerprint of the default env, so that every later change can prove the default game did not move.
- `pytest tests/regression/test_default_env_golden_regression.py -v` passes for board 8, 30 and 60 turns, 2 and 3 players, seeded legal play.
- The digest covers integer game state only (a test asserts no float tensor enters it).
- A test fails when the committed digest is changed without an `ENV_VERSION` bump.
- The negative control (an edit to env behaviour that makes the test fail) is recorded in the PR body.
- Play-path diff is empty.

**GM-K1 (kernel core).** As a rules author, I want a validated, hashed, data-driven rule definition, so that a rule set has a stable identity.
- `python -m alpha_zero.rules validate science-v1` and `python -m alpha_zero.rules hash science-v1` are exit 0 (the CLI itself ships in GM-K2; until then the same checks run through the test named in the PR).
- One failing fixture per validator rule (unique ids, prerequisites resolve, acyclic, capstone reachable, costs positive, unlock references resolve), in the test named in the PR.
- The hash is equal for CRLF and LF input and for reordered keys (test named in the PR).
- Loading a built-in rule set with a wrong pinned hash raises.
- `pytest tests/regression/test_debt_ratchet.py -v` passes with the baseline unchanged.
- `pytest tests/scripts/test_release_play_path_closure.py -v` passes, and the play-path diff is empty.
- `python scripts/check_doc_sync.py` exits 0 with the new `docs/MODULE_REFERENCE.md` row.
- `pytest tests/ --cov=alpha_zero` reports at least 85%.

**GM-K2 (kernel tooling).** As a rules author, I want a CLI and a GameInfo importer, so that I can validate, hash, pace and import from my own game without committing its data.
- Exit codes of `validate`, `hash`, `pace` and `import-gameinfo` are asserted by tests (0 pass, non-zero on a bad input).
- `import-gameinfo` exits non-zero and writes nothing when the output path is inside the repository.
- The pacing test asserts the `science-v1` bands in section 3.3 on the model's output.
- `python -m alpha_zero.rules pace` runs zero games.
- Play-path diff is empty.

**GM-B0 (generated Lua bundle).** As the maintainer, I want `AlphaZeroMain.lua` generated from the module sources, so that source and loaded code cannot drift.
- `python scripts/build_lua_bundle.py --check` exits 0 and `pytest tests/scripts/test_lua_bundle_sync.py -v` passes.
- `git diff` on `mod/AlphaZeroAI/Scripts/AlphaZeroMain.lua` is empty after the first generation (byte-identical).
- The PR body lists the 27 differing source lines.
- Negative control: editing a module source without regenerating makes `--check` exit non-zero (recorded in the PR body).

**GM-B1 (Lua test harness).** As a bridge developer, I want the exporter to run against stubbed game objects, so that Lua behaviour is tested, not text-parsed.
- `lupa` is declared in its own test extra and CI installs it.
- `tests/fixtures/lua/civ6_stub.lua` exists, and a test named in the PR runs the exporter against it and asserts one decoded field.
- That test is collected by a CI job (`pytest tests/regression/test_suite_wiring_regression.py -v` passes).

**GM-B2 (exporter parity).** As the agent, I want the real game's research state exported and research offered, so that live play can see and use it.
- Under the GM-B1 harness, with a stubbed player that has researched techs and civics, the exported state lists them by type name; with the calls absent, the exporter still returns (each new call is `pcall`-guarded).
- The state carries current research with cost, progress and boost flag, per-city buildings and districts, and a `schema_version`.
- Research choices appear in the legal actions when `CanResearch` is true, and not otherwise.
- `executeResearchTech` and `executeResearchCivic` call `UI.RequestPlayerOperation` and fall back to the setter when it fails (stub tests).
- A blocked end-turn is reported as blocked and the state carries the blocking type (stub test).
- `python scripts/build_lua_bundle.py --check` exits 0 after the change.

**GM-B3 (Python wiring).** As the agent, I want the encoder to receive the exported research state, so that its tech and civic slots carry information.
- `protocol.py` changes are additive: a test parses a payload without the new fields.
- With `bridge/tech_registry.py`, the state converter fills non-zero tech and civic slots on a synthetic fixture (test named in the PR; no real tables or name lists committed).
- With no registry, the converter logs a warning and increments a counter, and the test asserts both.
- The action converter maps a research action from the new legal-action entry.
- `pytest tests/unit -k bridge` passes.

**GM-B4 (GameInfo snapshot).** As the maintainer, I want one command that snapshots my game's tables locally, so that real numbers can feed the rule kernel.
- The Lua `export_gameinfo` handler is covered by a GM-B1 stub test.
- A test shows `gameinfo_capture.py` refuses to write when `CIV6AZ_DATA_ROOT` is unset or lies inside the repository, and writes under it otherwise.
- A hygiene test fails if a snapshot file appears under tracked paths.
- No real GameInfo content is committed.

**GM-B5 (search proxy and `--mcts`).** As a user of the live bridge, I want search mode to stop silently playing END_TURN, so that I am not misled about what the agent does.
- A test that does not mock the search reproduces the mismatch (84-channel network, 13-channel simulator) and is committed first.
- The proxy test shows a settler can found a city and a mountain blocks movement inside the search.
- `python -m alpha_zero.civ6_local --mode civ6 --mcts` exits non-zero at start-up with a message naming the gap id, and `python -m alpha_zero.civ6_local --mode civ6 --dry-run` still exits 0.
- `pytest tests/unit/test_bridge_mcts_policy.py -v` passes (updated for the refusal).
- The CLAUDE.md command row for `--mcts` is updated in the same PR.
