# Civ6 Bridge

## Purpose

Defines what the real-game bridge (``alpha_zero/bridge/`` and the mod under
``mod/AlphaZeroAI/``) exports from a running game, which commands it offers and
carries out, and what it refuses. The mod is read-only on the game's official
surfaces, and nothing here has been verified in a real game unless a requirement
says so.

## Requirements

### Requirement: The loaded Lua bundle is generated from its module sources

`python scripts/build_lua_bundle.py --check` SHALL exit 0 only when the committed
`mod/AlphaZeroAI/Scripts/AlphaZeroMain.lua` equals the generator's output from
the module files. The first generated bundle MUST be byte-identical to the bundle
committed before this change.

#### Scenario: Module edited without regeneration

- **WHEN** a module file under `mod/AlphaZeroAI/Scripts/` changes and the bundle is not regenerated
- **THEN** `--check` exits non-zero and `tests/scripts/test_lua_bundle_sync.py` fails

#### Scenario: First generation

- **WHEN** the generator first runs on the reconciled sources
- **THEN** its output is byte-identical to the previously committed bundle

### Requirement: The exporter runs under a Lua test harness

Tests SHALL execute the exporter under a Lua 5.1-compatible runtime against
`tests/fixtures/lua/civ6_stub.lua`. In the CI job that installs the Lua test
extra, a missing runtime MUST fail the job rather than skip.

#### Scenario: Harness in CI

- **WHEN** the Lua test extra is absent in the CI job configured to require it
- **THEN** the Lua tests fail instead of reporting skipped

### Requirement: The exporter reports research state

The exported state SHALL carry a `schema_version`, the local player's researched
techs and civics as type names, the current research with cost, progress and boost
flag, and each city's buildings and districts. Every new game call MUST be
`pcall`-guarded, so a failing call leaves its field empty and the state still exports.

#### Scenario: Researched techs on the stub

- **WHEN** the stub player has researched two techs and the exporter runs
- **THEN** `techs_researched` lists exactly those two type names

#### Scenario: A failing game call

- **WHEN** the stub makes the civics call raise
- **THEN** the state exports with an empty civics list and a valid `schema_version`

### Requirement: Research is offered and executed through official surfaces

Research SHALL appear in the legal actions exactly for the techs and civics the
game reports as open (`CanResearch`, `CanProgress`; the prerequisite table where
`CanProgress` is missing), leaving out the one already in progress. Executing it
MUST use `UI.RequestPlayerOperation` with the row's hash and an exclusive insert
mode, as the game's own choosers do, with the GameCore setter as a fallback, and
MUST refuse a choice the player already has or the game reports as closed.

#### Scenario: Researchable techs

- **WHEN** the stub reports two researchable techs and none is in progress
- **THEN** the legal actions include one research-tech action whose items are exactly those two, by row index and by type name

#### Scenario: A request the game drops

- **WHEN** a research command is sent while end-turn is blocked on a research choice and the game does not apply the request
- **THEN** the mod dismisses no notification and END_TURN still reports blocked

#### Scenario: No UI request available

- **WHEN** `UI.RequestPlayerOperation` is missing and a research command is executed
- **THEN** the GameCore setter is called with the row index and the command succeeds

### Requirement: A blocked end of turn is reported, not hidden

END_TURN SHALL report a blocked turn with the blocking type instead of returning
success unconditionally, and MUST NOT send the request to the game while the turn
is blocked. The Python agent loop MUST handle a blocked report without raising
and count it.

#### Scenario: Unanswered research choice

- **WHEN** the stub blocks end-turn on a research choice and END_TURN is executed
- **THEN** the acknowledgement reports blocked with the research blocking type, and the agent loop increments its blocked counter without an exception

### Requirement: Wire changes are additive

New protocol fields SHALL be optional with defaults, and no existing field MUST
be renamed, retyped or removed, with one stated exception: `player.current_research`,
which the mod wrote as a raw row index although the reader declares a type name,
is a type name from schema 2 on (the index is in `research_status.index`). A
payload without the new fields, or without `schema_version`, MUST parse.

#### Scenario: Legacy payload

- **WHEN** a state payload recorded before this change is parsed
- **THEN** it parses, and every new field takes its default

### Requirement: Encoder tech and civic slots are filled from a registry

The state converter SHALL fill the 84-channel encoder's tech slots from a registry
built through `alpha_zero.rules.registry`, and its civic slots from the ordered
civic list given with that registry (the `Civics` table of a GameInfo export),
because the rules kernel has no civic tree yet. When no registry is available,
the slots MUST stay zero, a WARNING MUST be logged once and a counter
incremented. A registry that is configured and cannot be loaded MUST be an error,
not a silent fall-back to no registry. A researched item with no slot, or with a
slot the encoder has no position for, MUST be counted.

#### Scenario: Fixture with a registry

- **WHEN** a fixture state with researched techs is converted with a registry present
- **THEN** the corresponding encoder tech and civic slots are non-zero

#### Scenario: Missing registry

- **WHEN** the same state is converted with no registry
- **THEN** the slots are zero, one WARNING is logged and the counter reads 1

#### Scenario: A configured registry that cannot be loaded

- **WHEN** `CIV6AZ_BRIDGE_RULESET` names a rule set that does not load and a converter is built
- **THEN** construction fails with an error naming the rule set

### Requirement: Offered research reaches the action mask

The action converter SHALL set a research action in the legal-action mask for
each research choice the mod offers, and MUST refuse a research action that was
not offered. With a registry the research action of a technology or civic SHALL
be its registry slot and the command SHALL name it by the game's type name;
without a registry it SHALL be the game's own row index. A choice with no slot,
or beyond the action space's research actions, MUST be counted.

#### Scenario: A choice made through the converters

- **WHEN** the mod offers one technology, the mask is built, that action is converted and the command is executed by the mod
- **THEN** the mod requests that technology and, once the game applies it, the next state shows it as the current research

#### Scenario: Research that was not offered

- **WHEN** a research action is converted with legal actions that do not offer it
- **THEN** the conversion is refused

### Requirement: GameInfo snapshots stay out of the repository

`gameinfo_capture.py` SHALL write a snapshot only under `CIV6AZ_DATA_ROOT`, in the
input format of `python -m alpha_zero.rules import-gameinfo`, and MUST refuse a
path inside the checkout. No tracked file MUST be a GameInfo snapshot, except
synthetic fixtures on an explicit allowlist in the hygiene test: files written by
hand for tests, holding no data captured from a game.

#### Scenario: Path inside the checkout

- **WHEN** a capture is requested with a destination under the repository root
- **THEN** it is refused and nothing is written

#### Scenario: Hygiene

- **WHEN** the hygiene test lists tracked files
- **THEN** none outside the synthetic-fixture allowlist matches the snapshot format

#### Scenario: A snapshot outside the allowlist

- **WHEN** a file in the snapshot format is tracked at a path the allowlist does not name
- **THEN** the hygiene test fails and names the path

### Requirement: Live search mode refuses until an action translator exists

`python -m alpha_zero.civ6_local --mode civ6 --mcts` SHALL exit non-zero at
start-up, before any game connection, with a message naming the gap id registered
for the bridge action translator. The search proxy MUST seed `MicroCivEnv` using
only its existing rules, with no file under `alpha_zero/envs/` changed. An un-mocked test MUST
record the current failure before the refusal lands. The refusal MUST stay until the
search is also given a network and checkpoint matching the simulator's observation
shape and an un-mocked end-to-end search test passes; a translator alone does not lift it.

#### Scenario: Starting live search

- **WHEN** `civ6_local` starts with `--mode civ6 --mcts`
- **THEN** it exits non-zero without connecting and the message names the translator gap id

#### Scenario: Dry run without search

- **WHEN** `python -m alpha_zero.civ6_local --mode civ6 --dry-run` runs
- **THEN** it exits 0 as before
