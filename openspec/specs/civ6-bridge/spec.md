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
is blocked. The game's list of blockers is the only authority: the mod MUST NOT let
`UI.CanEndTurn()` overrule a listed blocker, because the live game answers true while
one is active (seen 2026-11-21). The Python agent loop MUST handle a blocked report
without raising and count it.

#### Scenario: Unanswered research choice

- **WHEN** the stub blocks end-turn on a research choice and END_TURN is executed
- **THEN** the acknowledgement reports blocked with the research blocking type, and the agent loop increments its blocked counter without an exception

#### Scenario: The game says the turn can end while a blocker is listed

- **WHEN** `UI.CanEndTurn()` answers true and the blocker list names a blocker
- **THEN** END_TURN reports blocked and sends nothing

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

### Requirement: Unit commands are requested the way the game's own UI requests them

The mod SHALL resolve a unit action's operation by type name from the game's database, use that
row's hash, and request it only after the game's strict check accepts it. A move MUST carry no
attack modifier. A name the game's database does not have, or an operation the strict check
refuses, MUST be answered as a failure that names the reason, and nothing is requested.

#### Scenario: A unit moves

- **WHEN** a `move` command names an own unit and a tile the game accepts
- **THEN** the unit is on that tile after the game applies the request, and the answer is success

#### Scenario: The game refuses

- **WHEN** the strict check refuses the operation (no moves left, an impassable tile)
- **THEN** the answer is a failure with the reason, and the game state does not change

#### Scenario: An operation name the game does not have

- **WHEN** the type name is not a row of the game's operation or command tables
- **THEN** the answer is a failure naming it, not a success

### Requirement: A move or an attack does not start a war

Before a move or an attack the mod SHALL ask the game whether it would change the war state, and
MUST refuse the command when the game says it would or when the game cannot be asked. This
requirement covers moves and attacks only. Two gaps are registered and not closed by it: an
answer from the game in a form other than the expected list is read as "no change", and the
separate declare-war command has no gate.

#### Scenario: An attack on a player at peace

- **WHEN** an `attack` command targets a unit of a player the agent is at peace with
- **THEN** the answer is a failure saying it would start a war, and nothing is requested

#### Scenario: The game cannot be asked

- **WHEN** the game's war-state check is missing or raises
- **THEN** the move or attack is refused and nothing is requested

### Requirement: A production order names the item by the game's hash

A production command SHALL be sent with the database row's hash and the exclusive insert mode,
whether the command names the item by type name or by row index. An item that needs a plot MUST
be refused.

#### Scenario: Build from a name

- **WHEN** a production command names `UNIT_WARRIOR` for a city that can build it
- **THEN** the city is building it after the request is applied, replacing what it was building

#### Scenario: An item that needs placement

- **WHEN** a production command names a district
- **THEN** the answer is a failure saying placement is not supported

### Requirement: The state after a command reflects the command

After a command that did not end the turn, the mod SHALL export the next state on a later tick
than the request, and SHALL keep listening for commands for a window measured from the last
command handled, up to a fixed number of commands per turn.

#### Scenario: A move is visible in the next state

- **WHEN** a move is accepted
- **THEN** the next exported state shows the unit on its new tile

#### Scenario: A long turn

- **WHEN** commands arrive for longer than the window, each within the window of the one before
- **THEN** the mod is still listening

### Requirement: Offered actions are ones the game accepts

Every unit action the exporter offers SHALL be one the game's strict check accepts at export
time; move targets MUST be the game's own adjacent plots; an attack MUST NOT be offered where it
would start a war; a city's production choices SHALL be asked by row hash.

#### Scenario: Move targets are neighbours

- **WHEN** a unit stands on an even or an odd row
- **THEN** every offered move target is exactly one step away

### Requirement: The loop does not resend what the game refused

Within one game turn the agent loop SHALL NOT send a command the mod refused in that turn again,
SHALL choose research and each city's production at most once unless the choice had no effect,
SHALL send only `end_turn` after a fixed number of commands, and SHALL be able to stop after a
configured number of game turns (`CIV6AZ_AGENT_MAX_GAME_TURNS`). A choice that cannot be sent
MUST be re-selected, not answered by ending the turn. When the game keeps blocking the end of
turn and nothing sent changes that, the loop SHALL declare a stall and stop sending, not spin.

#### Scenario: A refused command

- **WHEN** the mod answers a command with a failure
- **THEN** the same command is not sent again before the turn ends

#### Scenario: A turn that cannot be ended

- **WHEN** the game answers `end_turn` with a blocker three times running and the state does not change
- **THEN** the loop reports a stalled turn and exits with code 3 (or, with `on_stall=wait`, sends nothing more in that turn)

#### Scenario: A limit in game turns

- **WHEN** `max_game_turns` is 3 and each turn takes several commands
- **THEN** commands are sent in exactly three game turns

### Requirement: The mod follows what the live game was seen to do

Where the live session of 2026-11-21 showed the game behaving differently from the reading
the mod was written from, the mod SHALL follow the game: a city's item counts as buildable
only when both the loose and the strict form of the game's check accept it; a city's current
build is read from its type hash when the game does not return the item itself; tile yields
are read under the yield names the live game uses; and whole numbers are written to JSON
without loss above 2^31. Each MUST have a test that fails on the earlier behaviour.

#### Scenario: A building that is not unlocked yet

- **WHEN** the game's loose check accepts a building and its strict check does not
- **THEN** the building is not offered and a `city_produce` for it is refused

#### Scenario: A large whole number

- **WHEN** a value above 2^31 is exported
- **THEN** the JSON carries every digit

### Requirement: The log-tailing transport hands out every newest state once

`LogTailer.wait_for_state` SHALL return the newest state that has not been handed
out yet, without waiting when such a state has already been read, and MUST NOT
return the same state twice. A state read in the same batch of log lines as the
ack before it MUST still be returned to the next caller. States replaced by a
newer one before being read SHALL be counted.

#### Scenario: Ack and state arrive together

- **WHEN** an ack line and the state line behind it are read in one batch, and the caller waits for the ack and then for a state
- **THEN** the state is returned at once, not after a timeout

#### Scenario: No newer state

- **WHEN** a state has been handed out and no newer one has been read
- **THEN** waiting for a state ends in a timeout, not in the same state again

### Requirement: A start in mid-turn can see the state already written

At start the tailer SHALL keep the last complete state in the existing log and
make it available only when a heartbeat read after start shows the mod is
listening; a state read after start replaces it. Acks and heartbeats written
before start MUST NOT be used.

#### Scenario: Started while the mod waits for a command

- **WHEN** the log already holds a complete state and a heartbeat is read after start
- **THEN** that state is handed out as the current one

#### Scenario: Old heartbeat only

- **WHEN** the log holds a state and a heartbeat written before start, and nothing is written after start
- **THEN** no state is handed out and the heartbeat check stays false

### Requirement: The search's answer is translated into an offered command

`SearchTranslator.translate` SHALL return, for a simulator action, either a live command
that the game offered in the current state or a reason, and MUST NOT raise. Units and
cities SHALL be identified by live id, never by position in a list. A move MUST NOT be
translated into an attack.

#### Scenario: A unit the simulator left out comes first in the list

- **WHEN** the live unit list holds a unit with no simulator stand-in before the acting unit
- **THEN** the command names the acting unit's own live id

#### Scenario: Nothing offered lies toward the destination

- **WHEN** no offered move target is nearer the destination cell than the unit's own tile
- **THEN** the action is reported as not translatable and is left out of the search's root

### Requirement: The live search only weighs what can be played

`Civ6MctsPolicy.select_command` SHALL restrict the root of its search to simulator actions
that translate, SHALL act on the search's own halving winner, and SHALL return `end_turn`
with a recorded reason when nothing translates. A seeded state with no other player in it
MUST NOT be treated as a finished game.

#### Scenario: Nobody else is in the seeded state

- **WHEN** the live state holds only the acting player's units and cities
- **THEN** a step in the simulator does not end the game and carries no win or loss reward

#### Scenario: Same seed, same state

- **WHEN** `select_command` is called twice with the same state and `decision_seed`
- **THEN** the command and the record are identical

### Requirement: The agent loop can run a policy that answers with a command

`BridgeAgentLoop` SHALL ask a policy whose class defines `select_command` for a command,
handing it the turn ledger as a veto and a seed derived from the game turn and the
command's number in it, and SHALL send the answer through the same path as any other
command. A policy that names nothing or raises MUST be counted (`search_fallbacks`,
`search_failures`) and MUST NOT be answered with an `end_turn` the game is refusing while
another command may still be sent. `civ6_local --mode civ6 --mcts` SHALL remain refused.

#### Scenario: The search cannot answer a research blocker

- **WHEN** the game blocks the end of turn on a research choice and the search names nothing
- **THEN** the loop sends the offered research command in preference to any unit action
  nobody chose, and the turn can then be ended

#### Scenario: The policy raises

- **WHEN** `select_command` raises
- **THEN** the failure is counted, the loop goes on, and no command the ledger rules out is sent

### Requirement: The search's network comes from the checkpoint's own record

`build_search_policy` SHALL build a network of the simulator's shape at the board and
horizon the checkpoint records, MUST refuse a checkpoint that records no horizon, and MUST
refuse a network that is not of the simulator's shape (the full-game network is not). An
untrained network SHALL be built only on an explicit request.

#### Scenario: A checkpoint without a recorded horizon

- **WHEN** the checkpoint's metadata has no `max_turns`
- **THEN** `build_search_policy` raises and names the missing horizon

### Requirement: Live search mode is refused at start-up

`python -m alpha_zero.civ6_local --mode civ6 --mcts` SHALL exit non-zero at
start-up, before any game connection, with a message naming a registered gap id.
The search proxy MUST seed `MicroCivEnv` using only its existing rules, with no file
under `alpha_zero/envs/` changed. The translator, a network built from a checkpoint's
own record and an un-mocked end-to-end test against the stand-in exist; the refusal
MUST stay until a trained simulator checkpoint is loaded, the search path has been
seen working in a real game, and the registered comparison against a random policy
has been run.

#### Scenario: Starting live search

- **WHEN** `civ6_local` starts with `--mode civ6 --mcts`
- **THEN** it exits non-zero without connecting and the message names a registered gap id

#### Scenario: Dry run without search

- **WHEN** `python -m alpha_zero.civ6_local --mode civ6 --dry-run` runs
- **THEN** it exits 0 as before

### Requirement: Other players are exported only as far as the player sees them

With `OBSERVE_ALL_PLAYERS` on, the mod SHALL export another player's unit only when
the game reports both its tile and the unit itself as visible to the local player,
and another player's city only while its tile is visible now. A foreign unit MUST
NOT carry its remaining moves, remaining attacks or promotions, and a foreign city
MUST NOT carry its production. When the game cannot be asked about visibility,
nothing of another player SHALL be exported. With the setting off, no other
player's unit or city SHALL be exported.

#### Scenario: A stealth unit on a visible tile

- **WHEN** another player's unit stands on a tile the player sees but the game reports the unit as not visible
- **THEN** the unit is not in the exported state

#### Scenario: Barbarians and city-states

- **WHEN** a barbarian unit (player id 63) and a city-state unit are in sight
- **THEN** both are exported, whatever the number of major players

#### Scenario: A city seen earlier

- **WHEN** another player's city stands on a tile that was revealed but is not visible now
- **THEN** the city is not in the exported state

### Requirement: The state says what each other owner is to the player

The state SHALL carry `observes_others` and, for every owner of an exported foreign
unit or city, an `other_players` entry with `kind` (major, minor, barbarian, free)
and `at_war`. `at_war` MUST be absent, not false, when the game cannot be asked.
The Python converter SHALL put only hostile owners (at war, barbarian, free city)
on the enemy side and SHALL count the units and cities it leaves out. A state that
carries no `observes_others` SHALL be converted as before.

#### Scenario: A city-state at war

- **WHEN** a city-state the player is at war with has a unit in sight
- **THEN** its entry has `kind` minor and `at_war` true, and the unit is on the enemy side

#### Scenario: A neutral scout

- **WHEN** a major the player is at peace with has a unit in sight
- **THEN** the unit is not on the enemy side and is counted as left out

### Requirement: Nothing the player cannot see now is given away

A tile that is not visible now MUST NOT carry its owner, improvement, resource or
yields. When the game cannot be asked about visibility, no tile SHALL be reported
as visible. An attack MUST NOT be offered or carried out on a plot whose only
foreign occupant the player cannot see.

#### Scenario: A tile in fog

- **WHEN** a tile was revealed earlier and is not visible now
- **THEN** it is exported with its terrain and features and without owner, improvement, resource or yields

#### Scenario: A hidden unit next to an archer

- **WHEN** the only foreign unit on a plot in range is one the player cannot see
- **THEN** no attack on that plot is offered, and one sent anyway is refused

### Requirement: Killing what is visible is not winning the game

The live search's simulator holds only what the player can see. It MUST NOT let the
acting player win by elimination, and MUST NOT pay a win at the turn limit unless an
enemy city is in the simulator. The acting player with no unit and no city left SHALL
have lost, also when its last unit removed the last visible enemy. Each decision's record SHALL say what of the other
players was seeded and on how many lines of play a win was taken back.

#### Scenario: A lone hostile unit next to a warrior

- **WHEN** the acting player's attack removes the only enemy unit in the simulator and the enemy has no city there
- **THEN** the simulated game goes on, the move pays the per-kill reward and no win reward, and no win is paid at the turn limit either

#### Scenario: The acting player loses everything

- **WHEN** the acting player's last unit is removed and it holds no city
- **THEN** the simulated game ends as a loss
