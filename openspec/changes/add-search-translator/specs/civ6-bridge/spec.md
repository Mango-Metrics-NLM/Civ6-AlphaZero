# civ6-bridge -- deltas

## ADDED Requirements

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
- **THEN** the loop sends the offered research command, and the turn can then be ended

#### Scenario: The policy raises

- **WHEN** `select_command` raises
- **THEN** the failure is counted, the loop goes on, and no command the ledger rules out is sent

### Requirement: The search's network comes from the checkpoint's own record

`build_search_policy` SHALL build a network of the simulator's shape at the board and
horizon the checkpoint records, and MUST refuse a checkpoint that records no horizon. An
untrained network SHALL be built only on an explicit request.

#### Scenario: A checkpoint without a recorded horizon

- **WHEN** the checkpoint's metadata has no `max_turns`
- **THEN** `build_search_policy` raises and names the missing horizon
