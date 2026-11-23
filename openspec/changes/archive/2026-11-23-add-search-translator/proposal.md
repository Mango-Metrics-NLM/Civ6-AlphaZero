# Turn the live search's answer into a command the game offered

## Why

The live search (`Civ6MctsPolicy`) plays in the MicroCiv simulator seeded from the
live state, so its answer is one of the simulator's actions. Nothing turned that into
a live command, which is why `civ6_local --mode civ6 --mcts` is refused (GM-B5). This
is slice GM-B6a of `kb/specs/20261120-live-command-layer-plan.md`: the translation and
the search-side corrections it needs. The agent loop is not changed and the refusal
stays; that is GM-B6b.

## What Changes

- New `alpha_zero/bridge/search_translator.py`: one simulator action to one command the
  game offered in this state, or a reason. Units and cities by live id; a move toward
  the destination cell by the offered tile that gets nearest it; an attack only into
  that cell (melee offer first, else ranged); production by simulator class.
- `alpha_zero/bridge/mcts_policy.py`: the search's root is restricted to actions that
  translate; a seeded state with nobody else in it is not a finished game; a city that
  is already building is seeded busy; the simulator's clock is clamped under the
  horizon; `Civ6MctsPolicy.select_command` returns the command and a record of the
  decision, acting on the search's own halving winner.

## Impact

- Capability: `civ6-bridge` (two requirements added).
- Bridge only; nothing on the release play path. No mod change, no wire change.
- Known limit, registered as GM-B8: with the mod's default settings only the acting
  player's own units and cities are exported, so the simulator never contains another
  player. The search then plans on an empty board and never considers an attack.
- Not verified in a real game. No trained checkpoint exists, so this proves plumbing,
  not playing strength.

## Amended at archive (2026-11-23)

The baseline requirement "Live search mode refuses until an action translator exists" is removed and replaced by "Live search mode is refused at start-up": the translator exists since this change, and the refusal now rests on a missing trained checkpoint, a missing live check and the registered comparison that has not run. Tasks 3.1 and 3.2 are archived open (3.2 was done by `add-visible-other-players`).
