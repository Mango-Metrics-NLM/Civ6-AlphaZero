# DEV-SQE to Pre-PR: GM-B8b, the search's simulator with an opponent in it

- **Date:** 2026-11-22
- **Session:** session_01CXusP9Ct9R7eg25YLcY7R7
- **Branch:** `claude/gm-b8b-search-opponent` = GM-B6b (#163) + GM-B8a + this slice. **No PR:**
  #162 and #163 hold both slots and may not merge before the live sitting.
- **Spec:** OpenSpec `add-visible-other-players` (tasks 2.x); gap register row B8b.

## What was built

`alpha_zero/bridge/mcts_policy.py` only:
- the acting player never wins by elimination in the simulator; at the turn limit a win is
  taken back unless an enemy city is in the simulator; with nothing left of its own it has lost
  (a mutual kill included);
- `elimination_suppressed` counts lines of play on which a win was taken back, over all clones;
- the decision record gains `others` (what the converter seeded and left out) and that count;
- the simulator is seeded from the decision seed (review of #162, merged in).

## How it was checked

- Tests from a written contract by `sdlc-qa-sdet` (87 tests); writing them found the win coming
  back at the turn limit. 28 of them fail under the old rule.
- Modelling review by `sdlc-ml-scientist`: the rule removes the fake win (same roots, seeds and
  network: attack chosen 9 of 18 without the rule, 3 of 18 with it). Findings fixed with tests:
  mutual kill scored a draw; a win at the limit stood with one enemy unit and no enemy city in
  sight; stale info; the measurement as a with-and-without test.
- Bridge, Lua, script and regression suites pass apart from seven sandbox-only failures;
  governance passes; `ruff check .`; single-file type check; validators; play-path diff empty.

## Not verified

Nothing ran in a real game; no trained checkpoint. The search reads only the winner at a
terminal, never step rewards, so its value is the network's estimate on a board of visible
pieces. The turn-limit rules are not reached at today's search budgets.

## Open

- The value head's reading of "no enemy city in sight" as nearly won: only training changes it.
- The opponent-mover sign case of the reward correction works (reviewer ran it) and has no test.
