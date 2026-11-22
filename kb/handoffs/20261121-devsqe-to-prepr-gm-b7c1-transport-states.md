# DEV-SQE to Pre-PR: GM-B7c-1, the socket transport keeps every state

- **Date:** 2026-11-21
- **Session:** session_01CXusP9Ct9R7eg25YLcY7R7
- **Branch:** `claude/gm-b7c1-transport-states` (from the default branch; no mod change)
- **Spec:** `kb/specs/20261120-live-command-layer-plan.md`; OpenSpec `fix-live-state-delivery`

## What was built

- `alpha_zero/bridge/log_tailer.py`: states carry a sequence number and are handed out by it.
  Before, `wait_for_state` cleared its event and then waited, so a state read in the same batch
  of log lines as the ack before it was discarded and the agent loop waited out its timeout
  while the mod gave the turn up.
- The state already in the log at start is released by a heartbeat that reports its own game
  turn (a start in mid-turn). Unread acks are bounded.
- A state sent in pieces keeps a character or a space cut at a piece boundary.
- `alpha_zero/bridge/live_bridge.py`: `require_new`, `backlog_state()`.
- `alpha_zero/bridge/state_diff.py`; `scripts/live_command_check.py` (one command to a running
  game, the answer and what changed).

## How it was checked

- Tests first by `sdlc-qa-sdet` from a written contract, with a real tailer on a real file; the
  defect test fails on the old code with a timeout.
- Independent review by `sdlc-architect`: the first version could release a state from an
  earlier game load as current (any heartbeat was taken as "the mod is listening"), lose a line
  written between the scan and the tail, and hand one state to two waiters. Fixed with tests.
- Fourteen planted defects each fail the suite.
- After merging the default branch (with GM-B7a): bridge, Lua and script suites 4,159 passed,
  26 skipped; `ruff check .`, single-file type checks, doc-sync, OpenSpec and kb validators,
  bundle check clean; play-path diff empty. `make verify`, the full type check and coverage
  were not run; CI is the gate.

## Not verified

Nothing ran in a real game. That the lost state is the cause of the live turn stall (LF-H1) is
a hypothesis; `docs/LIVE_CIV6.md` item 7 is the check (a command's next state arrives in a
second or two, not after thirty).

## Open

- A state sent in pieces has no export id; a restarted game whose log has grown past the old
  read position is read from the middle. Both registered; the first needs a mod change.
- A reload of the same save at the same turn cannot be told from a continuing turn.
- Next: GM-B7c-2 (the loop's per-turn ledger), on `claude/gm-b7c2-turn-ledger`.
