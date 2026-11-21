# Keep every state the game writes on the socket transport

## Why

On the log-tailing transport (`LiveBridge`), `LogTailer.wait_for_state` cleared its
"state arrived" event and then waited. Log lines are read in batches, so an ack and
the state behind it are parsed before the caller asks for the state; the event is
then cleared and the state is lost. The agent loop asks for a state right after
every ack, so after most commands it saw nothing, waited its full timeout, and the
mod, which stops listening 30 seconds after the last command, gave the turn up.
This is a second lead on the live turn-cycle stall (LF-H1), beside the blocked end
of turn that GM-B2 reports. It is a hypothesis until seen in a real game.

A start in the middle of a turn had the same shape: the tailer read from the end
of the log and never saw the state the mod was already waiting on.

This is slice GM-B7c-1 of `kb/specs/20261120-live-command-layer-plan.md`. It is
independent of the mod changes in `fix-live-command-layer` (GM-B7a, GM-B7b) and
changes no Lua.

## What Changes

- `alpha_zero/bridge/log_tailer.py`: states carry a sequence number;
  `wait_for_state` hands out the newest state not handed out yet, at once when one
  is there; states replaced before being read are counted. At start the last
  complete state already in the log is kept and made available when the first live
  heartbeat shows the mod is listening. Unread acks are bounded.
- `alpha_zero/bridge/live_bridge.py`: `require_new=True` means "newer than the one
  last handed out"; `backlog_state()`.
- New `alpha_zero/bridge/state_diff.py`: what changed between two states, and
  whether a command's expected effect is visible.
- New `scripts/live_command_check.py`: send one command to a running game and print
  the answer and what changed, for the live checklist in `docs/LIVE_CIV6.md`.

## Impact

- Capability: `civ6-bridge` (two requirements added).
- No change to the mod, the wire format, the file-polling transport, or anything on
  the release play path.
- Not verified in a real game.
