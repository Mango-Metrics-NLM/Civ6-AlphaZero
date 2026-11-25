# The kick re-arms the mod and never ends the turn

## Why

`CIV6_BRIDGE_KICK_START` existed because the mod listens for commands only for `COMMAND_TIMEOUT`
(30 s) after the last one, and `PollCommands` then returns at its first line until the next turn
begins: an agent started later in the turn gets no state. The kick "fixed" that by running
`UI.RequestAction(ACTION_ENDTURN)` through FireTuner, which bypasses the mod's end-of-turn
blocker check. The live session of 2026-11-21 showed that an end of turn sent while a blocker is
active deactivates the turn (LF-H1), so the kick was a third route to the stall the bridge track
has been closing, and in the best case it gave the agent's turn away unplayed.

A mod loaded in the middle of its own turn (a reload) has the same problem: the turn-start
event has already gone by, so it never starts listening.

## What Changes

- `SocketTunerBridge.request_rearm(context=None)` sets `ExposedMembers.AlphaZeroRearm` to a
  fresh nonce through the same Lua context as commands (`GameCore_Tuner` preferred) and returns
  the nonce. Its Lua is the module constant `REARM_LUA`, which names no game action.
- `SocketTunerBridge.kick_end_turn` stays as a deprecated alias (`DeprecationWarning`) of
  `request_rearm`; `LiveBridge.kick_turn` calls `request_rearm`.
- The mod's `PollCommands` reads the flag before its early return. It accepts it only when the
  local player is the bridge's player, `Players[id]:IsTurnActive()` answers true, the window is
  closed, no command is queued or awaits its state export and no `end_turn` ran this turn; then it
  reopens the command window, exports the state once and logs `Re-arm accepted <nonce>`.
  Otherwise it logs `Re-arm ignored <nonce>: <reason>`. Each answer is also the machine line
  `AZ_REARM:<nonce>:accepted|ignored:<reason>`. It clears the flag when read; a turn start,
  `Reset` and initialisation drop a pending one.
- `Initialize` starts the turn when the mod is loaded during the bridge player's own active turn.
- `civ6_local` logs a failing kick at WARNING and still runs the agent loop.

## Impact

- Live behaviour of the bridge changes (the kick no longer ends a turn). Opt-in: only with
  `CIV6_BRIDGE_KICK_START` set, on the socket transport; the mid-turn start needs
  `IsTurnActive`, which the stand-in game used before this change does not have.
- **Not run in a real game.** Whether `IsTurnActive` exists in the UI context is a live
  checklist item (`docs/LIVE_CIV6.md` 33); where it does not, every re-arm is ignored.
- Nothing on the release play path changes.
