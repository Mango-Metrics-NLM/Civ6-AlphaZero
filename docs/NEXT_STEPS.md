# Next Steps — Live Bridge (v1.8.0)

## Status

Live command delivery to Civilization VI is **proven, headless, end to end**. The
FireTuner socket transport (`alpha_zero/bridge/tuner_socket_bridge.py`) discovers
the `GameCore_Tuner` Lua state over `127.0.0.1:4318` and sets
`ExposedMembers.AlphaZeroCommand` directly, with no GUI, no `.ltp` relay panel,
and no `commands.json` file hop. On 2026-07-12 this was verified against a real
running game: **4 autonomous turns** were delivered and acknowledged
(`Command received via ExposedMembers` -> `Executing: END_TURN` -> `AZ_ACK`).

The encoder/network channel bug that previously forced the agent to fall back to
`end_turn` every turn (an 84-channel observation collapsed to 1 channel in
`agent_loop._build_observation`, plus a bridge network defaulting to 13 channels)
is fixed, so **the agent now runs its actual policy** (`policy_logits` over the
demo action space) rather than a hardcoded fallback. State decoding continues to
work correctly every turn, with zero regression of the earlier "missing width"
bug.

## Known Limitations

- **Turn-cycle stall between automated turns.** After a handful of automated
  end-turns, the turn cycle stalls in what looks like a Civ6 UI-automation
  limbo rather than a bridge defect:
  - The local player's `TURN_ACTIVE` flag reads `false` with no popup blocking
    it.
  - No headless mechanism has yet been found to force-advance the turn from
    this state.
  - The socket command path itself keeps working (commands are still
    delivered and acknowledged); it is the game's own turn-processing state
    that gets stuck.
- **`msg_id` int32 overflow.** `LiveBridge.next_msg_id` derives the message id
  from `int(time.time() * 1000)`, which gets truncated to int32 in the ack
  (observed as `-2147483648`). Cosmetic today — acks still correlate correctly
  in practice — but it is a latent correctness bug worth fixing before message
  ids are relied upon for anything stricter.
- **Sustained real-move demo still needed.** All live verification so far has
  been short (4 turns, mostly `END_TURN`). A longer run with the agent making
  and executing real (non-end-turn) actions has not yet been demonstrated, and
  requires a fresh single-player game to retry now that the channel fix makes
  real moves possible.

## Recommended Next Steps

1. **Run a sustained-play validation on a fresh game.** Start a new
   single-player Civ6 game and let the agent play 20+ turns via the socket
   transport, confirming it issues real policy-driven actions (not just
   `END_TURN`) and that state decoding stays correct throughout.
2. **Add mod-side end-turn robustness.** Investigate and handle the
   "unit needs orders" / turn-not-advancing condition on the mod side so the
   turn cycle self-sustains instead of stalling in UI limbo between automated
   turns. This is the most likely fix for the turn-cycle stall described
   above.
3. **Fix `msg_id` to a 32-bit-safe monotonic counter.** Replace the
   `int(time.time() * 1000)`-based id in `LiveBridge.next_msg_id` with a
   monotonically increasing counter (or an id scheme that does not overflow
   int32) so acks are unambiguous.
4. **Load a trained checkpoint for live play.** Point the live agent at a
   trained model checkpoint (see the GCS model store, v0-v218+) instead of an
   untrained/random-init network, so live demos reflect actual learned play
   rather than random policy output.
5. **Optional: retire the file transport once the socket path is
   battle-tested.** `alpha_zero/bridge/tuner_socket_bridge.py`
   (`SocketTunerBridge`) is a drop-in replacement for the file-based
   `TunerBridge`. Once step 1 has been repeated successfully across multiple
   fresh games, consider deprecating the `commands.json` file transport
   (`BridgeConfig.command_transport = "file"`) and making `"socket"` the sole
   supported path.
