# Sign MuZero's latent search and training targets by mover

## Why

T18 (`openspec`-less, recorded in CLAUDE.md §5.2 2026-11-12) extended the
forced-win oracle to `MuZeroMCTS` and found it is not two-player sign-correct.
That was recorded as a known defect (§5.3) and the alternating oracle cases were
`xfail(strict=True)`, because changing search semantics is CONFIRM-FIRST
(CLAUDE.md §3.2). MuZero is self-played on two-player MicroCiv
(`scripts/run_muzero_training.py`, `run_muzero_selfplay` in
`cloud/gcp/selfplay_worker.py`), where rewards are relative to a player, yet
three frames disagreed:

1. **Unsigned backup.** `_backpropagate_with_rewards` summed rewards along a
   latent path that crosses opponent turns with no sign flip, so an opponent's
   gain counted as the root player's. The non-default PUCT path
   (`use_gumbel=False`) alternated signs every ply and ignored rewards.
2. **Inconsistent target frame.** `run_muzero_training.py` labelled values from
   **player 0's** perspective, summed the env's shaped rewards across player
   changes, and added the outcome again on top of the terminal reward that
   already carried it. The cloud path stored mover-relative outcome values
   beside unsigned shaped reward targets.
3. **Edge reward and normaliser.** In-tree selection scored `child.value`
   without the edge reward (the root was fixed in T18), against a `[0, 1]`
   normaliser fixed per simulation and never updated.

## What Changes

The maintainer approved this design on 2026-11-12, choosing the recommended
option on each of three questions:

- **Rule-based turn model.** No learned head. `MicroCivEnv` changes
  `current_player` only on END_TURN, and the search knows every action on its
  path. `muzero_tree.EndTurnModel` (two-player; END_TURN = action 0) is the
  default, and `MuZeroMCTS(turn_model=...)` injects another rule. Every node
  carries `to_play`. `search(..., to_play=)` roots the tree, and
  `search_from_env` passes `env.current_player`.
- **One sign rule.** The backup, the root Q and in-tree selection all use
  `edge_q = perspective_sign(parent, child) * (reward + discount * value)`.
  The edge reward is in the child mover's frame, as `MicroCivEnv` scores it,
  so it combines with the child's value before the sign. Selection normalises
  with MuZero's running `MinMaxStats`, reset each search.
- **Outcome-only targets, board-game style.** `training/muzero_targets.py` is
  shared by both self-play paths. The value at `t` is the outcome for the
  mover at `t`. The reward is zero except on the final transition, which
  carries the outcome for the player to move after it. The absorbing state is
  worth 0. Shaped rewards are dropped: `CITY_FOUND_REWARD` is `+1.0`, the same
  size as a win.
- **Both search paths.** The PUCT path shares the signed backup and is now
  oracle-gated.
- `MuZeroNode` and the tree helpers move to `core/mcts/muzero_tree.py`
  (layer 1, beside `base`). `muzero.py` drops from 997 to under 900 lines,
  below the 1,000-line ratchet.

## Impact

- `MuZeroMCTS` output changes for every position whose search crosses END_TURN.
  Seeded MuZero runs from before this change do not reproduce, and MuZero
  checkpoints trained on the old targets were trained against a different
  value frame and reward head.
- Sequential `GumbelMCTS`, which A1 uses, is **byte-identical**: 84 of 84
  seeded MicroCiv records match before and after (policy bytes, root value and
  sampled action, over A1's shape and two other budgets).
  `gumbel.py`, `gumbel_ops.py`, `backup.py` and `batch_mcts.py` are untouched.
- Out of scope, recorded as follow-ups:
  - `MuZeroTrainingBridge.self_play_game` in `training/muzero_integration.py`
    still stores the env's shaped rewards and root values. It also calls
    `select_action` with three arguments, so it cannot run against the real
    search.
  - The value targets are undiscounted while the search discounts at `0.997`.
    The sign is consistent, but the magnitudes are not.
  - Pre-existing, found in review: the deployed trainer builds no reward
    tensor (`training_worker.prepare_dataset`), so `train_muzero_epoch` trains
    the reward head toward 0 and the new `reward_targets` are not read there.
    The cloud search therefore gets its terminal signal from the value head
    alone.
  - `EndTurnModel` is two-player only and now raises for any other mover
    rather than returning `-1`.
