# Make the MicroCiv move mask agree with the move executor

## Why

`MicroCivActionSpace.get_valid_mask` (`alpha_zero/core/civ6_actions.py`) marked a
move legal whenever the target tile was on the board and not a mountain. The
executor (`MicroCivEnv._is_valid_move`) additionally rejects any tile occupied by
a unit, own or enemy, and on rejection sets the mover's `moves_remaining` to 0.
Masked policy loss and every MCTS variant treat the mask as ground truth, so each
disagreement was a wasted action that also forfeited the unit's turn
(CLAUDE.md §5.3, `docs/evidence/draw_rate_sweep/report.md`). This is defect
LF-B1 of the local-first plan and must land before any learning-evidence run.

Measured before the fix: seeded random play over mask-legal actions produced
5,748 "Invalid move" rejections in 20 games (48,779 steps) and 50,450 in 200 games.

## What Changes

- The move branch of `get_valid_mask` excludes every tile holding a unit (any player).
- No other mask branch changes: attack, skip, found-city and produce already match
  the executor's rejection conditions.
- Fog of war is not touched: the mask is built from the same (true) state as before.
  Hidden-unit semantics are LF-B6 and need their own change.

## Impact

- Behaviour: fewer legal move actions; seeded action sequences differ from earlier runs.
- Affected specs: `selfplay-trajectories`.
- Code: `alpha_zero/core/civ6_actions.py`; tests in `tests/unit/test_action_mask_collision.py`.
- Every mask consumer shares the one builder (`MicroCivEnv.get_valid_actions`,
  `get_observation`, `EnhancedMicroCivEnv`, `VectorizedMicroCivEnv`, `BridgeMCTSEnv`).
