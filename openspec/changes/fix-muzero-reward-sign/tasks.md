# Tasks

## 1. Design (CONFIRM-FIRST)
- [x] 1.1 Maintainer chose a rule-based turn model, outcome-only targets, and fixing the PUCT path too (2026-11-12)

## 2. Tests
- [x] 2.1 `tests/unit/test_forced_win_oracle.py`: one MuZero oracle over both edge types and both search paths (48 cases); the six `xfail(strict=True)` cases removed
- [x] 2.2 The stand-in dynamics score the outcome in `MicroCivEnv`'s frame (the player to move after the transition); `test_fixture_dynamics_match_the_environment` holds it against the env
- [x] 2.3 `tests/unit/test_muzero_turn_model.py`: `EndTurnModel` predicts every mover change in seeded MicroCiv games; the env's terminal reward is in the next mover's frame; hand-computed signed backup; both self-play paths write mover-frame targets
- [x] 2.4 Negative controls: dropping the sign in `muzero_tree` fails 12/48 (every alternating case at odd depth, both paths); plain-negamax `perspective_sign` fails 12/48 (every same-player case)
- [x] 2.5 `test_muzero_mcts.py` (completed-Q target re-derived with the sign) and `test_search_budget.py` (the toy game's same-player turn model) stay green

## 3. Fix
- [x] 3.1 `core/mcts/muzero_tree.py`: `MuZeroNode`, `TurnModel`/`EndTurnModel`, `MinMaxStats`, `edge_q`, `backpropagate_with_rewards`
- [x] 3.2 `MuZeroMCTS`: `turn_model`, `to_play`, signed root Q and selection, one backup for both paths, `end_turn_action` config
- [x] 3.3 `training/muzero_targets.py`, used by `run_muzero_training.self_play_game` and `run_muzero_selfplay`
- [x] 3.4 Sequential `GumbelMCTS` byte-identical on seeded MicroCiv positions (84/84)

## 4. Documentation
- [x] 4.1 CLAUDE.md §5.3 row resolved, §5.4 bug row, §5.2 decision row
