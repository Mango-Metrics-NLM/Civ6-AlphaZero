# DEV-SQE → Pre-PR Handoff

**Date:** 2026-11-13
**Spec:** kb/specs/20261111-local-first-and-branching-plan.md (follow-up to T18; the CONFIRM-FIRST MuZero item in CLAUDE.md §5.3). OpenSpec: `openspec/changes/fix-muzero-reward-sign`.
**PR:** draft #136, `claude/muzero-reward-sign-fix` → `claude/civ6-alphazero-implementation-vMnun`, head 220da9e (3 commits on the #135 merge, cd89a00)

**Design authorization:** CONFIRM-FIRST under CLAUDE.md §3.2. The maintainer approved it in the session on 2026-11-12, choosing the recommended option on each of three questions:
- a rule-based turn model, not a learned `to_play` head;
- outcome-only, board-game-style training targets;
- fixing and oracle-gating the PUCT path as well.

**Files Changed:** 17 files, +1,116 / −321.
- `alpha_zero/core/mcts/muzero_tree.py` (new, layer 1):
  - `MuZeroNode`, moved here;
  - `TurnModel` / `EndTurnModel`, the two-player turn model (END_TURN = action 0, raises for a third mover);
  - `MinMaxStats`, which passes Q through raw until there is spread;
  - `edge_q`;
  - `backpropagate_with_rewards`.
- `alpha_zero/core/mcts/muzero.py`:
  - `turn_model`, `search(to_play=)`, and `search_from_env` rooted at `env.current_player`;
  - signed root Q and selection;
  - one signed reward backup for both the Gumbel and PUCT paths;
  - `MuZeroMCTSConfig.end_turn_action`;
  - `from_network` via `dataclasses.replace`;
  - 997 → 878 lines.
- `alpha_zero/training/muzero_targets.py` (new): `outcome_only_targets`, the shared target rule.
- `scripts/run_muzero_training.py`: value targets were in player 0's frame, plus shaped n-step returns that double-counted the outcome. They are now outcome-only targets in the mover's frame, and the absorbing state is worth 0.
- `alpha_zero/cloud/gcp/selfplay_worker.py` (`run_muzero_selfplay`): the root mover is passed into the search, and the shared targets are used. Net-zero line change to hold the coverage-omit ratchet.
- Tests:
  - `test_forced_win_oracle.py`: one MuZero oracle with 48 cases; the stand-in reward moves to `MicroCivEnv`'s frame.
  - `test_muzero_turn_model.py` (new).
  - `test_muzero_mcts.py`: the completed-Q re-derivation is now signed.
  - `test_search_budget.py`: the toy game's turn model.
  - `test_mcts_split_backcompat.py`: layering.
- Docs: CLAUDE.md §5.2/§5.3/§5.4, `docs/MODULE_REFERENCE.md` counts, CHANGELOG `[Unreleased]` Fixed, OpenSpec change.

**Tests Added:**
- `tests/unit/test_muzero_turn_model.py` (26):
  - `EndTurnModel` vs seeded MicroCiv games, step by step;
  - the terminal reward is in the next mover's frame (decisive-game guard);
  - a hand-computed signed backup;
  - MinMaxStats and the equal-Q selection case;
  - every backed-up node carries the turn model's mover, on both paths;
  - both self-play paths' target wiring, via a spy that forces a decisive game.
- `tests/unit/test_forced_win_oracle.py`: the MuZero oracle is parametrised over path (gumbel, puct) × edge type × depth × winning index, giving 48 cases. The 6 `xfail(strict=True)` cases are removed.

**Negative controls** (each planted, observed failing, then reverted):
1. Drop the sign in `edge_q` / `backpropagate_with_rewards`: 12/48 oracle cases fail (every alternating case at odd depth, both paths).
2. Make `perspective_sign` always −1 (plain negamax): 12/48 fail (every same-player case).
3. Feed the script's targets the last mover (`players[-1]`) as `final_to_play`: the script wiring test fails.
4. Restore the 0.5 no-spread selection fallback: the equal-Q selection test fails.

Note: changing the stand-in's reward frame alone moves the parity at which an unsigned search fails, from even depth to odd. The former xfails turning green is therefore not the evidence; controls 1 and 2 are.

**Test Results:**

| Gate | Result |
|---|---|
| Full local run at 220da9e (unit, cloud, regression, scripts, governance; `-m "not slow"`; Git Bash; worktree on PYTHONPATH) | 13,345 passed, 50 skipped, 8 failed — all environment-only, see Known Issues |
| Focused MuZero, oracle, budget, layering, ratchet and cloud-MuZero suites | 414 passed |
| Sequential `GumbelMCTS` byte-identity (A1's search) | 84/84 seeded MicroCiv records identical before/after (policy bytes, root value, sampled action) at (8,24) and (16,64) × 3 policy targets |
| Debt ratchet | passes; `coverage_omitted_lines` unchanged at 66,852 |
| CI on PR #136 at 220da9e | 23 passing, 0 failing, 1 skipped; mergeable, clean |

Coverage was not measured locally (the coverage instrumentation crash in CLAUDE.md §5.3); CI's coverage job is the gate. `cloud/*` is outside the global coverage gate, so the cloud change cannot move that number.

**Lint Status:** clean (`ruff check .`, `ruff format`).

**Type Check Status:** no errors in the touched files. 23 errors remain repo-wide locally: the pre-existing numpy/jax stub set recorded in the #135 handoff, in files this PR does not touch. CI Lint & Type Check is green.

**Smoke Tests:**
- `python -m alpha_zero.training_civ6 --dry_run` exits 0.
- `scripts/run_muzero_training.py` has no dry-run flag. Its self-play path is exercised by `test_run_muzero_training.py` and the new wiring test.

**Known Issues:**
- **Environment-only local failures (8):**
  - 6 in `tests/scripts/test_deployment_infra.py::TestEnvFile` (no `.env`; CLAUDE.md §5.3);
  - `test_seed_is_stable_across_processes` (its subprocess imports the main checkout's editable install);
  - `test_audit_import_graph.py::test_workflow_reference_counts_as_a_deploy_ref` (Windows `\` vs `/` in a path).

  None touch this change, and all pass in CI.
- **Breaking for MuZero:** seeded MuZero runs and MuZero checkpoints from before this change do not carry over (different value frame and reward targets). Do not resume them. A1 does not use MuZero.
- **Follow-ups, out of scope (recorded in CLAUDE.md §5.3 and the OpenSpec proposal):**
  - `MuZeroTrainingBridge.self_play_game` still stores shaped rewards and calls `select_action` with three arguments.
  - Value targets are undiscounted while the search discounts at 0.997. The sign is consistent; the magnitudes are not.
  - Pre-existing: the deployed trainer builds no reward tensor, so the cloud reward head trains toward 0.
- **Reviews:** an independent fresh-context review found no sign or wiring defect. Its vacuous-test finding was fixed (control 3). Copilot's two findings, the ratchet and the no-spread selection bias, were fixed in 220da9e, and both threads are resolved.
- After merge, `openspec-archive` moves `fix-muzero-reward-sign` into the archive and applies the new `muzero-search` capability to the baseline.
