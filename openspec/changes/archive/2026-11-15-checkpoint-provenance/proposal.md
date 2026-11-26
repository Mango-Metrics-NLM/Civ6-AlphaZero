# Record where a checkpoint came from, and make iteration 0 optional

## Why

A checkpoint trained at one `max_turns` and read at another loads, plays and
produces a number that means something else. Nothing recorded the horizon, the
search, the policy target or the code revision, so no reader could refuse it.
Separately, iteration 0 was hard-wired to random-rollout SimpleMCTS, so a run
could never train its first data on the search under study, and no file held the
untrained network that the A1 protocol compares against. These are defects
LF-A15 and LF-A16 of the local-first plan and must land before any learning-evidence run.

## What Changes

- `Civ6Network.save(path, *, metadata=None)` writes primitive-only provenance
  `{board_size, max_turns, num_players, mcts, gumbel_k, gumbel_sims, c_scale,
  policy_target, bootstrap, seed, git_sha, env_version}`, read from the resolved config.
- `utils/checkpoint.py` adds `HorizonMismatch`, `build_provenance`,
  `check_provenance`; `load_network(..., expected_max_turns=)`,
  `resume_network_weights(..., max_turns=, board_size=)` and the benchmark loader
  call it. A legacy file (no metadata) warns and passes.
- `--bootstrap {simple_mcts,none}`, default `simple_mcts` (behaviour unchanged).
- `checkpoint_0.pt` is saved before any training, except when resuming.
- `ENV_VERSION = 2` in `envs/microciv.py`.

## Impact

- Behaviour: checkpoints gain a `metadata` key; a horizon mismatch is now an error
  where a horizon is supplied. `--dry_run` and default runs also write `checkpoint_0.pt`.
- Affected specs: `selfplay-trajectories`.
- Code: `utils/checkpoint.py`, `core/civ6_network.py`, `training_civ6.py`,
  `evaluation/benchmark.py`, `envs/microciv.py`; tests in
  `tests/unit/test_checkpoint_provenance.py`.
