# Pin the default environment's seeded game with a golden fingerprint

## Why

The game-mechanics program will refactor `MicroCivEnv` (`alpha_zero/envs/microciv.py`).
A refactor of the env has to show that the default game did not change. Today nothing
records what the default game looks like. The determinism gate compares two runs of the
same code, so it cannot see a change that alters every run the same way. The
clone-fingerprint test only checks that search does not change terrain. If a refactor
quietly changes the rules, CI stays green, and every registered run and evidence artifact
stops matching the code that claims to reproduce it.

`ENV_VERSION` exists to mark a deliberate change to the game, but nothing ties it to
behaviour. A rule change can land without a bump, and nothing notices.

## What Changes

- New regression test `tests/regression/test_default_env_golden_regression.py`. It plays
  seeded legal games on the default `MicroCivEnv` at board 8, with 30 and 60 turns and
  2 and 3 players. It reuses the seeded-play pattern of
  `tests/unit/test_action_mask_collision.py` and the fingerprint pattern of
  `tests/unit/test_microciv_clone.py`.
- It computes one digest per configuration from integer game state only: units, cities,
  turn, current player, winner and the action sequence. It reads no float tensors and no
  observations, so the digest does not depend on the platform's float arithmetic or on
  the encoder.
- A committed fixture under `tests/fixtures/golden/` records each configuration's digest,
  plus the `ENV_VERSION` and numpy version it was generated with.
- The test fails when a digest differs while `ENV_VERSION` is unchanged. It also fails
  when `ENV_VERSION` differs from the fixture's, so a bump has to come with a regenerated
  fixture. The failure message says which of `ENV_VERSION` and the numpy version moved
  (or that neither did).
- A negative control is recorded in the test module: a one-line rule change in a scratch
  copy, or a perturbed action sequence, fails the test.

## Impact

- Tests only. New files go under `tests/regression/` and `tests/fixtures/golden/`, plus
  this OpenSpec change. Nothing under `alpha_zero/` changes.
- The release play path is untouched: no file matched by `scripts/release_play_path.txt`
  is modified. No finished or registered run goes stale, and the section 7.1 digest is
  unchanged.
- No game budget: the test plays only a handful of short in-process games and runs no
  runner.
- From now on, any change to the default game's rules or RNG use must bump `ENV_VERSION`
  and regenerate the fixture in the same change. A numpy upgrade that changes seeded
  draws is reported as a numpy move, not as an env change.
- Affected specs: `training-reproducibility`.

## Archive note (2026-11-23)

Merged with PR #143 (GM-S0a). The task boxes were never ticked in the change's own PR; they are ticked here because the test, its fixture and its negative controls are on the default branch and run in CI's regression job.
