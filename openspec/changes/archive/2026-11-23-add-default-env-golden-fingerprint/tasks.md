# Tasks

## 1. Golden fingerprint
- [x] 1.1 `tests/regression/test_default_env_golden_regression.py`: seeded legal play on the default `MicroCivEnv` at board 8, 30 and 60 turns, 2 and 3 players; a digest over integer game state only (units, cities, turn, current player, winner, action sequence)
- [x] 1.2 Fixture under `tests/fixtures/golden/` recording each configuration's digest, `ENV_VERSION` and the numpy version; a documented way to regenerate it
- [x] 1.3 The test fails on a changed digest with an unchanged `ENV_VERSION` and on an `ENV_VERSION` that differs from the fixture's; the failure message names which of `ENV_VERSION` and the numpy version moved
- [x] 1.4 The test passes twice in a row and is collected by the CI job that runs `tests/regression/` (suite-wiring gate green)

## 2. Negative control
- [x] 2.1 Negative controls recorded in the test module: a perturbed action sequence, and a one-line rule change in a scratch copy of the env, each fail the test with the "neither version moved" message; a faked `ENV_VERSION` mismatch fails with the version message

## 3. Gates
- [x] 3.1 `python scripts/validate_openspec.py` exits 0
- [x] 3.2 No file matched by `scripts/release_play_path.txt` is modified (`git diff --name-only` against the merge base)
- [x] 3.3 `ruff check .` and `OMP_NUM_THREADS=2 python3 -m pytest tests/regression/test_default_env_golden_regression.py -q` pass
