# DEV-SQE → Pre-PR Handoff

**Date:** 2026-11-19
**Spec:** openspec/changes/add-default-env-golden-fingerprint/proposal.md (program spec `kb/specs/20261119-game-mechanics-plan.md` is in PR #142, not on this branch)
**PR:** draft #143, branch `claude/gm-s0a-golden-fingerprint`
**Files Changed:**
- `tests/regression/test_default_env_golden_regression.py`
- `tests/fixtures/golden/default_env_v2.json`
- `openspec/changes/add-default-env-golden-fingerprint/proposal.md`
- `openspec/changes/add-default-env-golden-fingerprint/tasks.md`
- `openspec/changes/add-default-env-golden-fingerprint/specs/training-reproducibility/spec.md`

**Tests Added:** `tests/regression/test_default_env_golden_regression.py` (34 tests)
**Test Results:** 34 pass in the new module; 60 pass with `tests/regression/test_suite_wiring_regression.py` and `tests/unit/test_microciv_clone.py`; `tests/regression/test_debt_ratchet.py` passes. 0 fail. Coverage: not applicable, no file under `alpha_zero/` changed.
**Lint Status:** `ruff check .` clean; the new module is formatted. Locally `ruff format --check .` flags two files this change does not touch (one is `docs/templates/CLAUDE_TEMPLATE.md`); CI's lint job was green on this branch.
**Type Check Status:** not run locally for this change (no file under `alpha_zero/` changed); CI's job was green.
**Smoke Tests:** not run; nothing they exercise changed.
**Verification not done:** `make verify` and `make test-all` in full. CI is the gate; it was green on `52b1efa`, and the head `c9d6e55` (atomic fixture writer) was pushed after that.
**Negative controls:** in the suite: warrior cost +1 moves 16 of 20 digests; settler auto-found 0.10 to 0.15 moves 3 of 20; one perturbed action moves the digest. Throwaway: a production constant +1 fails `test_digest_matches_fixture` with the "neither moved" diagnosis. Atomic writer: with the old writer restored, the new failed-regeneration test fails.
**Known Issues:**
- The OpenSpec scenario "numpy is upgraded" speaks of the numpy version; the test keys on the Generator stream canary.
- `tasks.md` 2.1's scratch-copy control exists as the throwaway record, not as a permanent test.
- If numpy's stream and the environment change in one commit, `--regenerate` would rewrite the fixture.
- Rewards, observations and the encoder are outside the digest by design.
- This branch and PR #142 each append to `kb/ledger/ledger.jsonl`; the second to merge will conflict there until the default branch is merged into it.
