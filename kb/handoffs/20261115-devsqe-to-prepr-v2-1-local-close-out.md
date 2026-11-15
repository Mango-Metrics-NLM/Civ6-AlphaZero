# DEV-SQE → Pre-PR Handoff

**Date:** 2026-11-15
**Spec:** `kb/specs/20261111-local-first-and-branching-plan.md` (v2.1); OpenSpec `openspec/changes/add-gpu-first-run-cards` (tasks 4.8–4.10, section 4c)
**PR:** #140 (draft), branch `claude/cool-lovelace-o02vn9`, base `claude/civ6-alphazero-implementation-vMnun` at `6d548d8`
**Code head when written:** `2e51c2c` (the commit that adds this file changes `kb/` only)

## What this is

The v2.1 work that needs no GPU: the three follow-ups left open on PR #138 / #139, one
hazard found while planning them, and post-merge housekeeping. The scope is the plan the
maintainer approved in the session ("v2.1 local close-out").

1. **Run cards rehearse first (task 4.10).** `run_card` runs the pin, the preflight and the
   device check in a throwaway `runs/probe-<card_id>-<stamp>` directory, and claims
   `runs/<card_id>` only when that passes. A used directory is refused before the
   rehearsal. `run.json` gains `rehearsal` and `launched` (written before each child
   starts). The claim is one atomic `mkdir`.
2. **Torch wheel record (task 4.9).** `pip install --report`; the torch entry's URL and
   sha256 go into the venv marker and `run.json` as `torch_wheel`. Record only.
3. **No torn files.** `alpha_zero/utils/atomic_io.py`; `Civ6Network.save`, the benchmark's
   `result.json`, the in-loop `eval_<iter>.json` and `run.json` are written through a
   temporary file.
4. **Driver record.** A card that pins torch and names a CUDA device is refused when the
   NVIDIA driver version cannot be read.
5. **The release verifier applies the "After a VOID" rule (task 4.8).**
   `reproduction compare --out` names both runs and carries its digests;
   `build_release_bundle.py --compare`; `scripts/release_reproduction.py` holds the rule;
   `verify_release_evidence.py` recomputes the comparison and the verdict. A role's
   attempts are ordered by `started_at` alone; the role's non-VOID attempt must be the last
   of its role to start; no two attempts of a role may tie; nothing may start after a
   non-reproducing re-run.
6. **OpenSpec archive.** Five merged changes archived; baselines `search-budget` and
   `training-evaluation` created.

**No edit inside the hashed region of `docs/EVALUATION.md`**; the digest is still
`e460a353651a24c26d5e1891b6ad5bd27c18acac4fd9fd6367bc262175237c17` and the file has no
diff against the base.

## Files Changed

19 commits, 61 files (+5,177 / −216) before this handoff commit.

- Package: `alpha_zero/local/run_card.py`, `pinned_env.py`, `devices.py`;
  `alpha_zero/utils/atomic_io.py` (new); `alpha_zero/core/civ6_network.py`;
  `alpha_zero/evaluation/benchmark.py`, `reproduction.py`, `a1_protocol.py`;
  `alpha_zero/training_civ6.py`.
- Scripts: `scripts/release_reproduction.py` (new), `scripts/build_release_bundle.py`,
  `scripts/verify_release_evidence.py`.
- Docs and governance: `CHANGELOG.md`, `CLAUDE.md`, `ROADMAP.md`,
  `docs/evidence/SCHEMA.md`, `docs/runbooks/branching-and-release.md`,
  `docs/MODULE_REFERENCE.md`, `docs/architecture/C4_ARCHITECTURE.md`, two `Agent.md` files,
  `openspec/` (five archives, two new baselines, the open change's tasks and deltas).

## Tests Added

- `tests/scripts/test_run_card_rehearsal.py` (new)
- `tests/unit/test_atomic_io.py` (new)
- `tests/scripts/test_verify_release_evidence_reproduction.py` (new)
- `tests/scripts/test_verify_release_evidence_ordering.py` (new)
- additions to `tests/unit/test_pinned_env.py`, `tests/unit/test_local_devices.py`,
  `tests/unit/test_reproduction.py`, `tests/unit/test_a1_verdict.py`,
  `tests/scripts/test_release_evidence_e2e.py`, `tests/scripts/test_verify_release_evidence.py`,
  `tests/scripts/release_evidence_fixture.py`, `tests/scripts/test_run_card.py` (in place),
  `tests/scripts/test_release_play_path_closure.py` (one exemption)

## Test Results

CPU-only container, Python 3.11, `OMP_NUM_THREADS=2`, pytest without the coverage plugin
(it crashes here). No file under `alpha_zero/`, `tests/unit/` or `tests/integration/`
changed after the unit and integration runs.

- Unit (`-m "not slow"`): 10,546 passed, 1 failed, 58 skipped, 6 xfailed
- Integration + e2e + smoke + sanity + inference: 740 passed, 16 skipped, 1 xfailed
- Regression + governance + scripts + security + cloud: 2,998 passed, 7 failed, 9 skipped (with the two verifier unit modules re-run on the final head)
- Coverage on changed modules, measured by path with each module's own tests:
  `run_card.py` 92%, `pinned_env.py` 93%, `devices.py` 100%, `atomic_io.py` 96%,
  `civ6_network.py` 96%, `benchmark.py` 90%, `reproduction.py` 96%, `a1_protocol.py` 96%,
  `release_reproduction.py` 98%, `build_release_bundle.py` 94%,
  `verify_release_evidence.py` 86%. `training_civ6.py` (one changed call) was not measured
  by path.
- The global 85% gate and the cloud floor were **not** measured here. CI measured them:
  all 24 checks on `d44f7ee` completed, 23 passed and `pip-audit` was skipped by design.

Failures in this container that also fail on the base commit and are outside this change
(8): `tests/unit/test_pinned_env.py::test_git_archive_of_head_is_extracted_and_read_only`
(the container runs as root); six `tests/scripts/test_deployment_infra.py::TestEnvFile`
tests (no `.env`); `tests/scripts/test_gcp_diagnose.py::TestCheckPythonAuth::test_adc_available`
(no Google credentials).

**Lint Status:** `ruff check .` and `ruff format --check .` clean (981 files)
**Type Check Status:** `python -m mypy alpha_zero/` clean (346 files)
**Smoke Tests:** `python -m alpha_zero.training_civ6 --dry_run` exit 0;
`python -m alpha_zero.orchestrator --preset fast --dry_run` exit 0.
`python -m alpha_zero.distributed.coordinator --dry_run` times out in this container on
the base commit too and was not used as a gate.
**Other gates:** `make verify-registration` exit 0; `validate_openspec`, `validate_skills`,
`validate_agents`, `validate_claude_agents`, `check_doc_sync`, `validate_kb --git` exit 0;
debt ratchet passes with no baseline change; `make security-secrets` no leaks; card prices
660 / 980 / 980 / 8.

## Negative controls

Every new test was seen to fail against the defect it targets; the lists are in the test
modules' docstrings. Run cards and rehearsal 10, wheel record 8, atomic writes 4, driver 3,
re-run rule 44 (re-run after the last rule change by `sdlc-developer`).

## Independent reviews (separate contexts)

1. `sdlc-ml-scientist`, on whether the first verifier matched the registered text. HIGH:
   attempts were ordered by `started_at` with the bundle's order as a fallback, so an
   interrupted attempt could be left needing no record. MEDIUM: a second draw after a
   non-reproducing re-run could PASS. Also: torch wheel not compared; three overstatements
   in `docs/evidence/SCHEMA.md`.
2. `sdlc-qa-sdet`, on tests and negative controls: "ready with fixes". Fixed: atomic
   claim; tests for a failure after a passed rehearsal, an interrupted rehearsal, two
   interrupted attempts before one re-run, a report naming torch twice, a driver differing
   on one GPU, null identities in a record.
3. `sdlc-ml-scientist` again, on the fix. HIGH: with one edited `started_at` on the second
   draw the first fix was got around. MEDIUM: ties let the listing pick the re-run. Fixed
   by one rule (the role's non-VOID attempt is the last of its role to start) and by
   refusing ties.
4. The same reviewer confirmed the three findings closed at `8207eae`, by reading (its
   shell cannot run pytest). The edited-timestamp arrangements that still verify are all
   equivalent to leaving the non-reproducing re-run out of the listing, which is a stated
   limit.

## Known Issues

- **Nothing here has run on a GPU or on Windows.** The rehearsal, the driver check and the
  pip report are exercised with fakes and one real CPU `pip install --report`.
- An attempt that is not listed is invisible to a bundle verifier, and `started_at` is the
  operator's own record. Checkpoint digests in a compare record are not anchored to the
  bundle. All stated in `docs/evidence/SCHEMA.md`.
- Not modelled: the bundle section 7.1 requires after a non-reproducing re-run (OpenSpec
  task 4c.7). Not checked: three items in task 4c.11.
- Conservative readings the maintainer may overrule (`docs/evidence/SCHEMA.md`): a re-run
  at another commit, device plan, torch wheel or driver cannot PASS even when nothing was
  left to compare; a VOID by device-plan departure or an `MCTS_*` override has no re-run
  that can PASS; the cutoff after a non-reproducing re-run is its `started_at`; a pinned
  GPU card is refused when the driver cannot be read.

## Next

Maintainer: review PR #140; on the GPU machine,
`python scripts/run_card.py configs/run_cards/t0b.yaml --probe`, then the T0b card. v2.2
needs its own task plan.
