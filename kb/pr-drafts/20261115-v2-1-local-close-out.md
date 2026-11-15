# PR: v2.1 local close-out: rehearsal-first run cards, the verifier's re-run rule, torch wheel record, atomic writes, OpenSpec archive

**PR:** [#140](https://github.com/Mango-Metrics-NLM/Civ6-AlphaZero/pull/140) (draft) ·
**Branch:** `claude/cool-lovelace-o02vn9` → `claude/civ6-alphazero-implementation-vMnun`
**Spec:** [`kb/specs/20261111-local-first-and-branching-plan.md`](../specs/20261111-local-first-and-branching-plan.md), OpenSpec `openspec/changes/add-gpu-first-run-cards`
**Handoff:** [`kb/handoffs/20261115-devsqe-to-prepr-v2-1-local-close-out.md`](../handoffs/20261115-devsqe-to-prepr-v2-1-local-close-out.md)

## Target
- **Base branch:** `claude/civ6-alphazero-implementation-vMnun` (the current default; `dev` after the cut-over)
- **PR type:** feature (squash). First row of the table in the `pre-pr-handoff` skill; the type and base have not changed since the PR was opened, so no release-manager dispatch was made.

The sections below are the PR description as written at code head `2e51c2c`.

## Summary

The v2.1 work that does not need the GPU machine, following #138 and #139: the three follow-ups recorded as open there (OpenSpec `add-gpu-first-run-cards` tasks 4.8, 4.9, 4.10), one hazard found while planning them, and post-merge housekeeping.

- **A run card rehearses before it claims its run directory (4.10).** `run_card` used to claim `runs/<card_id>` before the pinned environment, the preflight and the device check, so a refused device or a failed preflight used up the `card_id`. A run now rehearses first, exactly as `--probe` does, in a throwaway `runs/probe-<card_id>-<stamp>` directory, and claims the card's directory only when that passes. A failure after the rehearsal passed still uses up the card; an interrupted rehearsal does not. The claim is one atomic `mkdir`. `run.json` gains `rehearsal` and `launched` (invocation names written before each child starts; `invocations` holds only finished ones).
- **The torch wheel's sha256 is recorded (4.9).** The pinned install asks pip for its installation report; the torch entry's URL and sha256 go into the venv marker and `run.json` as `torch_wheel`. It is a record: a missing report never fails a run, and it is not part of the torch pin the verifier compares with the registration.
- **An interrupted run leaves no torn file.** Checkpoints (`Civ6Network.save`) and the benchmark's `result.json` are written through a temporary file (`alpha_zero/utils/atomic_io.py`). File contents are unchanged. This matters because the re-run rule compares what an interrupted attempt left behind, and a torn file would read as "does not reproduce".
- **A pinned GPU run records its driver.** A card that pins torch and names a CUDA device is refused when `nvidia-smi` cannot report the driver version, because a re-run is compared under the same driver.
- **The release verifier applies the "After a VOID" re-run rule itself (4.8).** `reproduction compare --out` now names both runs and carries the digests it compared. `build_release_bundle.py --compare FILE` attaches it. `verify_release_evidence.py` requires a record for an attempt that was interrupted after it launched something, recomputes the comparison from the record's digests, checks those digests against the bundled arena results and iteration records, and reads a PASS from a non-reproducing re-run as INCONCLUSIVE ("non-reproducing re-run"). A role's attempts are ordered by `started_at` alone: every listed attempt must record one, no two attempts of a role may tie, the role's non-VOID attempt must be the last of its role to start, and nothing may start after a non-reproducing re-run. The shared logic is in `scripts/release_reproduction.py`, outside the release play path on purpose.
- **Five merged OpenSpec changes are archived** and their requirements folded into the baseline specs (two new capabilities: `search-budget`, `training-evaluation`).

**No edit inside the hashed region of `docs/EVALUATION.md`.** The registration stays at digest `e460a353…7c17` (`make verify-registration` exits 0; the file has no diff against the base).

**Nothing here has run on a GPU or on Windows.** The rehearsal, the driver check and the pip report are exercised with fakes and one real CPU `pip install --report` (pip 24.0).

## Target branch and PR type

- [x] Feature / fix / chore / docs -> `dev` (squash)

The cut-over has not happened, so this targets the current default branch, `claude/civ6-alphazero-implementation-vMnun`.

## Game budget

No run. Card prices are unchanged: `t0b.yaml` 660, `a1.yaml` 980, `a1r.yaml` 980, `smoke.yaml` 8.

## Tracking issue

None. These are the follow-ups recorded as open on #138 / #139 plus post-merge housekeeping.

## Linked OpenSpec change

`openspec/changes/add-gpu-first-run-cards` (tasks 4.8–4.10 and section 4c; added requirements for the rehearsal, the driver and wheel record, the verifier's re-run rule and torn-file safety; two follow-ups left open as 4c.7 and 4c.11). Archived in this PR: `benchmark-evidence-producer`, `checkpoint-provenance`, `eval-decoupling-diagnostics`, `fix-gumbel-budget`, `fix-move-mask`.

## Hot files touched

`CLAUDE.md` (§5.1 rows, one §5.2 decision row, four §5.4 rows, §13, archive citations, footer date), `ROADMAP.md` ("Now"), `CHANGELOG.md` (`[Unreleased]`), `kb/ledger/ledger.jsonl` (two appended lines).

## Verification

In a CPU-only container (Python 3.11, `OMP_NUM_THREADS=2`), on the final code:

- [x] `ruff check .`, `ruff format --check .`: clean (981 files)
- [x] `python -m mypy alpha_zero/`: no issues in 346 files
- [x] Unit (`-m "not slow"`): 10,546 passed, 1 failed, 58 skipped, 6 xfailed
- [x] Integration, e2e, smoke, sanity, inference: 740 passed, 16 skipped, 1 xfailed
- [x] Regression, governance, scripts, security, cloud: 2,998 passed, 7 failed, 9 skipped
- [x] `make verify-registration`: exit 0 (digest unchanged)
- [x] `validate_openspec`, `validate_skills`, `validate_agents`, `validate_claude_agents`, `check_doc_sync`, `validate_kb --git`: exit 0
- [x] Debt ratchet: passes, no baseline change
- [x] `make security-secrets`: no leaks
- [x] `python -m alpha_zero.training_civ6 --dry_run` and `python -m alpha_zero.orchestrator --preset fast --dry_run`: exit 0
- [x] Coverage on the changed modules, measured by path with each module's own tests: `run_card.py` 92%, `pinned_env.py` 93%, `devices.py` 100%, `atomic_io.py` 96%, `civ6_network.py` 96%, `benchmark.py` 90%, `reproduction.py` 96%, `a1_protocol.py` 96%, `release_reproduction.py` 98%, `build_release_bundle.py` 94%, `verify_release_evidence.py` 86%
- [ ] `make verify` as one command and the global 85% gate: not run here (the coverage plugin crashes in this container); CI is the gate, and its coverage job passes

The eight failures also fail on the base commit in this container and are outside this change: `test_pinned_env.py::test_git_archive_of_head_is_extracted_and_read_only` (the container runs as root), six `test_deployment_infra.py::TestEnvFile` tests (no `.env`), and `test_gcp_diagnose.py::TestCheckPythonAuth::test_adc_available` (no Google credentials). No file under `alpha_zero/`, `tests/unit/` or `tests/integration/` changed after the unit and integration runs.

## Negative control recorded

Each defect was planted, the listed tests failed, and the change was restored. They are listed in the test modules' docstrings.

| Area | Planted defects |
|---|---|
| Rehearsal (`tests/scripts/test_run_card_rehearsal.py`) | no rehearsal; no early refusal of a used directory; rehearsal not recorded; `launched` written after the child; CLI line removed; the claim letting `FileExistsError` escape; the real pass reporting `claimed=False`; an interrupted rehearsal falling through to the run |
| Wheel hash (`tests/unit/test_pinned_env.py`) | `--report` dropped; marker without the key; parser reading `hashes` only; parser raising on a malformed report; key made part of the venv identity; `run.json` dropping it; the first of two torch entries taken; no warning for an environment built before the record |
| Atomic writes (`tests/unit/test_atomic_io.py`) | write straight to the target; create that overwrites; checkpoint saved directly; `result.json` written in place |
| Driver (`tests/unit/test_local_devices.py`) | an error record read as "no GPUs"; the driver demanded without a pin; `run_card` not passing the report |
| Re-run rule (`tests/scripts/test_verify_release_evidence_reproduction.py`, `test_verify_release_evidence_ordering.py`, `tests/unit/test_reproduction.py`, `test_a1_verdict.py`) | 44 controls, all re-run after the last rule change. Among them: the anchoring of record digests to bundled files removed (a record that left out the differing arm then verified with PASS); a non-reproducing PASS not demoted, or demoted without its detectable-effect numbers; the non-VOID attempt not required to be last (a second draw dated before a non-reproducing re-run then verified with PASS); ties not refused; `started_at` not required; nothing checked after a non-reproducing re-run; the torch wheel not compared; a null identity matching a null; only the first GPU's driver compared |

## CONFIRM-FIRST items and who authorized

The scope is the plan the maintainer approved in this session ("v2.1 local close-out"), which listed each item below as the maintainer's to overrule.

- **The verifier change spans more than three modules** and makes a bundle with an interrupted attempt require a comparison record. No release bundle exists yet, so nothing shipped is affected.
- **A crash that needs a code fix costs the seed pair.** The registered text covers a re-run at the same commit only; it is implemented literally. A re-run at another commit, device plan, torch wheel or driver cannot PASS, even when nothing had been compared.
- **After a non-reproducing re-run** section 7.1 requires a new registration and a bundle listing both attempts. The verifier does not model that bundle yet (OpenSpec task 4c.7); any attempt listed as started after a non-reproducing re-run fails verification until it does.
- **A GPU card is refused when the driver version cannot be read.**
- **Atomic checkpoint and result writes** touch the play path (`alpha_zero/core/civ6_network.py`, `alpha_zero/evaluation/benchmark.py`). Contents are unchanged; this should land before the release candidate, or a finished run would read as stale.

Two more conservative readings came out of the reviews and are stated in `docs/evidence/SCHEMA.md` ("Conservative readings"), also the maintainer's to overrule:

- A VOID caused by a `device_plan` departure or an `MCTS_*` override has no re-run that can PASS, because its correct re-run necessarily runs under other conditions. Section 7.1 is silent on these two cases. Neither arises from an unedited registered card (`run_card` refuses an `MCTS_*` override and applies the card's device plan).
- The cutoff after a non-reproducing re-run is the re-run's `started_at`, not when it finished.

## Independent review

Four review passes ran in separate contexts; each finding was fixed on this branch.

1. **`sdlc-ml-scientist`** on the first verifier. HIGH: attempts were ordered by `started_at` with the bundle's listing order as a fallback, so recording the re-run as started earlier left the interrupted attempt needing no record. MEDIUM: a second draw after a non-reproducing re-run could PASS. Also the torch wheel was not compared, and `docs/evidence/SCHEMA.md` overstated three things.
2. **`sdlc-qa-sdet`** on tests and negative controls: "ready with fixes". Fixed: the run-directory claim is atomic; tests for a failure after a passed rehearsal, an interrupted rehearsal, two interrupted attempts before one re-run, a report naming torch twice, a driver differing on one GPU, and null identities in a record.
3. **`sdlc-ml-scientist`** again, on the fix. HIGH: one edited `started_at` on the second draw got around the first fix. MEDIUM: a tie let the listing order pick the re-run. Fixed by one rule (a role's non-VOID attempt is the last of its role to start) and by refusing ties.
4. **The same reviewer** confirmed those findings closed. It did so by reading the code; its shell cannot run pytest. The tests reproduce its arrangements and show the same bundles verifying with PASS when the order rules are switched off.

## Known limits

All stated in `docs/evidence/SCHEMA.md`.

- An attempt that is simply not listed in the bundle is invisible to a bundle verifier, and `started_at` is the operator's own record. The order rules refuse the orderings they name; a forged clock that stays consistent with them is not detectable.
- Checkpoint tensor digests in a comparison record are not anchored to anything in the bundle (checkpoints are not bundled).
- Not checked (OpenSpec task 4c.11): the torch wheel comparison is skipped when either run has no record; a control-VOID attempt's second draw is not required to be at another commit; a `started_at` is not checked against the commit date of its `sha`.
- The rehearsal directories (`runs/probe-…`) are records and may be deleted by hand.

## Notes

- T0b remains valid on the current default branch. After this merges, a card refused at its rehearsal no longer costs its `card_id`.
- When a re-run follows an interruption, run `reproduction compare` as soon as it finishes and before starting any other card (the runbook says so).
- Not in this PR: v2.2 (the local training loop), which needs its own task plan.

## Breaking Changes

None for a shipped artifact. For tooling: a release bundle that lists an interrupted attempt now needs a comparison record, every listed attempt needs a `started_at`, and a pinned GPU card needs a readable driver version. No release bundle exists yet.

## Checklist
- [x] All verification commands pass (the eight failures listed above also fail on the base commit in this container)
- [x] Coverage >= 85% on changed modules, by path; the global gate is CI's
- [x] CLAUDE.md updated
- [x] No hardcoded values (API keys, paths, magic numbers)
- [x] Type hints on all new public interfaces
