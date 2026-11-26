# DEV-SQE → Pre-PR Handoff

**Date:** 2026-11-14
**Spec:** openspec/changes/add-gpu-first-run-cards/proposal.md (plan approved by the maintainer in session; decisions: v2.1 close-out runs, runs on the maintainer's PC, GPU for every phase, both GPUs)
**PR:** #138 (draft) into `claude/civ6-alphazero-implementation-vMnun`, branch `claude/cool-lovelace-o02vn9`
**Session:** session_01CXusP6jtsBXpQcjpC9W3b8

## What changed

Registered runs are GPU-first: `cuda:0` trains, `cuda:1` judges, on a pinned CUDA torch
build. Six commits, 52 files (+6,391 / −299) on top of `cd89a00`.

- **Driver.** Run cards can pin a torch build (`torch_spec`, `torch_index_url`), name the
  GPU expected at each ordinal (`gpus:`), set `cuda_visible_devices`, and be rehearsed with
  `--probe`. The device each invocation uses is read from the runner's own argument parser,
  and a card that pins torch must name every device. The preflight records the build's
  architecture list, each GPU's compute capability, cuDNN and driver versions, and runs one
  real CUDA operation per device.
- **Determinism.** A seeded run on CUDA turns on deterministic algorithms
  (`enforce_deterministic_backend`); a no-op on CPU, so the CPU gates are unchanged.
- **Registration.** Section 7.1 of `docs/EVALUATION.md` is re-registered (digest
  `dba90481…5d91`) with a machine-readable `a1-devices` block; the three cards equal it by
  test; the release verifier voids a run whose device plan departs from it.
- **Re-run rule.** A re-run after an interruption is a reproduction check
  (`alpha_zero/evaluation/reproduction.py`); one that differs cannot PASS.
- **GPU determinism check.** `scripts/check_determinism.py` (runner `determinism_check`) is
  T0b's first invocation: an A1-shaped job twice, each repeat in its own process, in the
  pinned environment.
- **Structure.** `alpha_zero/local/` split into `card.py`, `devices.py`, `run_card.py`
  (877 lines, from 1,004), with no debt-baseline change.
- **Harness and docs.** Skill `.claude/skills/a1-reregister`, `make verify-registration`,
  `a1_protocol digest`, C4 component view, README, ROADMAP, MODULE_REFERENCE, both
  `Agent.md` files, CLAUDE.md, CHANGELOG, runbook, OpenSpec change.

**Files Changed:** `alpha_zero/local/{card,devices,run_card,pinned_env}.py`,
`alpha_zero/utils/{device,seeding,data_root,game_budget}.py`, `alpha_zero/training_civ6.py`,
`alpha_zero/evaluation/{a1_protocol,a1_registration,benchmark,benchmark_evidence,reproduction}.py`,
`scripts/{check_determinism,calibrate_throughput,verify_release_evidence}.py`,
`configs/run_cards/{a1,a1r,t0b}.yaml`, `docs/EVALUATION.md`, `docs/architecture/C4_ARCHITECTURE.md`,
`docs/runbooks/branching-and-release.md`, `docs/MODULE_REFERENCE.md`, `README.md`, `ROADMAP.md`,
`CHANGELOG.md`, `CLAUDE.md`, `Makefile`, `.claude/README.md`, `.claude/skills/a1-reregister/SKILL.md`,
`alpha_zero/{local,evaluation}/Agent.md`, `openspec/changes/add-gpu-first-run-cards/`.

**Tests Added:** `tests/unit/test_local_devices.py`, `tests/unit/test_reproduction.py`,
`tests/scripts/test_check_determinism.py`, `tests/scripts/test_verify_release_evidence_devices.py`;
rewritten `tests/regression/test_cuda_determinism_regression.py` (needs CUDA; a skip is not a
pass); extended `tests/scripts/test_run_card.py`, `tests/unit/test_pinned_env.py`,
`tests/unit/test_cuda_determinism_wiring.py`, `tests/scripts/test_a1_registration.py`,
`tests/scripts/test_t0b_calibration.py`, `tests/regression/test_a1_registration_digest_regression.py`,
`tests/unit/test_benchmark_producer.py`, `tests/unit/test_game_budget.py`.

## Test Results (head `68b6c5c`, this container: Python 3.11, torch 2.15.0 CPU, 2 vCPUs, `OMP_NUM_THREADS=2`)

| Suite | Passed | Failed | Skipped |
|-------|--------|--------|---------|
| unit | 10,468 | 1 (pre-existing) | 58 (+6 xfailed) |
| regression + governance + scripts + security | 1,927 | 7 (pre-existing) | 7 |
| integration + e2e + cloud + smoke + sanity + inference | 1,559 | 0 | 18 (+1 xfailed) |

The eight failures also fail on the base commit `cd89a00` in this container and are not
caused by this branch:
`tests/unit/test_pinned_env.py::test_git_archive_of_head_is_extracted_and_read_only` (the
container runs as root, so a read-only tree is still writable), six
`tests/scripts/test_deployment_infra.py::TestEnvFile::*` (need a local `.env`), and
`tests/scripts/test_gcp_diagnose.py::TestCheckPythonAuth::test_adc_available` (needs Google
credentials). CI is the gate for these.

**Coverage on changed modules** (path-based `coverage run --include`, because pytest-cov
crashes in this sandbox; CI's coverage job is the gate for the global 85%):
`devices.py` 100%, `seeding.py` 100%, `isolation.py` 100%, `a1_protocol.py` 96%,
`a1_registration.py` 96%, `reproduction.py` 95%, `card.py` 94%, `pinned_env.py` 91%,
`run_card.py` 90%; 94% over the nine files (1,599 statements).

**Lint Status:** clean (`ruff check .`, `ruff format --check .`)
**Type Check Status:** clean (`python -m mypy alpha_zero/`, 345 files)
**Validators:** `validate_openspec`, `validate_skills`, `validate_agents` (22 warnings, all
pre-existing), `validate_claude_agents`, `check_doc_sync`, `validate_kb --git` exit 0;
`make verify-registration` exit 0 (three digests agree, 98 tests); debt ratchet passes with
no baseline change; gitleaks: no leaks over 776 commits.
**Smoke Tests:** `python -m alpha_zero.training_civ6 --dry_run` exit 0;
`python -m alpha_zero.orchestrator --preset fast --dry_run` exit 0;
`python -m alpha_zero.distributed.coordinator --dry_run` times out at 150 s with
`BrokenPipeError` in the actor processes **on the base commit as well as on this head**
(pre-existing in this sandbox).
**Card prices:** `t0b` 660, `a1` 980, `a1r` 980, `smoke` 8 (all under the 1,000-game cap).

## Negative controls

Every new gate was shown to fail on a planted defect before it was accepted. The ones that
decided a design:

| Planted defect | Test that fails |
|----------------|-----------------|
| Device read from argv tokens instead of the runner's parser (`--dev cuda:1`) | `test_run_card.py` device-resolution tests |
| A pinned card with an invocation that names no device | `refuse_implicit_devices` tests |
| Build reports CUDA available but has no kernels for the GPU's architecture | `test_local_devices.py` architecture tests |
| A card's torch pin or GPU names drift from section 7.1 | `test_a1_registration.py` card-equals-registration tests |
| Run recorded on other hardware or another torch build | `test_verify_release_evidence_devices.py` |
| A `--probe` record offered as release evidence | `test_verify_release_evidence_devices.py` |
| `enforce_deterministic_backend` removed from the trainer or the arena | `test_cuda_determinism_wiring.py` |
| One checkpoint tensor, one iteration record or one arena game changed | `test_reproduction.py` |
| Edit inside the hashed region without a registration-log row | `test_a1_registration_digest_regression.py` |

## Independent reviews of record

Four contexts that did not write the code reviewed it: a general QA review, `sdlc-architect`,
`sdlc-ml-scientist`, and `sdlc-developer` (which built the reproduction tooling to the ML
scientist's finding). Every must-fix and high finding is addressed; CLAUDE.md section 5.2
(2026-11-14, "What four independent reviews changed") records what each found.

## CONFIRM-FIRST items

- **GPU re-registration of section 7.1**: instructed by the maintainer in session
  (GPU for every phase, both GPUs). Allowed because no A1-family run has started.
- **A re-run that fails to reproduce cannot PASS** (reads INCONCLUSIVE and needs a new
  confirmatory seed pair): taken in the approved plan; the maintainer may overrule.
- **T0b's CPU and second-GPU timings are informational**: no automatic switch back to CPU.
- **`download.pytorch.org` as a package source** for the pinned environment on the
  maintainer's machine: named in the approved plan.

## Known Issues

- **Nothing on this path has run on a GPU or on Windows.** The first execution is the
  maintainer's `python scripts/run_card.py configs/run_cards/t0b.yaml --probe`, then the T0b
  card, whose first invocation must report `REPRODUCED`.
- The release verifier does not yet read the reproduction-compare record; until it does, the
  re-run rule is enforced by review of the committed output.
- The torch wheel's sha256 is not recorded (version and CUDA tag are).
- A card refused at preflight still uses up its single-use run directory; probe first.
- Arena game records hold only win, draw or loss, so the arena half of the determinism check
  is coarse.
- The cards' `sha` is still all zeros; pinning it and filling the "pending T0b" slots is one
  later PR, through the `a1-reregister` skill.

## For Pre-PR

1. Confirm CI on the pushed head (the eight local failures above should pass or skip there).
2. Check that the PR body's CONFIRM-FIRST section matches this document.
3. Do not run T0b, A1 or A1-R: they need the maintainer's GPU machine.
