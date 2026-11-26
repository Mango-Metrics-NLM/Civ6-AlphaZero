# DEV-SQE → Pre-PR Handoff

**Date:** 2026-11-12
**Spec:** kb/specs/20261111-local-first-and-branching-plan.md (gap register: kb/specs/20261111-local-first-gap-register.md)
**PR:** draft #135, `claude/local-deployment-branching-plan-51a0cb` → `claude/civ6-alphazero-implementation-vMnun`, head 825e5c3

**Files Changed:** 98 commits, 235 files (+38,630 / −1,163). By task:
- **T0a** calibration harness, Windows baseline evidence
- **T5** `alpha_zero/utils/game_budget.py` plus wiring into every runner except the distributed coordinator (CONFIRM-FIRST; the guard asks)
- **T1** `alpha_zero/utils/data_root.py`, `alpha_zero/local/pinned_env.py`
- **T2** `alpha_zero/local/isolation.py`, plus guards in both GCS backends
- **T1b** `alpha_zero/local/run_card.py`, `scripts/run_card.py`, `configs/run_cards/`
- **T3a** `PROJECT_CHARTER.md` amendment A-1 (unratified), `ROADMAP.md`
- **T4** `.claude/hooks/pretooluse_guard.sh`, `.claude/settings.json`, four workflows
- **T6** `scripts/check_branch_direction.py`, `branch-policy.yml`, `.github/rulesets/*.json`, `release.yml` (verify-tag, wait-for-ci), `scripts/release_version.py`, `scripts/release_checks.py`, `scripts/maintainer/*`
- **T3** CONTRIBUTING, CLAUDE.md §3.2/§5/§15, `.claude/README.md`, release and pre-pr skills, PR template, `docs/runbooks/branching-and-release.md`
- **T7** move mask in `core/civ6_actions.py`
- **T8** `gumbel_ops.halving_schedule` / `planned_halving_sims`, `SearchStats`, five `training_civ6` search flags
- **T9** checkpoint provenance, `HorizonMismatchError`, `--bootstrap`, `checkpoint_0.pt`, `ENV_VERSION = 2`
- **T10** `--eval_interval` / `--eval_games`; `evaluate_against_random` now returns `ScoreSummary` (**BREAKING**, float shim kept); prequential diagnostics; `training/run_record.py`; `CIV6AZ_GIT_SHA`
- **T11** schema-2 benchmark evidence, pair/hull intervals, `--candidate`, `--shard`/`--merge`, `evaluation/benchmark_evidence.py`
- **T12** `scripts/probe_target_sharpness.py`, completed-Q and seed-overlap memos
- **T13** EVALUATION.md section 7.1 (sha 79a4ba54…8a4b), `configs/run_cards/a1.yaml` / `a1r.yaml`, `evaluation/a1_protocol.py`, `scripts/check_learning_trend.py`; two OpenSpec changes archived
- **T14** `scripts/verify_release_evidence.py`, `evaluation/a1_registration.py`, `release-evidence.yml`, `main.json` Release Evidence
- **T15** eight `.claude/agents/sdlc-*.md`, read-only, qa and write guards, `scripts/validate_claude_agents.py`
- **T16** loopback bind and CORS (`utils/network.py`, web, serving)
- **T18** forced-win oracle over batch_gumbel/MuZero; batch sign, lockstep and target fixes; MuZero root-Q and target fixes
- **Final wave** CHANGELOG `[Unreleased]`; `scripts/build_release_bundle.py`; `configs/run_cards/t0b.yaml` plus `a1_protocol calibrate`; play-path import-closure gate

**Tests Added:** 57 test files added or changed. Main suites:
- `test_game_budget`, `test_data_root`, `test_pinned_env`, `test_local_isolation`, `test_run_card`
- `test_pretooluse_guard`, `test_readonly_shell_guard`, `test_validate_claude_agents`
- branch-direction and rulesets regressions, `test_release_scripts`, `test_verify_release_evidence`
- `test_action_mask_collision`, `test_search_budget`, `test_checkpoint_provenance`, `test_eval_summary_and_diagnostics`
- benchmark producer, A1 registration, digest and seed-disjointness tests, `test_check_learning_trend`
- `test_forced_win_oracle`, `test_local_exposure`
- the real-producer end-to-end release test and the play-path closure test

**Test Results** (`make verify` at 2a54941, Git Bash, worktree on PYTHONPATH):

| Gate | Result |
|---|---|
| Unit tests | 8,809 passed; 1 failed (environment only, see Known Issues) |
| Governance tests | 767 passed |
| Coverage | 89.82% (gate 85%) |
| Cloud coverage floor | 60.0% (floor 58%) |
| gitleaks | no leaks |
| LLM eval gate | PASS |

The final fix wave (2a54941..825e5c3) was verified by focused suites: 1,879 passed, and the 7 failures are local `.env` and Windows-path environment issues. The whole-branch re-review passed 483 tests.

**Lint Status:** clean (`ruff check .`; `ruff format --check .` with the pinned 0.15.8).

**Type Check Status:** 23 errors locally, all from local numpy 1.26.4 and jax stubs, and present before this branch. CI's Lint & Type Check is green on PR #135.

**Smoke Tests:**
- `training_civ6 --dry_run` passes, including with `--mcts gumbel --gumbel_k 8 --gumbel_sims 24` and `--bootstrap none`.
- `run_card --price_only` reports a1/a1r = 980 and t0b = 604.

**Known Issues:**
- `test_seed_is_stable_across_processes` fails locally because its subprocess imports the main checkout's stale editable install.
- Branch Direction is red on #135 by design until cut-over step 5.
- The T4 guard has never run live. Sessions load hooks from the main checkout, so the maintainer must probe it in a fresh session after merge.
- The `.claude/agents` frontmatter hooks also need one live dispatch check (`.claude/README.md`).
- The MuZero reward-sign defect is recorded in CLAUDE.md §5.3, xfailed with its true reason. The fix is CONFIRM-FIRST.
- The distributed coordinator is unpriced (CONFIRM-FIRST).
- M-9 (re-running the verifier in `release.yml`) is open as OpenSpec task 5.7.
- EVALUATION.md section 7.1 slots wait on T0b. The A5/A12 decision is the maintainer's; Option A is registered.
