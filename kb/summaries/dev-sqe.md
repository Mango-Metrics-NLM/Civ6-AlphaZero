# DEV-SQE Session Log

## 2026-07-13 - W&B Retirement Sweep (MLflow-only tracking)
- **Session:** session_01AAR7bMpmkZtman9kFaeqft (third batch)
- **Task:** Complete W&B→MLflow migration: toggle unification (resolve_tracker_enabled, legacy WANDB_ENABLED honored+warned), WandbConfig→MLflowConfig bridge, single-tracker orchestrator init, shim + dead-suite deletions (user-approved), 66 gap-filling tests landed before deletions
- **Live bugs fixed:** run_*.py --wandb TypeError; adapter.log() dropped numpy metrics; MLflow run-group propagation never worked (.name/.id vs .info.*); create_wandb_logger mangled list tags
- **Files Changed:** 50 (+1,070/−2,674)
- **Tests:** 66 new/ported; targeted suites green; full CI-equivalent suite + 85% coverage gate run at ship time
- **Status:** complete — new PR from restarted branch `claude/project-charter-template-b4n9fu`
- **Next:** Pre-PR: confirm CI green; follow-ups: run_sweep.py Optuna port, setup_secrets.sh cleanup

## 2026-07-13 - Gap Analysis + Tech-Debt Sweep (coverage 85% gate, ruff-to-zero, mypy semantic fixes)
- **Session:** session_01AAR7bMpmkZtman9kFaeqft (same session as charter, second batch)
- **Task:** Branch-wide gap analysis and tech-debt enforcement: coverage gate 80→85 (CI measured 89.69%, local re-verified 88.88%), ruff 292→0 across alpha_zero/tests/scripts/infra, all semantic mypy errors fixed (180→157; rest is version noise), PR #91 review findings addressed
- **Real bugs found & fixed:** (1) `Civ6MctsPolicy.select_action` unpacked `GumbelMCTS.search` as `(action, stats_dict)` but real contract is `(policy_probs, root_value)` — guaranteed AttributeError on the live `--mcts` bridge path; both test files had drifted mocks hiding it. (2) `tests/unit/test_bridge_file_bridge.py` duplicate `TestSendCommandWithRetry` class shadowed 14 tests (never ran; all pass revived). (3) `train_epoch` profiler annotated as `TrainingProfiler` but real API is `PerformanceProfiler`. (4) 7 dangling asyncio tasks in bridge tests.
- **API improvements:** `GameBridge` Protocol in agent_loop.py (loop now transport-agnostic: FileBridge/LiveBridge/test doubles); `ActionConverter` accepts `Sequence` (covariance); `MCTSEnvironment.action_space`/`current_player` now read-only properties (backwards compatible).
- **Key discovery:** torch IS installable in web sessions via plain PyPI (`pip install torch`) — only the download.pytorch.org index is blocked; CLAUDE.md §5.3 corrected. Full suite now runs locally.
- **Tests:** 11,690 passed, 159 skipped, 1 xfailed, coverage 88.88% (85% gate green); 1 failure found in drifted mock, fixed, re-verified
- **Lint/format:** ruff 0 errors (both 0.15.8 local and CI-pinned 0.8.0); `ruff format --check alpha_zero/` green under 0.8.0 (do NOT format file_bridge.py/transformer_network.py with newer ruff — version skew)
- **Status:** complete — on branch `claude/project-charter-template-b4n9fu` (PR #91, now ready-for-review)
- **Next:** monitor PR #91 CI; remaining debt: W&B→MLflow sweep, omit-list shrink, unused-ignore noise under newer mypy

## 2026-07-13 - Project Charter Instantiation (docs-only)
- **Session:** session_01AAR7bMpmkZtman9kFaeqft
- **Task:** Instantiate PROJECT_CHARTER.md (7 non-goals NG-1..NG-7, 7 enforced invariants, carve-out amendment protocol), add templates/PROJECT_CHARTER_TEMPLATE.md and ROADMAP.md (M1–M5 themes), pointer edits to CLAUDE.md (§6.1, §13) and README.md
- **Files Changed:** PROJECT_CHARTER.md (new), templates/PROJECT_CHARTER_TEMPLATE.md (new), ROADMAP.md (new), CLAUDE.md, README.md, kb/specs/20260713-project-charter.md (new)
- **Tests:** none touched (docs-only); pytest/torch unavailable in this restricted session — CI is the gate; `ruff check alpha_zero/` = 1 pre-existing C420 (multiplayer/mcts.py:310), identical on clean HEAD
- **Status:** complete — on branch `claude/project-charter-template-b4n9fu`
- **Next:** Pre-PR validation; follow-ups noted in PR: CLAUDE.md v1.7.0 footer drift, pre-existing C420

## 2026-04-02 - Phase 8 Entry Point + Test Suite Fixes
- **Session:** 48dca8d8-7e46-41dc-b7f5-1ecb487c8400
- **Task:** Implement `alpha_zero/civ6_local.py` unified local gameplay entry point (Phase 8); fix test regressions for Atari CLI, hierarchical controller, DDP fixtures
- **Files Changed:**
  - `alpha_zero/civ6_local.py` (new — 733 lines)
  - `tests/unit/test_civ6_local.py` (new — 891 lines, 109 tests)
  - `tests/unit/test_atari/test_atari_cli.py` (fix: mock AtariTrainer.train to avoid MCTS hang)
  - `tests/unit/test_hierarchical/test_controller.py` (fix: fast_config fixture)
  - `tests/fixtures/mock_distributed.py` (fix: CPU-compatible DDP mock setup)
  - `kb/specs/20260402-full-civ6-local-gameplay.md` (new)
  - `kb/handoffs/20260402-planning-to-devsqe-full-civ6-local.md` (new)
- **Tests:** 428 pass, 2 skip (civ6_local: 109/109, atari: 186/186, DDP: 33/33 + 2 skip); ruff: 1 pre-existing error in hex_encoder.py (not in changed files)
- **Status:** complete — Phase 8 done; Phases 0–7 verified already implemented
- **Next:** Run Pre-PR validation; confirm convergence validation (Phase 0) with `scripts/validate_v130_training.py`

## 2026-03-01 (v1.3.0 Convergence Validation + Doc Reconciliation)

- Created `scripts/validate_v130_training.py` — local training convergence validator
- Added 4 regression tests in `tests/cloud/unit/test_training_worker.py` for weight loading
- Rewrote `NEXT_STEPS.md` — reconciled docs with actual 153K-line codebase state
- Updated `CLAUDE.md` sections 5.2 (arch decisions), 7 (project structure), 13 (next steps)
- Updated KB session state files (planning, dev-sqe, phase-6)
- Key finding: Many "planned" features already implemented (PBT, tournament, web, bridge, hierarchical)
- Key finding: v1.2.0 prod run had 0 promotions; v1.3.0 fix untested in production
- Status: **DEV-COMPLETE** (validation script ready for execution)

## 2026-02-08 (Phase 5 - Cost Optimization)

- Implemented `CostEstimator` in `alpha_zero/cloud/gcp/cost.py`.
- Verified preemption handling in `training_worker.py` (added `SignalHandler`).
- Integrated cost reporting into `vertex_orchestrator.py`.
- Created `configs/wandb_dashboard.json`.
- Passed 251 unit/integration tests.
- Status: **DEV-COMPLETE** (Handing off to Pre-PR)
