# DEV-SQE → Pre-PR Handoff

**Date:** 2026-07-13
**Spec:** kb/specs/20260713-project-charter.md (batch 2: user-requested gap analysis + tech-debt sweep on the same branch/PR)
**Files Changed:** 105 files (+413/−358) — highlights:
- `pyproject.toml` + `.github/workflows/ci.yml` — coverage gate 80 → 85 (CI measured 89.69% on run 29214300752; local full-suite re-verified 88.88%)
- `alpha_zero/bridge/mcts_policy.py` — **bug fix**: `select_action` now unpacks `GumbelMCTS.search`'s real `(policy_probs, root_value)` contract (was `(action, stats_dict)` → AttributeError on live `--mcts` path); `BridgeMCTSEnv` now satisfies `MCTSEnvironment` (added `action_space` property, protocol-conformant `reset`/`step`)
- `alpha_zero/bridge/agent_loop.py` — new `GameBridge` Protocol; loop is transport-agnostic (FileBridge/LiveBridge/test doubles), fixes the LiveBridge arg-type error
- `alpha_zero/bridge/action_converter.py` — `legal_actions` params widened `list` → `Sequence` (covariant, backwards compatible)
- `alpha_zero/core/civ6_mcts.py` — `MCTSEnvironment.action_space`/`current_player` → read-only protocol properties (plain attributes still conform)
- `alpha_zero/cloud/gcp/training_worker.py` — `train_epoch` profiler annotation corrected to `PerformanceProfiler` (the API actually called); duplicate TYPE_CHECKING block deduped
- `alpha_zero/hierarchical/mcts/hierarchical_mcts.py` — Optional-narrowing on `select_option_child` (defensive break on None)
- `alpha_zero/civ6_local.py`, `alpha_zero/orchestrator/chess_advisor.py`, `alpha_zero/core/jax_checkpoint.py`, `alpha_zero/optimizations/benchmarks.py` — remaining semantic mypy fixes
- `tests/unit/test_bridge_file_bridge.py` — revived 14 shadowed tests (duplicate class name; renamed `TestSendCommandWithRetryIntegration`)
- `tests/unit/test_bridge_mcts_policy.py`, `tests/unit/test_bridge_mcts_policy_gaps.py` — mocks aligned to real `search` contract (they had drifted and hid the bug)
- `tests/unit/test_live_bridge.py`, `tests/unit/test_log_tailer.py` — 7 dangling `asyncio.create_task` calls now stored + awaited
- ~80 test files — mechanical ruff fixes (RUF059 unused-unpack renames, import sorts, raw-string `match=` patterns)
- Charter/roadmap/template — PR #91 review findings (create_mcts → `core/mcts_factory.py`, `NEXT_STEPS.md` naming, robust ROADMAP table)
- `CLAUDE.md` — coverage refs 80→85, v1.8.0 reconciliation (§13 + footer), stale torch known-issue corrected

**Tests Added:** none new; 14 shadowed tests revived, 2 drifted mocks fixed
**Test Results:** 11,690 passed, 159 skipped, 64 deselected, 1 xfailed; coverage **88.88%** — **85% gate green** (CI-equivalent selection, run locally with torch 2.13.0 from PyPI). The single failure found (drifted mock in `test_bridge_mcts_policy_gaps.py`) was fixed and re-verified (21/21).
**Lint Status:** clean — 0 ruff errors across `alpha_zero/ tests/ scripts/ infra/` (was 292); verified under BOTH local ruff 0.15.8 and CI-pinned 0.8.0
**Type Check Status:** mypy 180 → 157 errors; **all semantic errors fixed** (arg-type, assignment, attr-defined, override, var-annotated, redundant-cast = 0). Remainder is version/stub noise (`unused-ignore`, `import-untyped`, `no-any-return`) that differs between local mypy and CI's pinned 1.13.0 — deliberately untouched to avoid regressing CI's view. CI mypy step is non-blocking (`continue-on-error: true`).
**Smoke Tests:** `ruff format --check alpha_zero/` green under CI's ruff 0.8.0 (345/345 files)
**Known Issues:**
- Do NOT run `ruff format` on `alpha_zero/bridge/file_bridge.py` / `alpha_zero/core/transformer_network.py` with ruff ≥0.15 — formatter opinion differs from CI's 0.8.0 and would break the CI format gate.
- W&B→MLflow sweep, coverage omit-list shrink, and `unused-ignore` noise remain as tracked debt (ROADMAP M4).
