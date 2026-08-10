# CI Fix Reference - PR #50

## PR: Phase 16: Batch Inference, Vectorized Self-Play & Cloud Orchestrator

### Code Review Issues (All Resolved)

| # | Severity | Source | Issue | Fix | Status |
|---|----------|--------|-------|-----|--------|
| 1 | High | gemini-code-assist | `_batch_sequential_halving` evaluates leaves individually (batch-size-1) | Refactored to collect `pending_evals` list per round, then single `_batch_evaluate()` call | Committed |
| 2 | Medium | gemini-code-assist | `except ImportError: pass` silently swallows wandb import failure | Changed to `logger.debug("wandb not installed, skipping batch metrics logging")` | Committed |
| 3 | Medium | gemini-code-assist | Same pattern in second `except ImportError` block | Applied same fix with descriptive log message | Committed |
| 4 | Medium | Copilot | `rewards` referenced on line 669 may be undefined if loop breaks early | Added `rewards = np.zeros(num_parallel_games, dtype=np.float32)` before the while loop (line 602) | Committed |

### CI Pipeline Fixes

| Stage | Check | Issue | Fix | Status |
|-------|-------|-------|-----|--------|
| Lint | `ruff format --check` | 6 web module files not formatted | Ran `ruff format` on all 6 files | Committed & Pushed |
| Lint | `ruff check` | N/A | All checks passed | Pass |
| Lint | `mypy` | `sentry_sdk.consts.VERSION` assertion error | Known upstream issue; `continue-on-error: true` in CI | Non-blocking |

### Files Modified for CI Fix

```
alpha_zero/web/__main__.py       # ruff format
alpha_zero/web/adapter.py        # ruff format
alpha_zero/web/config.py         # ruff format
alpha_zero/web/routes/game.py    # ruff format
alpha_zero/web/serializers.py    # ruff format
alpha_zero/web/session.py        # ruff format
```

### Commit History

1. `43d85b5` - `fix(ci): apply ruff format to web module files`
   - Pushed to `feature/phase-16-batch-inference`

### Verification Commands

```bash
# Lint check
ruff check alpha_zero/ --output-format=github

# Format check
ruff format --check alpha_zero/

# MyPy (continue-on-error in CI)
mypy alpha_zero/ --ignore-missing-imports --show-error-codes

# Unit tests (Phase 16)
pytest tests/unit/test_batch_mcts.py tests/unit/test_vectorized_env.py \
       tests/unit/test_cuda_graph_cache.py tests/unit/test_inference_benchmark.py \
       -v --tb=short -x

# Cloud unit tests
pytest tests/cloud/unit/ -v --tb=short -x
```
