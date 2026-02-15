# Merge Plan: PR #32 into `claude/strategic-planning-sdk-rEoM0`

## Summary

PR #32 (`src/` SDK) and our branch (`alpha_zero/strategic/`) have **zero source-code overlap**. They live in completely separate directories. The merge requires resolving **4 infrastructure file conflicts** — all additive and straightforward.

---

## Step 1: Merge PR #32 into our branch

```bash
git merge pr-32 --no-edit
# Will produce 4 conflicts — resolve in steps 2-5
```

---

## Step 2: Resolve `.github/workflows/ci.yml`

**Strategy: Keep both CI jobs (ours + PR #32's SDK job)**

Take our linting block (pinned versions, ruff formatter check, `continue-on-error` on mypy) AND PR #32's additions (`ruff check src/`, `mypy src/`, `test-sdk` job, `--ignore` flags for test isolation).

Concrete resolution:
- Linting: keep our pinned `ruff==0.8.0 mypy==1.13.0 bandit==1.8.0`, add `pydantic structlog` to the install
- Add PR #32's `ruff check src/` and `mypy src/` steps after ours
- Test job: keep our `pytest tests/unit/ -v --tb=short -x -m "not slow"` and our sanity tests
- Add PR #32's `--ignore` flags and `test-sdk` job block wholesale
- Add PR #32's SDK integration test step

---

## Step 3: Resolve `pyproject.toml`

**Strategy: Combine all optional-dependency groups**

Keep all groups from both branches:
- Our `strategic` extra → keep
- PR #32's `sdk` extra → keep
- PR #32's `rag`/`rag-full` extras → keep (they existed in the base)
- `all` extra → include both `alpha_zero[strategic]` AND `alpha_zero[sdk]`
- Ruff per-file-ignores → keep ours (reward_shaping, mechanics, human_interface, integration) AND PR #32's (`src/*` ANN401)
- Mypy overrides → keep all our module overrides AND PR #32's `src.*` override

---

## Step 4: Resolve `setup.py`

**Strategy: Take PR #32's minimal shim**

PR #32 correctly moves all metadata to `pyproject.toml` and reduces `setup.py` to a 3-line shim. Our branch has a full `setup()` call with duplicated dependency lists. Since `pyproject.toml` is the source of truth, adopt PR #32's approach:

```python
"""Thin shim for editable installs."""
from setuptools import setup
setup()
```

---

## Step 5: Resolve `tests/conftest.py`

**Strategy: Take PR #32's `try/except` guard**

PR #32 wraps torch imports in `try/except ModuleNotFoundError` so SDK-only tests can run without torch. This is strictly more robust than our bare import. Adopt their pattern:

```python
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ModuleNotFoundError:
    _HAS_TORCH = False
```

Then guard the `SimpleTestNetwork` class and torch-dependent fixtures with `if _HAS_TORCH:` blocks. PR #32 already does this — take their version and ensure our fixtures (bridge_config, game_state, trajectory) are preserved (they will be since they're below the conflict zone).

---

## Step 6: Verify no test regressions

Run in sequence:

```bash
# 1. Clear pycache (critical for torch import stability)
find /usr/local/lib/python3.11/dist-packages/torch -name "__pycache__" -type d | xargs rm -rf
find /home/user/Civ6-AlphaZero -name "__pycache__" -type d | xargs rm -rf

# 2. Our strategic SDK tests (325 tests)
python -m pytest tests/unit/test_strategic/ tests/integration/test_strategic/ -q --tb=short

# 3. PR #32's SDK tests
python -m pytest tests/unit/core/ tests/unit/environments/ tests/unit/adapters/ tests/unit/test_cli.py -q --tb=short

# 4. PR #32's integration tests
python -m pytest tests/integration/test_sdk_integration.py -q --tb=short

# 5. PR #32's smoke tests
python -m pytest tests/smoke/ -q --tb=short
```

---

## Step 7: Verify coverage >= 70% on `alpha_zero/strategic/`

```bash
find . -name "__pycache__" -type d | xargs rm -rf
python -m pytest tests/unit/test_strategic/ tests/integration/test_strategic/ \
  --cov=alpha_zero/strategic --cov-report=term-missing -q --tb=short
```

Expected: 86%+ (unchanged since no strategic source files were modified).

---

## Step 8: Commit and push

```bash
git add -A
git commit -m "merge: integrate PR #32 SDK training infrastructure

Merge src/ SDK (training pipeline, mini-games, curriculum learning)
alongside alpha_zero/strategic/ (multi-objective planning, explainability).

Conflict resolutions:
- ci.yml: combined both lint + test jobs
- pyproject.toml: merged dependency groups
- setup.py: adopted minimal pyproject.toml shim
- conftest.py: adopted torch try/except guard

Both test suites pass independently."

git push -u origin claude/strategic-planning-sdk-rEoM0
```

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| torch pycache import error | Medium | Clear pycache before each test run |
| PR #32 tests need pydantic/structlog | Low | Already in `[sdk]` extras, pip install |
| Conftest guard breaks torch-dependent tests | Low | `_HAS_TORCH` only skips when torch truly absent |
| CI workflow syntax error | Low | YAML validation before push |

---

## What This Does NOT Do (Future Work)

- Does not create cross-package adapters (e.g., wrapping `alpha_zero/strategic` environments for `src/` training pipeline)
- Does not unify the two environment protocols
- Does not add multi-objective support to PR #32's `MCTSEngine`
- Does not merge the two CLI entry points into one
