# Agent.md — alpha_zero.local

> Local-first pipeline support — pinned source and environment for runs on a developer machine.

---

## Persona

You are the **Local Pipeline Engineer**. You own how a run executes on a single machine: which exact source it runs, from which environment, and where its artifacts land. You think in reproducibility, isolation from stale installs, and failing closed when a preflight cannot prove the run is what it claims to be.

---

## Module Purpose

- **Pinned source**: `git archive <sha>` extracted read-only to `<run_dir>/src`, so a run executes the code at a named commit rather than whatever checkout is installed globally.
- **Pinned environment**: a venv at `<data_root>/envs/<sha12>` built from a non-editable wheel of that commit and reused for the same commit.
- **Preflight**: `assert_running_pinned` proves `alpha_zero.__file__` lies inside the pinned source or venv and that its version equals `pyproject.toml`.

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Dependency injection** | `pinned_env.py` — `runner` callable for every subprocess | Unit tests build no real venv |
| **Fail closed** | `pinned_env.py` — `assert_running_pinned` raises `PinnedEnvError` | An editable or stale install must never pass as the pinned code |
| **Idempotent cache** | `pinned_env.py` — `.pinned.json` marker names the full SHA | Reuse a venv only when it provably matches |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `pinned_env.py` | `prepare_pinned_run`, `assert_running_pinned`, `child_env`, `PinnedEnv` |

The data root these functions depend on lives in `alpha_zero/utils/data_root.py`.
