# Agent.md — alpha_zero.local

> Local-first pipeline support — pinned source and environment for runs on a developer machine.

---

## Persona

You are the **Local Pipeline Engineer**. You own how a run executes on a single machine: which exact source it runs, from which environment, and where its artifacts land. You think in reproducibility, isolation from stale installs, and failing closed when a preflight cannot prove the run is what it claims to be.

---

## Module Purpose

- **Pinned source**: `git archive <sha>` extracted read-only to `<run_dir>/src`, so a run executes the code at a named commit rather than whatever checkout is installed globally.
- **Pinned environment**: a venv at `<data_root>/envs/<sha12>` built from a non-editable wheel of that commit and reused for the same commit. A card may pin a torch build (`torch_spec` + `torch_index_url`); that exact build is installed first and the venv is then `<sha12>-t<pin hash>`, so a CPU venv for the same commit is never reused for a CUDA card.
- **GCP isolation**: `enter_local_mode` scrubs cloud variables and returns the child environment minus a deny list (`*_KEY`, `*_TOKEN`, ...; not an allowlist); `GCSBackend` (constructor and lazy `client`) raises `LocalModeViolation` (a `BaseException`) under `CIV6AZ_LOCAL_RUN=1`.
- **Preflight**: `assert_running_pinned` proves `alpha_zero.__file__` lies inside the pinned source or venv and that its version equals `pyproject.toml`.
- **Devices**: `devices.py` decides whether the pinned environment can honour the devices a card names. The pinned child reports its own torch build, GPU names, architectures and one real CUDA operation per device (`TORCH_REPORT_CODE`); `device_problem` refuses a missing ordinal, a build without kernels for the GPU, a GPU that is not the one the card expects at that ordinal, or a torch other than the pinned one. It never substitutes another device. A card that pins torch must name every invocation's device explicitly.
- **Run cards**: `run_card` prices every invocation of a `configs/run_cards/*.yaml` card with the runner's own parser and refuses a total over `max_games` without `approval_ref`, hardens only a data root it creates, enters local mode, pins the SHA, preflights a real pinned child, checks the card's devices, then runs the invocations sequentially and records `<root>/runs/<card_id>/run.json` (env var names only; the `device_plan`; VOID on interrupt or resume, FAILED on a non-zero child, with the child's last log lines as `error_tail`). `--probe` rehearses everything up to the device check in a throwaway `runs/probe-<card_id>-<time>` directory and launches nothing.

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Dependency injection** | `pinned_env.py` — `runner` callable for every subprocess | Unit tests build no real venv |
| **Fail closed** | `pinned_env.py` — `assert_running_pinned` raises `PinnedEnvError` | An editable or stale install must never pass as the pinned code |
| **Fail closed** | `isolation.py` — `guard_gcs`, `LocalModeViolation(BaseException)` | `except Exception` state saves must not swallow a GCP touch |
| **Idempotent cache** | `pinned_env.py` — `.pinned.json` marker names the full SHA and the torch pin | Reuse a venv only when it provably matches |
| **Fail closed** | `devices.py` — `device_problem`, `implicit_device_problem` | The runners fall back to CPU with a warning; a card is evidence about a named device, so it refuses instead |
| **Ask the source** | `run_card.py` — `invocation_devices` reads the runner's own parsed namespace | Only the parser knows abbreviations (`--dev`) and defaults; a token scan disagrees with the child |
| **Fail closed** | `run_card.py` — unknown runner, unparseable argv or over-cap total refuses before anything is created | An unpriceable run is the run the cap exists to catch |
| **Registry** | `run_card.py` — `RUNNERS` maps a runner to its module and argv pricer | Adding a runner is one entry; tests swap the launched module |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `pinned_env.py` | `prepare_pinned_run`, `assert_running_pinned`, `child_env`, `PinnedEnv`, `TorchPin` |
| `devices.py` | `TORCH_REPORT_CODE`, `device_problem`, `implicit_device_problem`, `architecture_supported`, `apply_device_environment` (pure functions; imports no torch) |
| `isolation.py` | `enter_local_mode`, `guard_gcs`, `is_local_mode`, `LocalModeViolation` (stdlib-only) |
| `card.py` | `RunCard`, `Invocation`, `load_card`, `RunCardError`: the card model and its validation (unknown keys refused) |
| `run_card.py` | `price_card`, `prepare_data_root`, `invocation_devices`, `run_card` (`probe=`), `device_report_lines`, `main` (CLI wrapper: `scripts/run_card.py`, `--price_only` / `--probe`) |

The data root these functions depend on lives in `alpha_zero/utils/data_root.py`; CUDA ordinal parsing in `alpha_zero/utils/device.py` (`cuda_ordinal`); the cuBLAS setting in `alpha_zero/utils/seeding.py`.

---

## Skills

- **Reproducible execution**: run the exact code at a named SHA from a read-only tree and a SHA-keyed venv
- **Isolation**: keep a workstation run away from GCP credentials, buckets and user-site packages
- **Budgeting**: price a multi-invocation run card before anything is created or spawned
- **Device honesty**: prove the pinned environment has the GPUs a card names, or refuse with the reason

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Run-Card Reviewer** | Check a card's SHA, invocations and priced total against the 1,000-game cap | Before running a new card |

---

## Tools

| Tool | Purpose |
|------|---------|
| `python scripts/run_card.py CARD --price_only` | Price a card; creates nothing |
| `python scripts/run_card.py CARD --data_root PATH` | Run a card and write `run.json` |
| `pytest tests/unit/test_pinned_env.py tests/unit/test_local_isolation.py -v` | Pinned env and isolation tests |
| `pytest tests/scripts/test_run_card.py -v` | Run-card driver tests |

---

## Invariants

1. A card is priced in total, across every invocation, before the data root or run directory is touched
2. A runner the driver cannot price is refused, never waved through
3. An existing data root is never re-ACLed; only a root the driver creates is hardened
4. A run directory is single-use; an interrupted or resumed card is VOID
5. `run.json` records environment variable names, never values
6. `isolation.py` stays stdlib-only (`alpha_zero/cloud/gcp/storage.py` imports it)
7. Each child is launched with `--max_games <its priced games>`, and the parent re-verifies the pinned preflight report with its own code
8. On Windows a hard kill cannot be caught: a lingering `RUNNING` record is read as VOID

---

## Dependencies

- **Internal**: `alpha_zero.utils.data_root`, `alpha_zero.utils.game_budget`; the runners priced by `run_card.py` (`training_civ6`, `evaluation.benchmark`, `orchestrator`) are imported lazily
- **External**: `pyyaml` (cards); stdlib otherwise
- **Consumed by**: `scripts/run_card.py`; `alpha_zero.cloud.gcp.storage` (`guard_gcs`)
