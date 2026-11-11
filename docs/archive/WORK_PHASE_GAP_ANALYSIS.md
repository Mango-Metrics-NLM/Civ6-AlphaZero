# Work-Phase Gap Analysis — `claude/civ6-alphazero-implementation-vMnun`

_Generated 2026-05-11 against local working tree._

## TL;DR

The in-flight work on this branch is a **MuZero rebrand + production hardening** pass on the GCP training pipeline. The named scope (per modified-file surface) is:

1. Rename `civ6az-` → `civ6az-mzwm-` across job display names, container image tags, and the default W&B project.
2. Harden `load_muzero_model()` against checkpoints with foreign architecture (e.g. ViT) — strips unknown config keys and falls back to fresh random init when >20% of `state_dict` keys mismatch.
3. Add a `--model_version_start` orchestrator flag so MuZero runs can deliberately start at version 0 (fresh) instead of auto-detecting an incompatible older checkpoint.
4. Fix the `mcts.search()` call signature in `run_muzero_selfplay` (was passing `env`, now passes `(obs, action_mask)`).
5. Add 8 ad-hoc debug-utility scripts born from production firefighting (mostly Cloud Logging + Vertex job inspectors that work around a Windows IPv6 routing bug).

## Drift risk

- **Branch is 75 commits behind `origin/claude/civ6-alphazero-implementation-vMnun`.** Anything that touches `vertex_orchestrator.py`, `vertex_jobs.py`, `selfplay_worker.py`, or `cli.py` risks merge conflicts on rebase. Before this work is committed, rebase + re-test.
- 9 YAML configs are modified but currently unreadable due to OneDrive sync stubs — `git diff` errors with `mmap failed: Invalid argument`. Force-download with `attrib +P` or click "Always keep on this device" in Explorer, then re-verify.

## Audit results

### Modified files (with diff line counts — actual, not the earlier hallucinated values)

| File | Lines | What changed |
|---|---:|---|
| `alpha_zero/cloud/gcp/cli.py` | 2 | List-jobs filter `civ6az-` → `civ6az-mzwm-` |
| `alpha_zero/cloud/gcp/vertex_jobs.py` | 12 | WANDB_PROJECT default + 5 display-name infixes |
| `alpha_zero/cloud/gcp/vertex_orchestrator.py` | 12 | `--model_version_start` flag + branch |
| `alpha_zero/cloud/gcp/selfplay_worker.py` | 35 | Unknown-key filter, mismatch fallback, `mcts.search` signature |
| `tests/cloud/unit/test_selfplay_worker_muzero.py` | 196 | 6 new tests across 3 classes |
| `scripts/_fetch_job_logs.py` | net +3 | Rewritten to ADC + IPv4 + clean OAuth flow |
| `scripts/build_and_push.py` | ~8 | Image name `civ6az-training` → `civ6az-muzero` |
| `scripts/cloud_build_container.py` | unread | OneDrive stub |
| `scripts/gcp_diagnose_and_submit.py` | unread | OneDrive stub |
| `dockerfiles/cloudbuild.yaml` + 8 YAMLs | unread | OneDrive stubs |

### Untracked files (purpose)

| File | Purpose |
|---|---|
| `scripts/_gcp_lib.py` ⭐ NEW | Shared IPv4 patch + ADC token helper (added by this audit) |
| `scripts/_cancel_job.py` | Cancel one or more Vertex jobs by ID |
| `scripts/_fetch_job_logs.py` (mod) | Fetch Cloud Logging entries by job ID + workerpool0-0 filter |
| `scripts/_fetch_worker_logs.py` | Fetch logs for a specific self-play worker job |
| `scripts/_get_job_logs.py` | Logs for most-recent Vertex job, multi-filter cascade |
| `scripts/_poll_build.py` | Poll Cloud Build status by build ID |
| `scripts/fetch_vertex_logs.py` | Job details + logs, `--latest` and per-id flows |
| `scripts/gcp_provision_resources.py` | Create Artifact Registry repo + GCS bucket |
| `scripts/run_tests_ipv4.py` | Pytest wrapper that patches socket BEFORE importing torch |
| `tests/cloud/unit/test_orchestrator_muzero_config.py` | 7 (now 11) tests for build_selfplay_extra_args + _wandb_env_vars + --model_version_start |
| `PLANNING.md`, `CLAUDE_CREATIVITY_NN_INTEGRATION.md`, `docs/GCP_INFRA.md`, `docs/architecture/`, `docs/prd/` | Planning + ADR docs (unreadable due to OneDrive sync) |

### Resolved tech debt (this audit)

| ID | Issue | Resolution |
|---|---|---|
| TD-1 | IPv4-socket patch duplicated across 8 scripts | Centralised in `scripts/_gcp_lib.force_ipv4_only()` (idempotent, auto-installs on import). `run_tests_ipv4.py` keeps its inline patch by design — must run BEFORE any imports. |
| TD-2 | ADC refresh-token exchange duplicated across 8 scripts | Centralised in `scripts/_gcp_lib.get_access_token()` + `auth_headers()`. Surfaces clearer errors when ADC file is missing or is a service-account key. |
| TD-3 | `scripts/_fetch_logs_tmp.py` debug leftover (`_tmp` suffix; functionally identical to `_fetch_job_logs.py`) | **Deleted.** |
| TD-4 | Hardcoded default job IDs in `_fetch_worker_logs.py:14` and `_poll_build.py:21` referencing specific past runs | Removed; now require explicit CLI arg, exit 2 with usage on missing arg. |
| TD-5 | Hardcoded project number `<project-number>` (line 41 of `_fetch_job_logs.py`) | Replaced with `f"projects/{PROJECT}"` driven by `CIV6AZ_VERTEX_PROJECT` env. |
| TD-6 | `test_search_call_in_selfplay_uses_obs_and_mask` used hardcoded absolute Windows path `C:\Users\iansh\OneDrive\Documents\Civ6-AlphaZero\...` — would fail in any other checkout, worktree, or CI | Now resolves via `inspect.getfile(selfplay_worker)`. Portable. |
| TD-7 | Lack of structured logging in utility scripts (mix of `print` for results + nothing for errors) | All refactored scripts now use `logger` for diagnostics on stderr and `print` for tool output on stdout; `configure_script_logging()` provides consistent format. |
| TD-8 | Missing module docstrings + type hints in utility scripts | Every refactored script now has a module docstring, usage banner, typed `main()`, and explicit return code semantics. |

### Test-coverage additions (this audit)

| ID | Behaviour | New test |
|---|---|---|
| TC-1 | `--model_version_start` flag: default `None`, accepts `0` (distinguishable from None), accepts positive int, rejects non-integer | 4 tests in `TestModelVersionStartFlag` |

### Pre-existing test coverage (already strong)

`tests/cloud/unit/test_selfplay_worker_muzero.py` already covers:
- MuZeroConfig field filtering (`TestConfigFiltering`, 2 tests)
- Architecture-mismatch fallback (`TestArchitectureMismatchFallback`, 2 tests)
- MCTS-search API regression (`TestMCTSSearchAPI`, 2 tests — signature + call-site AST check)

`tests/cloud/unit/test_orchestrator_muzero_config.py` already covers:
- `build_selfplay_extra_args` MuZero override (3 tests)
- `_wandb_env_vars` propagation (4 tests)

### Remaining tech debt (deferred, low-risk)

| ID | Issue | Recommendation |
|---|---|---|
| TD-9 | Hardcoded `civ6az-mzwm-` prefix scattered across `cli.py`, `vertex_jobs.py` | Factor into a module-level constant `JOB_NAME_PREFIX` in `vertex_jobs.py`; `cli.py` filter imports it. Defer until upstream rebase to avoid conflict thrash. |
| TD-10 | Existing `TODO` at `vertex_jobs.py:740` ("multi-node rendezvous args") | Track in issue; not in current scope. |
| TD-11 | Unread YAML configs | Force OneDrive download and re-audit; expected to be mechanical `civ6az-` → `civ6az-mzwm-` renames. |
| TD-12 | 8 utility scripts have zero tests | The smaller `_*.py` debug-only ones don't need tests. `fetch_vertex_logs.py` and `gcp_provision_resources.py` are bigger and worth a smoke test (mock `requests`). Suggested unit-test scaffolding deferred. |

## Strong recommendation: rebase before commit

Current state — 75 commits behind origin on an active feature branch — means **any commit you make here will need to be rebased**. The actual diffs in `vertex_orchestrator.py` and `selfplay_worker.py` are small and localized, so conflict risk is moderate, not catastrophic. Recommended sequence:

```
git stash -u                                            # set aside everything, incl. untracked
git pull --rebase origin claude/civ6-alphazero-implementation-vMnun
git stash pop                                           # bring work back; resolve any conflicts
# re-run: pytest tests/cloud/unit/test_orchestrator_muzero_config.py \
#                tests/cloud/unit/test_selfplay_worker_muzero.py
```

If the rebase produces non-trivial conflicts in `vertex_orchestrator.py`, the `--model_version_start` flag is the most likely point of friction — it lives near other `parse_args()` additions that upstream may have changed.

## Verification (this audit)

- ✅ `scripts/_gcp_lib` imports cleanly; module-level constants populate from env.
- ✅ All 6 refactored scripts import without error.
- ✅ Missing-arg paths in `_cancel_job`, `_fetch_worker_logs`, `_poll_build` exit 2 with usage banners.
- ⏳ `pytest tests/cloud/unit/test_orchestrator_muzero_config.py tests/cloud/unit/test_selfplay_worker_muzero.py` — run via `scripts/run_tests_ipv4.py` (Windows IPv6 hangs without it; documented in MEMORY.md).
