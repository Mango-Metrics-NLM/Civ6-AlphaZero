# PR: Post-merge Windows & dual-GPU E2E hardening, zero-warning `Agent.md` burndown, and deterministic validator contracts

**Branch:** `fix/post-merge-gpu-e2e-aqa-hardening` → `origin/claude/civ6-alphazero-implementation-vMnun` (`439ee9c4`)
**Spec:** [`kb/specs/20261111-local-first-and-branching-plan.md`](../specs/20261111-local-first-and-branching-plan.md)

## Target branch and PR type

- [x] Feature / fix / chore / docs -> `dev` (squash)

Note: `origin/dev`, `origin/qa`, and `origin/main` do not exist remotely yet (local `main` is at `35334ea5`, 410 commits behind), so this PR targets the current remote default branch `origin/claude/civ6-alphazero-implementation-vMnun` (`439ee9c4`) per `branching-and-release.md`.

## Summary

- **Dual-GPU & Windows E2E verification and runtime fixes:** Verified live dual-GPU determinism (`scripts/check_determinism.py --train-device cuda:0 --arena-device cuda:1`: `REPRODUCED`, 0 differences across 3 checkpoints, 2 iterations, 1 arena on RTX 5060 Ti + RTX 5060 with `torch 2.11.0+cu128`). Fixed `alpha_zero/utils/rag_client.py` (`_call_safely` over `_FATAL_BASE_EXCEPTIONS`) so native PyO3/ChromaDB `BaseException` panics degrade gracefully while preserving `KeyboardInterrupt`/`SystemExit`/`GeneratorExit`; updated `tests/unit/test_seeding.py` to inherit `os.environ` alongside `PYTHONHASHSEED` on Windows.
- **DRY tree discovery & worktree/PEP 405 pruning:** Extracted shared `iter_python_files(..., skip_parts=...)` and `_skip_reason` in `scripts/audit_import_graph.py`, removed the duplicated `os.walk` / `SKIP_PARTS` copy from `scripts/measure_debt.py`, and pruned PEP 405 virtual environments (`pyvenv.cfg`, `.venv*`, `venv*`) and linked git worktrees (`.git` files, `.claude/`, `.worktrees/`, `worktrees/`) across both `audit_import_graph.py` and `alpha_zero/utils/agent_tools.py:CodeExplorer`.
- **Portable Windows/POSIX harness, Makefile, and hygiene:** Updated `Makefile` (`PYTHON ?=`, `PIP ?=`, and removed redundant inline `OMP_NUM_THREADS=2` in `verify-registration`) and `.claude/hooks/*.sh` to resolve a real Python >= 3.11 interpreter (avoiding the Windows Store `python3` alias) and route `CITATION.cff` to `check_doc_sync.py`. Extracted `tests/governance/_bash.py` (`find_bash` / `resolve_bash` + package `__init__.py`) so all Windows bash hook and maintainer script tests resolve Git Bash instead of the `System32\bash.exe` WSL stub. Scoped `.gitignore`, `.dockerignore`, and `.gitleaks.toml` for `.venv*`, `venv*`, worktrees, and `.claude/scheduled_tasks.lock` without exempting `.env*`.
- **Enterprise deterministic validators & `Agent.md` warning burndown (`22 -> 0`):** Added sorted findings, structured `--verbose` logging, and canonical `sort_keys=True` `--json` output to `scripts/validate_skills.py` and `scripts/validate_claude_agents.py`; synced `.claude/hooks/readonly_shell_guard.sh` option allowlists; completed all six incomplete `Agent.md` docs and fixed factual drifts (`alpha_zero/monitoring/Agent.md`, `alpha_zero/cloud/Agent.md`, `alpha_zero/opponents/Agent.md`, `alpha_zero/utils/Agent.md`, `tests/Agent.md`, plus `docs/architecture/C4_ARCHITECTURE.md` `log_evaluation_result()` 15-metric count), ratcheting `tests/governance/agent_warning_baseline.json` `max_warnings` to `0`.
- **Documentation & regression/AQA expansion:** Updated `README.md`, `CHANGELOG.md`, `ROADMAP.md`, `CLAUDE.md`, `docs/README.md`, `docs/architecture/C4_ARCHITECTURE.md`, `docs/runbooks/secret-scanning.md`, `.claude/README.md`, and `.claude/skills/{debt-audit,doc-sync,verify}/SKILL.md`; expanded `tests/regression/test_gpu_e2e_and_worktree_aqa.py`, `tests/governance/`, `tests/regression/`, `tests/scripts/`, and `tests/unit/`.

## Changes

- **Runtime & Utilities:** `alpha_zero/utils/agent_tools.py`, `alpha_zero/utils/rag_client.py`
- **Validators & Scripts:** `scripts/audit_import_graph.py`, `scripts/measure_debt.py`, `scripts/validate_claude_agents.py`, `scripts/validate_skills.py`
- **Harness, Build, Hygiene:** `.claude/hooks/{_resolve_python,posttooluse_governance,posttooluse_lint,precompact_state,readonly_shell_guard,sessionstart_bootstrap}.sh`, `.claude/skills/{debt-audit,doc-sync,verify}/SKILL.md`, `.claude/README.md`, `Makefile`, `.gitignore`, `.dockerignore`, `.gitleaks.toml`
- **Documentation & Persona Docs:** `README.md`, `CHANGELOG.md`, `ROADMAP.md`, `CLAUDE.md`, `docs/README.md`, `docs/architecture/C4_ARCHITECTURE.md`, `docs/runbooks/secret-scanning.md`, `alpha_zero/{cloud,hexcrystal,inference,monitoring,opponents,serving,utils}/Agent.md`, `tests/Agent.md`
- **Tests & Governance Ratchets:** `tests/governance/{__init__.py,_bash.py,agent_warning_baseline.json,test_governance_determinism.py,test_guard_review_round1.py,test_hook_guard_json.py,test_pretooluse_guard.py,test_readonly_shell_guard.py,test_validate_claude_agents.py,test_validate_skills.py}`, `tests/regression/{test_gpu_e2e_and_worktree_aqa.py,test_repo_hygiene_regression.py,test_validator_cli_regression.py}`, `tests/scripts/{test_audit_import_graph.py,test_maintainer_scripts.py}`, `tests/unit/{test_agent_tools.py,test_factories.py,test_seeding.py}`

## Game budget

- `python -m alpha_zero.training_civ6 --dry_run`: 8 games (cap 1,000)
- `python -m alpha_zero.orchestrator --preset fast --checkpoint-dir <tmp> --dry_run`: 0 training games
- `python scripts/check_determinism.py --train-device cuda:0 --arena-device cuda:1`: 16 games (cap 1,000)
- No registered card (`t0b.yaml` 660, `a1.yaml` 980, `a1r.yaml` 980) was executed; `docs/EVALUATION.md` §7.1 hashed region untouched (`REGISTRATION_DIGEST e460a353651a24c26d5e1891b6ad5bd27c18acac4fd9fd6367bc262175237c17`).

## Tracking issue

None — post-merge stabilization, GPU/Windows verification, and enterprise governance hardening branch (`fix/post-merge-gpu-e2e-aqa-hardening`).

## Linked OpenSpec change

`openspec/changes/add-gpu-first-run-cards` (references updated for live `check_determinism.py` verification while formal T0b/A1 run-card execution remains open); no new OpenSpec change required.

## Hot files touched

`CLAUDE.md`, `README.md`, `CHANGELOG.md`, `ROADMAP.md`, `Makefile`, `.gitignore`, `.dockerignore`, `.gitleaks.toml`, `docs/architecture/C4_ARCHITECTURE.md`, `kb/ledger/ledger.jsonl`.

## Verification

- [x] `uvx ruff==0.15.8 check .` and `uvx ruff==0.15.8 format --check .`: clean (`986` files)
- [x] `uvx --python 3.11 --from mypy==1.19.1 ... mypy alpha_zero/ --show-error-codes`: `Success: no issues found in 346 source files`
- [x] `python -m bandit -r alpha_zero/ -s B101 --severity-level high --confidence-level high`: `No issues identified` (`121,932` lines)
- [x] `gitleaks dir .` (`433.51 MB` in `6.96s`) + `gitleaks git . --log-opts="HEAD"` (`800 commits` in `3.02s`): `no leaks found`
- [x] `scripts/measure_debt.py --compare docs/evidence/tech_debt_baseline/baseline.json`: exit `0`
- [x] `make validate`: `openspec: OK`, `kb: OK`, `skills: OK`, `agents: OK` (`0` errors, `0` warnings), `claude-agents: OK`, `doc-sync: OK`
- [x] `make verify-registration`: `102 passed`, `REGISTRATION_DIGEST` matches (`e460a353...7c17`)
- [x] `scripts/run_llm_eval.py`: `QUALITY GATE: PASS` (`12/12` analyze, `8/8` report)
- [x] Standalone `tests/governance` (`.venv\Scripts\pytest.exe -c tests/governance/pytest.ini --confcutdir=tests/governance tests/governance`): `790 passed in 365.00s`
- [x] Fast suites (`tests/regression/`, `tests/sanity/`, `tests/security/`, `tests/smoke/`, `tests/inference/`, `tests/cloud/smoke/`): `543 passed`
- [x] Cloud coverage floor (`tests/cloud/unit/` `831 passed` + `tests/cloud/smoke/` `36 passed`): `alpha_zero/cloud: 60.1%` of `5538` statements across `26` files (floor `58%`)
- [x] CI coverage sweep (`tests/unit/`, `tests/sanity/`, `tests/integration/`, `tests/cloud/unit/`, `tests/regression/` `-m "not slow and not requires_env"`): `12,834 passed, 50 skipped, 60 deselected, 7 xfailed`, total coverage **`89.31%`** (gate `85%`)
- [x] Scripts + E2E (`tests/scripts/`, `tests/e2e/` `-m "not slow and not requires_env"`): `952 passed, 11 skipped, 8 deselected`
- [x] Live dual-GPU determinism (`scripts/check_determinism.py --train-device cuda:0 --arena-device cuda:1`): `status: REPRODUCED`, `differences: []` (`RTX 5060 Ti` + `RTX 5060`, `torch 2.11.0+cu128`, `cudnn 91900`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`)
- [x] Smoke tests: `training_civ6 --dry_run` exit `0`; `orchestrator --preset fast --checkpoint-dir <tmp> --dry_run` exit `0`

## Negative control recorded

- **Native panic boundary (`alpha_zero/utils/rag_client.py`):** Reverted `except BaseException` to `except Exception` in `_call_safely`; `pytest tests/regression/test_gpu_e2e_and_worktree_aqa.py tests/unit/test_factories.py -k "native_panic"` failed `2/2` (`NativeRustPanic` / `FakePanicError` escaped), then passed after restore.
- **PEP 405 & worktree pruning (`scripts/audit_import_graph.py`):** Removed `(path / "pyvenv.cfg").is_file() or (path / ".git").exists()` check from `_skip_reason`; `pytest tests/regression/test_gpu_e2e_and_worktree_aqa.py tests/scripts/test_audit_import_graph.py -k "pep405"` failed `3 failed, 2 passed` (`custom_env`, `nested_wt`, and AQA worktree test all lost `'alpha_zero.pkg.unused'` / `'alpha_zero.p.hidden'`), then passed `5/5` after restore (`tests/regression/test_gpu_e2e_and_worktree_aqa.py` also verified `7/7`).
- **Pre-existing/sibling controls preserved:** `test_repo_hygiene_regression.py` (`test_allowlist_covers_venvs_and_worktrees_but_never_dotenv`), `test_validator_cli_regression.py` (`test_json_output_is_valid_json`), and `test_suite_wiring_regression.py` (`TestOptionalDependencyImportsAreGuarded`).

## CONFIRM-FIRST items and who authorized

None — all changes stay within stabilization, Windows/GPU verification, validator determinism, hygiene, and documentation synchronization requested by the maintainer on branch `fix/post-merge-gpu-e2e-aqa-hardening`; `docs/EVALUATION.md` §7.1 hashed region and `kb/handoffs/**` are untouched.

## Notes

- Pre-existing unrelated local workspace artifacts (`uv.lock` CRLF/LF notice, `0.26.0`, `mypy_output.txt`, `.claude/scheduled_tasks.lock`) are excluded from this PR scope.
- `orchestrator --preset fast --dry_run` without `--checkpoint-dir` inspects `./checkpoints/orchestrator/`; on clean CI or isolated `--checkpoint-dir` it exits `0`, whereas stale local checkpoints of a different architecture fail weight load as expected.
- Formal T0b single-use `run_card.py` execution and section 7.1 slot fill-in remain for the maintainer.
