# PR: Post-merge GM + Lua bridge E2E GPU AQA, validator determinism, and pre-PR governance alignment

**Branch:** `fix/post-merge-gm-full-e2e-gpu-aqa` -> `origin/claude/civ6-alphazero-implementation-vMnun` (`976a2afd`)
**Spec:** [`kb/specs/20261119-game-mechanics-plan.md`](../specs/20261119-game-mechanics-plan.md)

## Target branch and PR type

- [x] Feature / fix / chore / docs -> `dev` (squash)

Note: Local `main` is at `35334ea5` (`[origin/main: gone]`), while the active remote default/integration branch is `origin/claude/civ6-alphazero-implementation-vMnun` (`976a2afd`, carrying merged PRs `#143` GM-S0a and `#145`–`#153` GM-B0..GM-B5).

## Summary

- **Off-play-path self-play search seeding & DRY cleanup (`alpha_zero/cloud/gcp/selfplay_worker.py`):** Post-merge dual-GPU slow-unit verification (`cuda:0` RTX 5060 Ti + `cuda:1` RTX 5060, `torch 2.11.0+cu128`) exposed `tests/unit/test_selfplay_seed_threading.py::test_same_seed_reproduces_the_same_games` failing because `run_selfplay` omitted `"seed"` from `mcts_kwargs`. Fixed off the release play path via `_apply_search_seed` (`derive_seed(seed, SeedStream.SELF_PLAY_SEARCH, game_idx)`), added optional `seed: int | None = None` to `run_muzero_selfplay`, and extracted shared `_sign_rewards` and `_cleanup_after_game` helpers (`54+/59-`, keeping `coverage_omitted_lines=66849 <= 66858`).
- **Zero-skip GM + Lua bridge & dual-GPU AQA (`tests/regression/test_gm_bridge_and_gpu_e2e_aqa.py`):** Added end-to-end AQA tests covering seeded `run_selfplay` and `run_muzero_selfplay` determinism across CPU and live `cuda:0`/`cuda:1`, un-mocked multi-device training/checkpoint invariants, and `alpha_zero.rules` -> `bridge` -> `Civ6NetworkV2` -> `lupa` Lua 5.1 command execution (`CIV6AZ_REQUIRE_LUA=1`).
- **Enterprise governance validator parity & hook/CI alignment (`scripts/`, `.claude/`, `Makefile`, `.pre-commit-config.yaml`):** Unified sorted findings (`issues.sort`), canonical sorted `--json` (`sort_keys=True`), and structured `--verbose` stderr logging across all six governance validators (`validate_kb.py`, `validate_openspec.py`, `check_doc_sync.py`, plus `issues.sort` / `sort_keys=True` in `validate_agents.py`, matching `validate_skills.py` and `validate_claude_agents.py`), exposed `--verbose` in `.claude/hooks/readonly_shell_guard.sh`, wired `scripts/build_lua_bundle.py --check` into `Makefile` (`check-lua-bundle` / `validate`), `.pre-commit-config.yaml`, and `.claude/hooks/posttooluse_governance.sh` (tested with negative control `exit 2`), and reported optional `lupa` in `.claude/hooks/sessionstart_bootstrap.sh`.
- **Hygiene, architecture docs, & regression/governance hardening:** Cleaned the duplicate empty comment header block in `.gitignore`, excluded `mlruns/`, `mlflow.db*`, `mlartifacts/`, and `eval_results/` in `.dockerignore`, allowlisted `mlartifacts/`, `eval_results/`, and `venv.*/` in `.gitleaks.toml`, added Level 3 GM + bridge C4 component diagram and container updates in `docs/architecture/C4_ARCHITECTURE.md`, and synchronized `CHANGELOG.md`, `ROADMAP.md`, `README.md`, `CLAUDE.md`, `alpha_zero/cloud/Agent.md`, `docs/{README,GAME_RULES,MODULE_REFERENCE}.md`, `.claude/README.md`, `.claude/skills/verify/SKILL.md`, `tests/governance/`, and `tests/regression/`.

## Changes

- **Runtime:** `alpha_zero/cloud/gcp/selfplay_worker.py`
- **Validators & Build/Harness/Hygiene:** `scripts/{validate_agents,validate_kb,validate_openspec,check_doc_sync}.py`, `.claude/hooks/{posttooluse_governance,readonly_shell_guard,sessionstart_bootstrap}.sh`, `.claude/skills/verify/SKILL.md`, `.claude/README.md`, `Makefile`, `.pre-commit-config.yaml`, `.gitignore`, `.dockerignore`, `.gitleaks.toml`
- **Docs & C4 Architecture:** `CHANGELOG.md`, `CLAUDE.md`, `README.md`, `ROADMAP.md`, `alpha_zero/cloud/Agent.md`, `docs/{README,GAME_RULES,MODULE_REFERENCE}.md`, `docs/architecture/C4_ARCHITECTURE.md`
- **Tests:** `tests/governance/{test_governance_determinism,test_guard_review_round1,test_hook_guard_json,test_readonly_shell_guard}.py`, `tests/regression/{test_gm_bridge_and_gpu_e2e_aqa,test_repo_hygiene_regression,test_validator_cli_regression}.py`

## Game budget

- No training or evaluation run cards (`configs/run_cards/{t0b,a1,a1r}.yaml`) or `scripts/check_determinism.py` runs were executed in this change (`0` live Civ6 FireTuner or training games beyond unit/regression micro-tests).
- `docs/EVALUATION.md` §7.1 hashed region is untouched (`REGISTRATION_DIGEST e460a353651a24c26d5e1891b6ad5bd27c18acac4fd9fd6367bc262175237c17` verified via `python -m alpha_zero.evaluation.a1_protocol digest`).

## Tracking issue

None — post-merge SQE verification and pre-PR governance/architecture/AQA readiness on `fix/post-merge-gm-full-e2e-gpu-aqa`.

## Linked OpenSpec change

- `openspec/changes/add-game-rules-kernel` (PRs `#146` GM-K1, `#147` GM-K2; CLI/data follow-up open until calibration)
- `openspec/changes/add-bridge-research-parity` (PRs `#148`–`#153` GM-B0..GM-B5; live FireTuner verification and GM-B6 action translator remain open)

## Hot files touched

`CHANGELOG.md`, `CLAUDE.md`, `README.md`, `ROADMAP.md`, `Makefile`, `.gitignore`, `.dockerignore`, `.gitleaks.toml`, `.pre-commit-config.yaml`, `docs/architecture/C4_ARCHITECTURE.md`, `kb/ledger/ledger.jsonl`.

## Verification

- [x] `ruff check .` (`All checks passed!`) & `ruff format --check .` (`1022 files already formatted`) on `Python 3.11.9`
- [x] `mypy alpha_zero/ --show-error-codes`: `Success: no issues found in 357 source files`
- [x] `bandit -r alpha_zero/ -s B101 --severity-level high --confidence-level high`: `No issues identified` (`123,892` lines)
- [x] `gitleaks dir` (clean committable copy: `20.18 MB`, `0` leaks) + `gitleaks git . --config .gitleaks.toml --redact --no-banner --log-opts="HEAD"` (`857 commits`, `27.86 MB`, `0` leaks)
- [x] `scripts/build_lua_bundle.py --check`: `AlphaZeroMain.lua: up to date`
- [x] `scripts/measure_debt.py --compare docs/evidence/tech_debt_baseline/baseline.json`: exited `0` (`coverage_omitted_lines=66849`, `pytest_skip_sites=150`, `noqa_comments=70`, `type_ignore_comments=148`)
- [x] `python -m alpha_zero.evaluation.a1_protocol digest`: `AGREE: the registration is consistent` (`e460a353...7c17`)
- [x] `scripts/run_llm_eval.py`: `QUALITY GATE: PASS` (`strategy_advisor` `12/12`, `game_report` `8/8`)
- [x] All 6 governance validators (`validate_skills.py`, `validate_agents.py`, `validate_claude_agents.py`, `validate_kb.py --no-git`, `validate_openspec.py`, `check_doc_sync.py`): exit `0` across default, `--json`, and `--verbose` modes
- [x] Track 1 (`C:\Program Files\Python311\python.exe`, `torch 2.11.0+cu128`, `numpy 1.26.4`, `lupa 2.6`, `CIV6AZ_REQUIRE_LUA=1`, Dual-GPU `cuda:0` RTX 5060 Ti + `cuda:1` RTX 5060):
  - `tests/governance/` (`-c tests/governance/pytest.ini --confcutdir=tests/governance`): `[100%]` exit `0` (includes `test_governance_determinism.py` across `PYTHONHASHSEED` in default/`--json`/`--verbose`)
  - `tests/regression/ tests/sanity/ tests/security/ tests/smoke/ tests/inference/`: `562 passed`
  - `tests/unit/ -m "not slow"`: `11789 passed, 47 skipped, 17 deselected, 6 xfailed`
  - `tests/unit/test_selfplay_seed_threading.py` (incl. `@pytest.mark.slow`): `11 passed`
  - `tests/integration/ -m "not slow and not requires_env"`: `605 passed, 5 skipped, 43 deselected, 1 xfailed`
  - `tests/scripts/ -m "not requires_env"`: `944 passed, 5 skipped, 8 deselected`
  - `tests/e2e/ -m "not slow and not requires_env and not requires_gcp"`: `70 passed, 1 skipped, 5 deselected`
  - `test-cloud-coverage` (`tests/cloud/unit/` `831 passed` + `tests/cloud/smoke/` `36 passed`): `alpha_zero/cloud: 60.2% of 5538 statements across 26 files (floor 58%) [OK]`
- [x] Track 2 (`.venv\Scripts\python.exe`, `numpy 2.4.6`, `torch 2.12.0+cpu`, `pytest 9.1.1`, without `lupa` / CUDA):
  - `tests/governance/` (`-c tests/governance/pytest.ini --confcutdir=tests/governance`): `[100%]` exit `0`
  - `tests/regression/ tests/unit/test_selfplay_seed_threading.py tests/scripts/test_lua_bundle_sync.py`: `544 passed, 4 skipped` (`test_cuda_determinism_regression.py` skipped on CPU-only `.venv`)
- [x] Post-rebase (`976a2afd`) verification (`C:\Program Files\Python311\python.exe`, `CIV6AZ_REQUIRE_LUA=1`, `OMP_NUM_THREADS=2`):
  - `ruff check .` & `ruff format --check .` (`1037 files`), `mypy alpha_zero/` (`359 source files`), `scripts/build_lua_bundle.py --check`, `scripts/measure_debt.py --compare`, all 6 governance validators (incl. `validate_kb.py --git`), and `394 passed` across targeted `tests/governance/` (`test_governance_determinism.py`, `test_hook_guard_json.py`, `test_readonly_shell_guard.py`, `test_validate_kb.py`, `test_guard_review_round1.py`) + full `tests/regression/`, `test_selfplay_seed_threading.py`, `test_civ6_local_search_refusal.py`, and `test_lua_bundle_sync.py` passing cleanly (`gitleaks` clean, `0` protected/release-play-path files touched).

## Negative control recorded

- **`posttooluse_governance.sh` Lua bundle guard (`tests/governance/test_hook_guard_json.py::test_posttooluse_governance_routes_lua_mod_edits_to_bundle_check`):** Exercises both valid `mod/AlphaZeroAI/Scripts/*.lua` edit routing (`exit 0`) and a failing `--check` fake runner (`exit 2`, proving sync errors block hook completion).
- **`selfplay_worker.py` search seeding & validator determinism (`tests/unit/test_selfplay_seed_threading.py`, `tests/governance/test_governance_determinism.py::test_negative_control_detects_a_nondeterministic_validator`, `tests/regression/test_gm_bridge_and_gpu_e2e_aqa.py`):** Verified post-fix determinism across CPU/CUDA and negative control against shuffled validator output and CLI option abbreviations (`test_guard_review_round1.py`).

## CONFIRM-FIRST items and who authorized

None — all changes stay within post-merge stabilization, enterprise governance alignment, documentation sync, and non-play-path test/AQA hardening requested by the maintainer on `fix/post-merge-gm-full-e2e-gpu-aqa`. All 28 pathspecs in `scripts/release_play_path.txt`, `docs/EVALUATION.md` §7.1, `kb/handoffs/**`, baselines, and run cards remain untouched.

## Notes

- Pre-existing unrelated local workspace artifacts (`uv.lock` modified, untracked `0.26.0` and `mypy_output.txt`) are outside PR scope and excluded from secret scan staging/commit.
- Live Steam FireTuner (`127.0.0.1:4318`) and live GCP (`requires_gcp`) paths were not exercised (no live game/cloud environment active in local workstation CI pass).
- **Pre-existing upstream GM-B4 (`976a2afd`) Windows-host findings (follow-up outside this PR's scope):** running the full bridge/Lua unit sweep on Windows (`Python 3.11.9`) surfaced `15` failures in unmodified upstream PR #152 (`GM-B4`) files (`2,858 passed, 15 failed, 6 skipped`): `mod/AlphaZeroAI/Scripts/JsonEncoder.lua` (`[%c]` matching multi-byte UTF-8 continuation bytes under Windows MSVCRT in `test_lua_json_encoder.py::TestRoundTrip::test_a_json_reader_gets_the_string_back[café 中]`) and `14` Windows path-separator / CRLF / unprivileged `WinError 1314` symlink assertions across `tests/unit/test_bridge_gameinfo_capture{,_review,_review2}.py`. Left untouched here so this PR stays strictly scoped without unapproved `mod/` edits.
