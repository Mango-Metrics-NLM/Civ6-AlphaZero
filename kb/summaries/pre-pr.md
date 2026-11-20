# Pre-PR Summary

> Rolling summary for the Pre-PR workflow role (newest entries at the top, capped
> at 20). Seeded so the `pre-pr-handoff` skill has a target to prepend to.

<!-- entries below -->

## 2026-11-20 - Pre-PR post-merge GM+Bridge E2E/GPU AQA & governance alignment (`fix/post-merge-gm-full-e2e-gpu-aqa`)
- **Outcome:** `PR-READY` draft created at `kb/pr-drafts/20261120-post-merge-gm-e2e-and-prepr-hardening.md` (rebased onto `976a2afd`, GM-B0..GM-B5); all CI suites passed on both Python 3.11 (`torch 2.11.0+cu128`, `numpy 1.26.4`, `lupa 2.6`, Dual-GPU `cuda:0`/`cuda:1`) and `.venv` (`numpy 2.4.6`, `torch 2.12.0+cpu`) with `alpha_zero/cloud` coverage at `60.2%` (`>= 58%` floor), zero release-play-path edits, zero `# noqa` / `# type: ignore` / skip additions, and clean `ruff`/`mypy`/`bandit`/`gitleaks`/`measure_debt`/`eval-llm`/6-validator runs.
- **Key changes:** Fixed off-play-path `selfplay_worker.py` `SeedStream.SELF_PLAY_SEARCH` seeding (`run_selfplay` + optional `seed` on `run_muzero_selfplay`) with DRY helpers; added zero-skip `tests/regression/test_gm_bridge_and_gpu_e2e_aqa.py`; unified `issues.sort`, `sort_keys=True` `--json`, and `--verbose` stderr logging across all 6 governance validators (`validate_agents.py`, `validate_kb.py`, `validate_openspec.py`, `check_doc_sync.py`); wired `build_lua_bundle.py --check` into `Makefile` (`check-lua-bundle`/`validate`), `.pre-commit-config.yaml`, and `posttooluse_governance.sh`; synced `C4_ARCHITECTURE.md` (Level 2 + Level 3 GM/Bridge diagram incl. GM-B4/B5), `.gitignore`, `.dockerignore`, `.gitleaks.toml`, `CHANGELOG.md`, `ROADMAP.md`, `README.md`, `CLAUDE.md`, `.claude/`, and `docs/`.

## 2026-11-19 - PR #145 triage: merge PR #142 base, fix K6 state JSON, `_resolve_python.sh`, and Linux/LF hook tests
- **Session:** 31c93d11-adb3-4751-9d30-11395c290068
- **Validating:** Merged `origin/claude/civ6-alphazero-implementation-vMnun` (`690fe16`, PR #142) cleanly into `fix/post-merge-gpu-e2e-aqa-hardening` (resolved `ROADMAP.md` and `kb/ledger/ledger.jsonl` preserving both streams), repaired corrupted `kb/sessions/pre-pr.state.json` (eliminating CI K6 failure across `Governance Gates` and `Test Python 3.11/3.12/3.13`), guarded unset `$VIRTUAL_ENV` and enumerated multi-hit `PATH` interpreters via `type -aP` in `.claude/hooks/_resolve_python.sh`, and enforced explicit LF newlines + POSIX non-python `safe_bin` isolation in `tests/governance/test_hook_guard_json.py`.
- **Status:** Verified locally after final edit; ready for push to PR #145.

## 2026-11-18 - Post-merge Windows & dual-GPU E2E hardening, zero-warning `Agent.md`, and deterministic validators
- **Session:** 31c93d11-adb3-4751-9d30-11395c290068
- **Validating:** Branch `fix/post-merge-gpu-e2e-aqa-hardening` against `origin/claude/civ6-alphazero-implementation-vMnun` (`439ee9c4`). Fixed `RAGClient._call_safely` native panic handling, shared `audit_import_graph.iter_python_files` + PEP 405/worktree pruning, portable Windows Python/Git Bash resolution across `Makefile`, `.claude/hooks/*.sh`, and `tests/governance/_bash.py`, deterministic sorted `--json`/`--verbose` validators + synced `readonly_shell_guard.sh`, `.gitignore`/`.dockerignore`/`.gitleaks.toml` virtualenv/worktree exclusions (without `.env*`), C4 `log_evaluation_result()` 15-metric alignment, and burned down all `Agent.md` warnings (`22 -> 0`). `docs/EVALUATION.md` §7.1 untouched (`REGISTRATION_DIGEST e460a353651a24c26d5e1891b6ad5bd27c18acac4fd9fd6367bc262175237c17`).
- **Lint:** pass (`uvx ruff==0.15.8 check .` + `format --check .`, 986 files)
- **Types:** pass (`uvx mypy==1.19.1 alpha_zero/ --show-error-codes`, 346 files, 0 issues)
- **Security:** pass (`bandit` 0 high issues across 121,932 lines; `gitleaks dir` 6.96s & `gitleaks git --log-opts=HEAD` 3.02s -> 0 leaks)
- **Tests:** governance `790 passed`; regression/sanity/security/smoke/inference/cloud-smoke `543 passed`; CI coverage sweep (`unit`/`sanity`/`integration`/`cloud/unit`/`regression`, `-m "not slow and not requires_env"`) `12,834 passed, 50 skipped, 60 deselected, 7 xfailed`, total coverage `89.31%` (gate 85%); `scripts`+`e2e` `952 passed, 11 skipped, 8 deselected`; cloud floor `831 + 36 passed` (`60.1%` vs `58%` floor); registration `102 passed`; `eval-llm` `12/12 + 8/8` PASS; `measure_debt` & all 6 `make validate` checks pass.
- **Smoke Tests & Live Dual-GPU:** `training_civ6 --dry_run` exit 0; `orchestrator --preset fast --checkpoint-dir <tmp> --dry_run` exit 0; `scripts/check_determinism.py --train-device cuda:0 --arena-device cuda:1` -> `REPRODUCED` (`differences: []`, RTX 5060 Ti + RTX 5060, `torch 2.11.0+cu128`).
- **Review:** Isolated `/narrow-critic` subagent review completed; resolved both surfaced hook findings via shared `.claude/hooks/_resolve_python.sh` (`${CLAUDE_PROJECT_DIR:-.}/.venv` + `>= 3.11` check across all 4 Python hooks, `cd "$CLAUDE_PROJECT_DIR"` before resolution in `posttooluse_governance.sh` / `precompact_state.sh`) with outside-cwd regression test in `tests/governance/test_hook_guard_json.py`.
- **Status:** PR-ready; awaiting human push and PR creation against `origin/claude/civ6-alphazero-implementation-vMnun`.
- **Next:** Human push of `fix/post-merge-gpu-e2e-aqa-hardening`; maintainer T0b single-use `run_card.py` execution + A1/A1-R runs + branch cut-over.

## 2026-11-15 - v2.1 local close-out (PR #140)
- **Session:** session_01CXusP6jtsBXpQcjpC9W3b8
- **Validating:** 19 commits on `claude/cool-lovelace-o02vn9` over `6d548d8`: rehearsal-first run cards, the torch wheel record, atomic checkpoint and result writes, the driver record, the release verifier applying the "After a VOID" rule, and the archive of five merged OpenSpec changes. No edit inside the hashed region of `docs/EVALUATION.md` (digest `e460a353…7c17`).
- **Lint:** pass (`ruff check .` + `ruff format --check .`, 981 files)
- **Types:** pass (`python -m mypy alpha_zero/`, 346 files)
- **Tests:** unit 10,546 passed; integration + e2e + smoke + sanity + inference 740 passed; regression + governance + scripts + security + cloud 2,998 passed. Eight failures, all also failing on the base commit in this container (root, no `.env`, no Google credentials). Coverage by path on changed modules 86-100%; the global gate is CI's (all 24 checks on `d44f7ee` completed: 23 passed, `pip-audit` skipped by design). `make verify` as one command was not run: the coverage plugin crashes here.
- **Smoke Tests:** pass (`training_civ6 --dry_run` exit 0; `orchestrator --preset fast --dry_run` exit 0)
- **Review:** in isolated subagents, four passes: `sdlc-ml-scientist` and `sdlc-qa-sdet` on the first version, `sdlc-ml-scientist` on the fix (it found one more route around the ordering rule), then its confirmation that the findings are closed (by reading; its shell cannot run pytest).
- **Status:** PR-ready as a draft; awaiting human review. Not marked ready for review and not merged: both are the maintainer's.
- **Next:** watch PR #140 (CI on the final head, review comments). Maintainer: the CONFIRM-FIRST items and conservative readings in the PR body; then T0b on the GPU machine.

## 2026-09-04 - Correctness-first branch: search fix, seeding, and the gates that were fail-open (PR #118)
- **Session:** session_01KZPsP2Mvg333p1akDqpLTG
- **Validating:** six commits on `claude/2026-standards-audit-7tgk7e` -- the 2026 standards audit, the correctness-first plan, one sign-correct value backup shared by all five MCTS variants, the Danihelka completed-Q policy target, per-stream seed derivation, `MicroCivEnv.clone()` terrain sharing, and the harness/supply-chain fixes (SHA-pinned actions, CI PR trigger, coverage-floor ratchet, dead-code ratchet vs build artifacts, `chess` out of the `dev` extra).
- **Lint:** pass (`ruff check .` + `ruff format --check .`, whole repo)
- **Types:** pass (`python -m mypy alpha_zero/` -> 330 files, 0 errors)
- **Tests:** 10,045 unit+sanity+regression pass / 58 skipped; integration 580; e2e 70; smoke 30; scripts 178; cloud 846+36; security 18; governance 55+. Coverage gate (>=85%) enforced and **green in CI** on head `b715582` (runs 449 push / 450 pull_request, both success); the local coverage run was inconclusive in this sandbox and CI is cited as its gate.
- **Smoke Tests:** pass (`training_civ6 --dry_run` exit 0; `orchestrator --preset fast --dry_run` exit 0)
- **Status:** PR-ready (CI green on the current head; two Copilot review findings addressed in `b715582`)
- **Next:** Maintainer-only items remain open and are named in the plan's §9.3 -- W&B key rotation, the `allUsers` inference binding, tagging `v1.8.0`/`v1.9.0`, and the S5 A/B evidence seed-overlap decision. WS-S item S5 and the `policy_target`/`greedy_threshold` config plumbing (reachable in code only) are the open engineering follow-ups.

## 2026-07-25 - Code hygiene sweep + gate hardening (PR #104)
- **Session:** session_018npcgPmnuoRAww3eMeffk7
- **Validating:** hygiene phases 1-6 (`harden-lint-and-type-gates` OpenSpec change): dead-code deletions (~3,350 ln), five-way pin alignment (ruff 0.15.8 / mypy 1.19.1), whole-repo lint gate, mypy 130→0 + blocking gate, eval-harness enforcing.
- **Lint:** pass (`ruff check .` + `ruff format --check .`, 863 files)
- **Types:** pass (`mypy alpha_zero/` exits 0, flag-free, canonical env)
- **Tests:** 11,728 pass / 0 fail / 57 skipped / 1 xfailed (CI-equivalent sweep); coverage delegated to CI job (local numpy-2.4 tracer issue), deleted modules already omitted
- **Smoke Tests:** dry-run smoke running at handoff time; CI test jobs green on run 30169685310
- **Independent review:** ran in an isolated subagent (review of record). Verdict: contract holds; 2 blocking findings fixed in `91c9b7c` (validate_kb K5 archive fallback — governance job was red; ruff extend-exclude anchoring — alpha_zero/interfaces/ was silently de-gated) + Makefile/CLAUDE.md scope drift + faiss fail-fast restore.
- **Status:** PR-ready (draft PR #104; mypy gate flipped after green advisory run 30169685310)
- **Next:** watch PR #104 CI (post-fix run must show governance green + blocking mypy green); merge; post-merge run `openspec-archive` for `harden-lint-and-type-gates`; follow-up PRs: dependency reconciliation, W&B naming sweep (user-approved), module merges, library hygiene + mypy strict expansion.

## 2026-07-25 - Enterprise docs overhaul + governance gates (PR #93)
- **Session:** session_01Rw3tQDw76pEDHNHxvCPEgW
- **Validating:** phases A-F of `add-enterprise-docs-and-gates` (OpenSpec).
- **Lint:** pass (ruff on `alpha_zero/` + governance code)
- **Tests:** 48 governance tests pass (`tests/governance/`); full suite runs in CI
- **Gates:** `make validate` (openspec/kb/skills/doc-sync) green; `pre-commit run --all-files` green
- **Status:** PR-ready (draft PR #93, awaiting CI + human review)
- **Next:** confirm CI green; post-merge run `openspec-archive`; decide on tagging v1.8.0
