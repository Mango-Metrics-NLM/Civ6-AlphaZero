# Pre-PR Summary

> Rolling summary for the Pre-PR workflow role (newest entries at the top, capped
> at 20). Seeded so the `pre-pr-handoff` skill has a target to prepend to.

<!-- entries below -->

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
