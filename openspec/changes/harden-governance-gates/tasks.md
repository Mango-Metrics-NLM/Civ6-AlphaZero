# Tasks

## 1. Secret scanning
- [x] 1.1 Add `.gitleaks.toml` extending the default ruleset, with an artifact-path
      allowlist and value-scoped fixture exemptions
- [x] 1.2 Measure the adoption baseline (0 findings / 525 commits) before making it blocking
- [x] 1.3 Add `make gitleaks-install` / `make security-secrets` with a pinned, overridable version
- [x] 1.4 Add `make print-gitleaks-version` and have CI read the pin from it
- [x] 1.5 Add tree + history scan steps to the CI `security` job with `fetch-depth: 0`

## 2. Agent.md validation
- [x] 2.1 Write `scripts/validate_agents.py` (A1–A6, `--path`/`--quiet`/`--json`/`--verbose`)
- [x] 2.2 Fix every error it reports (monitoring rewrite, new `hub/Agent.md`, core paths, tests gate)
- [x] 2.3 Wire into PostToolUse hook, pre-commit, `make validate`, CI lint list, CI run step
- [x] 2.4 Add `tests/governance/test_validate_agents.py` in house style

## 3. Skills tree coverage
- [x] 3.1 Scan `.agents/skills/` alongside `.claude/skills/`
- [x] 3.2 Parse YAML block scalars so folded `description:` values are real
- [x] 3.3 Add S5 (skill directory with no `SKILL.md`)
- [x] 3.4 Fix the violations this surfaces

## 4. Function-size guard
- [x] 4.1 Enable ruff `PLR0915` at `max-statements = 100`
- [x] 4.2 Record the three over-limit `main` functions as an explicit burndown list
- [ ] 4.3 Split `vertex_orchestrator.main` (248 stmts) and remove its entry
- [ ] 4.4 Split `training_worker.main` (168 stmts) and remove its entry
- [ ] 4.5 Split `run_enhanced_training.main` (134 stmts) and remove its entry

## 5. Repository hygiene
- [x] 5.1 Add `tests/regression/test_repo_hygiene_regression.py`
- [x] 5.2 Fix the five tracked-but-ignored files
- [x] 5.3 Dedupe `.gitignore` (83 entries) and `.dockerignore` (110 lines), behaviour-verified
