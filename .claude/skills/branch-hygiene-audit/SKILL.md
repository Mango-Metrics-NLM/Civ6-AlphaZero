---
name: branch-hygiene-audit
description: Run a deterministic pre-PR SQE/SWE/Architect branch audit covering play-path isolation, AST god-file & tech-debt ratchets, coverage floors, ruff/mypy/bandit/gitleaks checks, narrow-critic security/style review, all governance validators, and doc/C4/CHANGELOG/KB sync. Use before opening or updating any PR.
allowed-tools: Bash, Read, Edit
---

# Branch Hygiene Audit Skill

Provides a repeatable, deterministic pre-PR verification workflow that catches CI blockers, play-path drift, god-file growth, coverage regressions, and harness/documentation drift before pushing.

## Prerequisites

- Python 3.11+ with `.[dev,lua]` installed (`CIV6AZ_REQUIRE_LUA=1`, `OMP_NUM_THREADS=2`).
- Base ref to compare against (for example `origin/claude/civ6-alphazero-implementation-vMnun` or `origin/dev`).

## Steps

### 1. Check play-path isolation

Read non-comment lines of `scripts/release_play_path.txt` and run `git diff --name-only <base>...HEAD -- <pathspecs>`. Pre-`v2.1.0` work outside the registered learning run must produce an empty hit list (`PLAY_PATH_HITS: []`).

### 2. Inspect tech-debt metrics & god-file LOC

```bash
python scripts/audit_import_graph.py --json
python scripts/measure_debt.py --compare docs/evidence/tech_debt_baseline/baseline.json
```

When edits reduce a ratcheted metric (`files_over_1000_lines`, `runtime_asserts`, `except_pass_handlers`, `dead_modules`), regenerate `docs/evidence/tech_debt_baseline/baseline.json` in the same commit with `--write`. Never raise a ratcheted baseline to bypass `tests/regression/test_debt_ratchet.py`.

### 3. Run static analysis, import smoke check, and security/secret scans

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy alpha_zero/ --show-error-codes
python -m bandit -r alpha_zero/ -s B101 --severity-level high --confidence-level high
make security-secrets  # when gitleaks / POSIX shell is available
```

Also import every module under `alpha_zero/` to ensure zero broken or circular imports, and verify zero hardcoded environment values or production `assert` statements on untrusted I/O paths.

### 4. Enforce per-module coverage and package floors

```bash
python -m pytest <modified-tests> --cov=<modified-module> --cov-report=term-missing --cov-fail-under=85 -q
python -m pytest tests/cloud/unit/ tests/cloud/smoke/ --cov=alpha_zero.cloud --cov-config=.coveragerc-cloud --cov-report=
python -m coverage json --rcfile=.coveragerc-cloud -o coverage-cloud.json
python scripts/check_coverage_floors.py --report coverage-cloud.json --package alpha_zero/cloud
```

### 5. Run all deterministic repo validators and regression/governance suites

```bash
python scripts/build_lua_bundle.py --check
python scripts/validate_openspec.py
python scripts/validate_kb.py --git
python scripts/validate_skills.py
python scripts/validate_agents.py
python scripts/validate_claude_agents.py
python scripts/check_doc_sync.py --verbose
python scripts/check_release_readiness.py --verbose
OMP_NUM_THREADS=2 CIV6AZ_REQUIRE_LUA=1 python -m pytest tests/regression/ -q --tb=short
python -m pytest -c tests/governance/pytest.ini --confcutdir=tests/governance tests/governance -q --tb=short
```

### 6. Sync documentation, C4 architecture, ignore rules, and KB session state

Before pushing:
- Update `docs/architecture/C4_ARCHITECTURE.md`, `docs/LIVE_CIV6.md`, package `Agent.md`, `README.md`, and `CHANGELOG.md` (`## [Unreleased]`) for any module splits, new gates, or live evidence.
- Verify `.gitignore`, `.dockerignore`, and `.gitleaks.toml` remain aligned on generated scratch artifacts without allowlisting `.env*` secrets.
- Sync `kb/sessions/{dev-sqe,pre-pr}.state.json` (`resume_pointers.next_steps` <= 5 non-empty strings) and record a Tier B objective gate artifact before requesting human merge review.

## Related

- `verify` — full CI parity gate.
- `debt-audit` — AST import graph and tech-debt baseline workflow.
- `public-release-audit` — deterministic R1–R6 release readiness checks.
