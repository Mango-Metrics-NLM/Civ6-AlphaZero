---
name: verify
description: Reproduce every CI gate locally in one pass — lint, format, types, all CI test suites (with CIV6AZ_REQUIRE_LUA=1), security, cloud coverage floor, governance, all six validators plus Lua bundle sync, and StrategyAdvisor offline eval. Use before any handoff, before opening or updating a PR, and after any change that could affect CI. Wraps `make verify`.
allowed-tools: Bash, Read
---

# Verify Skill

One command that reproduces what CI actually gates on, so "green locally" means
something.

## Why this exists

The verification sequence used to be reassembled by hand each session — six
pytest invocations plus ruff, mypy and the validators — and the hand-assembled
version drifted from CI on four independent axes: the coverage gate (80 local
vs 85 in CI), the coverage scope (whole tree vs CI's filtered subset), the mypy
flags (`--strict` locally, plain in CI), and `validate_kb` (with and without
`--git`). Passing the local sequence therefore proved nothing about CI.

`make verify` is now the single definition. If CI changes, change it there.

## Steps

### 1. Run the gate

```bash
make verify
```

This runs, in CI's order (with `CIV6AZ_REQUIRE_LUA=1` and `OMP_NUM_THREADS ?= 2`):
`lint-check` (ruff check + format --check), `typecheck` (`mypy alpha_zero/`), all CI test
suites (`unit`, `sanity`, `regression`, `integration`, `smoke`/`inference`, `scripts`, `e2e`,
`cloud/unit`, `cloud/smoke`), `test-security` (security pytest + high/high `bandit`),
`security-secrets` (`gitleaks`), `test-governance`, `validate` (all six governance
validators — `openspec`, `kb --git`, `skills`, `agents`, `claude-agents`, `doc-sync` —
plus `check-lua-bundle`), `test-all` (85% global coverage gate), `test-cloud-coverage`
(per-package cloud floor check), and `eval-llm` (StrategyAdvisor offline output-contract gate).

### 2. If the environment is suspect, check it first

```bash
bash .claude/hooks/sessionstart_bootstrap.sh
```

It reports missing dependencies and — importantly — whether `mypy` or `ruff`
resolve to an isolated tool venv that cannot see this project's type stubs. If
it reports a tool/interpreter split, use `python3 -m mypy` (or `python -m mypy`
on Windows where `python3` may resolve the Store alias), not `mypy`; the bare
command reports dozens of phantom `import-untyped` errors. `make typecheck` handles
interpreter selection automatically.

### 3. What it covers

`make verify` runs lint, format, types, all CI test suites (`test`, with `CIV6AZ_REQUIRE_LUA=1`
via `make install-lua`), security (pytest **and** the high/high bandit scan CI runs), secret
scanning (`security-secrets`), governance (`test-governance`), all six validators plus Lua bundle
sync (`validate`), global coverage (`test-all`, 85% gate), cloud-package coverage floor check
(`test-cloud-coverage`), and the StrategyAdvisor offline eval check (`make eval-llm`).

It did not always. It shipped omitting coverage, bandit and `eval-llm` while its
own echo claimed "all CI gates reproduced locally" — the exact drift this target
exists to prevent, in the target itself. If you add a CI job, add it here too;
when the two lists diverge, passing locally stops meaning anything. Note that `alpha_zero/cloud/*` is in the coverage
omit list, so changes there cannot move the number — never cite coverage as
evidence that cloud-path work is tested.

### 4. Report honestly

State pass/fail counts per suite. If a suite could not run (a missing optional
dependency, a sandbox limitation), say so explicitly and name CI as the gate
for it. Do not report a suite as green that you did not run.
