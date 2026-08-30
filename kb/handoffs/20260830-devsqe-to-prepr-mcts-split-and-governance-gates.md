# DEV-SQE → Pre-PR Handoff

**Date:** 2026-08-30
**Spec:** `openspec/changes/harden-governance-gates/proposal.md`
**Branch:** `claude/god-files-decomposition-gojlo3`
**PR:** #115 (draft)
**Session:** session_0161DvFczCzPQqumjKqYdrXy

## What shipped

Two things, in four commits, each independently revertible.

### 1. `core/civ6_mcts.py` decomposed (`bedb371`, `5b41ca4`)

2,443 lines holding five search variants, four protocols and three config
dataclasses → `alpha_zero/core/mcts/`, seven modules, largest 876 lines.
`civ6_mcts.py` is a permanent re-export shim with explicit `__all__`.
`mcts_factory.py` is deliberately **untouched** — its lazy import resolves
through the shim, so the diff outside the moved blocks is the shim and docs.

Layering, corrected during design by reading call sites rather than assuming:
`gumbel` and `muzero` carry their own node/config types and depend on neither
`base` nor each other. The Gumbel-Top-k helpers went into their own leaf module
`gumbel_ops` precisely because `MuZeroMCTS` calls them.

### 2. Governance gates (`35ff6db`)

Three gaps, each of which let a class of defect through unnoticed:

- **No secret scanning.** gitleaks now scans tree + all 525 commits. Pin has one
  definition (Makefile); CI reads it via `make print-gitleaks-version`. Baseline
  0 findings → blocking, no burndown.
- **30 `Agent.md` persona docs had zero validator coverage.** New
  `scripts/validate_agents.py` (A1–A6). First run found 16 real errors.
- **`.agents/skills/` was ungated** — routed to the validator by the hook but
  scanned by neither pre-commit nor CI. Now covered, plus an S5 rule and a
  frontmatter parser that handles YAML block scalars.

Plus repo hygiene: five tracked-but-ignored files, 83 duplicate `.gitignore`
entries, a 110-line `.dockerignore` duplicate, a percent-regex bug in two
validators, and the `.claude/settings.json` permissions that the `$(PYTHON) -m`
Makefile routing broke.

### 3. Makefile interpreter routing (`b2b2d4b`)

`make test-all` failed with `ModuleNotFoundError: numpy` even with `.[dev]`
installed — the recipes invoked a bare `pytest`, which resolved to a uv tool venv
that cannot see the project. Same hazard CLAUDE.md §15.3 already records for
mypy. 18 recipe lines now go through `$(PYTHON) -m`.

**Consequence larger than the broken target:** with the invocation fixed, the
coverage gate runs here. Local coverage had been written off as impossible on the
strength of a failure whose actual cause was PATH resolution.

## Files changed

**New:** `alpha_zero/core/mcts/{__init__,mcts_protocols,gumbel_ops,base,puct,simple,gumbel,muzero}.py`,
`alpha_zero/hub/Agent.md`, `scripts/validate_agents.py`, `.gitleaks.toml`,
`tests/unit/test_mcts_split_backcompat.py`, `tests/governance/test_validate_agents.py`,
`tests/regression/test_repo_hygiene_regression.py`,
`openspec/changes/harden-governance-gates/`.

**Modified:** `alpha_zero/core/civ6_mcts.py` (→ shim), `scripts/validate_skills.py`,
`alpha_zero/cloud/gcp/container_healthcheck.py`, `Makefile`,
`.github/workflows/ci.yml`, `.pre-commit-config.yaml`, `.gitignore`,
`.dockerignore`, `.claude/settings.json`, `.claude/hooks/posttooluse_governance.sh`,
`.agents/skills/*/SKILL.md` (3), `alpha_zero/{core,monitoring,inference,tournament}/Agent.md`,
`tests/Agent.md`, `docs/architecture/C4_ARCHITECTURE.md`,
`docs/NEXT_STEPS_CASE_2026-08.md`, `docs/MODULE_REFERENCE.md`,
`kb/specs/20260217-multiplayer-gap-analysis.md`, `CLAUDE.md`, `README.md`,
`CHANGELOG.md`, `pyproject.toml`.

## Tests added

`test_mcts_split_backcompat.py` (41), `test_validate_agents.py` (13),
`test_repo_hygiene_regression.py` (10). Total 64.

## Test results

| Gate | Result |
|---|---|
| `pytest tests/{unit,sanity,integration,cloud/unit,security,regression}` | **12,381 passed**, 66 skipped, 1 xfailed, 0 failed (5m40s) |
| Coverage (`make test-all` equivalent) | **89.30%** against the 85% gate — 12,327 passed, 12m14s |
| `ruff check .` / `ruff format --check .` | clean, 937 files |
| `mypy alpha_zero/` | clean, 367 files |
| Validators (openspec/kb/skills/agents/doc-sync) | all exit 0 |
| `tests/governance` | 64 passed |
| `make security-secrets` | 0 findings, tree + 525 commits |
| `python -m alpha_zero.training_civ6 --dry_run` | exit 0 |
| CI on PR #115 @ `5b41ca4` | 7/8 green; Coverage Report was still running |

**Coverage caveat:** 89.30% was measured at `5b41ca4`. The governance commit that
followed adds only tests, docs and config — no `alpha_zero/` runtime code except a
one-line addition to `container_healthcheck.py`, which is coverage-omitted — so the
number is not expected to move, but was **not** re-measured after `35ff6db`.

## Negative controls

Every guarantee was reproduced as a failure before landing (`negative-control` skill):

| Injected defect | Test that fired |
|---|---|
| Shim re-implements `SimpleMCTS` instead of re-exporting | `test_shim_reexports_are_identical` |
| `gumbel` imports `base` (absolute) | `test_independent_variants_do_not_depend_on_base` |
| `gumbel` imports `base` (`from .base import`) | same — **after** the Copilot fix; missed before it |
| `gumbel` imports `base` (`from . import base`) | same |
| Name dropped from the shim's `__all__` | `test_shim_all_is_complete_and_resolvable` |
| Consumer repointed off the shim (dead `mock.patch`) | `test_patching_the_shim_reaches_a_function_local_consumer` |
| 120-statement function in a non-exempt file | ruff `PLR0915` |
| `docker-compose.yml` negation removed | `test_no_tracked_file_is_ignored` |
| Duplicate `.gitignore` entry | `test_no_duplicate_ignore_entries` |
| Catch-all gitleaks allowlist | `test_allowlist_is_not_a_catch_all` |
| A4/A6/A5/A3/A1/A2 violations in `Agent.md` | 13 governance tests |

**One control failed and taught something:** the first gitleaks negative control
used AWS's own documentation keys (`AKIAIOSFODNN7EXAMPLE`) and did **not** fire —
gitleaks allowlists those upstream. The config was only proven to work after
retesting with randomly generated values. Recorded in CLAUDE.md §5.2.

## Lint / type-check status

Clean. `ruff check .` includes the newly-enabled `PLR0915`; three `main` functions
exceed it (248/168/134 statements) and sit in `per-file-ignores` as an explicit
burndown list — delete an entry when its function is split, never raise the limit.

## Known issues / carried forward

- **The cloud trio is deliberately NOT decomposed.** Adversarial review found 38
  `mock.patch` string targets against those modules; a shim preserves `old is new`
  while a patch on it becomes a silent no-op once the patched symbol's consumer
  moves. Prerequisites named in the plan and in CLAUDE.md §5.2.
- **Three pre-existing defects surfaced by review, filed not fixed** (this PR's
  non-goal is pure relocation): `gumbel_ops` divides by `temperature` with no
  zero guard (yields `-inf`, degrading Gumbel-Top-k to uniform sampling — not the
  `ZeroDivisionError` the reviewer claimed); `SimpleMCTS.select_action` uses the
  global RNG while the class holds `self._rng`; a `gumbel.py` docstring describes
  a Gumbel-weighted heuristic the code does not implement.
- **`MuZeroNetworkProtocol` is defined twice** (`mcts/mcts_protocols.py` and
  `core/losses.py`) and `core/__init__.py` re-exports both, so one shadows the
  other. Also `MCTSConfig` collides with `orchestrator/config.py::MCTSConfig`.
- **22 A3 warnings remain** — five `Agent.md` files missing conventional sections.
  Deliberately non-blocking; a visible burndown.
- `.[dev]` still fails on `chess` in sandboxes whose pip resolves a wheel-less
  version. This is an environment artifact, **not** a repo defect —
  `pip install --only-binary=:all: chess` resolves it.

## Notes for Pre-PR

- Four Copilot review threads on #115: one fixed and resolved (relative-import
  blind spot in my own test), three replied to with verification that they are
  pre-existing and out of scope for a relocation commit. Those three are
  intentionally left **open** for the author to triage.
- `openspec/changes/harden-governance-gates/` must be linked in the PR's
  "Linked OpenSpec change" field. Archiving is a separate post-merge PR.
- Tasks 4.3–4.5 in that change (splitting the three oversized `main` functions)
  are deliberately unchecked — they are the burndown, not this PR's scope.
