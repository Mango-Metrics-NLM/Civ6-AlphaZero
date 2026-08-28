# Contributing to Civ6 AlphaZero

Thanks for contributing. This guide covers the development loop, the quality
gates, and the spec-driven change workflow.

## Prerequisites

- Python **3.11+**
- A POSIX shell (Linux/macOS) or Windows with Git Bash

## Setup

```bash
git clone https://github.com/Mango-Metrics-NLM/Civ6-AlphaZero.git
cd Civ6-AlphaZero
python -m venv venv && source venv/bin/activate
pip install -e '.[dev]'
make hooks-install          # installs the pre-commit and pre-push git hooks
```

## Development loop

```bash
make verify        # EVERY CI gate, in CI's order — run this before opening a PR
```

`make verify` is the single definition of the gate. Prefer it over assembling the
sequence by hand: the hand-assembled version drifted from CI on four axes (coverage
threshold, coverage scope, mypy flags, and whether the KB immutability check ran), so
passing it proved nothing about CI. The individual targets remain available for a
faster inner loop:

```bash
make lint          # ruff check/format, whole repo (0.15.8, 100-char lines)
make typecheck     # mypy, exactly as CI runs it (typecheck-strict is local-only)
make test          # the four suites CI runs, with CI's marker filters
make test-all      # coverage, mirroring CI's coverage job (gate: 85%)
make validate      # governance validators (openspec, kb, skills, doc-sync)
make eval-llm      # LLM eval gate (offline)
```

The installed git hooks run ruff/format and the governance validators on commit,
and a quick unit-test pass on push.

## Quality gates

- **Coverage:** CI enforces **85%** on `alpha_zero/` (`--cov-fail-under=85`,
  matching `pyproject.toml` `fail_under`). Keep new code covered.
- **Lint:** ruff check + format must pass on the **whole repository** (tests/,
  scripts/, infra/, and root runners included; unmaintained fringe dirs are
  excluded centrally in `pyproject.toml`). Line length is 100.
- **Types:** mypy is a **blocking** CI gate as of 2026-07-25 — `mypy alpha_zero/`
  must exit 0 in the dev-extra environment; several packages
  (`alpha_zero.core/cloud/training/tournament/pbt/web/evaluation/envs`) are held
  to stricter settings — keep them clean.
- **Security:** bandit runs in CI on `alpha_zero/`.
- **Docs:** `check_doc_sync.py` fails the build on version drift, stale module
  counts, retired-tooling references in the README, unknown extras, or broken
  links. Run `make validate` before pushing docs.

## Branching & pull requests

- Never commit or push to the default branch — the harness guard blocks it. Work
  on a feature branch and open a **draft** PR.
- Fill in the PR template. Ensure the verification checklist passes.
- Force-pushes are blocked (use `--force-with-lease` only when genuinely needed).

## OpenSpec change workflow

Substantive changes — anything that alters a capability's behavior, adds a
requirement, or touches a governance gate — are proposed under
`openspec/changes/<change-id>/` **before** implementation:

1. Scaffold with the `openspec-propose` skill (or by hand): `proposal.md`,
   `tasks.md`, optional `design.md`, and spec deltas.
2. Implement against `tasks.md`.
3. After merge, apply the deltas to the baseline with the `openspec-archive` skill.

The binding gate is `python scripts/validate_openspec.py`. See
[`openspec/AGENTS.md`](openspec/AGENTS.md) for details. The upstream npm
`openspec` CLI is an optional convenience; the vendored Python validator is
authoritative.

## Releases

Versioning is semantic (`MAJOR.MINOR.PATCH`). To cut a release, use the `release`
skill (or follow it by hand): bump the version in `pyproject.toml` and
`alpha_zero/__init__.py`, roll `CHANGELOG.md`'s `[Unreleased]` section into
`[X.Y.Z]`, and push a `vX.Y.Z` tag. Pushing the tag triggers
`.github/workflows/release.yml`, which builds the GitHub release from the tag.
`check_doc_sync.py` enforces that the version and the changelog head agree.

## Governance & knowledge base

- Scope and invariants: [`PROJECT_CHARTER.md`](PROJECT_CHARTER.md) (amendment
  protocol in §7–8).
- Session continuity uses the `kb/` handoff protocol — see
  [`kb/README.md`](kb/README.md) and `CLAUDE.md` §14.
