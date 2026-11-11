# Contributing to Civ6 AlphaZero

Thanks for contributing. The first half of this guide is everything a human
contributor needs: set-up, the inner loop, the gates a pull request must pass,
and how to open one. The second half describes the spec-driven and agent
workflows the maintainer uses; you can ignore it unless you are changing a
governance gate or working with the coding agent.

## Prerequisites

- Python **3.11 or 3.12** (3.13 is exercised in CI as experimental)
- A POSIX shell (Linux/macOS) or Windows with Git Bash
- No GPU is needed; everything runs on CPU, slowly

## Set-up

```bash
git clone https://github.com/Mango-Metrics-NLM/Civ6-AlphaZero.git
cd Civ6-AlphaZero
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
make hooks-install          # optional: pre-commit + git pre-commit/pre-push hooks
```

`pip install -e '.[dev]'` installs the package, pytest, ruff, mypy and the
governance tooling. Some optional features need extras (`[monitoring]`,
`[tuning]`, `[hub]`, `[web]`, `[eval]`); the README lists what each enables.

If torch wedges on your machine (a `conv2d` that never returns on a small VM),
set `OMP_NUM_THREADS=2` before running anything that imports torch — the
OpenMP spin-wait, not the code, is the cause.

## The inner loop

```bash
make verify-fast   # lint, types, governance + regression tests, validators — minutes
make verify        # EVERY CI gate in CI's order — run this before opening a PR
```

`make verify` is the single definition of the gate; it mirrors CI exactly
(coverage threshold and scope, mypy flags, the KB immutability check). The
hand-assembled sequence drifted from CI on four axes once, so prefer the target.
The individual pieces remain available:

```bash
make lint                # ruff check/format, whole repo (0.15.8, 100-char lines)
make typecheck           # mypy, exactly as CI runs it (typecheck-strict is local-only)
make test                # the suites CI runs, with CI's marker filters
make test-all            # coverage, mirroring CI's coverage job (gate: 85%)
make test-cloud-coverage # the deployed path, against coverage-floors.json
make validate            # governance validators (openspec, kb, skills, agents, doc-sync)
make security-secrets    # gitleaks: committable tree + this branch's history
make eval-llm            # StrategyAdvisor output-contract gate (offline)
```

Run one test file directly with `pytest tests/unit/test_mcts.py -v`; the
governance tests need their own config:
`pytest -c tests/governance/pytest.ini --confcutdir=tests/governance tests/governance`.

## Quality gates (what CI checks)

- **Tests:** unit, integration, regression, e2e/smoke and cloud suites; every
  directory that holds collectable tests must be named by a CI job
  (`tests/regression/test_suite_wiring_regression.py` enforces this).
- **Coverage:** CI enforces **85%** on `alpha_zero/` (`--cov-fail-under=85`,
  matching `pyproject.toml` `fail_under`). Keep new code covered.
  `alpha_zero/cloud/` is omitted from that gate and measured separately against
  a per-package floor in `coverage-floors.json` (via `.coveragerc-cloud`). Floors
  are a ratchet: raise one when the measurement supports it, never lower one,
  and move the recorded `measured` value in the same PR.
- **Lint:** ruff check + format must pass on the **whole repository** (tests/,
  scripts/, infra/ included; fringe directories are excluded centrally in
  `pyproject.toml`). Line length is 100.
- **Types:** mypy is a **blocking** CI gate — `mypy alpha_zero/` must exit 0 in
  the dev-extra environment; several packages
  (`alpha_zero.core/cloud/training/tournament/pbt/web/evaluation/envs`) are held
  to stricter settings. Keep them clean.
- **Debt ratchet:** `tests/regression/test_debt_ratchet.py` holds the metrics in
  `docs/evidence/tech_debt_baseline/baseline.json`. A metric may fall — update
  the baseline in the same PR — but may never rise. Re-measure with
  `python scripts/measure_debt.py`; find newly dead modules with
  `python scripts/audit_import_graph.py`.
- **Security:** bandit on `alpha_zero/`; gitleaks on the tree and this branch's
  history; workflow actions are SHA-pinned (gated by a regression test).
- **Docs:** `scripts/check_doc_sync.py` fails the build on version drift, stale
  module counts, forbidden claims in `README.md` / `ROADMAP.md`, unknown extras,
  broken links in the top-level documents, a `docs/` page missing from
  [`docs/README.md`](docs/README.md), or an unreferenced roadmap "Now" bullet.
  Run `make validate` before pushing docs.

### Adding a test that guards a bug

Before you trust it, prove it fails: reintroduce the defect (or plant it in a
fixture) and watch the test go red, then fix and watch it go green. Several of
this repository's gates passed for months while the thing they guarded was
broken; the `negative-control` skill describes the ritual, and a PR that adds a
regression test should say what its negative control was.

## Branching and pull requests

- Work on a feature branch and open a **draft** PR against the default branch.
  Direct pushes to the default branch are not accepted (branch protection; the
  maintainer's agent harness also refuses them locally).
- Fill in the PR template and make sure its verification checklist is true, not
  just ticked.
- Do not force-push a branch someone else has reviewed; if you must rewrite,
  use `--force-with-lease`.
- A change that alters deployed behaviour, deletes more than a hundred lines,
  changes a public interface, or touches distributed-training coordination needs
  a maintainer decision **before** implementation — open an issue first
  (see [`GOVERNANCE.md`](GOVERNANCE.md)).
- Commit messages follow Conventional Commits where practical and keep the
  subject under 72 characters.

## Claims and evidence

This project separates *delivered* from *validated* (see the Validation column
in [`ROADMAP.md`](ROADMAP.md)). A pull request that claims a measurement —
a speed-up, a win rate, a learning trend — commits the artifact under
`docs/evidence/` following `docs/evidence/SCHEMA.md`, states whether it was
produced on code before or after the 2026-09-04 search fixes, and does not put
the number in `README.md` or `ROADMAP.md` until it is committed
(`check_doc_sync` D4b forbids unmeasured claims there).

## Releases

Versioning is semantic (`MAJOR.MINOR.PATCH`). To cut a release, use the `release`
skill (or follow it by hand): bump the version in `pyproject.toml`,
`alpha_zero/__init__.py` and `CITATION.cff`, roll `CHANGELOG.md`'s `[Unreleased]`
section into `[X.Y.Z]`, and push a `vX.Y.Z` tag. Pushing the tag triggers
`.github/workflows/release.yml`, which builds the GitHub release from the tag.
`check_doc_sync.py` enforces that the three versions and the changelog head agree.

---

## For maintainers and agents: the spec-driven workflow

Nothing below is required to send a bug fix or a test.

### OpenSpec change workflow

Substantive changes — anything that alters a capability's behaviour, adds a
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

### Knowledge base and handoffs

Session continuity for the coding agent uses the `kb/` handoff protocol —
[`kb/README.md`](kb/README.md) and `CLAUDE.md` §14. Handoff files are immutable
once created (gate K7) and the ledger is append-only; the identifiers in them are
opaque run ids. Humans do not need to write to `kb/`.

### Agent harness

The hooks, permission allowlist and their threat model are described in
[`.claude/README.md`](.claude/README.md). The guard that refuses pushes to the
default branch is a heuristic for the agent, not a substitute for branch
protection.

### Governance

- Scope and invariants: [`PROJECT_CHARTER.md`](PROJECT_CHARTER.md) (amendment
  protocol in §7–8); who decides what: [`GOVERNANCE.md`](GOVERNANCE.md).
- Decision rights for automated contributors (autonomous / confirm-first /
  prohibited): `CLAUDE.md` §3.2.
