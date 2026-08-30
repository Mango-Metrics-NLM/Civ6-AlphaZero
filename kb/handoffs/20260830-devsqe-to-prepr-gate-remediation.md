# DEV-SQE → Pre-PR Handoff (supersedes the earlier handoff of the same date)

**Date:** 2026-08-30
**Spec:** `openspec/changes/harden-governance-gates/proposal.md`
**Branch:** `claude/god-files-decomposition-gojlo3`
**PR:** #115 (draft)
**Session:** session_0161DvFczCzPQqumjKqYdrXy
**Supersedes:** `kb/handoffs/20260830-devsqe-to-prepr-mcts-split-and-governance-gates.md`

## Why this supersedes the earlier handoff

Handoff documents are immutable (validator check K7), so the earlier one stands as
written. It should not be read on its own. **It claimed a verification standard the
work did not meet**, and this document states what was actually true.

Specifically, it reported "10 passed" for `tests/regression/test_repo_hygiene_regression.py`
as evidence the hygiene invariants were guarded. They were not guarded by anything:
**no CI job and no make target ran that directory.** The tests passed when I invoked
them by hand and would never have run again. Three sibling defects had the same shape —
a validator exiting 0 on a missing `--path`, an allowlist test matching four literal
spellings, and a requirement asserting coverage the validator did not implement.

The lesson is recorded as a decision row in CLAUDE.md §5.2: *prove a new gate fails
when the thing it guards is broken, and prove something actually invokes it.*

## What shipped in this round (`f4cea7a`)

All 13 review findings, verified individually against the repo before fixing.

**Gates that did not gate (4):** `tests/regression/` wired into the CI `test` and
`coverage` jobs and into `make test` / `make test-all`; `A0` added so a missing or
misnamed `--path` fails; the allowlist test rewritten to compile the real patterns
against representative source; the A5 requirement narrowed to top-level packages with
its reasoning stated, plus a new `alpha_zero/core/mcts/Agent.md`.

**Gates weaker than claimed (4):** the layering walk widened to the full AST so a
deferred import cannot bypass it; `GITLEAKS_BIN` defaults to the managed pinned binary;
the installer resolves `uname` to the correct release artifact for linux/darwin ×
x64/arm64 plus windows, failing loudly on anything else; the `docs/archive/` path
exemption removed.

**Missing tests (2):** block-scalar frontmatter (folded, literal, and the empty case
that used to pass vacuously), and the `.agents/skills/` tree plus S5 — with `_point_at`
now patching `AGENT_SKILLS_DIR`, which it did not, so tmp-tree tests had been scanning
the repository's real second tree.

**Docs (3):** governance test files 4 → 5, C4 `gumbel_ops.py` 100 → 98, and the
`.gitleaks.toml` baseline comment rewritten to stop claiming the repository's history is
clean while the repo-wide audit has an open finding.

## Test results

| Gate | Result |
|---|---|
| Coverage (`make test-all` equivalent, now including `tests/regression/`) | **89.29%** against the 85% gate — 12,365 passed, 66 skipped, 1 xfailed, **0 failed**, 11m29s |
| `tests/regression/` standalone | 38 passed |
| `tests/governance` | **73** passed (was 64) |
| `ruff check .` / `ruff format --check .` | clean, 937 files |
| `mypy alpha_zero/` | clean, 367 files |
| Validators (openspec / kb / skills / agents / doc-sync) | all exit 0 |
| `make security-secrets` | 0 findings — tree and the 528-commit `HEAD` lineage |

Adding `tests/regression/` to the coverage run moved the number by 0.01 points
(89.30 → 89.29), as expected for a suite that exercises repository configuration
rather than `alpha_zero/`.

## Negative controls

Every fix was reproduced as a failure first.

| Injected defect | Test that fired |
|---|---|
| `'''^.*$'''` in the gitleaks allowlist | `test_allowlist_does_not_match_ordinary_source_paths` |
| `'''^alpha_zero/'''` (subtler over-broad entry) | same — strictly stronger than the old check |
| Function-local `from .base import MCTSNode` in `gumbel.py` | `test_independent_variants_do_not_depend_on_base[gumbel]` |
| Function-local **absolute** `from alpha_zero.core.mcts.base import …` | same |
| `--path` naming a missing file | `A0` |
| `--path` naming a non-`Agent.md` file | `A0` |
| `description: >` with an empty body | S1 non-empty check |
| Skill directory with no `SKILL.md`, in either tree | S5 |
| Skill under `.agents/skills/` with name ≠ directory | S1 |

The block-scalar case was additionally demonstrated by reconstructing the old parser:
it returned the literal `">"` for both a populated and an **empty** folded scalar, so
the non-empty check passed in both — the defect the fix removes.

## Lint / type-check status

Clean. `PLR0915` remains enabled with three `main` functions on the `per-file-ignores`
burndown (248 / 168 / 134 statements).

## Known issues / carried forward

- **UNREMEDIATED: a real W&B API key is in git history** — commit `87cd68a2`, on the
  unmerged `feature/sprint6-validation-and-nplayer-foundation`. Requires rotation by the
  maintainer; no code change remediates it. `docs/runbooks/secret-scanning.md`.
- **Three pre-existing defects filed, not fixed**, because this PR's non-goal is pure
  relocation and each is byte-identical to the pre-split original: `gumbel_ops`
  temperature-zero degeneracy (yields `-inf`, silently degrading Gumbel-Top-k to uniform
  sampling — not the `ZeroDivisionError` the reviewer claimed), `SimpleMCTS` using the
  global RNG while holding `self._rng`, and a `gumbel.py` docstring describing a
  Gumbel-weighted heuristic the code does not implement. Their threads are intentionally
  left open on the PR.
- **The cloud trio remains undecomposed**, blocked on a patch-target remapping strategy:
  38 `mock.patch` string targets, and an identity test cannot detect a dead patch.
- **22 A3 warnings** — five `Agent.md` files missing conventional sections. Deliberately
  non-blocking; a visible burndown.
- `MuZeroNetworkProtocol` is defined twice and `core/__init__.py` re-exports both.

## Notes for Pre-PR

- Nine of the eleven round-2 threads are resolved; the two already-resolved doc ones make
  eleven. The three round-1 threads are left open by intent — they are the author's call.
- `openspec/changes/harden-governance-gates/` must be linked in the PR's "Linked OpenSpec
  change" field. Tasks 4.3–4.5 (splitting the oversized `main` functions) are deliberately
  unchecked — that is the burndown, not this PR's scope.
- CI must be re-confirmed on `f4cea7a`: the regression step in the `test` job and the
  regression directory in the `coverage` job are both new and have never run in CI.
