# PR: Correctness-first — one sign-correct backup for every search, and the gates that were fail-open

**PR:** [#118](https://github.com/Mango-Metrics-NLM/Civ6-AlphaZero/pull/118) ·
**Branch:** `claude/2026-standards-audit-7tgk7e` → `claude/civ6-alphazero-implementation-vMnun`
**Spec:** [`kb/specs/20260904-correctness-first-plan.md`](../specs/20260904-correctness-first-plan.md)

## Summary

- **The default search was wrong on every deployed path.** `GumbelMCTS` negated the
  value on every ply, but 71% of MicroCiv steps do not change the player (measured),
  so the sign of a node's Q depended on the parity of the depth at which its leaves
  were evaluated. `base.py` was corrected for this in August and the fix never
  reached the other variants: `grep -c to_play` gave `base` 10, `puct` 9, and
  **`gumbel` 0, `simple` 0, `muzero` 0** — while `gumbel` is `DEFAULT_MCTS_TYPE`,
  what `selfplay_worker` takes by default, and what `arena.py` (the promotion judge)
  hard-defaults to. The data generator and the judge shared the defect. There is now
  one `backpropagate_path` and one `perspective_sign` in a leaf module used by all
  five variants.
- **The gate written to protect that fix did not work.** The forced-win oracle passed
  vacuously for Gumbel, the one variant it existed for: the returned policy was
  exactly `[1/3, 1/3, 1/3]` and `np.argmax` breaks a tie by returning the lowest
  index, which was the winning action by coincidence. Moving the win to index 1 or 2
  failed at every depth with and without the bug. The oracle now sweeps the winning
  index and rejects a policy uniform across two or more actions.
- **Visit counts are the wrong policy target for Gumbel.** Sequential halving splits
  the budget evenly among survivors, so the distribution is near-uniform and the
  answer lives in the completed Q values. The arena played a different move than its
  own search chose in 8 of 25 root positions before this was corrected.
- Plus the supply-chain and harness fixes the standards audit found, each with a
  negative control that was watched to fail first.

## Changes

| Area | Files | What |
|---|---|---|
| Search correctness | `core/mcts/backup.py` (new), `base.py`, `puct.py`, `gumbel.py`, `simple.py`, `muzero.py`, `core/batch_mcts.py` | One sign-correct backup + `perspective_sign` + `should_track_to_play`; five copies of the flip loop collapsed to one. MuZero keeps per-ply alternation (a latent model cannot observe the mover) and is byte-identical. |
| Policy target | `core/mcts/gumbel_ops.py`, `gumbel.py`, `batch_mcts.py` | `build_completed_q_policy` implements Danihelka's `softmax(logits + sigma(completedQ))` — also the first use of the `c_scale` field, declared since 2026 and read nowhere. `policy_target` (`completed_q`/`visit_counts`/`one_hot`) and `greedy_threshold` on both configs; `one_hot` reproduces prior runs exactly. |
| Reproducibility | `utils/seeding.py` (new), `training_civ6.py` | `derive_seed(base, stream, *coords)` via BLAKE2b. Self-play varies by iteration; evaluation is a fixed held-out set. `EVAL_SEED_OFFSET` deprecated, not deleted. |
| Performance | `envs/microciv.py` | `clone()` shares immutable terrain: 1,123 → 357 µs at board 8, 3,207 → 391 µs at board 16. |
| Packaging | `pyproject.toml`, `alpha_zero/opponents/__init__.py`, 13 test modules | `chess` out of the `dev` extra (its sdist aborts the whole install); `opponents` made lazy (PEP 562) so an optional extra is not mandatory for a subpackage import. |
| Supply chain | `.github/workflows/*.yml`, `.github/dependabot.yml`, `Makefile`, `.claude/settings.json` | All 27 `uses:` refs SHA-pinned; `pull_request` no longer filters on a branch that is not the default; Dependabot gains docker + grouping + cooldown; Makefile exports `OMP_NUM_THREADS`; the push guard now resolves this repo's real default branch. |
| New gates | `tests/regression/test_workflow_pinning_regression.py`, `test_coverage_floors_ratchet.py`, `test_suite_wiring_regression.py` (two new classes), `tests/unit/test_mcts_backup_correctness.py`, `test_seeding.py`, `test_microciv_clone.py`, six suites under `tests/governance/` | See below. |
| Docs | `CHANGELOG.md`, `README.md`, `docs/architecture/C4_ARCHITECTURE.md`, `NEXT_STEPS.md`, `ROADMAP.md`, `docs/MODULE_REFERENCE.md`, `CLAUDE.md`, `docs/audits/`, `kb/` | Audit report, correctness-first plan, and the branch sync. |

## Test Results

| Suite | Result |
|---|---|
| unit + sanity + regression | **10,045 passed**, 58 skipped, 17 deselected |
| integration | 580 passed, 9 skipped, 1 xfailed |
| e2e / smoke / scripts | 70 / 30 / 178 passed |
| cloud unit + smoke | 846 + 36 passed |
| security / governance | 18 / 55+ passed |
| Coverage (≥85%) | **green in CI** on `b715582` (runs 449, 450). The local run was inconclusive in this sandbox; CI is its gate. |
| Lint / format | clean, whole repo |
| Type check | clean, 330 files |
| Smoke | `training_civ6 --dry_run` exit 0; `orchestrator --preset fast --dry_run` exit 0 |
| Secrets | gitleaks clean: tracked tree + 542 commits on HEAD's lineage |

Every new gate was verified by reintroducing the defect:

| Gate | Reverting the fix produces |
|---|---|
| Forced-win oracle | 6 failures each for `gumbel`, `puct`, `simple` (Gumbel was **0** before the oracle was hardened) |
| Seed derivation | "iteration 0 and iteration 1 generated identical self-play data" |
| Dead-code ratchet vs `build/` | dead count 53 → 30 on an identical tree |
| Optional-import gate | names the unguarded module |
| Zero-assertion gate | names the planted assertion-free test |
| Coverage-floor ratchet | "coverage floors may only ever be raised" |

## Breaking Changes

- **`GumbelMCTS.search` and `BatchGumbelMCTS.search` return a different policy.**
  Previously a one-hot; now the completed-Q improved policy. This is the intended
  fix — the one-hot trained the policy head by cloning a single argmax — and
  `policy_target="one_hot"` reproduces the prior behaviour exactly for anyone
  replaying an existing run.
- **Seeded runs draw different streams.** Reproducibility is preserved (same seed →
  same run); the sequences differ from before, so artifacts produced under the old
  derivation are not bit-comparable.
- **`chess` is no longer installed by `.[dev]`.** Use `.[chess]`. The chess tests
  `importorskip` and skip cleanly without it.

## Checklist

- [x] All verification commands pass (lint, format, types, the CI suites, validators)
- [x] Coverage ≥85% — enforced and green in CI on the current head
- [x] `CLAUDE.md` updated (§5.2 decisions and §5.4 bug fixes)
- [x] No hardcoded values — seed streams are named constants, the permitted-import
      set is derived from `pyproject.toml`, repo-local roots from the tree, and
      `OMP_NUM_THREADS` is an overridable Make variable
- [x] Type hints on all new public interfaces
- [x] Every new gate merged with a recorded negative control

## Open, and deliberately not in this PR

Maintainer-only per the plan's §9.3: rotating the W&B key still reachable in
history, removing the `allUsers` binding on the inference service, and tagging
`v1.8.0`/`v1.9.0` (the changelog half is done; a tag fires `release.yml` with
`contents: write`). Engineering follow-ups: WS-S item S5 (the A/B evidence
seed-overlap memo) and plumbing `policy_target`/`greedy_threshold` out to a CLI or
YAML — they are reachable in code only today.
