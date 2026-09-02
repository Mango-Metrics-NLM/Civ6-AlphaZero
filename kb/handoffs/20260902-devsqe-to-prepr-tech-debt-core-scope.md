# DEV-SQE → Pre-PR Handoff

**Date:** 2026-09-02
**Spec:** `kb/specs/20260902-tech-debt-hardening.md` (v2)
**Branch:** `claude/tech-debt-hardening-plan-xrn9sp`
**PR:** #116 (draft)
**Session:** session_01EcSsbbTYiSkmZsMVyipm7G

## What shipped

The plan's **Core scope**, in five commits, each independently revertible.

### 1. WS-M — measurement first (`1b10237`)

`scripts/measure_debt.py` emits every metric the plan tracks as JSON;
`tests/regression/test_debt_ratchet.py` holds each to
`docs/evidence/tech_debt_baseline/baseline.json`. A metric may fall (and the
baseline falls with it in the same PR); it may never rise. Metrics needing an
absent tool report `null` rather than a guess.

### 2. WS4.6 — workflow and hook hardening (`9bf942b`)

Least-privilege `permissions:` on all three workflows, job timeouts,
`action-gh-release` pinned to a commit SHA, `--strict-markers`, pre-commit
scoped to `tests/regression/`, and the no-op `--ignore=tests/unit/test_distributed`
deleted. Codecov removed: no `CODECOV_TOKEN` exists, so `fail_ci_if_error: true`
would only turn tokenless rate limits into red builds.

**The guard never protected this repo.** `pretooluse_guard.sh` compared against
the literal `main`; the default branch is `claude/civ6-alphazero-implementation-vMnun`.
It now resolves at runtime. A second latent bug in the same rule:
`[ on-main ] || is_push && dest_is_main` parses as `(A || B) && C`, so a commit
while checked out on main was denied only if the command line also said `main`.
17 governance tests gate both.

### 3. WS3.1 + WS4.1–4.2 — the two real defects (`c492351`)

- **`action_space_size` defaulted to 381** in `NetworkConfig`, the live bridge,
  `MuZeroConfig`, the MuZero hub mixin and the N-player head, while the
  environment's width is 376 and `transformer_network.py` had warned it was
  wrong for a year. All six now derive from one source. An explicit 381 still
  wins, so 381-wide checkpoints load.
- **Any failed safe checkpoint load silently retried with `weights_only=False`.**
  Fixed in both loaders, including `utils/checkpoint.safe_load_checkpoint`,
  which six live call sites use and all six passed `allow_unsafe_fallback=True`
  to explicitly. The single unsafe call now lives in the leaf module
  `utils/safe_load.py` (a shared helper under `core/` would cycle).

Found while doing it: `MuZeroTrainer.save_checkpoint` pickled its config
dataclass, so its own checkpoints could only be reopened unsafely.

### 4. WS2 — Tier A deletion (`d5ed22f`)

44 modules / 21,903 lines with no live importer, plus the build files that
only served them (four Dockerfiles, `docker-compose.yml`, `setup.py`,
`sweep_config.yaml`, the Makefile targets). 13 test files deleted, 10 edited to
keep their live cases.

### 5. WS1 — the deployed path is measured (`f13e768`)

`cloud-tests` now measures `alpha_zero.cloud` against a floor of 58% (measured
59.7%). Plus a ratchet test closing three ways the omit list has been weakened
here before.

## Verification (all green, run locally)

| Gate | Result |
|---|---|
| Global coverage (CI job equivalent) | **89.84%**, gate 85%, 11,445 passed |
| `tests/unit` | 9,921 passed, 50 skipped |
| `tests/integration` | 580 passed, 9 skipped, 1 xfailed |
| `tests/cloud/unit` + `smoke` | 846 passed |
| `tests/regression` | 125 passed |
| `tests/governance` | 91 passed |
| `tests/scripts` + `tests/e2e` | 239 passed |
| `ruff check .` / `ruff format --check` | clean / 878 formatted |
| `mypy alpha_zero/` | no issues, 328 files |
| 4 governance validators | all exit 0 |
| Cloud floor check | 59.7% ≥ 58% |

**Sandbox note now in CLAUDE.md:** local suites need `OMP_NUM_THREADS=2`, or two
tests wedge in `torch.conv2d` on the OpenMP spin-wait and hit the timeout.

## Metric movement

| Metric | Before | After |
|---|---:|---:|
| Package lines | 168,224 | 146,321 |
| Dead modules (lines) | 79 (40,503) | 54 (26,129) |
| Dead modules inside coverage | 34 | 27 |
| Unsafe `torch.load` calls | 17 | 8 |
| Runtime asserts | 69 | 60 |
| Files > 1,000 lines | 27 | 24 |
| Coverage omit patterns | 45 | 39 |
| Dependency manifests | 6 | 5 |

## Every new gate has a recorded negative control

Debt ratchet (added a dead module → 5 metrics fail), action-space AST (restored
381 → 2 failures naming the line), unsafe-load AST (dropped one keyword → 1
failure), checkpoint hardening (default flipped back → 12 failures), tool pins
(bumped one → names the file), coverage floors (floor 95 → exit 1), omit-list
ratchet (all three controls fire).

## What Pre-PR must know

1. **Two behaviour changes ship here.** `GCSCheckpointStorage` is now safe-by-default;
   if an incumbent GCS checkpoint fails a safe load, the escape hatch is
   `CIV6AZ_ALLOW_UNSAFE_CHECKPOINT=1`. The spec's "inventory the champion lineage
   before flipping" could not be run — no GCS access from this session. **This is
   the one Core item not fully discharged and it needs the maintainer.**
   Checkpoints written by `MuZeroTrainer` before this change need that variable
   once and a re-save.
2. **A documented feature was deleted, honestly recorded.**
   `count_consecutive_non_promotions` lived only in the dead legacy evaluator.
   Promotion-stagnation detection on the deployed path is now unimplemented —
   a gap, not a relocation. CLAUDE.md §5.2 says so.
3. **Deprecations, not deletions:** the four hexcrystal re-export shims and
   `TournamentWandbLogger` warn on use (not on package import) for one release.
4. **Burndown allowlists exist and fail on stale entries:** 8 files hardcoding
   376 (including the deployed cloud path) and 8 remaining `weights_only=False`
   sites. Both are visible debt, not hidden.

## Not done (needs the maintainer, spec §9)

- **WS0 in full:** rotate the W&B key; close #72 (its head *is* the leaked
  commit), #69, #49, #32; merge the Dependabot backlog.
- **Tier B:** fourteen yes/no rows, 18,133 lines.
- **WS5** (seven decompositions), **WS3.2** (env prefix), **WS6**, **WS7**.
- Protected `release` environment and `v*` tag protection: repository settings.
