# Make the repository ready for public release

Tracking: #TBD (umbrella "Launch train" issue, created by the maintainer after merge)

## Why

The repository is about to be made public. A standards audit
(`docs/audits/2026-09-04-standards-audit.md`) and its correction
(`kb/specs/20260904-correctness-first-plan.md`) already cover hygiene and
search correctness. What they did not ask is what a first public reader sees,
and whether the project's claims match what it can show. A panel review on
2026-11-10 (`docs/audits/2026-11-10-public-release-review.md`) found:

- Every Hugging Face link in the README resolves to a 404; the only published
  artifact lives under a different namespace.
- Personal and cloud identifiers (a personal email, a GCP project number and
  service-account address, bucket names used as code defaults) are in the
  tracked tree, and the deployed inference server falls back to random weights
  when a download from the default bucket fails.
- Off-scope material is tracked (Atari replays and demo, a generic project
  template with its own build, a duplicate root `interfaces/` package).
- The package metadata has no licence expression, authors, URLs or
  classifiers; the project has never cut a tag, so `release.yml` has never run.
- The ROADMAP and charter say learning convergence was "validated", while every
  committed evidence artifact predates the 2026-09-04 search fixes and the
  only training run ends below random.
- At least seventeen sites compute a draw-blind win rate while the measured
  draw rate is 73.5%; the deployed promotion default cannot promote at that
  rate, and the runbook's recommended threshold is also unreachable at the
  deployed 20-game sample.

## What Changes

- **Public surface.** README rewritten for one audience (engineers building or
  auditing a self-play system) with a status table that distinguishes
  delivered from validated; one real Hub link; `CITATION.cff`, `GOVERNANCE.md`,
  `.editorconfig`, `.python-version`, `py.typed`; PEP 621 metadata completed.
- **Public-surface hygiene gate.** `tests/regression/test_public_readiness_regression.py`
  enforces: no personal emails, no cloud project numbers or service-account
  addresses, no bucket literals outside example files, a ratchet on project-id
  mentions, no tracked binaries over 100 KB outside an allowlist, README
  external links present in a manifest, `CITATION.cff` consistent with the
  package version, required pyproject metadata, no forbidden strength claims.
- **Documentation gates.** `check_doc_sync.py` gains D4b (forbidden claim
  patterns in README and ROADMAP), D6′ (link checking extended to the
  governance documents), D8 (every non-archived doc appears exactly once in
  the index) and D9 (every "Now" roadmap bullet carries an issue number).
- **Off-scope removal.** Atari replays and demo, the template scaffold's code,
  the root `interfaces/` package, an orphan root `Dockerfile`, a personal MCP
  config and a dead W&B dashboard are removed; root runners move to
  `scripts/`; the Rust/SpacetimeDB stubs move under `experimental/`.
- **Evaluation protocol.** `docs/EVALUATION.md` defines the metric vocabulary
  (`score_rate` with a ternary interval is the statistic for claims;
  `win_rate` is legacy and draw-blind), a benchmark protocol executed by the
  promotion judge's own arena function, an evidence-artifact contract with a
  pre/post-fix epoch label, and a promotion-feasibility report. Draw-aware
  metrics are added alongside every logged win rate; a regression gate
  prevents new draw-blind sites.
- **CI and supply chain.** Non-blocking security scans (bandit against a
  baseline, zizmor, dependency review, pip-audit against an expiring
  baseline, CodeQL, Scorecard); `persist-credentials: false` on every
  checkout; a checksum on the gitleaks download; `release.yml` builds and
  checksums artifacts and refuses a tag that does not match the version;
  the Spaces deploy workflow no longer fires on a branch push.
- **Release.** `[Unreleased]` rolls into `2.0.0` (four BREAKING sections), with
  a summarised entry and the full detail archived.

## Impact

- Specs: `documentation` (MODIFIED: README accuracy, Documentation index;
  ADDED: Public-surface hygiene), `validation-gates` (ADDED: Public-readiness
  regression gate; MODIFIED: CI governance job), new capability
  `evaluation-protocol` (ADDED only).
- Code: `alpha_zero/evaluation/` (benchmark protocol, feasibility report,
  Elo transform), `alpha_zero/cloud/gcp/arena.py` (search seed, paired seats),
  `alpha_zero/training_civ6.py` (`--mcts` flag, additive metrics), the
  draw-blind sites listed in `tests/regression/draw_aware_allowlist.toml`.
- Deleted: `replays/`, `demos/`, root `interfaces/`, `templates/{src,tests,pyproject.toml}`,
  `mcp_config.json`, `configs/wandb_dashboard.json`, root `Dockerfile`,
  `mod/AlphaZeroAI/Scripts/GeneratedPaths.lua` and their tests.
- Out of scope, tracked as issues: key rotation, IAM, default-branch rename,
  branch protection, tagging, the promotion-default flip, lock-file strategy,
  coordinator/orchestrator edits, Hub card update, distribution rename.
