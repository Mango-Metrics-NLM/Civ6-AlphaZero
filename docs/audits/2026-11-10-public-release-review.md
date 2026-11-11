# Public-release peer review — Civ6-AlphaZero

**Date:** 2026-11-10 · **Commit reviewed:** `96c9e57` (last change 2026-09-12) · **Branch:** `claude/cool-lovelace-o02vn9`
**Method:** a panel of five role reviews (SDLC/CI-CD/DevSecOps, ML/AI science, Product + Scrum, architect, integration lead), each run as an independent agent against the tree, the GitHub API and the Hugging Face API, and reconciled by argument rather than by vote. Every decision in the plan is recorded as *thesis → counter-argument → rebuttal → decision + gate*. This report is the record of the review; the resulting change is `openspec/changes/public-release-readiness/`.

Confidence tags: **[Certain]** reproduced by command in this session · **[Likely]** read in a file but not executed · **[Guessing]** inference.

This review does not repeat the 2026-09-04 standards audit (`2026-09-04-standards-audit.md`) or its correction (`kb/specs/20260904-correctness-first-plan.md`). §2 records what is still open from them. §3–§8 are the questions they were not asked: what a first public reader sees, whether the project's claims match what it can show, and what it costs to close the gap.

---

## 1. Verdict

**Not ready to flip to public at `96c9e57`; ready after the maintainer checklist in §9 and the changes in this branch.** The process machinery (negative-controlled gates, seeded reproducibility, ratchets) is above the 2026 bar. The public surface is below it: every advertised Hugging Face artifact is a dead link, personal and cloud identifiers are in the tree, off-scope binaries are tracked, the package has no licence metadata, no release has ever been cut, and the roadmap claims a validation that no committed artifact supports.

| Area | Before | After this branch | Remaining owner |
|---|---|---|---|
| Public surface (README, links, identifiers, clutter) | 1 | 4 | — |
| Packaging & release | 1 | 3 (tag + provenance are maintainer steps) | maintainer |
| CI & supply chain | 3 | 4 (scanners non-blocking by decision) | maintainer to promote to blocking |
| Documentation IA | 2 | 4 | — |
| Evaluation credibility | 2 | 3 (infrastructure; no post-fix learning evidence yet) | milestone v2.1 |
| Governance | 4 | 4 | — |

## 2. Status of the 2026-09-04 audit's items

| Audit item | Status at `96c9e57` | Handled where |
|---|---|---|
| 1 Rotate W&B key | **Open.** The leaked commit is the head of closed PR #72, so `refs/pull/72/head` keeps it reachable regardless of branch deletion. Rotation is the only remedy. | §9 maintainer checklist |
| 2 Close public inference endpoint | **Open** (4 `allUsers` bindings, `ingress: all`) | §9 |
| 3 CI PR trigger | Done (PR #118) | — |
| 4 `SKIP_PARTS` build dirs | Done | — |
| 5 Dev install / `chess` | Done | — |
| 6 pip-audit job | Open → added non-blocking | P5 |
| 7 SHA-pin actions / Dependabot | Done; 11 Dependabot PRs open, #125 fails CI | §9 PR dispositions |
| 8 Web CORS/host defaults | **Open** (CONFIRM-FIRST) | issue |
| 9 Push-guard hardening | **Open** (maintainer) | issue |
| 10 Deprecated import chain | Partly (not re-measured here) | issue |
| 11 Lock strategy | **Open**; `uv.lock` records 1.6.0 and is consumed by nothing | issue (CONFIRM-FIRST, deployed path) |
| 12 Safe-load migration | **Open** (45 sites) | issue |
| 13 Cut tags / merge `[Unreleased]` | Open → rolled to 2.0.0 here; tag is the maintainer's | P6, §9 |
| 14 Bandit widening / zizmor / Scorecard | Open → added (reporting) | P5 |
| 15 Extra ruff rules | Not addressed | issue |
| 16 Test determinism tooling | Not addressed | issue |
| 17 Split CLAUDE.md | Deferred (CONFIRM-FIRST); the citation-resolution gate that makes a later split safe lands here | P3 |
| 18 Package metadata / py.typed / matrix | Metadata and `py.typed` done here; 3.13 added as experimental | P3, P5 |
| 19–23 Dead modules, config surface, checkpoint API, test taxonomy, serving auth | Open per the tech-debt spec | issues |

## 3. Public surface (Product + DevSecOps)

**Findings [Certain unless noted].**
- All six Hugging Face badges and table rows in the README (`Mango-Metrics-NLM/civ6-hexcrystal-encoder`, `civ6-alphazero-transformer`, `civ6-muzero`, two Spaces, one dataset) return 404 from the Hub API. The only published artifact is `ianshank/civ6-hexcrystal-encoder` (74.6K parameters, MIT). The `Mango-Metrics-NLM` Hub organisation does not exist for the publishing account. The README says "Hub integration complete".
- Duplicate badge rows (README lines 4–5 and 8–10). A "Top-level runners" table advertises root scripts.
- A personal email address appears in four tracked files including `CODE_OF_CONDUCT.md`; a GCP project number and a compute service-account address appear in five files including one script; bucket names are code defaults (`alpha_zero/serving/inference_server.py:46`), and that server falls back to random weights when the download fails; the Terraform state backend is the training bucket. A Windows temporary path is committed in `mod/AlphaZeroAI/Scripts/GeneratedPaths.lua`, which nothing reads or generates.
- Tracked off-scope material: nine Atari Pong replays (2.2 MB of `.npz`), the Atari MuZero demo (5,600 lines plus 2,643 lines of tests), a generic project template with its own `pyproject.toml` (ungated), a root `interfaces/` package duplicating `alpha_zero/interfaces/` (4,383 lines, one test importer), Rust/SpacetimeDB stubs, a personal MCP configuration, a W&B dashboard JSON.
- Missing: `CITATION.cff`, `.editorconfig`, `.python-version`, `alpha_zero/py.typed`, a governance pointer, a trademark/non-affiliation statement.
- Five names for one thing: repository `Civ6-AlphaZero`, distribution `alpha_zero`, Hub library `alpha-zero-civ6`, card library `alpha-zero-hexcrystal`, mod "AlphaZero Civ6 Project".

**Dialectic that decided the README.** *Thesis:* lead with research honesty. *Counter:* that sinks the portfolio reader; lead with what works. *Rebuttal:* test each persona against what is true today — the modder cannot be served (four turns on random weights, Windows + FireTuner), the results-seeking researcher cannot be served (no post-fix evidence). What the repository uniquely has is a public, instrumented correctness post-mortem of a self-play system, so the one persona it can satisfy is the engineer building or auditing such a system, and candour serves that reader directly. *Decision:* README v2 for that persona, with a status table that separates *delivered* from *validated*.

**Dialectic that decided the scrub.** *Thesis:* replace every identifier. *Counter:* theatre while 605 commits keep them, and placeholders break the maintainer's scripts. *Rebuttal:* scrub by sensitivity. Project ids are not secrets; the sensitive items are the project number plus service and region (together they derive the unauthenticated endpoint URL — fixed by IAM, not grep), bucket names as code defaults (name takeover), and the personal email (the Code of Conduct needs a working private channel). *Decision:* remove the email, number and service-account addresses everywhere; make bucket defaults fail closed; keep docs-only project-id mentions under a ratchet; do not rewrite history, and say so.

## 4. Documentation (Product)

- `docs/README.md` lists `CLAUDE.md` under "start here" and omits `HF_PUBLISHING.md`, two runbooks, `evidence/` and `diagrams/`. Two six-line redirect stubs sit at the top of `docs/`. `NEXT_STEPS.md` is three-quarters live-bridge status written in internal workstream jargon. `ROADMAP.md` says training convergence was validated and refutes it on the same page; `PROJECT_CHARTER.md` repeats the claim and two speed-up figures, and its invariant 2 names Python ≥3.10 while `pyproject.toml` requires ≥3.11. `CHANGELOG.md` holds an 880-line `[Unreleased]` with four BREAKING sections (so the next version is 2.0.0 under the SemVer promise on its line 6), a duplicate `[1.3.0]`, and the file:line of the unrotated key.
- *Thesis:* prune and move. *Counter:* moves break archive-not-delete, charter links (the link gate only covers README and the index) and 76 `CLAUDE.md §N` citations. *Rebuttal:* the defect is the absence of labels and a single entry point, not file location; Diátaxis as labels in one index, moving only what removes a contradiction; no docs site for one maintainer. *Decision:* rebuilt index with audience and last-verified columns, a Records section whose rows say whether a measurement predates the 2026-09-04 search fixes, link checking extended to the governance documents, and an index-completeness gate.

## 5. Packaging, CI and supply chain (SDLC/DevSecOps)

**Findings.**
- `pyproject.toml` lacks `readme`, `license`, `authors`, `urls`, `classifiers`, `keywords`; `setuptools>=61` predates PEP 639; the `eval` extra is a git URL inside `all`, which blocks any index upload. Three `requirements*.txt` drift from pyproject; `uv.lock` is stale and unused; the deployed image installs `requirements-container.txt` with `PYTHONPATH`, so `requires-python` is never enforced in the image [Likely: NGC 24.09 ships 3.10].
- The coverage-floor ratchet never runs in CI: the test job checks out at depth 1, no base ref resolves, and the test skips [Certain by reading `test_coverage_floors_ratchet.py:30-96` and `ci.yml:77`].
- No vulnerability or SAST scanning; bandit runs at high/high only; every `actions/checkout` leaves `persist-credentials` at its default; the gitleaks binary is downloaded without a checksum; `release.yml` has never run and would dump the whole history into the first release's notes; `deploy_spaces.yml` would auto-publish scaffold Spaces the moment the default branch is renamed to `main`, its validate job installs an extra without pytest, and its `HF_TOKEN` step runs an unpinned install; `infra/cloudbuild-gameplay.yaml:93-111` has a canary that sleeps and echoes success; deploy manifests use `:latest`.
- `scripts/validate_kb.py` parses `--git` and never reads it; K7 is skipped silently on shallow clones.
- `.claude/settings.json` sets `enableAllProjectMcpServers: true` and allows `python -c`, which makes the `gcloud` deny decorative.

**Dialectics.** *Dependency truth:* thesis "pyproject is canonical" vs counter "the image never reads it" → rebuttal: two consumers, two artefacts — ranges for library users, an exact hashed export for the image generated from a refreshed lock with torch excluded; deleting `uv.lock` without a replacement is rejected; the export is CONFIRM-FIRST because it changes the deployed image. *Security scanning:* thesis "non-blocking" vs counter "warn-only gates decay" → rebuttal: tree-only tools (bandit with a baseline, zizmor) can block on *new* findings; feed-dependent tools (pip-audit) must never block a PR on an advisory published that morning — use dependency review for PRs and an expiring baseline for scheduled audits. *Delivery:* thesis "one PR" vs counter "unreviewable" → rebuttal: split by reviewability class (deletion-only, prose, CI, charter, release), merge commits, never squash.

## 6. Evaluation and scientific credibility (ML/AI science)

**Findings.**
- No committed evidence of learning. The only training run ends at a score of 0.25–0.40 against random; every evidence artifact predates the 2026-09-04 search fixes, including the encoder A/B whose headline Elo figure is live on the public Hub card. That A/B reused training seeds in 280 of 800 arena games, ran one training per arm, quotes two different confidence intervals, and its card says CPU while its JSON says CUDA.
- The local trainer cannot run the advertised Gumbel default (`training_civ6.py:95` has no CLI flag; local evaluation always uses PUCT at `:907`), so no local run can validate the default path.
- `run_arena` seeds the environment but not the search and plays each map from one seat.
- At least seventeen draw-blind win-rate computations exist, several at decision sites (`pbt/scheduler.py:280`, `tournament/league.py:73`), against a measured draw rate of 73.5%.
- The default promotion policy cannot promote at that draw rate; the runbook's recommended lower-bound policy at 0.52 needs an observed score near 0.65 at the deployed 20 games (about +107 Elo in one iteration), and a true +14 Elo needs roughly 1,160–1,470 games at 80% power. The runbook calibrates its threshold against the retracted A/B.
- The completed-Q policy target diverges from Danihelka et al. 2022 in four ways (`c_scale` 1.0 vs 0.1, no min-max normalisation, a non-convex completion, zero mass on unvisited actions), so the "fixed" target may still be close to one-hot in practice [Likely; measured in P7].
- The forced-win oracle covers depths 1–4 and not the batch or MuZero variants; there is no closed-loop test that the network prior learns from self-play.
- The LLM "eval gate" grades output structure even in online mode (mock judge) and passes silently when its extra is missing.
- The WS-D2 mask/executor mismatch is not evaluation-neutral: a rejected move forfeits the unit's turn, and the rate depends on the policy.

**Dialectics.** *Headline claim:* thesis "say learning is unvalidated" vs counter "that invites dismissal" → rebuttal: a cheap, seeded, CPU, sub-1000-game pre-registered run is feasible, but it can only validate the PUCT path until the `--mcts` flag exists; wording "evaluation infrastructure validated; no post-fix learning evidence yet". *Benchmark:* thesis "a new script" vs counter "nobody runs it; vacuous smoke" → rebuttal: a protocol and schema executed by the promotion judge's own arena function, with a non-vacuous check (a 64-simulation random-rollout search must beat random with its interval above 0.5). *Metrics:* thesis "route through score_rate" vs counter "renames break dashboards" → rebuttal: additive, enforced as a pairing. *Promotion gate:* thesis "don't flip" vs counter "a default that cannot promote is a credibility hole" → rebuttal: both the default and the recommended configuration are unreachable at the deployed sample size; make infeasibility visible at startup and size the sample honestly; the flip itself remains a recorded maintainer decision. *Elo:* bootstrap over sequential Elo measures order sensitivity, not strength; claims use pairwise score rate; Bradley-Terry is a follow-up.

## 7. Governance and the agent harness

- `kb/` handoffs are immutable under gate K7 and ledger entries require a session id; the session identifiers are opaque and already present in 201 commit messages. Scrubbing them would violate the repository's own rules for no reduction in exposure. *Decision:* frame, do not scrub — a `kb/README.md` preamble and a `.claude/README.md` threat model ("a heuristic guard, not a sandbox").
- Three backlogs exist (kb ledger, openspec tasks, and none on GitHub). *Decision:* one system per horizon — ROADMAP for themes and Now/Next/Later, GitHub Issues as the only actionable list, openspec for the scope of an in-flight change, kb for agent memory — and security items never become public issues.
- CLAUDE.md (1,455 lines) still says Python 3.10+ and is dated 2026-07-25; its split is deferred as CONFIRM-FIRST, but a test that every `§N` citation resolves lands now so the split cannot silently orphan a citation later.

## 8. What this branch changes

Commits on the branch, by reviewability class (merge commits, never squash; the pull-request
description carries the same map):

| Commit | Class | Contents |
|---|---|---|
| `d1306b7` | deletion-only | Atari replays and demo with their tests, root `interfaces/` and its test, template code, personal MCP config, W&B dashboard JSON, orphan root `Dockerfile`, generated Lua path file; coupling config; 91 files, 20,553 lines removed |
| `8ea973e` | docs | This review, the OpenSpec change, the evidence contract and index with every artifact backfilled and banner-labelled pre-fix, `docs/EVALUATION.md` with its recomputed power table, `CITATION.cff`, `GOVERNANCE.md`, `.editorconfig`, `.python-version`, `py.typed`, model-card corrections |
| `5e63152` | docs + gates | README v2, `docs/README.md` index, `docs/LIVE_CIV6.md`, ROADMAP validation column, community files, `kb/README.md` and `.claude/README.md`, minimal CLAUDE.md edits; doc-sync D4b/D6′/D8/D9/CFF and the CLAUDE.md citation gate |
| `c66f12b` | charter (ratify) | Links, M1 wording, speed-up claims, Python floor |
| `335d5e7` | release | `[2.0.0]` summary with archived detail, version 2.0.0 everywhere, PEP 621 metadata |
| `f19ef71`, `057546b` | CI | CPU-only torch on runners, non-blocking scanners (GHAS-dependent ones skipped while private), release build and checksums, Spaces deploy without a branch trigger, `validate_kb --git` enforced, pre-commit and Makefile fixes, pinning and Makefile gates |
| `6069f13`, `8394eee` | tests | Windows portability of two regression gates (UTF-8 reads, POSIX exemption keys), found by the new contributor smoke |
| `b408083`, `dceb793`, `af29b52`, `7843d4c`, `51f948b` | code + config | Runner moves, identifier scrub by sensitivity, fail-closed bucket defaults, `*.example.yaml` configs, the public-readiness gate and its manifest; follow-ups the gate and CI found |
| `1cb1169` | evaluation | Seeded, paired-seat arena; `BenchmarkProtocol` over `run_arena`; additive `score_rate`/`draw_rate` everywhere with an AST gate; promotion feasibility report and rewritten runbook; `--mcts` on the local trainer; learning oracle; Gumbel target entropy measurement |

**Negative controls.** Every new gate was shown failing on a planted defect before it was kept.
The controls are recorded in the commit messages above and in the worker reports they
summarise; the planted-defect tests that run on every CI pass are the `TestNegativeControls` /
`TestTheGateItselfDiscriminates` classes in each regression module.

## 9. Maintainer checklist before the flip (in order)

1. Rotate the W&B key; then remove the leak's file:line from `CHANGELOG.md` and `docs/runbooks/secret-scanning.md`.
2. Remove the four `allUsers` IAM bindings and set Cloud Run ingress to internal, or confirm the services are not deployed.
3. Merge P5 (which removes the Spaces push trigger) **before** renaming the default branch to `main`; then drop the legacy branch name from `ci.yml`.
4. Branch protection: required status checks, **zero** required approvals (a sole author cannot approve their own PR), no force-push; a `v*` tag ruleset; enable private vulnerability reporting.
5. Confirm `ianshank/Agents` (the `eval` extra) is public, or remove `eval` from `make verify`.
6. Ratify the charter commit (P4).
7. Decide Discussions (SUPPORT.md promises them) and a Code of Conduct contact alias.
8. Tag `v2.0.0` on the merge commit of P6 and check the release artifacts.
   The first Cloud Build after the merge pushes to `civ6az/alphazero-training` (and
   `civ6az/alphazero-inference`), a new Artifact Registry path; set
   `CIV6AZ_CONTAINER_IMAGE_URI` to keep launching an image built under the old path.
9. Triage the open pull requests: merge #124, #129, #131 (workflow SHA bumps); close #125 (fails CI; the pin moves at five sites together), #120 (orphan Dockerfile removed), #95 (codecov no longer used), #99–#102 (superseded by the Dependabot rewrite), #32 and #49 (charter NG-6), #69 (per the recorded disposition), #106 (close and open a salvage issue — the atomic-IPC work never landed); hold #130 as `status:needs-decision`.
10. Archive stale remote branches, disable the wiki, set the repository description and topics.

## 10. Issues to open after the flip (public backlog; security items excluded)

| Title | Labels |
|---|---|
| Flip the promotion default to `score_rate_lower_bound` once the C1 memo is recorded | area:search-eval, status:needs-decision |
| First post-fix learning run (pre-registered protocol in `docs/EVALUATION.md`) and first benchmark baseline | area:search-eval |
| Re-run the encoder A/B on post-fix code with disjoint seeds; update the Hub card | area:search-eval, status:needs-decision |
| Bradley-Terry ratings with bootstrap intervals for multi-opponent tables | area:search-eval |
| Thread `--seed` through `vertex_orchestrator` and `selfplay_worker`; store `score_rate` in orchestrator state | area:cloud, status:needs-decision |
| WS-D2: the move mask admits moves the executor rejects (property test) | bug, area:env |
| Re-measure the draw rate on post-fix search; decide `max_turns` | area:env |
| Lock strategy: refreshed `uv.lock` + hashed container export excluding torch | area:ci-supply-chain, status:needs-decision |
| Promote bandit/zizmor/dependency-review to blocking after two clean weeks; enable provenance attestation | area:ci-supply-chain, status:needs-decision |
| Replace the sleep-and-echo canary and `:latest` deploys; separate the Terraform state bucket | area:cloud, status:needs-decision |
| Narrow `.claude/settings.json` (`enableAllProjectMcpServers`, `python -c`, `npx`) and harden the push guard | area:governance |
| Split CLAUDE.md into `.claude/rules/` and ADRs (citation gate already in place) | area:governance, status:needs-decision |
| Rename the distribution to `civ6-alphazero` before any index upload | area:governance, status:needs-decision |
| Checkpoint safe-load migration (45 `weights_only=False` sites) | area:governance, status:needs-decision |
| Salvage the atomic-IPC work from PR #106 | area:bridge |
| Align the completed-Q target with Danihelka et al. or record the deviation as a design decision | area:search-eval, status:needs-decision |

## 11. Reproduction log

Every row names the command and the observed result on the branch head `1cb1169`
(Python 3.11.x venv, `OMP_NUM_THREADS=2`, unshallowed clone).

| Check | Result |
|---|---|
| Hugging Face API lookups for the six README links at `96c9e57` | all six not found; `ianshank/civ6-hexcrystal-encoder` found (74.6K params, MIT) |
| GitHub API at `96c9e57`: open pull requests / issues / tags | 15 (11 Dependabot) / 0 / 0 |
| `python scripts/validate_openspec.py --change public-release-readiness` | OK |
| `python scripts/check_doc_sync.py` (D1–D9 incl. the new rules) | OK |
| `python scripts/validate_kb.py --git` (full history, K7 enforced) | OK |
| `ruff check .` / `ruff format --check .` | clean / 896 files formatted |
| `python -m mypy alpha_zero/` | no issues in 331 source files |
| `pytest tests/regression tests/cloud/unit tests/governance tests/scripts/test_evaluation_power_table.py` | 1,241 passed, 2 skipped |
| `pytest tests/unit -m "not slow"` | 9,714 passed, 58 skipped (the 8 failures seen mid-integration were fixed before the final slice; the three affected files re-run 69 passed) |
| `python -m alpha_zero.training_civ6 --dry_run` | exit 0 in 7.5 s |
| `python -m alpha_zero.orchestrator --preset fast --dry_run` | exit 0 |
| `make verify` (every CI gate: lint, format, types, CI's suites, security incl. gitleaks over the tree and HEAD's lineage, governance, validators, coverage, cloud floor, advisor contract gate) | exit 0; global coverage 87% against the 85% gate; `alpha_zero/cloud` 60.0% of 5,525 statements against the 58% floor |
| CI on the branch | each failure on an intermediate head was this branch's own and was fixed in the next commit (a version mismatch, two Windows portability defects, a required `--project`, a comparator default, a baseline value); the runner-side install failures before any test ran were removed by the CPU-only torch install |
