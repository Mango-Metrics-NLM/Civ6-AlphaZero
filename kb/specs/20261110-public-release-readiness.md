# Civ6-AlphaZero public-release review and launch plan (team edition)

> Format: every decision is argued **Thesis → Counter-argument → Rebuttal (a third
> perspective, not a defence) → Decision + gate**. Panel: SDLC/DevSecOps lead, ML/AI science
> lead, Product lead + Scrum master, architect, and an integration lead (this session) who
> owns the plan, the hot files and the `kb/ledger` (single-writer rule). Authorization
> vocabulary from CLAUDE.md §3.2: **A** autonomous · **C** CONFIRM-FIRST · **M** maintainer-only.

## 0. Context

The repo (private today, 1,349 tracked files, 0 tags, 0 issues, 15 open PRs, last commit
2026-09-12) is about to go public. A standards audit (2026-09-04) and its correction
(`kb/specs/20260904-correctness-first-plan.md`) already exist and PR #118 shipped the
search-correctness fixes, SHA pins, Dependabot rewrite and CI-trigger fix. This plan does not
redo that audit. It answers: what will a first public reader see; what does the project
claim versus what it can show; which remaining gaps are cheap to close now; and how should
a small agent team deliver it without unreviewable PRs.

**User decisions (2026-11-10):** implement autonomous items as draft PRs · link only the HF
artifact that exists (`ianshank/civ6-hexcrystal-encoder`) · delete off-scope clutter, move the
rest · keep `kb/`, `openspec/`, `.claude/`, `.agents/` **without** scrubbing (frame instead) ·
scanners non-blocking first · roll `[Unreleased]` into **2.0.0** with a summarised entry
(detail archived) · keep `templates/*.md` under `docs/templates/`, delete the template code.

## 1. Verified findings (HEAD 96c9e57) — short form

- **Must happen before the flip, maintainer-only:** rotate the W&B key (commit `87cd68a2` is
  the head of *closed* PR #72, so `refs/pull/72/head` keeps it forever; deleting the branch
  fixes nothing); remove 4 `allUsers` IAM bindings and Cloud Run `ingress: all` (project
  number + service + region make the unauthenticated GPU URL derivable); rename default branch
  to `main` **only after** `deploy_spaces.yml`'s push-on-main trigger is removed (it would
  auto-publish scaffold Spaces); branch protection with required checks and **0 required
  approvals** (a sole author cannot approve their own PR); confirm `ianshank/Agents` (the
  `eval` git-URL extra) is public or pull `eval` out of `make verify`.
- **Public surface:** 6/6 HF links 404, only `ianshank/civ6-hexcrystal-encoder` exists (the
  `Mango-Metrics-NLM` HF org does not exist for this account); duplicate badges; false "Hub
  integration complete"; personal email in 4 tracked files; project number + SA email in 5
  files incl. `scripts/gcp_diagnose_and_submit.py:84`; bucket names as *code defaults*
  (`serving/inference_server.py:46` falls back to random weights on download failure);
  Terraform state lives in the training bucket; Windows temp path in
  `mod/AlphaZeroAI/Scripts/GeneratedPaths.lua` (nothing reads or generates it); `replays/atari`
  binaries; `demos/atari` (5,600 + 2,643 test lines), root `interfaces/` (4,383), `templates/`
  code, `server-module/`, `spacetimedb_module/`, `mcp_config.json`, `configs/wandb_dashboard.json`;
  no `CITATION.cff`/`.editorconfig`/`.python-version`/`py.typed`; five names for one thing
  (`Civ6-AlphaZero`, `alpha_zero`, `alpha-zero-civ6`, `alpha-zero-hexcrystal`, "AlphaZero Civ6 Project").
- **Docs:** two redirect stubs; superseded doc at top level; `docs/README.md` omits 5 docs and
  lists CLAUDE.md as "start here"; `NEXT_STEPS.md` is 75% bridge status in workstream jargon;
  ROADMAP says "convergence validated" and refutes it on the same page; PROJECT_CHARTER repeats
  "proven", "13×", "5–10×" and invariant 2 says ≥3.10 while pyproject says ≥3.11; CHANGELOG
  880-line `[Unreleased]` with 4 BREAKING sections, duplicate `[1.3.0]`, and it publishes the
  leaked key's commit + file:line (`:489`); `openspec/project.md` stale (ruff 0.8, mypy advisory).
- **Packaging/CI:** no `license/authors/urls/classifiers/readme`; `setuptools>=61`; `eval` git
  URL inside `all`; 3 drifting requirements files, `uv.lock` stale/unused (and
  `measure_debt.py` counts them: deleting one moves the debt baseline); the coverage-floor
  ratchet **never runs in CI** (test job checks out depth 1, so no base ref resolves and it
  skips); no vuln/SAST scanning; bandit high/high only; zizmor would flag all 11 checkouts
  (`persist-credentials` unset); gitleaks binary downloaded without checksum; `release.yml`
  never ran, no build/attest/version check, first tag would dump 605 subjects into notes;
  `deploy_spaces.yml` validate job broken, unpinned `pip install` on the `HF_TOKEN` step, no
  `environment:`; fake canary in `infra/cloudbuild-gameplay.yaml:93-111` (`sleep 120; echo passed`)
  and `:latest` deploys; `validate_kb.py --git` parsed but never read; `.claude/settings.json`
  `enableAllProjectMcpServers: true` and `Bash(python -c:*)` defeat the `gcloud` deny;
  2 OpenSpec changes unarchived; 11 Dependabot PRs (#125 fails CI).
- **Evaluation:** no post-fix evidence of learning; **the local trainer cannot run the Gumbel
  default** (`training_civ6.py:95` `use_gumbel_mcts=False`, no CLI flag; local eval always PUCT
  `:907`); `run_arena` seeds the env but not the search (`arena.py:174-191`) and plays each map
  from one seat; ≥17 draw-blind win-rate sites (not 8), several of them *decision* sites
  (`pbt/scheduler.py:280`, `tournament/league.py:73`); default promotion policy cannot promote
  at 73.5% draws, and the runbook's recommended `score_rate_lower_bound@0.52` needs an observed
  ≈0.65 at n=20 (≈+107 Elo) — +14 Elo at 80% power needs **n≈1,160–1,470**; completed-Q target
  diverges from Danihelka et al. (`c_scale` 1.0 vs 0.1, no min-max Q normalisation, non-convex
  completion, zero mass on unvisited) so it may still be near one-hot; the "+31 Elo" A/B is
  pre-fix, 280/800 seed overlap, one run per arm, two CIs quoted, CPU/CUDA contradiction, and is
  live on the public Hub card; forced-win oracle covers depths 1–4 and not Batch/MuZero; no
  closed-loop learning test; the LLM "eval gate" grades structure even `--online` and passes
  silently when the extra is missing; WS-D2 mask bug is not evaluation-neutral (forfeits turns).

## 2. Panel decisions (dialectical)

### 2.1 Audience and positioning (Product)
- **Thesis:** research-honesty-first README with a proven / built-not-validated / scaffold table.
- **Counter:** serves the RL researcher, sinks the portfolio reader; lead with what works.
- **Rebuttal:** test each persona against what is *true today*: the modder cannot be served
  (4 turns on random weights, Windows + FireTuner), the results-seeking researcher cannot be
  served (no post-fix evidence). What the repo uniquely has is a public, instrumented
  correctness post-mortem of a self-play system. The one persona it can satisfy is **the RL
  engineer building or auditing a self-play system**; candour serves them, and the portfolio
  reader as a side effect.
- **Decision (A):** README v2 for that persona (first 15 lines fixed in §4); harness material
  → CONTRIBUTING + a 10-line `GOVERNANCE.md` pointer to charter §7.5; bridge → `docs/LIVE_CIV6.md`
  labelled demo-grade; "not affiliated with Firaxis, 2K, Take-Two, Google DeepMind"; soften
  "reverse-engineered" wording. Names: repo `Mango-Metrics-NLM/Civ6-AlphaZero`, distribution
  `civ6-alphazero` (**C**, import name unchanged, no PyPI release exists so this is the cheapest
  moment), HF canonical `ianshank/…` until an org exists, `HF_ORG` required with no default.
  Gate: readiness test asserts README first 40 lines contain "Status" and "not affiliated" and
  none of `production|complete|13x|5–10x`; new `check_doc_sync` **D4b** forbids claim patterns in
  README and ROADMAP.

### 2.2 Documentation IA (Product + architect)
- **Thesis:** prune stubs, archive two docs, fold NEXT_STEPS into ROADMAP, rewrite the index.
- **Counter:** moving files breaks archive-not-delete, charter links (D6 only checks README and
  docs/README.md, so CI would stay green on the dead links) and 76 `CLAUDE.md §N` citations.
- **Rebuttal:** the defect is *no labels and no single entry point*, not file location. Use
  Diátaxis as labels in one index; move only what removes a contradiction. A docs site is not
  justified for one maintainer.
- **Decision (A):** `docs/README.md` rebuilt as: Start here (ordered 1–5: README,
  `docs/EVALUATION.md`, C4, ROADMAP, CONTRIBUTING) · Tutorials/How-to/Reference/Explanation
  tables with audience + last-verified · Records (audits, `docs/evidence/*`, ADRs, PRDs, `kb/specs`)
  with a "measured before/after 2026-09-04" column · Agent harness ("not required for human
  contributors"). Delete the two 6-line stubs; `NEXT_STEPS.md` → `docs/LIVE_CIV6.md`;
  `docs/NEXT_STEPS_CASE_2026-08.md` → archive; `templates/*.md` → `docs/templates/`. Charter
  link fixes go in the charter PR (§3 P4). Gates: **D6′** extends link checking to CHARTER,
  ROADMAP, CONTRIBUTING, SUPPORT, SECURITY, GOVERNANCE; **D8** every tracked `docs/**/*.md` outside
  `archive/` appears exactly once in the index; negative controls: plant an unindexed doc, then a
  dead charter link.

### 2.3 Roadmap and charter honesty (Product)
- **Thesis:** downgrade M1 to "in validation".
- **Counter:** a milestone ledger records what shipped; rewriting it is its own dishonesty.
- **Rebuttal:** shipping and validating are different axes and the table conflates them;
  "convergence validated" is a factual claim the same page refutes, so correcting it is not
  rewriting history.
- **Decision:** ROADMAP gains a **Validation (evidence)** column (M1 ❌ not demonstrated; M2 ⚠️
  operationally exercised, pre-fix; M3 ⚠️ command delivery proven, demo-grade; M4 "5–10×" is a
  target) and a Now/Next/Later section linked to issues; delete "269 model versions" as a
  success signal (A). Charter corrections (§5 M1 "proven", "~13×/5–10×" at `:54-55`, invariant 2
  "≥3.10" vs pyproject ≥3.11, link to `templates/`) are a **separate maintainer-ratified PR**
  per charter §7.5. Gates: D4b; **D9** every "Now" bullet carries `#\d+`.

### 2.4 Identifier scrub (DevSecOps)
- **Thesis:** replace project id, number, SA email, buckets, personal email in ~60 files.
- **Counter:** theatre while 605 commits keep it; breaks maintainer scripts; charter invariant 5
  wants explicit defaults; §3.3 caps a pass at 20 files.
- **Rebuttal:** scrub by *sensitivity*. Project ids are not secrets. Sensitive: (1) project
  number + service + region (derivable endpoint URL — fixed by IAM, not by grep), (2) bucket
  names as code defaults (bucket-name takeover → `inference_server.py` loads an attacker's model
  or silently falls back to random weights), (3) Terraform state in the training bucket
  (design finding), (4) the personal email (CoC needs a working private channel).
- **Decision:** (A) personal email → Security Advisories + handle; project number and SA emails
  → zero everywhere incl. `docs/archive` (redacting an identifier is not "maintaining" an
  archived doc; say so in `docs/archive/README.md`); code defaults fail closed (`_gcp_lib.py:45`,
  `inference_server.py:46` default `None`, use raises — `None` is an explicit default);
  `cloudbuild.yaml` → built-in `$PROJECT_ID`; `configs/vertex_*.yaml` → `*.example.yaml` + gitignored
  `configs/local/`; delete `GeneratedPaths.lua`; `$HOME` in `launch_prod_v1.6.0.sh`. Leave `kb/`
  and docs-only project-id mentions alone (ratchet their count instead). History is not
  rewritten; the report says so. (**C**) `env_loader.py` required-without-default behaviour.
  Gate: readiness test §5.

### 2.5 Harness as public artifact (SDLC + AI)
- **Thesis:** keep and scrub (original user choice).
- **Counter:** shipping an allowlist with `python -c`/`npx`/`pip install`, a guard with known
  wrapper bypasses, and `enableAllProjectMcpServers: true` invites copy-paste of fail-open
  controls; CLAUDE.md contradicts the tree.
- **Rebuttal:** the harness is the most novel thing here for the MLOps reader, but only as a
  documented, tested component with a threat model, not a working directory. Session IDs are
  opaque and already public in commit messages; scrubbing them violates K7 and the ledger rule.
- **Decision:** no `kb/` byte changes; `kb/README.md` preamble (what the IDs are, handoffs
  immutable by gate, fix the 7-vs-8 event drift and the "sessions gitignored" error) (A);
  `.claude/README.md` with an explicit "heuristic guard, not a sandbox" threat model (A);
  settings narrowing (`enableAllProjectMcpServers: false`, drop `python -c`/`npx`/`node`) is
  **M** (N8); CLAUDE.md minimal edits (3.10→3.11, torch floor, token example, `run_*.py` paths,
  footer) (A); the split stays **C** but its *gate lands now*: a governance test that every
  `CLAUDE.md §N(.M)` citation in tracked files resolves to an existing heading, plus D3 targets
  extended (A). CODEOWNERS rows for `/.claude/ /.github/ /Makefile /kb/` (A).

### 2.6 Delivery shape (Scrum + SDLC)
- **Thesis:** one branch, stacked slices, one draft PR.
- **Counter:** ~21k deleted lines + prose + CI + eval in one PR is unreviewable; squash kills
  bisectability; CI runs twice per push (push + PR events, different concurrency keys).
- **Rebuttal:** split by **reviewability class**, not by topic: deletion-only diffs are safe at
  any size when gates are green; prose and code need line review; charter needs ratification;
  release goes last; deployed-path edits need the maintainer's Cloud Build smoke.
- **Decision:** PR train in §3, merge commits never squash, one umbrella "Launch train" issue
  holding the DoD table (§6). PR bodies = template + Tracking, Depends-on, Hot files, C-items
  and who authorized, `git diff --stat`. Gate for the deletion PR: `--diff-filter=D` only plus
  debt baseline moved; gate for the public-surface PR: a ~20-line test that its diff touches no
  file the image build reads. Fix the coverage ratchet to use `GITHUB_BASE_REF` + `git merge-base`
  with `fetch-depth: 0` on the test job, negative control = lower a floor in a PR.

### 2.7 Dependency truth (SDLC)
- **Thesis:** pyproject canonical; `requirements.txt` a pointer.
- **Counter:** the deployed image never reads pyproject; `requires-python` is never enforced in
  the image (NGC 24.09 likely ships 3.10); naive hash pinning would replace NVIDIA's torch.
- **Rebuttal:** two consumers, two artefacts: ranges for library users (pyproject); an exact
  hashed export for the image generated from a refreshed lock with `--no-emit-package torch`.
- **Decision:** (A) `requirements.txt` → pointer, Makefile `install*` → `.[dev]`, cache keys
  → pyproject, Cloud Build step asserting `sys.version_info >= (3,11)` inside the image;
  (**C**, deployed path) regenerate `uv.lock`, `uv export --generate-hashes --no-emit-package torch`
  → `requirements-container.txt` with a CI diff check; PEP 735 groups deferred (changes the
  install command in 5 places + the SessionStart hook). Deleting `uv.lock` without replacement
  is rejected.

### 2.8 Security scanning (DevSecOps)
- **Thesis:** non-blocking pip-audit/CodeQL/zizmor/dependency-review (user choice).
- **Counter:** warn-only gates decay into the fail-open pattern the audit condemned.
- **Rebuttal:** split by what the result depends on. Tree-only tools (bandit `-b baseline`,
  zizmor `--offline`) can block on *new* findings now; feed-dependent tools (pip-audit) must
  never block PRs on advisories published that morning — use `dependency-review-action`, which
  diffs base vs head natively, for PR blocking; pip-audit runs scheduled against an **expiring**
  baseline (`{id, package, reason, expires}`).
- **Decision (A, shape approved as non-blocking first):** `security-scan.yml` with bandit
  medium/medium over `alpha_zero scripts infra` + baseline, zizmor, dependency-review (PR only),
  pip-audit scheduled + `scripts/diff_audit_baseline.py`; CodeQL in its own workflow with
  job-level `security-events: write`; Scorecard workflow file now (reports once public);
  `persist-credentials: false` on all checkouts; sha256 for the gitleaks download. Promotion to
  blocking after two clean weeks is a separate **C**. Negative control: plant
  `subprocess.call(x, shell=True)` → bandit red → baseline → green.

### 2.9 Release engineering (SDLC + Product)
- **Thesis:** roll to 1.9.0; maintainer tags.
- **Counter:** four BREAKING sections under SemVer ⇒ 2.0.0; verbatim 880 lines are unreadable;
  first tag dumps whole history into notes; "1.8.x supported" was never released.
- **Rebuttal:** the release is a *claim*; make every part of it checkable: tag == version,
  notes from the CHANGELOG section, artifacts with provenance.
- **Decision:** `[2.0.0]` ≈40 lines (Breaking / Fixed: search correctness / Security / Removed)
  + `docs/archive/changelog-2.0.0-detail.md`; fix duplicate `[1.3.0]` and `[0.9.0]` ordering;
  `release.yml`: assert tag == pyproject version, `python -m build`, sha256sums, notes from the
  section, first-tag-safe range, `attest-build-provenance` (**C**: new `id-token`/`attestations`
  permissions); D7 extended (one `[Unreleased]`, non-increasing versions); SECURITY.md → 2.0.x
  after the tag; CITATION.cff version = pyproject. Tag is **M**, on the merge commit, last.

### 2.10 Public claim of learning (ML + Product)
- **Thesis:** status table says learning is unvalidated post-fix.
- **Counter:** "does not learn" invites dismissal; say "evaluation infrastructure validated,
  learning run pending".
- **Rebuttal:** the question is whether one cheap run turns "pending" into "first
  measurement". It is feasible (CPU, seeded, ≈700 games, ≈2.7 h incl. controls, NG-7 compliant)
  **but** the local trainer only runs PUCT, so it cannot validate the advertised Gumbel default.
- **Decision:** wording "Evaluation infrastructure validated; no post-fix learning evidence
  yet" (A). Add `--mcts {civ6,gumbel}` to `training_civ6.py` (A, S). Optional **P7b**: a
  pre-registered local run (protocol written first: primary endpoint final checkpoint vs
  random-init network, 100 paired games, ternary CI, claim only if `ci_low > 0.5`; negative
  control random-init vs random-init CI ∋ 0.5; MDE stated ≈0.088 ≈ +61 Elo), labelled by search
  path, reported whichever way it comes out, run in `--resume_from` chunks ≤2 h (A). Gate: the
  evidence contract (§2.13) + a ledger `VALIDATION-RUN` event.

### 2.11 Benchmark and draw-aware metrics (ML)
- **Thesis:** `scripts/benchmark_suite.py`; route 8 sites through `score_rate`.
- **Counter:** another script nobody runs; renaming metrics breaks dashboards and
  `vertex_orchestrator` threshold compares; the count is ≥17 and includes decision sites.
- **Rebuttal:** the benchmark is a **protocol + schema executed by the promotion judge's own
  function** (`run_arena` → `scoring.summarize` → `EvaluationResult v2`), after fixing
  `run_arena`'s unseeded search and single-seat maps; metrics are **additive** and enforced as a
  *pairing* (every logged `*win_rate` is accompanied by `score_rate` and `draw_rate`).
- **Decision:** (A) `alpha_zero/evaluation/benchmark.py` `BenchmarkProtocol` (board 8, 60 turns,
  25 sims, Gumbel k=16, paired seats `seed + i//2`, `derive_seed(base,"bench.env",pair)`,
  opponents random/heuristic/simple_mcts/checkpoint:sha256, `BaselineType.SIMPLE_MCTS`),
  envelope `{benchmark_schema_version, protocol, evidence, results[]}`; smoke test asserts schema
  **and** SimpleMCTS(64) vs random `ci_low > 0.5` at n=40 (non-vacuous), seat-sign negative
  control, same-seed determinism once the MCTS seed is threaded. (A) E1 additive at the
  non-coordination sites + `tests/regression/test_draw_aware_metrics_regression.py`: AST scan of
  `BinOp(Div)` with a `win`-named left operand over `alpha_zero/` and `scripts/`, TOML allowlist
  keyed `path::qualname` with reasons (arena `win_rate`, `LegacyWinRatePolicy`, chess NG-6),
  allowlisted classes must define `score_rate`; pairing check on `log_metrics`/`mlflow.log_metric(s)`
  dict literals; controls on in-memory source strings. (**C**) `distributed/coordinator.py:493`
  and orchestrator-state `score_rate` (E5) → issue. Drop bootstrap-over-sequential-Elo
  (measures order sensitivity, not strength); claims use pairwise `score_rate` CI; Elo labelled
  "ladder rating"; Bradley-Terry + bootstrap → issue (M) with a synthetic-recovery gate.

### 2.12 Promotion gate (ML)
- **Thesis:** don't flip (C); log verdict + CI.
- **Counter:** a public default that cannot promote is a credibility hole.
- **Rebuttal:** the runbook's recommended config is *also* unreachable at n=20, and its
  "known winner" calibration is circular (uses the retracted A/B). The fix is to make
  infeasibility visible and size n honestly.
- **Decision:** (A) `PromotionConfig.feasibility(n_games, expected_draw_rate) → (min_observed_score,
  implied_elo)` logged at INFO at eval start, WARNING above `CIV6AZ_PROMOTION_MAX_IMPLIED_ELO`
  (default 50); unit tests `feasibility(20,0.7)` ⇒ >90 Elo, `feasibility(1500,0.7)` ⇒ <20;
  runbook rewritten (superiority test at 0.5 with n from the table vs an Elo bar; drop A/B
  calibration); persist `verdict.to_dict()`. (**C**) default flip, startup hard-refusal → issues.

### 2.13 Evidence, cards, citation (ML)
- **Thesis:** banners on pre-fix artifacts; fix the encoder card; CITATION.cff.
- **Counter:** annotation in-repo leaves "+31 Elo" live on the public Hub card.
- **Rebuttal:** one **evidence contract** is cheaper than five ad-hoc fixes, and the honest A/B
  statement is "compared which encoder better fits a corrupted target".
- **Decision (A):** `docs/evidence/SCHEMA.md` block (`schema_version, git_sha, generated_utc,
  search_epoch ∈ {pre-2026-09-04, post-2026-09-04}, seed|seed_null_reason, command,
  hardware{cpu_model,gpu_model,logical_cores,omp_num_threads}, python, torch, numpy, games_total,
  claims[], non_claims[]`); `docs/evidence/README.md` index; backfill existing dirs (hardware
  `null` + reason, all `pre`); `tests/regression/test_evidence_contract_regression.py`
  (every dir except `tech_debt_baseline` has one `report.md`, ≥1 JSON with a valid block, an
  index row; control: fixture missing `git_sha`). A/B paragraph (exact text in the ML review,
  §6) in the in-repo card, runbook and `ab_report.md`; Hub card update is **M**; readiness test
  forbids `+31` outside `docs/evidence/hexcrystal_ab/` and CHANGELOG. `docs/model_cards/README.md`
  ("only the encoder is published"); no dataset card (no dataset). `CITATION.cff` (cff 1.2.0,
  handle not email, version = pyproject, references: Silver 2018, Schrittwieser 2020, Danihelka
  2022, Karnin 2013, Silver 2017 — DOIs verified before commit). Gumbel divergences from the
  paper recorded in docstring + CHANGELOG with a target-entropy measurement test (A); aligning
  `c_scale`/normalisation is **C** (S3).

### 2.14 LLM eval harness (AI)
- **Thesis:** rename to "output-contract gate".
- **Counter:** legitimate contract test; renaming churns Make/CI/docs.
- **Rebuttal:** the name is the smallest problem: even `--online` grades structure (mock judge),
  `make verify` passes silently without the extra, and the extra is a git URL to a personal repo.
- **Decision (A):** keep the target; reword everywhere as "advisor output-contract gate (offline,
  stub LLM)"; fix the false comment in `configs/eval/strategy_advisor.yaml:9-10`; advisor marked
  Experimental/default-off (NG-2) in the status table; D4b forbids "LLM eval gate" outside
  CHANGELOG. (**M**) make `ianshank/Agents` public or remove `eval` from `verify`.

### 2.15 Backlog and systems of record (Scrum)
- **Thesis:** ~19 issues, `area:*`/`needs:*` labels, one milestone.
- **Counter:** a third backlog beside kb and openspec; 19 issues + 15 PRs reads as abandoned;
  public security issues are a disclosure.
- **Rebuttal:** one system per horizon: ROADMAP = themes/Now-Next-Later; GitHub Issues = the
  only actionable list; `openspec/changes` = scope of an in-flight change (`Tracking: #N`);
  `kb/specs` = frozen rationale; `kb/ledger` = agent memory. Sync rule: issue exists before an
  openspec task references it; closed at merge.
- **Decision:** milestones "v2.0.0 Public launch" (blocking only), "v2.1 First validated
  learning run", "M4 Throughput & N-player"; 11 labels (`bug, enhancement, documentation,
  good first issue, area:search-eval, area:env, area:bridge, area:cloud, area:ci-supply-chain,
  area:governance, status:needs-decision`); ≈10 curated public issues on day one; **security
  items never become public issues** (private checklist/advisory draft, closed pre-flip);
  `blank_issues_enabled: false`, add `question.yml` or enable Discussions (SUPPORT.md promises
  them; `has_discussions: false` today); PR template fixes; `kb/` stays agent-only (CODEOWNERS
  `kb/ @ianshank`). Issue creation is a write to an external service → **C**, bodies live in the
  report appendix.

### 2.16 Contributor experience (SDLC)
- **Thesis:** `make verify` is the contributor gate.
- **Counter:** Windows has no `make`; `verify` takes tens of minutes; `make lint` rewrites files;
  the `eval` extra may be unreachable; CI is Linux-only.
- **Rebuttal:** charter invariant 2 only requires Linux CI green plus a documented Windows
  deselect list; what breaks cross-OS is import-time behaviour and installs, which
  `tests/sanity` covers.
- **Decision (A, non-blocking):** `contributor-smoke` job {ubuntu, macos, windows} × 3.11 on
  pyproject path changes + weekly: `pip install -e .[dev]` then `pytest tests/sanity tests/regression -q`;
  `make verify-fast` (lint-check, typecheck, regression, governance); CONTRIBUTING rewritten
  plain-contributor-first. Blocking is **C**.

## 3. PR train (merge commits, in order, each cut from the default branch after the previous merges)

Pre-step (**M**): merge Dependabot #124/#129/#131 (they edit `ci.yml`) before cutting branches;
close #125 (fails CI; pin bumped across 5 sites in P5), #120 (orphan Dockerfile removed), #95,
#99–#102 (superseded by #118), #32/#49 (NG-6), #69 (per disposition), #106 (close + salvage
issue: atomic IPC never landed); hold #130 (`status:needs-decision`).

| PR | Contents | Class | Key gates |
|---|---|---|---|
| **P0** | `docs/audits/2026-11-10-public-release-review.md` (role sections, Sept-audit status table, maintainer checklist, issue bodies appendix, reproduction log with confidence tags); `openspec/changes/public-release-readiness/` (proposal with `Tracking:`, tasks per PR, deltas for `documentation`, `validation-gates`, new `evaluation-protocol`); umbrella issue body | docs | `validate_openspec.py` |
| **P1** | Deletions only: `replays/`, `demos/atari` + `tests/{unit,integration}/test_atari` + `scripts/run_atari_ui.py`, root `interfaces/` + `tests/unit/test_interfaces.py` + `configs/civ6_integration.yaml` check, `templates/{src,tests,pyproject}`, `mcp_config.json`, `configs/wandb_dashboard.json`, root `Dockerfile` + its Dependabot entry, `GeneratedPaths.lua`; `pyproject` excludes/per-file-ignores; suite-wiring parent list; `.dockerignore`; `measure_debt.py --write` baseline | deletion-only (user-approved **C**) | `git diff --diff-filter=D` check; suite-wiring; debt ratchet; `make verify` |
| **P2** | Moves + scrub + readiness gate: `run_*.py` → `scripts/` (6 test imports, per-file-ignores, `configs/sweep_muzero.yaml`, `openspec/project.md`, CLAUDE.md §3.1); `server-module`+`spacetimedb_module` → `experimental/spacetimedb/` + README; `templates/*.md` → `docs/templates/`; §2.4 scrub; fail-closed defaults; `*.example.yaml` + `configs/local/`; ignore-file gaps (`*.safetensors *.ckpt *.npz *.onnx *.bin`); `tests/regression/test_public_readiness_regression.py` + manifest (§5) shown **red before, green after** | code + config | readiness test; `make verify`; deployed-path smoke flagged for maintainer |
| **P3** | Public surface: README v2; `docs/README.md` index; stubs removed; `docs/LIVE_CIV6.md`; `docs/EVALUATION.md` (vocabulary, protocol, seeding contract, power table with a test that recomputes every cell); evidence SCHEMA/README/backfill/banners + A/B paragraph + model-card fixes + `docs/model_cards/README.md`; `CITATION.cff`, `.editorconfig`, `.python-version`, `GOVERNANCE.md`, `py.typed`; pyproject metadata (`readme`, `license="MIT"`, `license-files`, `authors`, `urls`, `classifiers`, `keywords`, `setuptools>=77`, `eval` out of `all`, `spaces` in); ROADMAP validation column + Now/Next/Later; SUPPORT/SECURITY/CoC/CONTRIBUTING/PR-template; `kb/README.md` + `.claude/README.md`; CLAUDE.md minimal edits; `openspec/project.md` refresh; `check_doc_sync` D4b/D6′/D8/D9 + CFF version + CLAUDE citation-resolution test | prose + gates | `check_doc_sync.py`; negative controls per new rule; `pip install -e .[all]` on a clean venv |
| **P4** | Charter-only: links, M1 "proven", 13×/5–10×, invariant 2 version floor | ratification | **M** merges |
| **P5** | CI & supply chain: `ci.yml` (push trigger lists real default + `main` with comment, cache keys, `fetch-depth: 0` on test, ratchet base fix, bandit widened + baseline, 3.13 `continue-on-error`, `contributor-smoke`); `security-scan.yml`, `codeql.yml`, `scorecard.yml` (non-blocking); `persist-credentials: false`; gitleaks sha256; `release.yml` per §2.9; `deploy_spaces.yml` (drop push trigger, pin installs, `environment:`, fix validate job, HF names unified); `validate_kb.py --git` fixed + K7 fails on shallow under `--git`; pre-commit bumps + gitleaks hook + patterns; Makefile dead targets + `install*` + `verify-fast`; pinning test extended (push triggers, unpinned `pip install`, secrets on unpinned steps); ruff 0.16.7 across 5 pin sites; Cloud Build py≥3.11 assert | CI | zizmor exit 0; pinning tests; negative controls (unpin one `uses:`, plant second `[Unreleased]`) |
| **P6** | Release roll: `[2.0.0]` summary + archived detail, version bump in pyproject/`__init__`/CITATION/SECURITY/README badge/MODULE_REFERENCE header, D7 extension | release | `check_doc_sync`; then **M** tags `v2.0.0` on the merge commit and watches `release.yml` |
| **P7** (milestone v2.1, not launch-blocking) | Evaluation plumbing per §2.10–2.13: `run_arena` MCTS seed + paired seats, `--mcts` flag, `evaluation_worker:323` seed, `BenchmarkProtocol` + smoke, E1 additive (non-coordination sites) + AST/pairing gate, promotion feasibility logging + runbook, target-entropy test, closed-loop learning oracle (self-play→train→re-search on a 2-move env; prior puts >0.8 on the winning action for PUCT and Gumbel; control: invert the value-target sign), stale docstrings; optional **P7b** pre-registered local run | code + evidence | all gates with negative controls; `make test-cloud-coverage` floor held |

**Must stay out (issues, owner noted):** key rotation, IAM, branch rename, protection, private
vulnerability reporting, Discussions, tag (M) · settings narrowing/push-guard (M, N8) · lock +
hashed container export (C) · coordinator/orchestrator E5 + seed threading (C) · promotion
default flip / hard refusal (C) · completed-Q alignment (C, S3) · WS-D2, D0/D1, fog (env
semantics, own OpenSpec) · any cloud or >1000-game run, Gumbel-arm learning run, A/B re-run
(NG-7/M) · Hub card push (M) · online LLM quality eval (NG-2) · CLAUDE.md split (C) · distribution
rename (C) · fake canary/`:latest` deploys/state-bucket split (C, deployed path) · scanners to
blocking (C) · Bradley-Terry (M effort) · `weights_only` migration (C).

## 4. README v2 — first 15 lines (fixed) and outline

```markdown
# Civ6-AlphaZero

[![CI](…/ci.yml/badge.svg)](…/ci.yml) [![Python 3.11+](…)](pyproject.toml) [![License: MIT](…)](LICENSE)
[![Status: research — learning not yet demonstrated](…)](docs/EVALUATION.md)
[![HF encoder](…)](https://huggingface.co/ianshank/civ6-hexcrystal-encoder)

An AlphaZero/MuZero-style self-play system for a simplified Civilization-like game (MicroCiv),
engineered so every claim about it can be checked, including the ones that turned out wrong.

- **What this is:** a CPU-reproducible, seeded MCTS + policy/value training stack with
  negative-controlled quality gates, a measured evaluation methodology, and a demo-grade bridge
  that sends commands into a live Civilization VI.
- **What this isn't (yet):** a strong agent. No post-fix run has shown learning; see
  [Project status](#project-status) and [`docs/EVALUATION.md`](docs/EVALUATION.md).
- **Not affiliated with** Firaxis Games, 2K, Take-Two Interactive or Google DeepMind.
  *Civilization VI* is a trademark of Take-Two Interactive Software, Inc.

## Project status
```
Then: status table (area · state · evidence link; "proven" = committed seeded artifact on
post-2026-09-04 code) → What this is + mermaid → Install (extras table; `all` note) → Quickstart
(kept measured table with pre-fix banner) → Evaluate (`BenchmarkProtocol`, EVALUATION.md,
evidence index) → Reproducibility & evidence → Cloud training (two lines, cost warning) → Live
Civilization VI (link to LIVE_CIV6.md) → Testing & quality gates → Documentation → Contributing ·
Citation · Security · Code of Conduct · License. Removed: duplicate badges, HF section
(→ HF_PUBLISHING.md), runners table, speed-up numbers. All external URLs listed in
`docs/public_links.toml` (manifest, offline-checked; `make check-links` hits the network).

## 5. Public-readiness regression test
`tests/regression/test_public_readiness_regression.py` + `tests/regression/public_readiness_manifest.json`,
house style of `test_repo_hygiene_regression.py` (pure predicates, planted controls through the
same predicate, reasoned non-stale allowlists, vacuity guards asserting >0 files inspected):
1. personal emails (allowlist empty; excludes noreply/example/gserviceaccount);
2. GCP project number / SA emails: zero tolerance; bucket literals outside `*.example.*`/archive: zero;
3. project-id ratchet (count may only fall; stale allowlist entries fail);
4. tracked binaries >100 KB outside allowlist (NUL sniff);
5. README badge/link URLs ⊆ manifest, manifest entries ⊆ README (D8-badge in doc-sync);
6. `CITATION.cff` parses, keys present, version = pyproject;
7. byte-compile every tracked `*.py` under lint-excluded dirs;
8. pyproject metadata keys; `license` is a string; `all` contains no `@ git+` extra;
9. CLAUDE.md: no `3\.10` outside an allow comment; footer date ≥ newest §5 date;
10. forbidden claims (`+31`, `13x`, `5–10x`, "LLM eval gate", `production`, `complete`) outside
    CHANGELOG/evidence dirs; README first 40 lines contain "Status" and "not affiliated";
11. absolute user paths (`C:\Users\`, `/home/<name>/`, `/Users/`) in non-archive files.

## 6. Launch Definition of Done (umbrella issue table)

| # | Item | Owner | Blocking | Evidence |
|---|---|---|---|---|
| 1 | Rotate W&B key; then scrub leak location text (CHANGELOG `:489`, runbook) | M | yes | W&B audit log |
| 2 | Remove 4 `allUsers`; ingress internal; confirm service state | M | yes | `get-iam-policy` |
| 3 | P5 merged (deploy trigger removed) **then** rename default → `main` | M | yes | API `default_branch` |
| 4 | Protection: required checks, 0 approvals, no force-push; tag ruleset `v*`; private vuln reporting on | M | yes | settings |
| 5 | Identifier scrub in tree; history accepted and documented | session | yes | readiness test |
| 6 | README v2, one live HF link, CITATION, GOVERNANCE pointer, trademark line | session | yes | D4b/D8 |
| 7 | Discussions enabled or SUPPORT claim removed; CoC role contact | M decides, session edits | yes | API |
| 8 | `ianshank/Agents` public or `eval` out of `verify` | M | yes | clean-venv `make verify` |
| 9 | Charter PR ratified | M | yes | P4 merged |
| 10 | 2.0.0 rolled; tag pushed; `release.yml` artifacts present | M | yes | release run |
| 11 | Dependabot triage; 0 PRs >30 days; stale branches archived; wiki off; description + topics set | M | yes/no | repo API |
| 12 | Security scanners running (non-blocking) | session | no | workflow run |
| 13 | Web UI accessibility | — | out of scope | — |

## 7. Agent-team execution model

Integration lead (this session): sole writer of `kb/ledger`, CHANGELOG, debt baseline,
MODULE_REFERENCE and every hot file (`pyproject.toml`, `README.md`, `CLAUDE.md`, `docs/README.md`,
`Makefile`, `ci.yml`); workers return hot-file edits as text. Wave 1 (parallel, isolated
worktrees, disjoint paths): **W-A** deletions (P1) · **W-C** CI (P5 files) · **W-D** evaluation
(P7 files). Wave 2 (sequential): **W-B** scrub + readiness test after W-A (allowlist must match
the final tree) · **W-E** docs after A/B/D (README cites moved scripts and the `score_rate`
vocabulary). Per PR: a fresh-context reviewer agent (`code-review` skill, `pre-pr-handoff` §2b)
plus a `negative-control` replay of every new gate; findings return as QA-GAP via the lead.
Forbidden for workers: `kb/`, `PROJECT_CHARTER.md`, `git push`. `dev-sqe-handoff` at the end.

## 8. Verification (per PR, CI mirrors)
```bash
OMP_NUM_THREADS=2 make verify
python scripts/check_doc_sync.py
python scripts/validate_openspec.py --change public-release-readiness
pytest tests/regression -q        # readiness, draw-aware, evidence-contract, pinning, ratchets
python scripts/validate_kb.py --git
python -m alpha_zero.training_civ6 --dry_run && python -m alpha_zero.orchestrator --preset fast --dry_run
```
Live negative controls recorded in each PR body: P2 re-add the personal email → red; P3 plant an
unindexed doc and a dead charter link → D8/D6′ red; P5 unpin one `uses:` and plant a second
`[Unreleased]` → red; P7 restore `wins / num_games` at `training_civ6.py:945` and flip one
evidence `search_epoch` → red. Clean-machine: `pip install -e '.[all]'`; `python -m build` wheel
contains `py.typed`; `git ls-files -z | xargs -0 file | grep -v text` lists only allowlisted
binaries. Effort: P0 S · P1 S · P2 M · P3 M · P4 S(M) · P5 M · P6 S · P7 L (P7b ≈3 h wall-clock).
