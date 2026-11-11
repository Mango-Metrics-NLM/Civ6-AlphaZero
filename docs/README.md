# Documentation Index

One entry point for everything under `docs/` and the top-level project files.
Pages are labelled by kind (tutorial · how-to · reference · explanation), by
**audience** (*user* runs the software · *contributor* changes it · *operator*
deploys it), by **status** (**Current** = maintained and expected to match the
tree · **Reference** = authoritative lookup, revised only when the thing it
describes changes) and by the date the page was last verified against the code.
Measurements carry one extra label: whether they were taken **before or after
2026-09-04**, the day the search-correctness fixes landed — any number measured
before it describes a defective search and is not evidence about learning.

This index is a gate: `scripts/check_doc_sync.py` (D8) fails if a tracked
`docs/**/*.md` outside `archive/` is missing from it or listed twice.

## Start here

In this order:

1. [`../README.md`](../README.md) — what the project is, what is proven and what is not, quick start.
2. [`EVALUATION.md`](EVALUATION.md) — how strength is measured here (draw-aware score rate, sample sizes, what a number can and cannot show). <!-- lead: row added by W-E because the file landed before hand-off; adjust wording freely -->
3. [`architecture/C4_ARCHITECTURE.md`](architecture/C4_ARCHITECTURE.md) — how the system is built (C4 context → containers → components).
4. [`../ROADMAP.md`](../ROADMAP.md) — milestones, with a *Validation (evidence)* column, and Now / Next / Later.
5. [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — set-up, the inner loop and the quality gates.

Project-level files that are not under `docs/`:

| File | Purpose |
|------|---------|
| [`../PROJECT_CHARTER.md`](../PROJECT_CHARTER.md) | Scope, non-goals, invariants and the amendment protocol (maintainer-ratified). |
| [`../GOVERNANCE.md`](../GOVERNANCE.md) | Who decides what; pointers to the charter, decision rights and release policy. |
| [`../CHANGELOG.md`](../CHANGELOG.md) | What shipped, by version. |
| [`../SECURITY.md`](../SECURITY.md) · [`../SUPPORT.md`](../SUPPORT.md) · [`../CODE_OF_CONDUCT.md`](../CODE_OF_CONDUCT.md) | Reporting a vulnerability · getting help · conduct. |

## Tutorials

There is no step-by-step tutorial beyond the quick start in the top-level
README. The closest thing is the seeded CPU run described there
(`python -m alpha_zero.training_civ6 --dry_run`, then `--seed 42`); a guided
walkthrough is a welcome contribution.

## How-to guides

| Doc | What it helps you do | Audience | Status | Last verified |
|-----|----------------------|----------|--------|---------------|
| [`LIVE_CIV6.md`](LIVE_CIV6.md) | Drive a running Civilization VI from the agent (demo-grade: four `END_TURN`s on random weights are all that has been verified). | user | Current | 2026-11-10 |
| [`HF_PUBLISHING.md`](HF_PUBLISHING.md) | Stage and publish the hex-crystalline encoder to the Hugging Face Hub. | contributor | Current | 2026-08-11 |
| [`INFERENCE_SERVICE.md`](INFERENCE_SERVICE.md) | Run and call the FastAPI inference server. | operator | Current | 2026-08-11 |
| [`GCP_INFRA.md`](GCP_INFRA.md) | Stand up the Vertex AI / GCS training infrastructure. | operator | Current | 2026-09-02 |
| [`GCP_TROUBLESHOOTING.md`](GCP_TROUBLESHOOTING.md) | Fix gcloud and Windows connectivity problems. | operator | Reference | 2026-02-21 |
| [`MUZERO_CLOUD_INTEGRATION.md`](MUZERO_CLOUD_INTEGRATION.md) | Run the MuZero variant on Vertex AI. | operator | Current | 2026-09-02 |
| [`runbooks/promotion-gate.md`](runbooks/promotion-gate.md) | Configure the model-promotion gate, and why the shipped defaults cannot promote at a 73.5% draw rate. | operator | Current | 2026-08-28 |
| [`runbooks/secret-scanning.md`](runbooks/secret-scanning.md) | Run gitleaks locally; triage of the open leaked-credential finding. | contributor | Current | 2026-08-30 |
| [`runbooks/v1.6.0-cloud-deploy.md`](runbooks/v1.6.0-cloud-deploy.md) | The v1.6.0 cloud deployment, step by step. | operator | Reference | 2026-05-22 |

## Reference

| Doc | What it is | Audience | Status | Last verified |
|-----|------------|----------|--------|---------------|
| [`MODULE_REFERENCE.md`](MODULE_REFERENCE.md) | The `alpha_zero/` module inventory, one row per subpackage (gated by `check_doc_sync` D2). | contributor | Reference | 2026-09-04 |
| [`architecture/README.md`](architecture/README.md) | Entry to the architecture tree (C4 model + ADRs). | contributor | Reference | 2026-07-25 |
| [`NAMING_CONVENTIONS.md`](NAMING_CONVENTIONS.md) | Environment variables, image tags and GCS path conventions. | operator | Reference | 2026-02-20 |
| [`model_cards/README.md`](model_cards/README.md) | Index of model cards and what a card must state. | user | Reference | 2026-11-10 |
| [`model_cards/hexcrystal_encoder.md`](model_cards/hexcrystal_encoder.md) | Model card for the published hex-crystalline encoder. | user | Reference | 2026-11-10 |
| [`evidence/SCHEMA.md`](evidence/SCHEMA.md) | The contract every committed measurement under `evidence/` must satisfy. | contributor | Reference | 2026-11-10 |

## Explanation

| Doc | What it explains | Audience | Status | Last verified |
|-----|------------------|----------|--------|---------------|
| [`HARNESS_ENGINEERING_ANALYSIS.md`](HARNESS_ENGINEERING_ANALYSIS.md) | Why the agent harness (hooks, permissions, KB workflow) is shaped the way it is. | contributor | Reference | 2026-08-11 |
| [`diagrams/hierarchical_mcts.md`](diagrams/hierarchical_mcts.md) | Design sketch: three-level hierarchical MCTS with goal propagation (2026-02 novelty write-up). | contributor | Reference | 2026-02-15 |
| [`diagrams/rag_enhanced_mcts.md`](diagrams/rag_enhanced_mcts.md) | Design sketch: retrieval-augmented MCTS. | contributor | Reference | 2026-02-15 |
| [`diagrams/reflection_memory.md`](diagrams/reflection_memory.md) | Design sketch: LATS-style reflection memory over game-state embeddings. | contributor | Reference | 2026-02-15 |

## Records

Audits, measurements, decisions and plans. These are not maintained; they are
what was true when written. The last column says whether a *measurement* was
taken on code **before** or **after** the 2026-09-04 search fixes
(`n/a` for documents that measure nothing).

### Audits and reviews

| Doc | What it records | Before / after 2026-09-04 |
|-----|-----------------|---------------------------|
| [`audits/2026-09-04-standards-audit.md`](audits/2026-09-04-standards-audit.md) | Coding-standards audit; its §0 records the review corrections and its remediation list is superseded by `kb/specs/20260904-correctness-first-plan.md`. | n/a (review) |
| [`audits/2026-11-10-public-release-review.md`](audits/2026-11-10-public-release-review.md) | Public-release review: status of the September items, the maintainer checklist and the post-flip backlog. | n/a (review) |
| [`audits/2026-11-11-local-first-plan-review.md`](audits/2026-11-11-local-first-plan-review.md) | Panel and two adversarial review rounds behind the local-first gap-closure and dev/qa/main branching plan (`kb/specs/20261111-local-first-and-branching-plan.md`). | n/a (review) |

### Evidence (committed measurements)

Index and contract: [`evidence/README.md`](evidence/README.md). Every artifact
below was measured on **pre-fix** code.

| Doc | What it measured | Before / after 2026-09-04 |
|-----|------------------|---------------------------|
| [`evidence/validation_iteration_zero/validation_iteration_zero_report.md`](evidence/validation_iteration_zero/validation_iteration_zero_report.md) | First end-to-end local training run (5 iterations, CPU): the loop runs; the checkpoint scored 0–10% wins vs random with no trend. | before |
| [`evidence/draw_rate_sweep/report.md`](evidence/draw_rate_sweep/report.md) | Draw rate under random play vs board size and turn limit (0.68 / 0.47 / 0.37 at 30 / 60 / 100 turns on 8×8). | before |
| [`evidence/seeded_probe_reproducibility/README.md`](evidence/seeded_probe_reproducibility/README.md) | Seeded probes of one checkpoint are identical (5/5); unseeded ones spread 0.30–0.45. | before |
| [`evidence/hexcrystal_ab/ab_report.md`](evidence/hexcrystal_ab/ab_report.md) | Encoder A/B arena (800 games); superseded as a strength comparison — it shows the harness runs. | before |
| [`evidence/hexcrystal_ab/summary_table.md`](evidence/hexcrystal_ab/summary_table.md) | The A/B counts and Wilson interval, as a table. | before |

### Architecture decision records

| Doc | Decision | Before / after 2026-09-04 |
|-----|----------|---------------------------|
| [`architecture/adr/README.md`](architecture/adr/README.md) | ADR index and lifecycle. | n/a |
| [`architecture/adr/0001-scaling-optimization.md`](architecture/adr/0001-scaling-optimization.md) | Scaling and optimisation architecture. | n/a |
| [`architecture/adr/0002-pipeline-hardening.md`](architecture/adr/0002-pipeline-hardening.md) | Retry, validation and observability in the cloud pipeline. | n/a |
| [`architecture/adr/0003-gcp-resource-deployment.md`](architecture/adr/0003-gcp-resource-deployment.md) | GCP resource deployment for the MuZero variant. | n/a |

### Plans and specifications

`kb/specs/` holds dated plans written for the agent workflow; they are frozen
rationale, not maintained documentation. The two that still sequence open work
are the 2026-09 pair.

| Doc | What it is | Before / after 2026-09-04 |
|-----|------------|---------------------------|
| [`../kb/specs/20260904-correctness-first-plan.md`](../kb/specs/20260904-correctness-first-plan.md) | **Current sequencing of open work**: search correctness first, then the draw-rate re-measure and the seeded learning run. | written on the fix day; sequences the post-fix runs |
| [`../kb/specs/20260902-validated-learning-plan.md`](../kb/specs/20260902-validated-learning-plan.md) | Content of each validation workstream (draw-rate fix, learning-trend run, promotion gate). | plan over pre-fix measurements |
| [`../kb/specs/20260902-tech-debt-hardening.md`](../kb/specs/20260902-tech-debt-hardening.md) | Tech-debt reduction plan; Core scope landed 2026-09-02, Tier B decisions and WS3–WS7 remain. | n/a |
| [`../kb/specs/20260713-project-charter.md`](../kb/specs/20260713-project-charter.md) | How the charter was instantiated. | n/a |
| [`../kb/specs/20260402-full-civ6-local-gameplay.md`](../kb/specs/20260402-full-civ6-local-gameplay.md) | The live-bridge plan behind v1.7.0–v1.8.0. | n/a |
| [`../kb/specs/20260217-multiplayer-gap-analysis.md`](../kb/specs/20260217-multiplayer-gap-analysis.md) | N-player gap analysis (G1–G10). | n/a |
| [`../kb/specs/20260208-phase-6-pbt.md`](../kb/specs/20260208-phase-6-pbt.md) | Population-based training specification. | n/a |
| [`../kb/specs/20260208-phase-5-cost-optimization.md`](../kb/specs/20260208-phase-5-cost-optimization.md) | Cloud cost optimisation and monitoring. | n/a |

<details>
<summary>Product requirement documents (2026-02 → 2026-03, historical; new proposals go through <code>openspec/changes/</code>)</summary>

| Doc | Area | Before / after 2026-09-04 |
|-----|------|---------------------------|
| [`prd/README.md`](prd/README.md) | PRD index with delivery status. | n/a |
| [`prd/prd-nplayer-microciv.md`](prd/prd-nplayer-microciv.md) | N-player MicroCiv. | n/a |
| [`prd/prd-batch-selfplay.md`](prd/prd-batch-selfplay.md) | Batch self-play cloud wiring (its throughput figure is a target, never measured). | n/a |
| [`prd/prd-multi-iteration-orchestrator.md`](prd/prd-multi-iteration-orchestrator.md) | Multi-iteration training orchestrator. | n/a |
| [`prd/prd-pbt-hyperparameter-tuning.md`](prd/prd-pbt-hyperparameter-tuning.md) | Population-based training. | n/a |
| [`prd/prd-pipeline-hardening.md`](prd/prd-pipeline-hardening.md) | Evaluation worker, backoff, validation. | n/a |
| [`prd/prd-circuit-breaker.md`](prd/prd-circuit-breaker.md) | Circuit breaker and retry hardening. | n/a |
| [`prd/prd-observability.md`](prd/prd-observability.md) | Failure alerting and dashboards (written for the since-retired W&B). | n/a |
| [`prd/prd-container-rebuild.md`](prd/prd-container-rebuild.md) | Container rebuild and training resumption. | n/a |
| [`prd/prd-emergency-recovery.md`](prd/prd-emergency-recovery.md) | Container rebuild, deployment and orchestrator fix. | n/a |
| [`prd/prd-version-validation.md`](prd/prd-version-validation.md) | Container version validation gate. | n/a |
| [`prd/prd-gcp-resource-deployment.md`](prd/prd-gcp-resource-deployment.md) | GCP resource deployment and validation. | n/a |
| [`prd/prd-rapid-diagnosis.md`](prd/prd-rapid-diagnosis.md) | Rapid diagnosis of GCP connectivity. | n/a |
| [`prd/prd-bypass-mitigation.md`](prd/prd-bypass-mitigation.md) | Bypass and mitigation of GCP connectivity issues. | n/a |
| [`prd/prd-resolution-verification.md`](prd/prd-resolution-verification.md) | Resolution and verification of GCP connectivity. | n/a |
| [`prd/prd-fix-tournament-test.md`](prd/prd-fix-tournament-test.md) | Fix for a stale tournament test. | n/a |
| [`prd/aqa-remediation-sprint.md`](prd/aqa-remediation-sprint.md) | AQA remediation sprint report. | n/a |

</details>

## Agent harness

**Not required reading for human contributors.** These files configure and
record the Claude Code workflow the maintainer uses; a human contributor needs
only `CONTRIBUTING.md`.

| Doc | What it is |
|-----|------------|
| [`../CLAUDE.md`](../CLAUDE.md) | Constraint specification for the coding agent: gates, permissions, decision log, command reference. |
| [`../.claude/README.md`](../.claude/README.md) | What the hooks and permission allowlist do, and their threat model (a heuristic guard, not a sandbox). |
| [`../kb/README.md`](../kb/README.md) | The Planning → DEV-SQE → Pre-PR handoff protocol and what the opaque session identifiers in `kb/` are. |
| [`../openspec/AGENTS.md`](../openspec/AGENTS.md) | The OpenSpec change workflow (propose → implement → archive). |
| [`../openspec/project.md`](../openspec/project.md) | OpenSpec project context: stack, entry points, conventions. |
| [`templates/CLAUDE_TEMPLATE.md`](templates/CLAUDE_TEMPLATE.md) | Blank constraint-specification template from which `CLAUDE.md` was instantiated; reuse it for another repository. |
| [`templates/PROJECT_CHARTER_TEMPLATE.md`](templates/PROJECT_CHARTER_TEMPLATE.md) | Blank charter template from which `PROJECT_CHARTER.md` was instantiated. |

## Archive

| Doc | What it holds |
|-----|---------------|
| [`archive/README.md`](archive/README.md) | Frozen, unmaintained documents kept for provenance — superseded plans, legacy READMEs, one-off analyses, and the 2026-08 next-steps reconciliation. Excluded from the documentation-sync gates. |
