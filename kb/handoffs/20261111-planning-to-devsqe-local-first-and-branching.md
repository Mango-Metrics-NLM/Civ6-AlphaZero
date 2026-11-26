# Planning → DEV-SQE Handoff

**Date:** 2026-11-11
**Spec:** kb/specs/20261111-local-first-and-branching-plan.md (program plan v3, after two adversarial review rounds)
**Register:** kb/specs/20261111-local-first-gap-register.md (82 `LF-` rows)
**Review record:** docs/audits/2026-11-11-local-first-plan-review.md
**Session:** session_9a6019a1-f6d9-4e79-9f63-04002716a56d
**Branch:** `claude/local-deployment-branching-plan-51a0cb`

**Why this plan exists:**
- The maintainer asked to close every remaining gap with everything running locally, Vertex excluded, under a professional dev/qa/main branching strategy.
- No learning evidence exists after the 2026-09-04 search fixes.
- The only real promotion loop is Vertex's, and the local CLI loop crashes on real training.

**Acceptance Criteria:** v2.1, spec §9 tasks T0a–T18. Each task names its failing test, its negative control and its verify command.
- **Branching (T4, T6, T3):**
  - The guard denies commits and pushes on dev/qa/main, tag creation, and `gh` write calls, in both Bash and PowerShell.
  - Branch Direction enforces spec §3.4 on a temporary-repo fixture.
  - `tag_release.py` and `release.yml` enforce spec §3.5.
  - Every ruleset check is a real job.
- **Isolation and pinning (T1, T2, T1b):**
  - `GCSBackend` raises `LocalModeViolation(BaseException)` under local mode.
  - Runs execute from pinned source and a pinned env.
  - A run card totalling more than 1,000 games is refused.
- **Science (T7–T13):**
  - B1: 0 invalid moves in 200 seeded games.
  - `SearchStats` equality across all three searches.
  - Provenance refuses a horizon mismatch.
  - Benchmark output passes the evidence contract and records per-game pairs.
  - EVALUATION.md section 7.1 is registered verbatim before any A1 run.
- **Evidence (T14):** the verifier recomputes the verdict and exits 0/1/2. A FAIL verdict still satisfies `learning_recorded`.
- **Release (T17):** v2.1.0 is tagged on main with a bundle under `docs/evidence/release_v2_1_0/`.

**Constraints:**
- Confirm-first items are marked CF in the register (A3, A11/T8, L0–L4, and others). Record each approval in the issue and in a handoff note.
- Maintainer-only:
  - tags;
  - merges into qa/main;
  - rulesets, settings, and any `gh api` write;
  - key revocation;
  - any run over 1,000 games.
- Every run is ≤1,000 games, priced per card, on CPU for confirmatory runs, with the data root `E:\civ6az-data`.
- Vertex/GCP code is frozen, not deleted. New code goes in `alpha_zero/local/`, which counts toward the 85% coverage gate.

**Priority:** high

**Blocked On:**
- Maintainer §10 actions 1–5:
  - revoke the W&B key and purge `refs/pull/72/head`;
  - IAM `allUsers` bindings and the user-scope bucket variable;
  - approve the plan;
  - create the machine user;
  - merge T3a/T4, then tag v2.0.0 on `c214daa`.
- The cut-over must happen before any further PRs target the new branch layout.
- The LF-A5/A12 decision gates T13.
