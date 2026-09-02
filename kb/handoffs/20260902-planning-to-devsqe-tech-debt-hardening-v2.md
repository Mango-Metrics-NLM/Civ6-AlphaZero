# Planning → DEV-SQE Handoff (v2)

**Date:** 2026-09-02
**Spec:** kb/specs/20260902-tech-debt-hardening.md (v2; supersedes the v1 handoff of the same date)
**Session:** session_01EcSsbbTYiSkmZsMVyipm7G
**Branch:** `claude/tech-debt-hardening-plan-xrn9sp`, PR #116

**Why a v2:** an independent adversarial review of v1 re-derived its numbers. The
dead-code census was 2.7x too small and had false positives (grep vs AST); 34 dead
modules are *measured* by the coverage gate; the live action-space bug is `381`, not
`356`; the "red draft PR" coverage mechanism is replaced by per-package floors; the
default branch is not `main` so the §3.2 guard is prose; the estimate was 6–7 weeks for
~52 PR-sized units. `scripts/audit_import_graph.py` (9 tests) ships with the plan so
the baseline is reproducible by command, not by hand.

**Acceptance Criteria:** spec §3, each with its gate:
- Tier A deleted (44 modules / 19,882 lines); every Tier B row decided; debt ratchet at 0 Tier A
- per-package coverage floor for `alpha_zero/cloud/gcp` in the `cloud-tests` job, set from
  the post-deletion measurement; omit list ≤ 15 patterns with same-line reasons; global ≥ 85%
- zero `torch.load` without `weights_only` (AST test); checkpoint inventory before the flag flips
- `S101` on for `alpha_zero/`, `-s B101` gone, zero runtime asserts
- one `action_space_size` source, zero `381`/`356` literals (AST test); one promotion
  threshold default; `CIV6AZ_` only with a two-release shim
- files > 1,000 lines ≤ 15, no decomposition child > 900, zero PLR0915 waivers
- guard reads the real default branch
- one manifest + one lock mechanism; open PRs ≤ 3 per spec §7
- every gate merged with a negative control

**Constraints:**
- CI is green; widen what it measures, never "fix" it. Floors come from measurement
  after Tier A lands, because dead code currently inflates the number.
- WS2 step 0 checklist is mandatory per deletion PR: Makefile, docker-compose, the three
  cloudbuild files, infra/, dockerfiles/, every Agent.md (validate_agents A6), non-archive
  docs, owning `__init__` re-exports, PROJECT_CHARTER, MODULE_REFERENCE/CLAUDE.md §7
  counts (check_doc_sync D2).
- Never delete Tier C (`multiplayer/*`, a §13 roadmap dependency).
- No hyperparameter, threshold value, or architecture change. No subpackage moves.
- Ratchets only tighten; net omit count only decreases.
- WS5.5–5.7 reverse a recorded 2026-08-30 decision and may start only after the
  `mock.patch` enumeration test passes on the current tree.
- Each PR states lines removed and coverage delta; not-independently-revertible items
  (WS3.2, WS3.4, WS4.1) ship with a shim or flag and a CHANGELOG entry.

**Priority:** high. Core scope first (WS0, WS-M, WS1, WS2 Tier A, WS3.1, WS3.4, WS4),
~28 units, 6 weeks. Full scope adds WS2 Tier B, WS3.2–3.7, WS5, WS6, WS7 (10–12 weeks).

**Blocked On:**
- Spec §9.1: Tier A blanket deletion authorization; fourteen Tier B yes/no rows.
- Spec §9.3: architectural authorization for WS3.2, WS3.4, WS5, WS7.1.
- Spec §9.5–9.6: maintainer-only actions (key rotation, closing #72/#69/#49/#32,
  Dependabot merges, default-branch decision, charter amendments).

Executable today without any of the above: WS0 (human), WS-M, WS1.1–1.4 on the current
tree, WS3.1 (a bug fix), WS4.1–4.2, WS4.6.
