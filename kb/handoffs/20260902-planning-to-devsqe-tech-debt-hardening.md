# Planning → DEV-SQE Handoff

**Date:** 2026-09-02
**Spec:** kb/specs/20260902-tech-debt-hardening.md
**Session:** session_01EcSsbbTYiSkmZsMVyipm7G
**Branch:** `claude/tech-debt-hardening-plan-xrn9sp`

**Acceptance Criteria:** spec §3, all mechanically verifiable:
- coverage omit list ≤ 15 `alpha_zero/` patterns with `cloud/gcp/*` measured, gate ≥ 85%
- ≥ 12,000 lines deleted with the `--seed 42` determinism gate byte-identical
- zero `torch.load` without `weights_only`, enforced by an AST-based security test
- `S101` enforced on `alpha_zero/`; zero runtime asserts remain
- one definition each for `action_space_size`, `promotion_threshold`, env-var prefix,
  each with a regression test that fails on a second definition
- files > 1,000 lines 27 → ≤ 20 (live files only, spec §4 WS5 table)
- one dependency manifest
- open PRs ≤ 3
- every new gate has a recorded negative control

**Constraints:**
- CI is already green; do not "fix" it. Widen what it measures (WS1) first, as a draft
  PR that is expected to go red so the real coverage number is reported.
- No training hyperparameter, threshold value, or network architecture changes.
- No subpackage moves. Decompose by responsibility with permanent re-export shims and a
  patchability test per shim (the #115 method).
- Ratchets only tighten: never raise `max-statements`, never add an omit entry without
  a same-line reason.
- Each PR independently revertible and states its line delta.
- Grep for a module's *string* name before deleting it (dynamic imports are invisible
  to the audit).
- Every new gate ships with its negative control (`negative-control` skill).

**Priority:** high. Order: WS0 → WS1.1 ‖ WS2.1–2.4 ‖ WS4.6–4.7 → WS3.1 → WS4.1–4.2 →
WS2.5 → WS1.2–1.6 ‖ WS3.2–3.7 ‖ WS6.1–6.2 → WS5 (one file per PR) ‖ WS4.3–4.5 ‖
WS6.3–6.7 → WS7.

**Blocked On:**
- Spec §8.1: maintainer authorization to delete the WS2 list (items 1–5, ≈ 9,000
  lines) and a separate yes/no on `alpha_zero/mechanics/` (5,863 lines).
- Spec §8.2: architectural authorization for WS3.2, WS5, WS7.2.
- WS0 (W&B key rotation) is a human action outside the codebase.

Without both authorizations the executable subset is WS0, WS1, WS4.1–4.2, WS4.6.
