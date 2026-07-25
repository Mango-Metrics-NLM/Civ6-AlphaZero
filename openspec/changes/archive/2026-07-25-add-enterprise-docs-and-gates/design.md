# Design: add-enterprise-docs-and-gates

Cross-cutting decisions for this change. Task-level detail lives in ``tasks.md``.

## 1. One umbrella change

All the work ships as one user-sanctioned PR, so it is one OpenSpec change with
its phase structure in ``tasks.md`` rather than several changes that would archive
simultaneously. This is a deliberate deviation from OpenSpec's small-change norm.
``CLAUDE.md``'s "≤20 files per pass" is honored **per phase/commit**; the PR total
(~95 files) is explicitly approved.

## 2. Baseline vs delta

``openspec/specs/`` holds only what is true today (``agent-harness``,
``kb-workflow`` — including the current *buggy* autofix behavior). Everything new
or changed is a delta, applied to the baseline post-merge by ``openspec-archive``.
The new capabilities ``documentation`` and ``validation-gates`` have no baseline
file, so their deltas are ADDED-only; the validator treats a missing baseline as
an empty requirement set (so ADDED-only is valid and MODIFIED/REMOVED against a
missing baseline is an OS022 error).

## 3. Stdlib-only vendored validators

The validators are the binding gate. The upstream npm ``openspec`` CLI is optional
because ``npx ...@latest`` is an unpinned network fetch, which a deterministic gate
must not depend on. Residual risk: the vendored validator may drift from upstream
format evolution; for this repo the vendored one is authoritative. The directory
layout (including ``openspec/AGENTS.md``) stays CLI-compatible.

## 4. Archive, never delete

Stale docs are ``git mv``-d to ``docs/archive/`` with a ``> Archived from <path> on
<date>; superseded by <link>`` header — preserving history and avoiding the
>100-line-deletion confirmation gate. The outgoing 1,495-line README is archived
too. Large superseded docs become 3-line pointer stubs. **Stub criterion:** stub a
file that is referenced by another in-repo doc or by an external PR/issue body;
otherwise move it without a stub.

## 5. Ledger grandfathering

``validate_kb.py`` freezes ``LEGACY_CUTOFF = "2026-07-25"``.
``CURRENT_EVENTS = {SESSION-END, PLANNING-HANDOFF, DEV-SQE-HANDOFF, PR-READY,
DESIGN-GAP, QA-GAP, SCOPE-CHANGE}`` (exactly the three handoff skills' union).
``LEGACY_EVENTS = {PLAN-LOCKED, DEV-COMPLETE}`` are accepted only before the
cutoff. ``session_id`` is required after the cutoff and fully optional before it
(four existing lines have no session key). This passes the 12 existing lines while
blocking new drift.

## 6. CI-enforcement sequencing

Each validator enters CI / ``make validate`` / pre-commit only in the phase where
the repo first passes it (openspec in A, kb in B, skills in C, doc-sync in D).
Every phase's verify battery lists only the gates that are green at that point.

## 7. Hook noise control

The ``PostToolUse`` governance hook validates only the edited file in single-file
mode (``--path``) — syntax/schema of that file only, no cross-file completeness or
cross-reference checks — so mid-scaffold edits do not spam failures. Full-strict
validation runs in pre-commit, CI, and the skill gates. No validator smoke runs in
``SessionStart`` (the governance hook already covers it without session-start
latency).

## 8. Explicit non-goals

- No docs-site generator: ``docs/README.md`` is the navigation layer; revisit if
  the contributor count grows.
- No glossary: ``docs/MODULE_REFERENCE.md`` serves that role.
- Repo-wide lint/format of ``scripts/``, root ``run_*.py``, ``interfaces/``,
  ``demos/``, and ``server-module/`` is out of scope (today's CI lints
  ``alpha_zero/`` only; these carry pre-existing debt) — tracked as follow-up.
- Bringing the legacy ``tests/scripts/`` suites into CI is out of scope (they need
  the ``[dev]`` extra, torch, and marker filtering).
