# Knowledge Base (`kb/`)

## What this directory is

`kb/` is the memory of the coding agent that the maintainer uses on this
repository. It carries state between agent sessions in a three-role workflow
(Planning → DEV-SQE → Pre-PR, CLAUDE.md §14): what was decided, what was handed
to whom, and where to resume. It is **not required reading for human
contributors** — everything a person needs is in `CONTRIBUTING.md` and
`docs/README.md`. The dated specs under `specs/` are the one part of this tree a
human might cite, as frozen rationale for a decision; the docs index lists them.

Three things to know before reading anything here:

- **The identifiers are opaque.** `session_…` ids and `claude.ai/code/…` URLs
  name the agent run that wrote an entry. They resolve to nothing outside the
  maintainer's own account and carry no information about the repository; they
  are kept because the ledger schema requires a run id per event and because
  they already appear in commit trailers.
- **Handoffs are immutable, by gate.** `scripts/validate_kb.py --git` (check
  K7) fails if any commit reachable from `HEAD` modifies, deletes or renames a
  file under `handoffs/`. So a handoff that is wrong is *superseded* by a later
  one, never corrected in place — read the newest first.
- **This is not the source for incident status.** Entries here record what a
  session knew at the time. For the current status of the open leaked-credential
  finding, read [`docs/runbooks/secret-scanning.md`](../docs/runbooks/secret-scanning.md);
  for anything else security-related, `SECURITY.md`.

Structure is validated by `scripts/validate_kb.py` (see `make validate`).

## Layout

| Directory | Contents | Mutability |
|-----------|----------|------------|
| `ledger/` | `ledger.jsonl` — append-only event log (one JSON object per line) | append-only |
| `summaries/` | Rolling per-role summaries (append-at-top, capped at 20 entries) | mutable |
| `sessions/` | `<role>.state.json` — resume pointers per role. The four role files (`planning`, `dev-sqe`, `pre-pr`, `phase-6`) are **tracked**; any other `*.state.json` is gitignored scratch (`.gitignore` ignores the pattern and negates those four). | mutable |
| `handoffs/` | Immutable cross-role handoff documents | **immutable** |
| `specs/` | Dated role specs (`YYYYMMDD-slug.md`) | append |
| `pr-drafts/` | Draft PR bodies produced by the Pre-PR role | append |

## Ledger schema (`validate_kb.py`)

Every line MUST be a JSON object with:

- `event` — one of the **eight current** event names, exactly the set in
  `CURRENT_EVENTS` in `validate_kb.py`:
  `SESSION-END`, `PLANNING-HANDOFF`, `DEV-SQE-HANDOFF`, `PR-READY`,
  `DESIGN-GAP`, `QA-GAP`, `SCOPE-CHANGE`, `VALIDATION-RUN`. The last records a
  training or evaluation run that produced committed evidence under
  `docs/evidence/` — the three workflow roles only ever logged process events,
  so without it the ledger could not answer "was this code ever run?".
- `timestamp` — ISO-8601 (a trailing `Z` is accepted).
- `session_id` — required on entries dated on/after the cutoff.

Other keys (`handoff_to`, `spec_file`, `pr_draft`, `files_changed`, `summary`,
`tests_added`, …) are role-specific and optional; any `spec_file` / `pr_draft`
path referenced must exist.

### Grandfathering (frozen)

`LEGACY_CUTOFF = 2026-07-25`. Entries dated **before** the cutoff may use the
retired event names `PLAN-LOCKED` / `DEV-COMPLETE` and may omit `session_id`
(some early entries used `native_session_id` or no session key). This keeps the
historical ledger valid while blocking the same drift in new entries. The cutoff
is a constant in `validate_kb.py` and is not meant to move.

## Handoff immutability

Files under `kb/handoffs/` are immutable once created. `validate_kb.py --git`
fails CI if any commit reachable from `HEAD` modifies, deletes, or renames a
handoff. A genuinely necessary amendment must be added — as a reviewed change —
to the `ALLOWED_AMENDMENTS` allowlist (keyed by path + blob sha) in
`validate_kb.py`; there is deliberately no other escape hatch.

## Session-state schema

`sessions/*.state.json` must be valid JSON. A full schema check is a WARNING
today because `pre-pr.state.json` predates the current schema; it is expected to
carry `last_session_id`, `current_focus`, `resume_pointers`, and `context_files`.
