# Knowledge Base (`kb/`)

Session continuity for the Planning → DEV-SQE → Pre-PR workflow (CLAUDE.md §14).
Structure is validated by `scripts/validate_kb.py` (see `make validate`).

## Layout

| Directory | Contents | Mutability |
|-----------|----------|------------|
| `ledger/` | `ledger.jsonl` — append-only event log (one JSON object per line) | append-only |
| `summaries/` | Rolling per-role summaries (append-at-top, capped at 20 entries) | mutable |
| `sessions/` | `<role>.state.json` — resume pointers per role (gitignored) | mutable |
| `handoffs/` | Immutable cross-role handoff documents | **immutable** |
| `specs/` | Dated role specs (`YYYYMMDD-slug.md`) | append |
| `pr-drafts/` | Draft PR bodies produced by the Pre-PR role | append |

## Ledger schema (`validate_kb.py`)

Every line MUST be a JSON object with:

- `event` — one of the **current** event names:
  `SESSION-END`, `PLANNING-HANDOFF`, `DEV-SQE-HANDOFF`, `PR-READY`,
  `DESIGN-GAP`, `QA-GAP`, `SCOPE-CHANGE`.
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
