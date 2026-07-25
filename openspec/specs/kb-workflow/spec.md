# kb-workflow Specification

## Purpose

The knowledge base (``kb/``) provides session continuity across the three-role
workflow pipeline — Planning → DEV-SQE → Pre-PR — through an append-only event
ledger, rolling summaries, mutable session state, and immutable cross-role
handoff documents. This baseline describes the behavior that exists **today**.

## Requirements

### Requirement: Append-only ledger

The workflow SHALL record role events in ``kb/ledger/ledger.jsonl`` as one JSON
object per line. Entries MUST only ever be appended; existing lines are never
rewritten or truncated.

#### Scenario: A handoff appends a ledger entry

- **WHEN** a handoff skill records a role event
- **THEN** exactly one valid JSON line is appended to ``kb/ledger/ledger.jsonl``
  and no prior line is modified

### Requirement: Session-end handoff

Each workflow role SHALL run its handoff skill before ending a session that did
meaningful work. The skill MUST update the role summary, append a ledger event,
and rewrite the role's session-state file.

#### Scenario: A role ends a working session

- **WHEN** the DEV-SQE role finishes a session that changed code
- **THEN** its summary, ledger, and session-state file are all updated

### Requirement: Handoff immutability

Cross-role handoff documents in ``kb/handoffs/`` SHALL be immutable once created;
a handoff file is only ever added, never modified in place.

#### Scenario: A handoff document is created

- **WHEN** a role produces a cross-role handoff
- **THEN** a new file is created under ``kb/handoffs/`` and no existing handoff
  file is changed
