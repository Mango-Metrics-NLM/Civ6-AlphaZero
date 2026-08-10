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
only additions are permitted. ``scripts/validate_kb.py`` MUST reject any commit
that modifies, deletes, or renames a handoff file unless it is listed in an
explicit, reviewed amendment allowlist.

#### Scenario: Modifying a handoff is rejected

- **WHEN** a commit changes an existing file under ``kb/handoffs/``
- **THEN** ``validate_kb.py --git`` reports a K7 violation and exits 1

### Requirement: Ledger schema validation

The ledger SHALL be schema-validated by ``scripts/validate_kb.py``. Every line
MUST parse as JSON and carry an ``event`` drawn from the current event set and an
ISO-8601 ``timestamp``; legacy event names and missing session keys are accepted
only for entries dated before a frozen cutoff.

#### Scenario: A malformed ledger line is rejected

- **WHEN** a line in ``kb/ledger/ledger.jsonl`` is not valid JSON
- **THEN** ``validate_kb.py`` reports a K1 violation and exits 1

#### Scenario: A new entry with a legacy event name is rejected

- **WHEN** a post-cutoff entry uses a legacy event name
- **THEN** ``validate_kb.py`` reports a K3 violation and exits 1
