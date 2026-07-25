# kb-workflow delta

## ADDED Requirements

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

## MODIFIED Requirements

### Requirement: Handoff immutability

Cross-role handoff documents in ``kb/handoffs/`` SHALL be immutable once created;
only additions are permitted. ``scripts/validate_kb.py`` MUST reject any commit
that modifies, deletes, or renames a handoff file unless it is listed in an
explicit, reviewed amendment allowlist.

#### Scenario: Modifying a handoff is rejected

- **WHEN** a commit changes an existing file under ``kb/handoffs/``
- **THEN** ``validate_kb.py --git`` reports a K7 violation and exits 1
