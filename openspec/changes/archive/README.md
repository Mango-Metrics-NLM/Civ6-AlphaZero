# Archived Changes

Completed OpenSpec changes are moved here after their work has merged **and**
their deltas have been applied to the baseline (``openspec/specs/``).

- Naming: ``YYYY-MM-DD-<change-id>/`` (the date the change was archived).
- Contents are frozen — an archived change is a historical record, not a working
  directory. Directories under ``archive/`` are skipped by
  ``scripts/validate_openspec.py``.
- Archiving is performed by the ``openspec-archive`` skill, which applies each
  delta's ``## ADDED | MODIFIED | REMOVED Requirements`` into the baseline and
  then runs the full validator as its gate.
