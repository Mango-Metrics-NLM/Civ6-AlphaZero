# Documentation

## Purpose

TBD.

## Requirements

### Requirement: README accuracy

The README SHALL describe the application as it actually is and MUST NOT reference
removed tooling or nonexistent files (the retired Weights & Biases tracker,
``alpha_zero/utils/wandb_logger.py``, or ``requirements-dev.txt``).

#### Scenario: Doc-sync gate rejects a stale reference

- **WHEN** the README contains a ``wandb`` install instruction
- **THEN** ``python scripts/check_doc_sync.py`` reports a D4 violation and exits 1

### Requirement: Version consistency

The declared version SHALL be consistent across the packaging metadata and the
importable package. ``pyproject.toml`` ``version`` and ``alpha_zero/__init__.py``
``__version__`` MUST be equal.

#### Scenario: Version drift is detected

- **WHEN** ``alpha_zero/__init__.py`` declares a different version than ``pyproject.toml``
- **THEN** ``check_doc_sync.py`` reports a D1 violation and exits 1

### Requirement: Module-inventory truth

``docs/MODULE_REFERENCE.md`` SHALL be the single source of truth for the module
inventory, and its per-module file counts and declared total MUST match the
filesystem.

#### Scenario: A drifted module count is detected

- **WHEN** a module's ``*.py`` count on disk differs from the count in the table
- **THEN** ``check_doc_sync.py`` reports a D2 violation and exits 1

### Requirement: Documentation index

The ``docs/`` tree SHALL provide an index (``docs/README.md``) that lists every
document with a purpose and a status, and every relative link from the README and
that index MUST resolve to an existing file.

#### Scenario: A broken documentation link is detected

- **WHEN** the README links a moved or deleted file with a relative path
- **THEN** ``check_doc_sync.py`` reports a D6 violation and exits 1

### Requirement: Release-changelog consistency

The changelog's first released version heading SHALL match the packaged version.

#### Scenario: Changelog and package version disagree

- **WHEN** the first released ``## [X.Y.Z]`` heading differs from ``pyproject.toml`` ``version``
- **THEN** ``check_doc_sync.py`` reports a D7 violation and exits 1
