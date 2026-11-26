# Documentation

## Purpose

Keeps the project's documentation mechanically truthful: the README, module
inventory, version strings, changelog, and documentation index must match the
code they describe, enforced by the ``check_doc_sync.py`` gate rather than by
convention. Covers accuracy requirements for the README, version consistency
across files, module-inventory counts, the docs index, and release/changelog
agreement.

## Requirements

### Requirement: README accuracy

The README SHALL describe the application as it actually is and MUST NOT reference
removed tooling or nonexistent files (the retired Weights & Biases tracker,
``alpha_zero/utils/wandb_logger.py``, or ``requirements-dev.txt``). It SHALL carry
a project-status section that distinguishes what is delivered from what is
validated by a committed evidence artifact, and it MUST NOT state an
unsupported strength or speed-up claim.

#### Scenario: Doc-sync gate rejects a stale reference

- **WHEN** the README contains a ``wandb`` install instruction
- **THEN** ``python scripts/check_doc_sync.py`` reports a D4 violation and exits 1

#### Scenario: Doc-sync gate rejects an unsupported claim

- **WHEN** the README or ROADMAP contains a forbidden claim pattern such as an unmeasured speed-up multiplier or ``production-ready``
- **THEN** ``python scripts/check_doc_sync.py`` reports a D4b violation and exits 1

#### Scenario: External links are declared

- **WHEN** the README links an external URL that is not listed in ``docs/public_links.toml``
- **THEN** ``pytest tests/regression/test_public_readiness_regression.py`` fails naming the URL

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
non-archived document exactly once with a purpose, an audience and a status, and
every relative link from the README, that index and the governance documents
(``PROJECT_CHARTER.md``, ``ROADMAP.md``, ``CONTRIBUTING.md``, ``SUPPORT.md``,
``SECURITY.md``, ``GOVERNANCE.md``) MUST resolve to an existing file.

#### Scenario: A broken documentation link is detected

- **WHEN** the README or a governance document links a moved or deleted file with a relative path
- **THEN** ``check_doc_sync.py`` reports a D6 violation and exits 1

#### Scenario: An unindexed document is detected

- **WHEN** a Markdown file exists under ``docs/`` outside ``docs/archive/`` and is absent from ``docs/README.md``
- **THEN** ``check_doc_sync.py`` reports a D8 violation and exits 1

### Requirement: Release-changelog consistency

The changelog's first released version heading SHALL match the packaged version.

#### Scenario: Changelog and package version disagree

- **WHEN** the first released ``## [X.Y.Z]`` heading differs from ``pyproject.toml`` ``version``
- **THEN** ``check_doc_sync.py`` reports a D7 violation and exits 1

### Requirement: Public-surface hygiene

The tracked tree SHALL contain no personal email address, no cloud project
number or service-account address, and no storage-bucket literal outside
example configuration, and it MUST NOT contain a binary file over 100 KB that
is not allowlisted with a reason.

#### Scenario: A personal identifier is reintroduced

- **WHEN** a tracked file gains a personal email address or a cloud project number
- **THEN** ``pytest tests/regression/test_public_readiness_regression.py`` fails naming the file

#### Scenario: A large binary is committed

- **WHEN** a tracked file larger than 100 KB with binary content is added outside the allowlist
- **THEN** the public-readiness regression test fails naming the file
