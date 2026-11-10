# documentation — deltas

## MODIFIED Requirements

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

- **WHEN** the README or ROADMAP contains a forbidden claim pattern such as ``13x`` or ``production-ready``
- **THEN** ``python scripts/check_doc_sync.py`` reports a D4b violation and exits 1

#### Scenario: External links are declared

- **WHEN** the README links an external URL that is not listed in ``docs/public_links.toml``
- **THEN** ``pytest tests/regression/test_public_readiness_regression.py`` fails naming the URL

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

## ADDED Requirements

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
