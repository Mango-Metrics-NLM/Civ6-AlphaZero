# agent-harness delta

## MODIFIED Requirements

### Requirement: Prohibited-action guard

The harness SHALL block prohibited Bash actions before they execute, via a
``PreToolUse`` hook (``.claude/hooks/pretooluse_guard.sh``) that emits a
``permissionDecision`` of ``deny``. The guard MUST cover direct commits or pushes
to ``main``, self-play runs exceeding 1000 games, and force-pushes (``--force`` /
``-f`` / a ``+``-refspec) in any argument position, while still allowing
``--force-with-lease``.

#### Scenario: A force-push is denied

- **WHEN** a Bash tool call runs ``git push origin main --force``
- **THEN** the guard emits a deny decision

#### Scenario: A lease-guarded force-push is allowed

- **WHEN** a Bash tool call runs ``git push --force-with-lease origin feature``
- **THEN** the guard does not deny it

### Requirement: Non-fatal lint autofix

The harness SHALL auto-format edited Python files via a ``PostToolUse`` hook, and
that hook MUST NOT fail the originating tool call nor remove unused imports
(``F401`` is deferred to commit-time gates so an in-progress multi-edit change is
never broken mid-sequence).

#### Scenario: Autofix preserves an as-yet-unused import

- **WHEN** an edit adds an import that a later edit will use
- **THEN** the autofix formats the file but does not strip the import

### Requirement: Permission governance

The harness SHALL declare an explicit allow/deny permission model in
``.claude/settings.json``. The deny list MUST refuse edits to ``.env`` files so
that secret material is not written through the file-editing tools.

#### Scenario: Editing .env is denied

- **WHEN** an ``Edit`` or ``Write`` tool call targets ``.env``
- **THEN** the operation is refused before execution

## ADDED Requirements

### Requirement: Governance validation hook

The harness SHALL run the relevant governance validator when a governed file is
edited, via a ``PostToolUse`` hook that validates only the edited file. A
validation failure MUST surface to the agent without blocking unrelated work.

#### Scenario: Editing a spec file triggers validation

- **WHEN** a ``Write`` or ``Edit`` tool call modifies a file under ``openspec/``
- **THEN** ``validate_openspec.py --path <file>`` runs and any failure is reported

### Requirement: Skill metadata

Every skill under ``.claude/skills/`` SHALL declare YAML frontmatter whose
``name`` matches its directory and whose ``description`` is non-empty.

#### Scenario: A skill without frontmatter is rejected

- **WHEN** a ``SKILL.md`` file lacks a ``name``/``description`` frontmatter block
- **THEN** ``validate_skills.py`` reports a violation and exits 1
