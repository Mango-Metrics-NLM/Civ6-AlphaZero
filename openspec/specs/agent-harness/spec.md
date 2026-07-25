# agent-harness Specification

## Purpose

The agent harness is the machine-enforced layer that governs how Claude Code
operates in this repository: the permission model and the lifecycle hooks defined
in ``.claude/settings.json`` and ``.claude/hooks/``. This baseline describes the
behavior that exists **today**; changes to it are tracked as deltas.

## Requirements

### Requirement: Prohibited-action guard

The harness SHALL block prohibited Bash actions before they execute, via a
``PreToolUse`` hook (``.claude/hooks/pretooluse_guard.sh``) that emits a
``permissionDecision`` of ``deny``. The guard MUST cover, at minimum, direct
commits or pushes to ``main`` and self-play runs exceeding 1000 games.

#### Scenario: Push to main is denied

- **WHEN** a Bash tool call contains ``git push origin main`` while the current
  branch is ``main``
- **THEN** the guard emits a deny decision and the command does not run

#### Scenario: Oversized self-play run is denied

- **WHEN** a Bash tool call runs an ``alpha_zero`` entry point with
  ``--num_games 5000``
- **THEN** the guard emits a deny decision citing the 1000-game cap

### Requirement: Session bootstrap

The harness SHALL verify the interpreter and prepare the environment at session
start, via a ``SessionStart`` hook. The hook MUST require Python 3.11 or newer and
install the development extra so tests and linters are runnable.

#### Scenario: Environment is bootstrapped at session start

- **WHEN** a new Claude Code session begins
- **THEN** the hook confirms Python >= 3.11 and installs ``.[dev]``

### Requirement: Non-fatal lint autofix

The harness SHALL auto-format edited Python files via a ``PostToolUse`` hook, and
that hook MUST NOT fail the originating tool call even when the formatter or
linter reports a problem.

#### Scenario: Autofix runs after a Python edit

- **WHEN** a ``Write`` or ``Edit`` tool call modifies a ``.py`` file
- **THEN** ruff formatting is applied and the tool call still succeeds

### Requirement: Permission governance

The harness SHALL declare an explicit allow/deny permission model in
``.claude/settings.json`` so that common safe operations run without prompting and
dangerous operations are refused.

#### Scenario: A denied operation is refused

- **WHEN** a Bash tool call matches a ``deny`` permission rule
- **THEN** the operation is refused before execution
