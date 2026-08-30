# validation-gates — deltas

## ADDED Requirements

### Requirement: Secret scanning over tree and history

The repository SHALL be scanned for committed secrets by a maintained,
rule-driven scanner covering both the tracked working tree and the full commit
history, because a secret removed in a later commit remains published in the
commit that introduced it. The scanner version MUST have exactly one definition,
which CI reads rather than restates.

#### Scenario: A credential is added to a tracked file

- **WHEN** a commit adds a value matching a secret rule to any non-allowlisted path
- **THEN** the secret-scanning gate fails and names the rule and file, with the value redacted

#### Scenario: The allowlist cannot neuter the gate

- **WHEN** the scanner configuration contains a catch-all allowlist pattern
- **THEN** the repository-hygiene regression suite fails, because a gate that
  allowlists everything reports success while detecting nothing

### Requirement: Deterministic validation of Agent.md persona documents

Every importable package under `alpha_zero/` SHALL carry an `Agent.md` whose
structure is machine-checkable: a heading naming a real module, a persona
statement, the required sections, coverage literals matching the canonical gate,
and file references that resolve.

#### Scenario: A persona document references a file that no longer exists

- **WHEN** an `Agent.md` names a backticked `*.py` path that is absent from both the
  documented module directory and the repository root
- **THEN** the agents validator reports A6 and exits non-zero, unless that line
  carries the `<!-- validate-agents: allow -->` marker

#### Scenario: A new package ships without a persona document

- **WHEN** a directory under `alpha_zero/` contains `__init__.py` but no `Agent.md`
- **THEN** the agents validator reports A5 and exits non-zero

### Requirement: Both skill trees are gated

Skill validation SHALL cover every skill tree the harness routes to it, and MUST
parse YAML block scalars so a folded value is validated on its content.

#### Scenario: A skill outside `.claude/skills/` violates a rule

- **WHEN** a `SKILL.md` under `.agents/skills/` declares a name that differs from its
  directory, or a coverage literal that differs from the canonical gate
- **THEN** the skills validator reports the violation and exits non-zero

### Requirement: Function size is mechanically bounded

A lint gate SHALL bound statements per function. Functions already over the limit
MUST be enumerated individually as a burndown list rather than accommodated by
raising the limit.

#### Scenario: A new oversized function is introduced

- **WHEN** a function outside the burndown list exceeds the configured statement limit
- **THEN** `ruff check .` fails with `PLR0915`
