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

Every **top-level** package under `alpha_zero/` SHALL carry an `Agent.md` whose
structure is machine-checkable: a heading naming a real module, a persona
statement, the required sections, coverage literals matching the canonical gate,
and file references that resolve.

Scoped to top-level deliberately. `alpha_zero/` holds 29 top-level packages and 44
importable packages in total; the nested 15 are implementation subdivisions
(`hierarchical/goals`, `rag/stores`, and so on) whose ownership is already stated
by their parent's persona document. Requiring a persona doc per nested package
would produce thin files that restate the parent. A substantial new subpackage
still warrants one — `alpha_zero/core/mcts/` has its own.

#### Scenario: A persona document references a file that no longer exists

- **WHEN** an `Agent.md` names a backticked `*.py` path that is absent from both the
  documented module directory and the repository root
- **THEN** the agents validator reports A6 and exits non-zero, unless that line
  carries the `<!-- validate-agents: allow -->` marker

#### Scenario: A new top-level package ships without a persona document

- **WHEN** a direct child directory of `alpha_zero/` contains `__init__.py` but no `Agent.md`
- **THEN** the agents validator reports A5 and exits non-zero

#### Scenario: A `--path` target does not exist

- **WHEN** the validator is invoked with `--path` naming a missing or non-`Agent.md` file
- **THEN** it reports A0 and exits non-zero, rather than silently validating nothing

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
