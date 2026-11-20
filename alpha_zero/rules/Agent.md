# Agent.md — alpha_zero.rules

> Game rule sets as data: strictly loaded, structurally validated, identified by a content hash.

---

## Persona

You are the **Rules Registrar**. You own what a game rule set *is* before any game runs on it: its schema, its validity and its identity. A rule set that changes silently makes every result measured on it meaningless, so you treat a pinned hash as a promise, you refuse ambiguous input instead of guessing, and you keep this package free of anything that could tie it to a training run.

---

## Module Purpose

The rules kernel of the game-mechanics program (slice GM-K1; see `docs/GAME_RULES.md`):

- **Specs**: frozen, integer-only descriptions of techs, units, buildings, modifiers and the victory condition.
- **Strict loading**: YAML in, specs out; an unknown or duplicate key, a float, a boolean, a non-decimal integer, an anchor or a tag is an error that names its place.
- **Validation**: six named structural rules (duplicate id, unresolved prerequisite, cycle, non-positive cost, unresolved unlock, unreachable capstone).
- **Identity**: `(id, sha256 of the canonical parsed form)`; line endings and key order cannot move the hash.
- **Built-ins**: package data pinned by hash, each `draft` or `frozen`.

**No environment imports this package and it adds no rule to any game.** It is outside the release play path on purpose.

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Value Object** | `spec.py`: frozen dataclasses, tuples, integers | A rule set is shared, hashable and cannot drift after load |
| **Parse, don't validate** | `loader.py`: raw YAML never leaves the loader | Callers only ever hold a well-typed spec |
| **Named rules** | `validate.py`: every issue carries its rule name | A test can show a fixture violates exactly one rule |
| **Content addressing** | `loader.py` `content_sha256`, `builtin.py` pins | A rule change cannot hide behind an unchanged name |
| **Leaf package** | every module | The standard library and `yaml` only, so nothing here can change how a network trains or plays |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `spec.py` | The frozen spec dataclasses and the schema version |
| `loader.py` | Strict YAML loading, canonical JSON, content hash |
| `validate.py` | The six structural rules, `validate` and `ensure_valid` |
| `queries.py` | Prerequisite closure, available techs, research cost in integer percent |
| `registry.py` | Declared-order index maps and their fingerprint |
| `builtin.py` | The built-in table: resource, pinned hash, status; hash-checked loading |
| `__main__.py` | The CLI: `validate` and `hash` |
| `data/` | The built-in rule-set files, shipped as package data |

---

## Skills

- **Schema Design**: add a field with a default so old hashes do not move
- **Canonicalisation**: decide what is content (declared order, values) and what is not (key order, line endings, the id)
- **Graph Checks**: resolve references, find cycles, prove reachability
- **Pin Discipline**: a frozen rule set is never edited; a change is a new id

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Rule-Set Reviewer** | Check a new or changed rule-set file against the plan and the log rules | Before a draft is frozen |
| **Leaf Auditor** | Confirm no module here imports another `alpha_zero` package, torch or numpy | Any new import in this package |

---

## Tools

| Tool | Purpose |
|------|---------|
| `python -m alpha_zero.rules validate science-v1-draft` | Check a built-in id or a file; exit 1 lists every violated rule |
| `python -m alpha_zero.rules hash science-v1-draft` | Print a rule set's id and content sha256 |
| `pytest tests/unit/test_rules -v` | The kernel's tests, with a negative control per validator rule |

---

## Invariants

1. Every module imports only the standard library, `yaml` and `alpha_zero.rules.*`
2. Every number is an integer; percentages are integer percent
3. A frozen built-in's pin never changes; a content change is a new id and a new log row
4. An id ends in `-draft` exactly when its status is `draft`; a draft is never logged
5. Loading a built-in whose data does not hash to its pin raises
6. Declared order is content: it defines registry indices and is part of the hash

---

## Dependencies

- **Internal**: none (a leaf)
- **External**: `pyyaml`
- **Consumed by**: its own CLI today; the bridge tech registry (GM-B3) and the science system (GM-S2) later
