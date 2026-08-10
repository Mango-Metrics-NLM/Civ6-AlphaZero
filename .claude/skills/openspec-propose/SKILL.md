---
name: openspec-propose
description: Scaffold a new OpenSpec change directory (proposal, tasks, design, and spec deltas) before implementing a substantive change. Use when a change alters a capability's behavior, adds a requirement, or touches a governance gate. Gated by scripts/validate_openspec.py.
allowed-tools: Read, Write, Edit, Bash
---

# OpenSpec Propose Skill

Scaffold `openspec/changes/<change-id>/` so a substantive change is specified
before it is implemented. See `openspec/AGENTS.md` for the full workflow.

## When to use

- **Use** when the work alters a capability's behavior, adds/changes a
  requirement, or changes a governance gate.
- **Skip** for direct edits that change no requirement (typos, dependency bumps,
  comment/docstring touch-ups, mechanical moves).

## Steps

### 1. Pick a change-id

Verb-led kebab-case, matching `^[a-z0-9][a-z0-9-]*$` (e.g. `add-x`, `fix-y`).

### 2. Scaffold the directory

Create `openspec/changes/<change-id>/` with:

- `proposal.md` — `## Why`, `## What Changes`, `## Impact`.
- `tasks.md` — a numbered checklist of `- [ ]` items.
- `design.md` — only if there are cross-cutting decisions to record.
- `specs/<capability>/spec.md` — one delta file per affected capability.

### 3. Write the spec deltas

Each delta uses `## ADDED Requirements`, `## MODIFIED Requirements`, and/or
`## REMOVED Requirements`. Every ADDED/MODIFIED requirement needs a
`### Requirement:` heading with a SHALL/MUST statement and at least one
`#### Scenario:` with `- **WHEN**` / `- **THEN**` lines. A MODIFIED/REMOVED name
must exist verbatim in the baseline `openspec/specs/<capability>/spec.md`; a
brand-new capability has no baseline, so its delta is ADDED-only.

## Final gate

The proposal is not ready to implement until:

```bash
python scripts/validate_openspec.py --change <change-id>
```

exits 0.
