---
name: openspec-archive
description: After an OpenSpec change has merged, apply its spec deltas into the baseline and move the change directory into the archive. Run this post-merge, not as part of the change's own PR. Gated by scripts/validate_openspec.py.
allowed-tools: Read, Write, Edit, Bash
---

# OpenSpec Archive Skill

Fold a merged change's deltas into the baseline (`openspec/specs/`) and archive
the change directory. Run **after** the change merges — never inside the change's
own PR.

## Steps

### 1. Apply each delta to the baseline

For every `openspec/changes/<id>/specs/<capability>/spec.md`, update
`openspec/specs/<capability>/spec.md` per delta section:

- **ADDED** — append the `### Requirement:` block. If the baseline file does not
  exist yet, create it with a `## Purpose` heading and a `## Requirements` section.
- **MODIFIED** — replace the existing requirement block: from its
  `### Requirement: <name>` heading through the line before the next `###`/`##`
  heading (or end of file) with the delta's version.
- **REMOVED** — delete that requirement block.

### 2. Move the change to the archive

```bash
git mv openspec/changes/<id> openspec/changes/archive/<YYYY-MM-DD>-<id>
```

Use the archive date. Archived directories are frozen and skipped by the validator.

### 3. Tick any remaining tasks

Mark the change's `tasks.md` items complete (now in its archived location).

## Final gate

The archive is not complete until the full validator passes — this confirms the
baseline is internally consistent and no delta was left dangling:

```bash
python scripts/validate_openspec.py
```

exits 0.
