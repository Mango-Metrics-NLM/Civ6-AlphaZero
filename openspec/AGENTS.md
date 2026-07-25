# OpenSpec Agent Instructions

> How an AI agent (or a human) drives a spec-managed change in this repository.
> This mirrors the `Fission-AI/OpenSpec <https://github.com/Fission-AI/OpenSpec>`_
> workflow; the vendored ``scripts/validate_openspec.py`` is the binding gate.

## The three-stage workflow

1. **Propose.** Create ``openspec/changes/<change-id>/`` with a ``proposal.md``
   (``## Why`` + ``## What Changes`` + ``## Impact``), a ``tasks.md`` checklist,
   an optional ``design.md`` for cross-cutting decisions, and spec deltas under
   ``specs/<capability>/spec.md``. Use the ``openspec-propose`` skill to scaffold
   this. The change-id is verb-led kebab-case (e.g. ``add-enterprise-docs-and-gates``).
2. **Implement.** Work the ``tasks.md`` checklist, ticking items as they land.
   Each substantive commit should keep the repo green under the gates active at
   that point.
3. **Archive.** After the change merges, apply its deltas into the baseline
   (``openspec/specs/``) and move the change directory to
   ``openspec/changes/archive/YYYY-MM-DD-<change-id>/``. Use the
   ``openspec-archive`` skill; it runs the full validator as its gate.

## When a change directory is required

- **Required** for anything that alters a capability's behavior, adds a
  requirement, or changes a governance gate — i.e. work a reviewer should be able
  to read as a spec before reading the diff.
- **Not required** for a direct edit that changes no requirement: typo fixes,
  dependency bumps, comment/docstring touch-ups, or purely mechanical moves.

## Baseline vs delta

``openspec/specs/`` holds only what is **true today**. A change never edits the
baseline directly; it records ``## ADDED | MODIFIED | REMOVED Requirements`` in
its delta files, and the archive step applies them. A brand-new capability has no
baseline file yet, so its delta is ADDED-only.

## The gate

```bash
python scripts/validate_openspec.py                 # whole tree
python scripts/validate_openspec.py --change <id>   # one change
python scripts/validate_openspec.py --path <file>   # single file (hook mode)
```

Exit 0 = pass, 1 = validation errors, 2 = usage error. A proposal is not ready to
implement, and a change is not ready to archive, unless the validator exits 0.
