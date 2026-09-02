---
name: doc-sync
description: Detect and fix documentation drift — version strings, module-reference counts, coverage-gate mentions, forbidden references, extras, and broken links. Use after changing code, docs, the version, the module layout, or the coverage gate. Gated by scripts/check_doc_sync.py.
allowed-tools: Read, Write, Edit, Bash
---

# Doc-Sync Skill

Keep the documentation consistent with the code. Run after any change that could
move a version string, a module count, the coverage gate, or a doc link.

## Steps

### 1. Run the checker

```bash
python scripts/check_doc_sync.py
```

### 2. Fix each reported finding

The error code tells you what to fix:

- **D1** — align the version across `pyproject.toml` and
  `alpha_zero/__init__.py` (the packaging version is authoritative). `setup.py`
  is checked too, but only if it exists — it was removed from this branch, so
  do not go looking for it.
- **D2** — update `docs/MODULE_REFERENCE.md` counts and the declared submodule
  count to match the filesystem.
- **D3** — make the coverage-gate number match `pyproject` `fail_under`; historical
  prose is not flagged, but a genuine gate declaration must agree.
- **D4** — remove the forbidden reference from the README (retired `wandb`
  tooling, `requirements-dev.txt`). A single legitimate legacy line may carry a
  `<!-- doc-sync: allow -->` marker.
- **D5** — use a real `pyproject` extra in any `pip install -e '.[X]'`.
- **D6** — fix or remove the broken relative link.
- **D7** — make the first released changelog heading match the packaged version.

### 3. Re-run until clean

## Final gate

```bash
python scripts/check_doc_sync.py
```

exits 0.
