---
name: debt-audit
description: Audit dead code and tracked tech debt — run the AST import-graph audit, tier what it finds, re-measure the debt metrics, and move the committed baseline in the same change. Use before or after deleting modules, when adding a module, and whenever the debt ratchet fails. Gated by tests/regression/test_debt_ratchet.py.
allowed-tools: Bash, Read, Edit
---

# Debt-Audit Skill

Two commands and one rule: measure before you cut, and never leave the baseline
describing a tree that no longer exists.

## Why this exists

Dead code was counted by hand three times in this project's history and the
three counts disagreed, because "unused" is not greppable — `from pkg import
Name` resolves through a package `__init__` re-export table, and a package
re-exporting its own submodule looks like a use while proving the opposite.
`scripts/audit_import_graph.py` resolves that with an AST import graph, so the
census is a command rather than an opinion.

`scripts/measure_debt.py` then turns the plan's tracked numbers into JSON, and
`tests/regression/test_debt_ratchet.py` holds every one of them to "never worse
than `docs/evidence/tech_debt_baseline/baseline.json`". The failure this
prevents is the ordinary one: a metric drifts up a module at a time, nobody
owns the drift, and the next census re-discovers the same debt at a larger size.

The full plan, including the tier tables, is
`kb/specs/20260902-tech-debt-hardening.md`.

## Steps

### 1. Take the census

```bash
python scripts/audit_import_graph.py            # table, largest module first
python scripts/audit_import_graph.py --json     # machine-readable
python scripts/audit_import_graph.py --package alpha_zero.mechanics   # one package
```

It is an audit, not a gate — it exits 0 whatever it finds. The gate is step 5.

### 2. Read the `kind` column before deleting anything

Every row is a module with **no live importer**, but only one of the three
kinds is a deletion candidate:

- **DEAD** — nothing imports it and nothing names it. A candidate.
- **entry-point** — `__main__`, or carries an `if __name__ == "__main__"`
  guard. It has no importer *by design*; deleting it removes a command.
- **deploy-ref** — its dotted name appears in a Dockerfile, `cloudbuild*.yaml`,
  the `Makefile`, `k8s/`, a shell script, `pyproject.toml` or a config. Wired
  by string, so the import graph cannot see it. The listed file is where to look.

`test_importers` counts test-only importers. A module imported *only* by its own
tests is still dead: the tests go with it.

### 3. Tier each DEAD candidate

The tiers are a decision record, not a size ranking (spec §2.3):

- **Tier A** — superseded, unwired, or scaffolding that no document claims as a
  feature. Deletable under one blanket authorization.
- **Tier B** — dead by graph but named in `CLAUDE.md`, `ROADMAP.md`,
  `docs/MODULE_REFERENCE.md` or a runbook. Deleting is a **product decision**,
  one yes/no per row; the alternative is a test that exercises the documented
  entry point. Do not delete a Tier B module on your own judgement.
- **Tier C** — keep. Dead today, named as the dependency of a planned item.

A newly discovered dead module belongs to whichever tier its documentation
status puts it in — check `CLAUDE.md` and `docs/MODULE_REFERENCE.md` before
assigning Tier A. If a public name is involved, announce the removal in
`CHANGELOG.md` before deleting it.

### 4. Re-measure and move the baseline in the same change

```bash
python scripts/measure_debt.py                  # JSON to stdout
python scripts/measure_debt.py --compare docs/evidence/tech_debt_baseline/baseline.json
```

`--compare` exits 1 if any metric regressed. Once the deletion is real:

```bash
python scripts/measure_debt.py --write docs/evidence/tech_debt_baseline/baseline.json
```

A baseline update belongs in the **same commit** as the change that earned it.
Lowering a number and leaving the baseline high means the ratchet stops holding
that ground; raising a baseline to make a red build green is the one thing this
gate exists to prevent — if a metric genuinely must rise, say why in the PR.

Metrics that need a tool this environment lacks are reported as `null` and
skipped by the ratchet rather than guessed. A `null` where the baseline holds a
number is a missing tool, not an improvement.

### 5. Final gate

```bash
pytest tests/regression/test_debt_ratchet.py -q
```

exits 0. It also fails when the metric *key set* drifts, so a renamed metric
cannot silently drop its own history.

The same gate runs at edit time: `.claude/hooks/posttooluse_governance.sh` fires
it when a file is added under `alpha_zero/` or a Bash command removes one, so a
new dead module surfaces in the session that created it rather than in CI.

## Related

- `verify` — the whole CI gate; the ratchet runs inside its `test` suite.
- `negative-control` — required before claiming any new debt gate works.
- `openspec-propose` — a deletion that changes a capability's behaviour needs a
  change directory first.
