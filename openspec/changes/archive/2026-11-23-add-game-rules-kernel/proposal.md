# Add a data-driven game rules kernel outside the play path

## Why

No tech tree, research or science yield exists anywhere in Python, while the
documentation claims one. The game-mechanics program plan
(`kb/specs/20261119-game-mechanics-plan.md`, slices GM-K1 and GM-K2) starts with
rule definitions as data: validated, hashed and queryable, in a leaf package that
nothing on the release play path imports. It can therefore land before v2.1.0
without staling any evidence run. Later rungs build on it, each under its own change.

## What Changes

- New leaf package `alpha_zero/rules/` (standard library plus `yaml`):
  `spec.py` (frozen, integer-only `TechSpec`, `UnitSpec`, `BuildingSpec`,
  `ModifierSpec`, `VictorySpec`, `RulesetSpec`), `loader.py` (strict load,
  canonical JSON, content hash), `validate.py`, `queries.py`, `registry.py`,
  `builtin.py`, `Agent.md`.
- `alpha_zero/rules/data/science_v1_draft.yaml`: the first-pass rule set of plan
  section 3.3, shipped as package data (`[tool.setuptools.package-data]`) and
  pinned in `builtin.py` as `science-v1-draft` with status `draft`. The id
  `science-v1` is reserved for the calibrated tree, frozen after v2.1.0.
- `python -m alpha_zero.rules` with `validate` and `hash` (GM-K1), then `pace`
  and `import-gameinfo` (GM-K2: `pacing.py`, `gameinfo.py`).
- An append-only hash log in `docs/GAME_RULES.md`, held equal to the frozen
  pins by a test. It stays empty in this change, which adds no frozen rule set.
- A `docs/MODULE_REFERENCE.md` row for `rules` (submodule count 30 to 31).

## Impact

- Affected specs: `game-rules` (new capability).
- Release play path: untouched. No file matched by `scripts/release_play_path.txt`
  changes, no play-path module imports `alpha_zero.rules`, and the A1 import
  closure is unchanged. The `default` preset and every environment are untouched:
  this change adds no rule to any environment and no preset.
- Confirm-first items carried: a new top-level package and a `pyproject.toml`
  package-data entry, both covered by the maintainer's approval of the plan
  (2026-11-19). No network-architecture, distributed, public-API or
  deletion change.
- Not in this change: a science system in the environment, the preset seam,
  per-file play-path pathspecs for the kernel. Each comes after v2.1.0 under its own change.

## Archive note (2026-11-23)

Merged with PRs #146 and #147 (GM-K1, GM-K2). Tasks 1.10 and 3.1 are ticked here: the play-path closure test and the debt ratchet pass on the default branch, and the CLAUDE.md rows exist. `make verify` cannot run in the agent's sandbox; CI is the record.
