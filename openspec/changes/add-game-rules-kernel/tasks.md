# Tasks

## 1. GM-K1: kernel core
- [x] 1.1 Tests first: an AST test that `alpha_zero/rules/**` imports only the standard library, `yaml` and `alpha_zero.rules.*`
- [x] 1.2 `spec.py`, `loader.py`: strict-load fixtures (unknown key, duplicate key, float, bool, negative cost) each raise; `canonical_json()` and `content_sha256()` stable under CRLF and key reordering
- [x] 1.3 `validate.py`: one failing fixture per rule (duplicate id, unresolved prerequisite, cycle, unreachable capstone, non-positive cost, unresolved unlock), each naming its rule
- [x] 1.4 `queries.py` (prerequisite closure, `available`, `research_cost` in integer percent) and `registry.py` (declared-order index maps, fingerprint)
- [x] 1.5 `builtin.py` (id, resource, sha256, status) and `data/science_v1_draft.yaml` as package data under id `science-v1-draft`, status `draft`; load through `importlib.resources`; a hash mismatch raises; status `draft` exactly when the id ends in `-draft`
- [x] 1.6 `__main__.py` with `validate` and `hash`, so every module has a live importer; `dead_modules` in `scripts/measure_debt.py` stays flat
- [x] 1.7 Hash log in `docs/GAME_RULES.md` (no row in GM-K1) and the test holding it equal to the frozen entries of `builtin.py`, with injected-table cases (missing row, extra row, hash mismatch, draft row) so it is not vacuous while the log is empty
- [x] 1.8 `Agent.md` (only existing paths backticked), `docs/MODULE_REFERENCE.md` row; `validate_agents.py` and `check_doc_sync.py` exit 0
- [x] 1.9 Negative controls recorded: disabling each validator rule fails its fixture test; editing the YAML without re-pinning makes the load raise; dropping the draft exclusion from the log comparison fails the draft-row case
- [ ] 1.10 Empty play-path diff; `tests/scripts/test_release_play_path_closure.py` and the debt ratchet pass; coverage at or above 85%

## 2. GM-K2: kernel tooling
- [x] 2.1 `pacing.py` and `pace`: science-v1-draft reports capstone path cost 580 and full-tree cost 780
- [x] 2.2 `gameinfo.py` and `import-gameinfo`: synthetic export fixture to a rule set that passes `validate`; refuses an output path inside the checkout
- [x] 2.3 CLI exit codes 0 / 1 / 2 tested for every subcommand
- [x] 2.4 `docs/MODULE_REFERENCE.md` `rules` file count updated

## 3. Close-out
- [ ] 3.1 CLAUDE.md section 5.1 rows for the CLI and a section 5.2 row; `make verify`
