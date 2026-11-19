# Tasks

## 1. GM-K1: kernel core
- [ ] 1.1 Tests first: an AST test that `alpha_zero/rules/**` imports only the standard library, `yaml` and `alpha_zero.rules.*`
- [ ] 1.2 `spec.py`, `loader.py`: strict-load fixtures (unknown key, duplicate key, float, bool, negative cost) each raise; `canonical_json()` and `content_sha256()` stable under CRLF and key reordering
- [ ] 1.3 `validate.py`: one failing fixture per rule (duplicate id, unresolved prerequisite, cycle, unreachable capstone, non-positive cost, unresolved unlock), each naming its rule
- [ ] 1.4 `queries.py` (prerequisite closure, `available`, `research_cost` in integer percent) and `registry.py` (declared-order index maps, fingerprint)
- [ ] 1.5 `builtin.py` and `data/science_v1.yaml` as package data; load through `importlib.resources`; a hash mismatch raises
- [ ] 1.6 `__main__.py` with `validate` and `hash`, so every module has a live importer; `dead_modules` in `scripts/measure_debt.py` stays flat
- [ ] 1.7 Hash log in `docs/GAME_RULES.md` and the test holding it equal to `builtin.py`
- [ ] 1.8 `Agent.md` (only existing paths backticked), `docs/MODULE_REFERENCE.md` row; `validate_agents.py` and `check_doc_sync.py` exit 0
- [ ] 1.9 Negative controls recorded: disabling each validator rule fails its fixture test; editing the YAML without re-pinning makes the load raise
- [ ] 1.10 Empty play-path diff; `tests/scripts/test_release_play_path_closure.py` and the debt ratchet pass; coverage at or above 85%

## 2. GM-K2: kernel tooling
- [ ] 2.1 `pacing.py` and `pace`: science-v1 reports capstone path cost 580 and full-tree cost 780
- [ ] 2.2 `gameinfo.py` and `import-gameinfo`: synthetic export fixture to a rule set that passes `validate`; refuses an output path inside the checkout
- [ ] 2.3 CLI exit codes 0 / 1 / 2 tested for every subcommand
- [ ] 2.4 `docs/MODULE_REFERENCE.md` `rules` file count updated

## 3. Close-out
- [ ] 3.1 CLAUDE.md section 5.1 rows for the CLI and a section 5.2 row; `make verify`
