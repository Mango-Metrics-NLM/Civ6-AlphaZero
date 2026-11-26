# DEV-SQE → Pre-PR Handoff

**Date:** 2026-11-20
**Spec:** kb/specs/20261119-game-mechanics-plan.md (slice GM-B3; OpenSpec `openspec/changes/add-bridge-research-parity`, tasks 3.5, 3.5a, 3.6)
**Branch:** `claude/gm-b3-research-wiring`, from the default branch with `claude/gm-b2-exporter-parity` (#150, which holds #148 and #149) and `claude/gm-k2-rules-tooling` (#147, which holds #146) merged in; its PR shows their commits until they merge
**Files Changed:**
- `alpha_zero/bridge/tech_registry.py`
- `alpha_zero/bridge/state_converter.py`
- `alpha_zero/bridge/action_converter.py`
- `alpha_zero/bridge/protocol.py`
- `alpha_zero/bridge/Agent.md`
- `tests/conftest.py`
- `tests/fixtures/research_wiring.py`
- `docs/LIVE_CIV6.md`
- `docs/MODULE_REFERENCE.md`
- `CHANGELOG.md`
- `CLAUDE.md`
- `.env.template`
- `docs/evidence/tech_debt_baseline/baseline.json`
- `openspec/changes/add-bridge-research-parity/tasks.md`
- `openspec/changes/add-bridge-research-parity/specs/civ6-bridge/spec.md`

**Tests Added:** `tests/unit/test_bridge_tech_registry.py` (81), `tests/unit/test_bridge_research_wiring.py`, `..._e2e.py`, `..._review.py` (155 together)
**Test Results:** 236 pass in those four files. `tests/unit` + `tests/integration` (`-m "not slow"`): 12,079 passed, 1 failed (`test_pinned_env.py::test_git_archive_of_head_is_extracted_and_read_only`: the sandbox runs as root; unrelated). `tests/regression`, `tests/governance`, play-path closure, bundle sync pass. Coverage not measured locally (the sandbox's numpy breaks the tracer); CI is the gate.
**Lint Status:** `ruff check .` clean; `ruff format --check` clean on `alpha_zero/` and `tests/`.
**Type Check Status:** `python -m mypy` on the four changed bridge files with `--follow-imports=silent`: no issues. `mypy alpha_zero/` cannot finish in this sandbox; CI is the gate.
**Smoke Tests:** end to end against the mod's Lua on the stand-in: export, mask, choice, command, the game's next frame, end of turn, in both modes and with a GameInfo-shaped rule set.
**Verification not done:** `make verify`; **nothing was run against a real game**; no network has been trained with these slots.
**Play path:** the diff against the default branch over `scripts/release_play_path.txt` is empty. The bridge is the first non-CLI importer of `alpha_zero.rules` and is outside the A1 import closure.
**Debt baseline:** `pytest_skip_sites` 149 -> 150 in the merge commit: #147 and #149 each added one site, as both their handoffs said.

## What changed
- `bridge/tech_registry.py` (new): `ResearchRegistry` gives each technology the slot of its position in a game rule set (`alpha_zero.rules.build_registry`) and each civic the slot of its row in a GameInfo export's `Civics` table. `CIV6AZ_BRIDGE_RULESET`, `CIV6AZ_BRIDGE_GAMEINFO`. A registry that is named and cannot be loaded is an error.
- `StateConverter`: with a registry, `technologies` and `civics` are slot numbers, which the full-game encoder reads. Without one the output is unchanged, counted and warned once.
- `ActionConverter`: offered research sets research bits in the mask; a research action is legal only if offered; with a registry the command names the item by the game's type name, without one the action is the game's row index.
- `protocol.py`: `LuaCommand.item_id` may be a type name, for research commands only.

## Review of record
- Tests by `sdlc-qa-sdet` against a written contract before the code: 134 of 148 failed or errored. After the review the contract changed and 57 failed on the then-current code.
- Negative controls: sending the slot instead of the type name fails 6 end-to-end tests; shifting every encoder slot by one fails 9. The test rule set's slots all differ from the stand-in's row indexes.
- Review by `sdlc-architect` (read-only): no slot/row mix-up found. Findings and what was done:
  1. Registry mode with the built-in rule set offers almost no research and no civics, so it clears a blocked turn worse than the default. **Documented plainly** (`docs/LIVE_CIV6.md`), a start-up warning when a registry has no civics, and a once-only warning naming the first choice that cannot be offered. Not changed: a choice with no slot cannot be offered in registry mode.
  2. Mode and fingerprint were recorded nowhere, and nothing checked the two converters agreed. **Fixed:** both log the mode; the state carries the fingerprint and the action converter refuses another registry's state.
  3. Every live turn counted twice. **Fixed:** a state is counted once.
  4. The counters are not surfaced by the agent loop. **Not fixed:** it needs `agent_loop.py`, a fifth module; the once-only warnings are the interim signal.
  5. A rule set with `mining` and `tech_mining`. **Fixed:** refused.
  6. String `item_id` accepted for every action type. **Fixed:** research only, identifiers only.
  7. Smaller: a non-int item mislabelled (fixed); no way to force no registry (fixed, `False`); tests depended on the developer's shell (fixed in `tests/conftest.py`); the encoder limit is a copied constant (pinned by a test, since the encoder has only a literal).

## Known issues and open questions
- The action space has 73 technology and 56 civic research actions; the current game has more. What does not fit is counted and cannot be chosen. Sizing the action space from the rule set is a play-path change and waits for v2.1.0.
- Without a GameInfo export there are no civic slots. Capturing the export is GM-B4.
- City buildings and districts, the research status and the end-turn blocker are parsed and not used as network inputs.
- Follow-up that needs the maintainer's word (more modules): the converters' `research_stats` in `BridgeAgentLoop.stats()`.
- **Merge order:** #146, #147, #148, #149, #150 before this one.
