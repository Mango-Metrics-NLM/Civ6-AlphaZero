# DEV-SQE to Pre-PR: GM-B6a, the search's answer becomes an offered command

- **Date:** 2026-11-21
- **Session:** session_01CXusP9Ct9R7eg25YLcY7R7
- **Branch:** `claude/gm-b6a-search-translator` (contains GM-B7c-1 #160 and GM-B7c-2 #161). No PR:
  both slots are taken, and nothing of GM-B6 merges before the maintainer's live sitting.
- **Spec:** `kb/specs/20261120-live-command-layer-plan.md`; OpenSpec `add-search-translator`

## What was built

- `alpha_zero/bridge/search_translator.py`: one simulator action to one command the game
  offered, or a reason. Units and cities by live id; a move to the offered tile nearest the
  destination cell; an attack only into that cell (melee offer, else ranged); production by
  simulator class.
- `alpha_zero/bridge/mcts_policy.py`: `BridgeMCTSEnv` restricts the search's root to what
  translates, does not end a game in which nobody else is seeded, seeds a building city as
  busy and clamps the simulator's clock; `Civ6MctsPolicy.select_command` acts on the search's
  halving winner and returns a record of the decision.
- The agent loop and the `--mcts` refusal are unchanged (GM-B6b).

## How it was checked

- Tests first by `sdlc-qa-sdet` from a written contract (152 tests); they found two defects in
  the first implementation (a loss penalty at a root with nobody in sight; units found by id
  alone, while ids are per owner).
- Independent review by `sdlc-architect`: no way to send an un-offered command; seven findings,
  six fixed with tests (`tests/unit/test_bridge_search_review.py`, 8 fail on the pre-review code),
  one registered as GM-B8 (below).
- Twenty-three planted defects each fail the suite.
- Bridge, Lua and script suites 4,656 passed, 26 skipped; regression 471 passed; `ruff check .`,
  single-file type checks, doc-sync and OpenSpec validators clean; play-path diff empty.
  `make verify`, the full type check and coverage were not run; no CI has run (no PR).

## Not verified

Nothing ran in a real game. No trained checkpoint exists: this proves plumbing, not strength.

## Open

- **GM-B8:** with the mod's default settings only the acting player's own units and cities are
  exported, so the search's simulator never contains another player. It needs a mod change and
  the maintainer's yes.
- GM-B6b: wire the agent loop to `select_command` (pass the turn ledger as `allowed`), and an
  un-mocked end-to-end test through the stand-in.
- With no `decision_seed` each call still builds its own search object; cost is not measured.
