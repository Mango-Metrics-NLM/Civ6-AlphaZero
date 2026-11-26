# DEV-SQE to Pre-PR: GM-B6b, the agent loop runs a policy that answers with a command

- **Date:** 2026-11-22
- **Session:** session_01CXusP9Ct9R7eg25YLcY7R7
- **Branch:** `claude/gm-b6b-search-seam` (contains GM-B6a, PR #162, until that merges). Nothing
  of GM-B6 merges before the maintainer's live sitting.
- **Spec:** `kb/specs/20261120-live-command-layer-plan.md`; OpenSpec `add-search-translator` (tasks 2.x)

## What was built

- `alpha_zero/bridge/command_policy.py`: the `CommandPolicy` protocol, the turn ledger as the
  search's veto (`allowed_by`), a seed per decision, and `choose`, which never raises and
  reports a policy that named nothing or failed.
- `alpha_zero/bridge/agent_loop.py`: a policy whose class defines `select_command` is asked
  through `choose`; the answer goes out through the ordinary send path. New statistics
  `search_decisions`, `search_fallbacks`, `search_failures`.
- `alpha_zero/bridge/offer_filter.py`: `first_to_unblock`, what goes out while the game blocks
  the end of turn and the policy named nothing (research, then other unblocking commands, then
  the rest).
- `alpha_zero/bridge/policy_factory.py`: the two builders moved out of `civ6_local.py` (old
  names kept) and `build_search_policy`.
- `civ6_local --mode civ6 --mcts` is still refused.

## How it was checked

- Tests first by `sdlc-qa-sdet` from a written contract: 119 tests in five files, among them the
  end-to-end chain through the stand-in game (the mod's Lua, real converters, a real Gumbel
  search over a real small network): a unit moved, a city founded, a build started, a turn
  ended, a research blocker answered, a declared stall, the same commands for the same seed.
- Independent review by `sdlc-architect`: four findings, three fixed with tests
  (`tests/unit/test_bridge_search_seam_review.py`, 18 tests), the fourth is a note for the PR
  (one existing test, `test_bridge_search_mode_wiring.py`, was corrected: the failed search is
  now counted and the turn ended through the ordinary path).
- Seventeen planted defects each fail the suite.
- Bridge, Lua, script and regression suites pass apart from seven sandbox-only failures (six
  need a `.env` file, one GCP credentials); governance suite passes; `ruff check .`, single-file
  type checks, doc-sync, kb and OpenSpec validators clean; play-path diff empty. `make verify`,
  the full type check and coverage were not run here.

## Not verified

Nothing ran in a real game. No trained checkpoint exists: this proves plumbing, not strength.
Three of the four end-to-end scenarios narrow the search's choices through the veto.

## Open

- Lifting the refusal: a trained checkpoint, a live check, and the registered comparison
  against a random policy (600 games in three arms, after A1). Not scheduled.
- GM-B8: the simulator contains no other player until the mod exports them (maintainer's yes).
- A wrapper policy that only forwards attributes is not recognised as a command policy.
