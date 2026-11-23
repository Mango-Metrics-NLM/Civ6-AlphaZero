# Tasks

## 1. GM-B6a: the translator
- [x] 1.1 Tests first from a written contract (`sdlc-qa-sdet`)
- [x] 1.2 `bridge/search_translator.py`: translation rules, `root_mask`
- [x] 1.3 `BridgeMCTSEnv`: own ids, root restriction, no-enemy root, busy cities, clock clamp
- [x] 1.4 `Civ6MctsPolicy.select_command` with a decision record
- [x] 1.5 Independent review (`sdlc-architect`) and its findings fixed with tests
- [x] 1.6 Docs: `docs/LIVE_CIV6.md`, `CHANGELOG.md`, `CLAUDE.md`, gap register (GM-B8)

## 2. GM-B6b: the seam
- [x] 2.1 Tests first from a written contract (`sdlc-qa-sdet`)
- [x] 2.2 `bridge/command_policy.py`: the `CommandPolicy` protocol, the ledger as the search's veto, a seed per decision
- [x] 2.3 `BridgeAgentLoop` asks a command policy through it; fallbacks and failures counted
- [x] 2.4 `bridge/policy_factory.py`: the builders moved out of `civ6_local.py`; `build_search_policy`
- [x] 2.5 Un-mocked end-to-end test through the stand-in game; the `--mcts` refusal stays
- [x] 2.6 Independent review (`sdlc-architect`); three findings fixed with tests (`tests/unit/test_bridge_search_seam_review.py`)

## 3. Not in this change
- [ ] 3.1 Lifting the `--mcts` refusal (needs a trained checkpoint, a live check and a registered comparison against a random policy: 600 games in three arms, after A1)
- [ ] 3.2 GM-B8: the mod exports other players' visible units and cities (mod change, maintainer's yes)
