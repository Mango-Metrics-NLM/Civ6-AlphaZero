# Tasks

## 1. GM-B6a: the translator
- [x] 1.1 Tests first from a written contract (`sdlc-qa-sdet`)
- [x] 1.2 `bridge/search_translator.py`: translation rules, `root_mask`
- [x] 1.3 `BridgeMCTSEnv`: own ids, root restriction, no-enemy root, busy cities, clock clamp
- [x] 1.4 `Civ6MctsPolicy.select_command` with a decision record
- [x] 1.5 Independent review (`sdlc-architect`) and its findings fixed with tests
- [x] 1.6 Docs: `docs/LIVE_CIV6.md`, `CHANGELOG.md`, `CLAUDE.md`, gap register (GM-B8)

## 2. Not in this change
- [ ] 2.1 GM-B6b: the agent loop takes commands from the search; un-mocked end-to-end test (the `--mcts` refusal stays)
- [ ] 2.2 GM-B8: the mod exports other players' visible units and cities (mod change, maintainer's yes)
