# DEV-SQE to Pre-PR: GM-B11, zero-game measurements

- **Date:** 2026-11-23
- **Session:** session_01CXusP9Ct9R7eg25YLcY7R7
- **Branch:** `claude/gm-b11-zero-game-measurements` (contains the close-out branch of #167). **No PR:** #167 and #168 hold both slots.
- **Spec:** `kb/specs/20261123-expressivity-census-preregistration.md`; gap register row B11

## What was built

- `scripts/census_search_expressivity.py` and `scripts/census_measures.py`: the census.
- `alpha_zero/bridge/log_tailer.py::iter_log_states`, `StateConverter.cell_of`, a 0-game entry
  in `alpha_zero/utils/game_budget.py`. No play-path file changed.
- `tests/unit/test_bridge_search_asymmetry_probe.py` with three fixture modules: the probe.
- A1 readiness, checked and not changed: the cards price at 660 / 980 / 980, the registration
  digest agrees in all three places, `make verify-registration` passes; the registered
  arithmetic was re-derived by hand and agrees with the text.

## Who did what

Pre-registration and probe design: `sdlc-ml-scientist`. Script: `sdlc-developer`. Tests, from
the documents: `sdlc-qa-sdet`. Review of the implementation against the pre-registration: its
author. Each round of findings was fixed by the role that owns the file.

## Results

- **Census: none.** Only hand-made states have been run, and the script labels that "NOT a
  census result". It needs a real `Lua.log`: 40 measured turns, 10 after turn 30, both boards,
  `--mod-version` and `--played-by`.
- **Probe: vacuous.** `kb/specs/20261123-three-arm-probe-result.md` has the numbers, the
  wording that may be used, and why no follow-up is registered.

## Checks

Census tests with 27 plus 11 planted defects, all caught; probe controls pass; `ruff`, single
file type checks, debt ratchet, suite wiring, evidence contract, doc-sync. One limit pinned by
a named test: an export that lost its tail joined to one of the same size that lost its head
cannot be told apart without an export id in the log.

## For the maintainer

One command on the next session's log, in `docs/LIVE_CIV6.md` under "Status at a glance".
