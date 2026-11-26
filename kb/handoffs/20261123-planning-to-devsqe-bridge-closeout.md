# Planning to DEV-SQE: the bridge track, closed out and frozen

- **Date:** 2026-11-23
- **Session:** session_01CXusP9Ct9R7eg25YLcY7R7
- **Branch:** `claude/gm-bridge-true-state` (docs, kb and OpenSpec only)
- **Spec:** gap register `kb/specs/20261119-game-mechanics-gap-register.md`, rows B9, B10, B11

## Why this exists

After GM-B8b merged (#165) the maintainer asked for an objective review of the whole track by
a team. Five roles reviewed in parallel: product owner, scrum master, architect, ML scientist,
AI engineer. Each was asked what is wrong or missing, not whether the work was good.

## What they concluded

1. The bridge was built for a policy that does not exist. Learning has never been shown on the
   corrected search, and A1 has not run. **Bridge features are frozen until it has.** The
   bottleneck is T0b, A1 and A1-R on the maintainer's GPU machine.
2. Thirteen slices were built against a stand-in for the game before the first live contact.
   The maintainer's session of 2026-11-21 (#166) then found five differences. The stand-in's
   defaults still model the old reading (row B10).
3. Four places where the mod goes ahead when it should refuse (row B9): the end-of-turn check
   when the game cannot be asked, a war-check answer that is not a list, the ungated
   declare-war command, attacks on barbarians offered to an agent that is not shown them.
4. Documents contradicted each other and #166. Corrected in this change; one status table in
   `docs/LIVE_CIV6.md` now carries built / stand-in / seen live per capability.
5. Nobody has measured how much of a real position the search's simulator can express
   (row B11). A board-8 network puts about 70 game tiles of a standard map in one cell.
6. The search can lose in its simulator and cannot win there; the predicted effect is avoiding
   contact. A three-arm probe (merged rule, raw simulator, symmetric rule) is proposed.
7. Process: 131 commits since 2026-11-19, 51 of them review fixes and 51 levelling merges.

## What this change did

- Verified the merged head `05b62bb`: the default branch's CI is green; locally the bridge, Lua,
  script, regression and governance suites pass apart from the sandbox-only tests that need
  `.env` or GCP credentials; bundle check, `ruff` and the six validators pass.
- Truth pass over `README.md`, `ROADMAP.md`, `CLAUDE.md`, `docs/LIVE_CIV6.md`,
  `docs/evidence/README.md`.
- Archived six OpenSpec changes: the four bridge ones into `civ6-bridge` (with the refusal
  requirement replaced, the war requirement reworded to what the code does, and #166's
  behaviour recorded), plus the golden fingerprint and the rules kernel. Three stay open
  because their remaining tasks are the maintainer's.
- Not changed, on purpose: anything under `alpha_zero/` or `mod/`. The refusal's message in
  `alpha_zero/local/gameplay_config.py` and a docstring in `bridge/mcts_policy.py` still name
  the missing translator; they go with B9.

## Next, in the approved order

1. **B9 + B10** (one PR, mod and stand-in, tests first by `sdlc-qa-sdet`, independent review).
   Its description carries the list of game calls still unseen live, ranked by what goes wrong.
2. **Zero-game measurements** (scripts and tests): A1 readiness and a run sheet; the census of
   B11 with its stop rule; the three-arm probe.
3. Harness changes, after those: the `negative-control` skill restores from a backup and never
   with `git checkout`; the push guard asks before a push to a branch whose PR is merged.

## Stop rule

After those, nothing is started without the maintainer's word, a red default branch, or a
finding from a live session.
