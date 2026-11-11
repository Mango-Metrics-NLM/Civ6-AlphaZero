# Draw rate vs board size and turn limit (random play)

> **Pre-fix artifact (superseded).** Generated before the 2026-09-04 search fixes (`e9d0572`: sign-correct value backup, completed-Q policy target, per-iteration seeds). The numbers below are a record of the mechanism and the wall clock, not evidence of strength. See [`docs/evidence/README.md`](../README.md) for the epoch convention.

Generated 2026-09-02 from commit `50144cc` (default-branch head), Python 3.11.15,
torch 2.14.0, numpy 2.4.6, CPU, `OMP_NUM_THREADS=2`. Reproduce with:

```bash
PYTHONPATH=. python docs/evidence/draw_rate_sweep/draw_rate_sweep.py 60 \
    docs/evidence/draw_rate_sweep/draw_rate_sweep.json
```

Every game is a seeded `MicroCivEnv` (seed `10000*board + 100*turns + game`) driven
by uniform random choice over the action mask; the outcome is `env.winner` at
termination and the final city counts as an **unordered pair** (player identity is
dropped, so (1,2) means "one player had one city, the other two"). 60 games per
cell, 720 games, 92.4 s wall.

## Why this was run

`docs/runbooks/promotion-gate.md` measured a 73.5% draw rate at the shipped arena
configuration (`board_size=8`, `max_turns=30`) and *hypothesised* that games are
too short, without measuring it. A code-reading review of `envs/microciv.py` on
2026-09-02 argued the opposite: that draws are a 1–1 city tie forced by geometry
(`GAME.MIN_CITY_DISTANCE = 3` leaves 8 legal second-city tiles on an 8×8 board once
both capitals are placed) and economics (a settler costs 50 production at a flat 3
production per turn, roughly 17 turns), so lengthening games would push *more*
games to 1–1. Both were arguments from code. This is the measurement.

## Result

| board | max_turns | draw rate | most common final city counts, unordered (of 60) | mean actions/game |
|---:|---:|---:|---|---:|
| 8 | 30 | **0.68** | (1,1) 41 · (0,1) 12 · (1,2) 6 | 209 |
| 8 | 60 | **0.47** | (1,1) 25 · (1,2) 20 · (0,1) 7 | 501 |
| 8 | 100 | **0.37** | (1,2) 26 · (1,1) 17 · (0,2) 5 | 939 |
| 10 | 30 | 0.85 | (1,1) 51 · (1,2) 7 · (0,1) 2 | 211 |
| 10 | 60 | 0.52 | (1,2) 27 · (1,1) 25 · (2,2) 6 | 560 |
| 10 | 100 | 0.27 | (1,2) 21 · (2,2) 12 · (1,3) 10 | 1197 |
| 12 | 30 | 0.83 | (1,1) 49 · (1,2) 9 · (2,2) 1 | 211 |
| 12 | 60 | 0.38 | (1,2) 29 · (1,1) 16 · (2,2) 7 | 564 |
| 12 | 100 | 0.30 | (2,2) 13 · (2,3) 13 · (1,3) 12 | 1290 |
| 16 | 30 | 0.87 | (1,1) 52 · (1,2) 8 | 211 |
| 16 | 60 | 0.42 | (1,2) 28 · (1,1) 18 · (2,2) 7 | 598 |
| 16 | 100 | 0.33 | (3,3) 10 · (2,3) 10 · (1,3) 7 | 1413 |

Raw counts: `draw_rate_sweep.json`.

## What it establishes

1. **The mechanism is confirmed.** At 8/30 the mass sits on a (1,1) city tie
   (41 of 60), and the random-play draw rate of 0.68 is within noise of the 0.735
   measured with trained Gumbel-MCTS policies in `docs/evidence/hexcrystal_ab/`.
   Trained play was not expanding better than random play.
2. **The geometry argument is wrong about the lever.** Second cities *are* founded
   on an 8×8 board given time: (1,2) goes from 6/60 at 30 turns to 26/60 at 100
   turns. Draw rate at board 8 falls 0.68 → 0.47 → 0.37 as `max_turns` goes
   30 → 60 → 100. The runbook's hypothesis was directionally right.
3. **Board size is the wrong lever at 30 turns.** Holding `max_turns=30`, a larger
   board makes draws *worse* (0.68 → 0.85 → 0.83 → 0.87), because it takes longer
   to reach a legal city site before the limit.
4. **Cost scales with the lever.** Actions per game grow ~2.4× at 60 turns and
   ~4.5× at 100 (board 8), so every self-play and arena figure in the repo is
   re-costed by the same factor.

## What it does not establish

- Random play is a lower bound on expansion, not a model of trained play. The
  decision in `kb/specs/20260902-validated-learning-plan.md` (WS-D) is to re-run
  this table under a trained checkpoint before changing any shipped default.
- Even at 100 turns, 27–37% of games draw, and the residual draws are still
  (1,1)/(2,2) ties: cities cannot be captured or destroyed in `MicroCivEnv`
  (`_execute_attack` only removes units, `envs/microciv.py:652-702`), so once both
  players hold a city, elimination is impossible and the game is decided by the
  city-count tie-break alone. Turn limit moves the draw rate; it does not remove
  the ceiling.

## A second finding, found in the run's own log

The run emitted **18,103** `Invalid move to (x, y)` warnings across 720 games — ~25
per game averaged over all cells, but that average is dominated by the long 100-turn
and 16×16 games. An independent re-run measured it per cell: **~3 per game at 8/20,
~6 at the shipped 8/30 (3% of actions), ~22 at 8/60**, and every rejection inspected
(5,700 actions) had a unit standing on the target tile. The action mask (`core/civ6_actions.py:629-635`) marks a move legal when
the target tile is on-board and not a mountain; the executor
(`envs/microciv.py:807-818`) additionally rejects tiles occupied by **any unit**,
and on rejection sets the moving unit's `moves_remaining` to 0
(`envs/microciv.py:549-551`). So the mask offers actions that fail, and choosing
one forfeits the unit's turn. Training's masked policy loss and MCTS both treat
that mask as ground truth. Recorded as WS-D2 in the plan; not fixed here.
