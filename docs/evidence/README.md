# Evidence index

Committed measurements, with the one fact a reader needs before quoting any of
them: whether the search had the 2026-09-04 correctness fixes when the number was
produced. Contract: [`SCHEMA.md`](SCHEMA.md).

| Artifact | Generated (commit) | Search epoch | Seeded | What it supports | Status |
|---|---|---|---|---|---|
| [`draw_rate_sweep/`](draw_rate_sweep/report.md) | 2026-09-02 (`50144cc`) | pre-fix | per cell | draw rate under *random* play vs board size and turn limit (0.68 / 0.47 / 0.37 at 30 / 60 / 100 turns on 8×8) | current for random play; re-measure under trained post-fix play is open |
| [`hexcrystal_ab/`](hexcrystal_ab/ab_report.md) | 2026-08-10 (`b14534a`) | pre-fix | 42 (280/800 arena games reused training seeds) | the A/B harness runs and is reproducible | **superseded** — not evidence that either encoder is stronger |
| [`seeded_probe_reproducibility/`](seeded_probe_reproducibility/README.md) | 2026-08-30 (`0b75871`) | pre-fix | yes | seeded probes are identical; unseeded probes spread 0.30–0.45 | current (reproducibility only) |
| [`validation_iteration_zero/`](validation_iteration_zero/validation_iteration_zero_report.md) | 2026-08-28 (`0c394f9`) | pre-fix | no | the local loop runs end to end on CPU; the checkpoint scores below parity against random | **superseded** as a learning measurement |
| [`windows_baseline_20261111/`](windows_baseline_20261111/README.md) | 2026-11-11 (`0eb7596`) | post-fix | 20261111 | machine snapshot (CPU, RAM, GPUs, Windows Update policy) and a harness smoke showing the calibration script runs; no throughput or ETA claim | current (harness smoke only; calibration is T0b) |
| [`target_sharpness_probe/`](target_sharpness_probe/report.md) | 2026-11-12 (`4edefa5`) | post-fix | 0–7, 42, 1234 | sharpness of the Gumbel policy target (entropy, max-p, share above `greedy_threshold`) for untrained networks at A1's k=8 / n=24: current settings, the paper's Go/chess constants under an affine and a per-node min-max reading, and `mctx`'s defaults; 0 games played | current (untrained networks only; input to the LF-A5/A12 decision) |
| [`tech_debt_baseline/`](tech_debt_baseline/baseline.json) | rolling | n/a | n/a | code-metric ratchet (`tests/regression/test_debt_ratchet.py`) | current |

No committed artifact measuring playing strength was generated on post-2026-09-04 code (`windows_baseline_20261111/` records a machine snapshot and a harness smoke only; `target_sharpness_probe/` measures search targets of untrained networks and plays no games). The first one is
the A1 learning run of milestone v2.1, pre-registered in `docs/EVALUATION.md` section 7.1.

## Recorded without a committed artifact

| What | When | Where it is written down | What it supports | What it does not |
|---|---|---|---|---|
| First live session against the real game, turns 1 to 7 | 2026-11-21 | the maintainer's report of the session, summarised in [`../LIVE_CIV6.md`](../LIVE_CIV6.md); PR #166 records the code changes that followed, not the session itself | the socket connection, the schema-2 export and the GameInfo capture work; the game acknowledged `research_tech`, `unit_move`, `unit_skip`, `unit_fortify`, `unit_found_city` and a warrior build; five behaviours of the game differ from what the mod assumed | the effect of each command in the following state; that the five fixes work (the rebuilt mod has not been run); anything about attacks, other players, several turns in a row, or how well an agent plays. No log excerpt is committed |

## Never measured

- Playing strength on post-2026-09-04 code (A1 is the first, not yet run).
- Anything about play in the real game: no trained policy has been loaded there.
- Whether a decision made by the search in its MicroCiv stand-in for a live state is a good
  decision in the real game, or how much of a real position that stand-in can express.
- How long one state export takes in the middle of a game.
