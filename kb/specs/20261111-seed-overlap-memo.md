# Memo: S5 seed overlap, and whether A1's streams can overlap (LF-A17)

**Status:** findings + recommendation · **Date:** 2026-11-11 · **Gap row:** LF-A17
("S5 seed-overlap memo", `kb/specs/20261111-local-first-gap-register.md`) · **Plan:**
`kb/specs/20261111-local-first-and-branching-plan.md` §5.4 (task T12) · **Owner:** ml-scientist
· **Authorization:** autonomous (A); the re-run-vs-annotate call in part 1 was maintainer-only
under the source plan.

---

## 0. The question, from its source

`kb/specs/20260904-correctness-first-plan.md` §3, success criterion **S5**:

> A decision memo on `docs/evidence/hexcrystal_ab/`: its arena seeds overlap its training seeds
> (reported overlap 280/800 games). Either re-run on disjoint seeds or annotate the artifact and
> every document quoting its +31.4 Elo. **Gate:** the artifact carries a train/test
> seed-disjointness statement.

and §4 row S5: "Re-run or annotate the A/B evidence for train/test seed overlap — maintainer
decides re-run vs annotate."

T12 extends the same question to the run this program depends on: **can A1's seeds overlap
across streams** (self-play vs arena maps vs evaluation), and can the exploratory seeds
1001–1003 collide with the confirmatory seeds 20261111 / 20261112 through `derive_seed`? Part 1
answers S5 as written; part 2 answers the A1 question.

## 1. S5 as written: the encoder A/B (`docs/evidence/hexcrystal_ab/`)

### 1.1 Current state

The annotate branch has already been taken. `docs/evidence/README.md` lists the artifact as
**superseded**, with the seed column reading "42 (280/800 arena games reused training seeds)";
`hexcrystal_ab/ab_report.md:52` and `summary_table.md:9` carry the statement "280 of the 800
evaluation games reused training seeds" next to the two search defects that also void it. The
S5 gate (a seed-disjointness statement on the artifact) is met in its negative form: the
artifact says its seeds were *not* disjoint. Outside `kb/`, the +31.4 Elo figure now appears only in
the artifact's own tables (`ab_report.md:18`, `summary_table.md:7`), under its superseded banner.

### 1.2 The 280/800 figure, re-derived

At the A/B's commit (`b14534a`):

- training: `scripts/run_encoder_ab.py:376` seeds iteration `it` with
  `seed + it * games_per_iter`, and `cloud/gcp/selfplay_worker.py:720` seeds game `i` with
  that `+ i`. At seed 42, 20 iterations × 14 games, the training maps are seeds **42..321**.
- arena: `cloud/gcp/arena.py` at `b14534a` (line 139) seeded game `g` with `seed + g`, so the
  800 arena games used seeds **42..841**.

Overlap: seeds 42..321 = **280 of 800 games**, as reported.

### 1.3 The defect is worse in today's code, not fixed

`arena.py:291` now seeds the map of **pair** `p` with `seed + p` (both seats play one map), and
`run_encoder_ab.py` is unchanged. Re-running the A/B as it stands at seed 42 would put training
maps 42..321 on 280 of the 400 arena pairs: **560 of 800 games** would reuse a training map.

Two further hazards apply to the planned re-run (LF-A9: "≥3 seeds, each arm ≤1k"):

- **Adjacent seeds share almost every arena map.** On the arithmetic path, seed `s` uses arena
  maps `s .. s+P-1`, so seeds 42/43/44 share 99 of 100 maps pairwise (measured for 1001/1002/1003
  and 20261111/20261112 below). Three "independent" seeds would be scored on one map set.
- **Adjacent seeds share training games.** `seed + it * games_per_iter + i` from seeds 42 and 43
  overlaps in 279 of 280 game seeds.

### 1.4 Recommendation for S5

Keep the annotation (it is accurate). **Do not re-run the A/B with `run_encoder_ab.py` as it
stands.** Before LF-A9 runs, move both of its seed sites to named streams: self-play through
`derive_seed(seed, <ab self-play stream>, iteration, game)` and the arena through
`run_arena(..., env_seeds=[derive_seed(seed, <ab arena stream>, p) ...])` (the benchmark already
does exactly this, `evaluation/benchmark.py:231-238`), with a test that the training and arena
map sets are disjoint and that two different base seeds share no arena map. That is a small
change to one script; it belongs to LF-A9 (v2.3), not to v2.1.

## 2. A1's streams

### 2.1 Where each stream's seed comes from

A1 as registered (plan §5.2): `training_civ6 --seed 20261111 --board_size 8 --max_turns 60
--num_iterations 20 --num_games 24 --mcts gumbel --gumbel_k 8 --gumbel_sims 24 --bootstrap none
--eval_games 0`, then `benchmark_suite.py --seed 20261111` for every arena arm.

| Stream | Seed of the map for one game | Code |
|---|---|---|
| self-play map, iteration `it`, game `g` | `derive_seed(S, "selfplay.env", it, g)` | `training_civ6.py:905` (env built at `:856`) |
| self-play search | `derive_seed(S, "selfplay.search", it)` | `training_civ6.py:862` |
| in-loop evaluation map | `derive_seed(S, "eval.env", g)` — **off in A1** (`--eval_games 0`) | `training_civ6.py:1100` |
| benchmark (arena) map, pair `p` | `derive_seed(S, "bench.env", p)` | `evaluation/benchmark.py:238`, passed as `env_seeds` at `:451/479` |
| benchmark searches / baseline opponents | `derive_seed(S, "arena.search" \| "arena.opponent", game, side)` | `cloud/gcp/arena.py:263, 273` |
| network initialisation | `torch.manual_seed(S)` | `training_civ6.py:1405` |

`derive_seed` (`utils/seeding.py:76-113`) hashes `"S|stream|coords"` with BLAKE2b and reduces
it modulo 2^31. Two seeds coincide only if two payloads collide in 31 bits; streams with
different names cannot be *made* to coincide by any choice of coordinates. A map is a pure
function of its integer seed (`MicroCivEnv.reset` re-creates `default_rng(seed)` and
`_generate_map` draws from it, `envs/microciv.py:274-276, 359-376`); the training path
(`reset(seed=s)`) and the arena path (`create_game_env(seed=s)` then `reset()`) were checked to
give the same map for the same seed.

### 2.2 Measured overlaps

Enumerated for all five A1-related seeds (exploratory 1001, 1002, 1003; confirmatory 20261111,
20261112), at A1's sizes: self-play 20 × 24 = **480 maps**, benchmark **100 pairs** (the primary
arm; every other arm is a prefix), in-loop evaluation **20** (counted although A1 turns it off).
Overlap was measured twice: on the integer seeds, and on a fingerprint of the initial state
(SHA-256 of the initial spatial + vector observation) — the second catches two seeds that happen
to generate the same map.

| Comparison | Seed overlap | Map overlap |
|---|---|---|
| A1 self-play (480) vs A1 benchmark (100) — **train vs test** | 0 | 0 |
| A1 self-play vs A1 in-loop eval (20) | 0 | 0 |
| A1 vs A1-R, every stream | 0 | 0 |
| any exploratory seed vs any confirmatory seed, every stream | 0 | 0 |
| exploratory seeds among themselves, every stream | 0 | 0 |
| **all 3,000 derived seeds (5 seeds × 600)** | 3,000 distinct | 3,000 distinct |
| *for contrast:* arithmetic `S + p` arena maps, 20261111 vs 20261112 | **99 / 100** | 99 / 100 |
| *for contrast:* arithmetic `S + p`, 1001 vs 1002 / 1002 vs 1003 / 1001 vs 1003 | 99 / 99 / 98 | same |

Map-space sanity: seeds 0..9,999 produce **10,000 distinct** initial states at board 8, so
distinct seeds give distinct maps in practice, and the seed-level and map-level results agree.
The a-priori chance of any collision among 3,000 31-bit seeds is about 3,000² / 2³² ≈ 0.2%;
none occurred.

Reproduction (≈25 s; the script is kept out of the tree because it answers a one-off question):

```python
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.utils.seeding import SeedStream, derive_seed
import hashlib
def fp(s):
    obs, _ = MicroCivEnv(board_size=8, max_turns=60).reset(seed=s)
    return hashlib.sha256(obs["spatial"].tobytes() + obs["vector"].tobytes()).hexdigest()
S = 20261111
train = {fp(derive_seed(S, SeedStream.SELF_PLAY_ENV, it, g)) for it in range(20) for g in range(24)}
bench = {fp(derive_seed(S, SeedStream.BENCH_ENV, p)) for p in range(100)}
assert not train & bench  # holds for 1001, 1002, 1003, 20261111, 20261112
```

### 2.3 Sharing that *is* present, by design

- **Arms of one card share maps and search seeds.** Every arena arm is invoked with
  `--seed 20261111`, so pair `p` is the same map in every arm and game `g` the same search seed.
  Symmetry (20 pairs), positive control (20), final-vs-random (20) and the secondary arms (30
  each) all play prefixes of the primary arm's 100 maps. This is common random numbers: it makes
  the 5/10/15/20 checkpoints comparable map for map, and each arm's interval is still computed
  from that arm's own games. What it does mean is that the arms are **not independent
  samples** — a map-specific pathology would show in the symmetry control and the primary arm
  together. Section 7.1 should say so in one sentence; no change is needed.
- **T0b's positive-control reference and the exploratory runs share maps.** T0b measures the
  heuristic-vs-random reference at seeds 1001–1003 × 200 games, i.e. on `bench.env` maps of
  1001–1003, which are the maps the exploratory A1 runs' arena arms use. The confirmatory
  positive control (seed 20261111) is on disjoint maps, so the reference it is judged against is
  out of sample. That is the right way round; record it.

### 2.4 Paths A1 does not use, but should not be pointed at A1's seeds

The arithmetic seed rules survive outside A1's path: `run_arena` without `env_seeds`
(`arena.py:291`, `seed + pair`), `run_selfplay`/`run_batch_selfplay` in
`cloud/gcp/selfplay_worker.py` (`seed + game_idx`, `:720`; `seed + i`, `:1097-1140`), and
`run_encoder_ab.py`. Adjacent base seeds on those paths overlap almost completely (§2.2
contrast rows). `cloud/gcp/evaluation_worker.py:347` derives its arena base from
`derive_seed(seed, "eval.env", iteration)`, so successive iterations land on unrelated ranges
there, but its maps are still `base + p`. None of this touches A1; it matters for LF-A9, LF-E2
("seeds through workers") and L2.

## 3. Findings

1. **S5's overlap is real, already annotated, and would recur — at 560/800 games — if the A/B
   were re-run with today's `run_encoder_ab.py`.**
2. **A1 has no cross-stream overlap**: 0 of 100 arena maps are training maps, for every one of
   the five seeds, at the seed level and the map level.
3. **Exploratory and confirmatory seeds cannot collide through `derive_seed`** at A1's sizes: all
   3,000 derived seeds and maps are distinct. The adjacency between 20261111 and 20261112 (and
   between 1001–1003) would be a 99% map overlap on any arithmetic path; A1 and A1-R use none.
4. Within one card, arms share maps by design (common random numbers); not an overlap defect.

## 4. Recommendation

- **A1 / T13:** register the seeds unchanged. Add to section 7.1: (a) the train/test
  disjointness statement with its reproduction snippet (§2.2), (b) one sentence that arena arms
  share maps by design and are not independent samples, (c) that the positive-control reference
  (T0b, seeds 1001–1003) is on maps disjoint from the confirmatory arm. No code change.
- **Guard (optional, autonomous, cheap):** a regression test asserting, at the registered seeds
  and sizes, that the `selfplay.env` and `bench.env` seed sets are disjoint and that the
  exploratory and confirmatory derived sets are disjoint. It runs in milliseconds on seeds alone;
  its negative control is to point the benchmark at `seed + p` and watch A1 vs A1-R overlap.
- **S5 / LF-A9:** keep the annotation; before any A/B re-run, move `run_encoder_ab.py` to derived
  streams with a disjointness test (§1.4). The maintainer's S5 choice reduces to "annotate now,
  re-run under LF-A9 after that change"; a re-run on the current script would reproduce the
  defect at twice the size.
- **Elsewhere (LF-E2 / L2):** when seeds are threaded through the cloud workers, use named
  streams, not `seed + i`, for the reason in §2.4.
