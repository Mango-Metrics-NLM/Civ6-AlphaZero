"""A1's seed streams cannot overlap (LF-A17; ``docs/EVALUATION.md`` section 7.1, "Seeds").

The seed-overlap memo (``kb/specs/20261111-seed-overlap-memo.md`` §4) proposed this
guard. At the registered seeds and sizes it asserts:

* every seed A1 derives -- self-play maps and searches, replay and loader
  shuffles, in-loop evaluation maps (off in A1, counted anyway), benchmark maps,
  arena searches and baseline opponents -- is distinct from every other, across
  all five registered base seeds (exploratory 1001-1003, confirmatory 20261111
  and 20261112);
* the self-play **maps** and the benchmark **maps** of each seed are disjoint, by a
  fingerprint of the initial observation (so two seeds that happen to generate one
  map would still be caught).

Negative control (in the file): the arithmetic ``seed + pair`` rule that
``run_arena`` uses without ``env_seeds`` overlaps on 99 of 100 maps between A1 and
A1-R; the overlap helper must report it, or the disjointness assertions could be
passing vacuously.

The benchmark's map seeds come from the production :meth:`BenchmarkProtocol.env_seeds`.
Self-play's are re-derived with the call ``training_civ6.run_self_play`` makes;
``test_self_play_still_derives_maps_this_way`` pins that call by its source text.
"""

from __future__ import annotations

import hashlib
import inspect
from collections import Counter

from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.evaluation.benchmark import BenchmarkProtocol
from alpha_zero.utils.seeding import SeedStream, derive_seed

EXPLORATORY = (1001, 1002, 1003)
CONFIRMATORY = (20261111, 20261112)
BASE_SEEDS = EXPLORATORY + CONFIRMATORY

# A1's sizes (section 7.1): 20 iterations x 24 self-play games; the primary arm's
# 200 games = 100 pairs (every other arm plays a prefix of those maps).
ITERATIONS, GAMES_PER_ITERATION = 20, 24
BENCH_GAMES = 200
EVAL_GAMES = 20
BOARD, MAX_TURNS = 8, 60


def self_play_map_seeds(base: int) -> list[int | None]:
    return [
        derive_seed(base, SeedStream.SELF_PLAY_ENV, it, g)
        for it in range(ITERATIONS)
        for g in range(GAMES_PER_ITERATION)
    ]


def bench_map_seeds(base: int) -> list[int | None]:
    return BenchmarkProtocol(n_games=BENCH_GAMES).env_seeds(base)


def a1_stream_seeds(base: int) -> list[int | None]:
    """Every seed one A1 card derives from ``base``."""
    seeds: list[int | None] = []
    seeds += [derive_seed(base, SeedStream.SELF_PLAY_ENV, it) for it in range(ITERATIONS)]
    seeds += self_play_map_seeds(base)
    seeds += [derive_seed(base, SeedStream.SELF_PLAY_SEARCH, it) for it in range(ITERATIONS)]
    seeds += [derive_seed(base, SeedStream.DATA_LOADER, it) for it in range(ITERATIONS)]
    seeds.append(derive_seed(base, SeedStream.REPLAY_SAMPLER))
    seeds += [derive_seed(base, SeedStream.EVAL_ENV, g) for g in range(EVAL_GAMES)]
    seeds += bench_map_seeds(base)
    for stream in (SeedStream.ARENA_SEARCH, SeedStream.ARENA_OPPONENT):
        seeds += [derive_seed(base, stream, g, side) for g in range(BENCH_GAMES) for side in "ab"]
    return seeds


def overlap(a: list[int | None], b: list[int | None]) -> int:
    """How many of ``b``'s values also occur in ``a``."""
    present = set(a)
    return sum(1 for value in b if value in present)


def _fingerprint(env: MicroCivEnv, seed: int | None) -> str:
    obs, _ = env.reset(seed=seed)
    return hashlib.sha256(obs["spatial"].tobytes() + obs["vector"].tobytes()).hexdigest()


def test_every_derived_seed_is_distinct_across_all_registered_seeds() -> None:
    every = [s for base in BASE_SEEDS for s in a1_stream_seeds(base)]
    assert None not in every
    duplicates = [seed for seed, n in Counter(every).items() if n > 1]
    assert not duplicates, f"{len(duplicates)} derived seeds collide, e.g. {duplicates[:3]}"


def test_training_and_arena_maps_are_disjoint_by_content() -> None:
    env = MicroCivEnv(board_size=BOARD, max_turns=MAX_TURNS)
    for base in BASE_SEEDS:
        train = {_fingerprint(env, s) for s in self_play_map_seeds(base)}
        arena = {_fingerprint(env, s) for s in bench_map_seeds(base)}
        assert len(arena) == BENCH_GAMES // 2, f"seed {base}: two pairs share a map"
        assert not train & arena, f"seed {base}: {len(train & arena)} arena maps are training maps"


def test_confirmatory_runs_share_no_arena_map() -> None:
    a1, a1r = (bench_map_seeds(base) for base in CONFIRMATORY)
    assert overlap(a1, a1r) == 0
    for exploratory in EXPLORATORY:
        assert overlap(bench_map_seeds(exploratory), a1) == 0


def test_negative_control_arithmetic_maps_would_overlap() -> None:
    """``seed + pair`` -- the rule A1 does not use -- overlaps almost completely."""
    pairs = BENCH_GAMES // 2

    def arithmetic(base: int) -> list[int | None]:
        return [base + p for p in range(pairs)]

    assert overlap(arithmetic(20261111), arithmetic(20261112)) == pairs - 1


def test_self_play_still_derives_maps_this_way() -> None:
    """If ``run_self_play`` changes how it seeds a game's map, this guard must follow it."""
    from alpha_zero import training_civ6

    source = " ".join(inspect.getsource(training_civ6.run_self_play).split())
    assert "derive_seed(config.seed, SeedStream.SELF_PLAY_ENV, iteration, game_idx)" in source
