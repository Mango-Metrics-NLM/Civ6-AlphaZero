"""A fixed, reproducible benchmark protocol over the shared arena.

Why this exists
---------------
Every strength claim in this repository so far was made under a different
recipe -- a different board, turn limit, simulation budget, seat assignment or
seed derivation -- so none of them could be compared with another, and most
could not be replayed at all (see ``docs/evidence/validation_iteration_zero/``).
A benchmark is a *protocol*: one frozen set of game parameters, one way of
naming maps, one way of pairing seats, and one envelope that records enough
provenance to reproduce the number it reports.

What it is not
--------------
It is not a new game loop. :func:`run_benchmark` calls the same
:func:`alpha_zero.cloud.gcp.arena.run_arena` the promotion gate uses, so a
benchmark result and an arena result differ only in protocol and provenance,
never in how a game is played or scored. The statistics come from
:mod:`alpha_zero.evaluation.scoring`, the SSOT.

Protocol in one paragraph
-------------------------
Games are played in pairs on the same map with the seats swapped (so first-mover
and map advantage cancel inside each pair), the map for pair ``p`` is
``derive_seed(seed, "bench.env", p)``, each side's search is seeded per game and
per side by the arena, the candidate plays greedily (temperature 0, no noise),
and the record is summarised as a draw-aware score rate with a ternary-variance
confidence interval, the implied Elo, and the minimum detectable effect the game
budget affords. ``n_games`` must be even.

Evidence producer (LF-D6)
-------------------------
Every result row keeps the per-game records ``{pair, seat, result}`` and reports,
at 95% and 99%, the iid ternary interval, the pair-clustered interval
(:func:`~alpha_zero.evaluation.scoring.pair_score_interval`) and their hull --
the bounds the pre-registered A1 decision rule reads. ``--shard i/N`` plays the
``i``-th contiguous block of whole pairs with the seeds the unsharded run would
use; ``--merge`` recombines shard files and recomputes every statistic, so the
merged file equals the unsharded one. ``--candidate heuristic|random`` fields a
baseline as the candidate (the positive control). ``result.json``'s ``evidence``
block follows ``docs/evidence/SCHEMA.md``.

Usage::

    python -m alpha_zero.evaluation.benchmark --checkpoint ckpt.pt \\
        --opponents random,heuristic,simple_mcts --seed 42 --out bench/
    python -m alpha_zero.evaluation.benchmark ... --shard 1/2 --out s1/
    python -m alpha_zero.evaluation.benchmark --merge s1/ s2/ --out merged/

The ``scripts/benchmark_suite.py`` wrapper is the same CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from alpha_zero.evaluation.benchmark_evidence import (
    EVIDENCE_SCHEMA_VERSION,
    SEARCH_EPOCH,
    SEARCH_SEMANTICS,
    benchmark_claims,
    cli_command,
    collect_evidence,
    facing_hull_ratio,
    hardware_summary,
    interval_summary,
    pair_scores,
    parse_shard,
    shard_pairs,
)
from alpha_zero.evaluation.scoring import (
    elo_from_score_rate,
    elo_interval,
    minimum_detectable_effect,
    summarize,
)
from alpha_zero.evaluation.scoring import win_rate as legacy_win_rate
from alpha_zero.storage.models import EvaluationResult
from alpha_zero.utils.game_budget import (
    EXIT_CODE_REFUSED,
    add_budget_arguments,
    enforce_cli,
)
from alpha_zero.utils.seeding import SeedStream, derive_seed, enforce_deterministic_backend

logger = logging.getLogger(__name__)

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "CANDIDATE_KINDS",
    "CHECKPOINT_OPPONENT_PREFIX",
    "DEFAULT_OPPONENTS",
    "EVIDENCE_SCHEMA_VERSION",
    "REPORTED_CONFIDENCE_LEVELS",
    "SEARCH_EPOCH",
    "BenchmarkProtocol",
    "benchmark_budget_inputs",
    "checkpoint_opponent_id",
    "collect_evidence",
    "facing_hull_ratio",
    "hardware_summary",
    "interval_summary",
    "main",
    "merge_shards",
    "pair_scores",
    "parse_shard",
    "render_report",
    "run_benchmark",
    "shard_pairs",
    "validate_envelope",
]

#: Bump when the envelope's shape changes incompatibly. 2 (LF-D6) added
#: ``candidate``, ``shard``, per-game records, the interval block and the
#: evidence block's ``schema_version`` / ``claims`` / ``non_claims``.
BENCHMARK_SCHEMA_VERSION = 2


OPPONENT_RANDOM = "random"
OPPONENT_HEURISTIC = "heuristic"
OPPONENT_SIMPLE_MCTS = "simple_mcts"
CHECKPOINT_OPPONENT_PREFIX = "checkpoint:"

DEFAULT_OPPONENTS: tuple[str, ...] = (OPPONENT_RANDOM, OPPONENT_HEURISTIC, OPPONENT_SIMPLE_MCTS)

#: Confidence level of the legacy ``ci_low`` / ``ci_high`` / ``elo_ci_*`` fields.
BENCHMARK_CONFIDENCE = 0.95

#: Levels the interval block reports. A1 reads 95% for its arms and 99% for the
#: symmetry control, so both are always present and no flag selects one.
REPORTED_CONFIDENCE_LEVELS: tuple[float, ...] = (0.95, 0.99)

CANDIDATE_CHECKPOINT = "checkpoint"
CANDIDATE_KINDS: tuple[str, ...] = (CANDIDATE_CHECKPOINT, "heuristic", "random")


# Envelope keys, named once so the validator and the writer cannot drift.
_TOP_KEYS = ("benchmark_schema_version", "protocol", "candidate", "shard", "evidence", "results")
_EVIDENCE_KEYS = (
    "schema_version",
    "git_sha",
    "generated_utc",
    "search_epoch",
    "seed",
    "command",
    "hardware",
    "python",
    "torch",
    "numpy",
    "games_total",
    "claims",
    "non_claims",
)
_HARDWARE_KEYS = ("cpu_model", "gpu_model", "logical_cores", "omp_num_threads")
_RESULT_EXTRA_KEYS = (
    "opponent_id",
    "draw_rate",
    "decisive_win_rate",
    "confidence",
    "mde_80pct",
    "elo_estimate",
    "elo_ci_low",
    "elo_ci_high",
    "games",
    "pairs",
    "intervals",
    "mde_alpha025_80pct",
    "hull_ratio",
    "mde_alpha025_80pct_hull",
)


@dataclass(frozen=True)
class BenchmarkProtocol:
    """The frozen game recipe a benchmark number is quoted under.

    Attributes:
        board_size: MicroCiv board edge. 8 is the shipped evaluation default.
        max_turns: Turn limit. 60 rather than the shipped 30: the draw-rate
            sweep (``docs/evidence/draw_rate_sweep/``) measured 0.68 draws at
            30 turns and 0.47 at 60 on board 8, and a benchmark whose games
            mostly tie cannot rank anything.
        num_simulations: Search budget per move for every searching player.
        gumbel_k: ``num_sampled_actions`` for the Gumbel-family searches.
        n_games: Games per opponent. Must be even: seats are paired.
        mcts_type: Search variant for the candidate (and a checkpoint opponent).
        gameplay_preset: Environment preset passed to the arena.
        paired_seats: Always ``True``; recorded in the envelope so a reader
            does not have to know the arena's convention.
    """

    board_size: int = 8
    max_turns: int = 60
    num_simulations: int = 25
    gumbel_k: int = 16
    n_games: int = 40
    mcts_type: str = "gumbel"
    gameplay_preset: str = "default"
    paired_seats: bool = True

    def __post_init__(self) -> None:
        errors: list[str] = []
        if self.n_games <= 0 or self.n_games % 2:
            errors.append(f"n_games must be a positive even number, got {self.n_games}")
        if self.board_size < 4:
            errors.append(f"board_size must be >= 4, got {self.board_size}")
        if self.max_turns < 1:
            errors.append(f"max_turns must be >= 1, got {self.max_turns}")
        if self.num_simulations < 1:
            errors.append(f"num_simulations must be >= 1, got {self.num_simulations}")
        if self.gumbel_k < 1:
            errors.append(f"gumbel_k must be >= 1, got {self.gumbel_k}")
        if not self.paired_seats:
            errors.append("paired_seats is the protocol; it cannot be disabled")
        if errors:
            raise ValueError("Invalid benchmark protocol: " + "; ".join(errors))

    @property
    def num_pairs(self) -> int:
        """Maps played: each from both seats."""
        return self.n_games // 2

    def env_seeds(self, seed: int | None) -> list[int | None]:
        """Map seeds for each pair: ``derive_seed(seed, "bench.env", pair)``.

        A *name*, not an offset, so the benchmark's maps cannot collide with a
        training run's self-play or evaluation streams under the same base
        seed. ``None`` propagates: an unseeded benchmark is unseeded.
        """
        return [derive_seed(seed, SeedStream.BENCH_ENV, pair) for pair in range(self.num_pairs)]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the envelope."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BenchmarkProtocol:
        """Rebuild from an envelope's ``protocol`` mapping."""
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def checkpoint_opponent_id(path: str | Path) -> str:
    """``checkpoint:<sha256>`` of the file, so an opponent is named by content.

    A path names a location; two files at one path on two days are two
    opponents. The digest is what makes a result against a checkpoint
    reproducible from the envelope alone.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return f"{CHECKPOINT_OPPONENT_PREFIX}{digest.hexdigest()}"


def _shard_games(protocol: BenchmarkProtocol, shard: tuple[int, int] | None) -> int:
    """Games one opponent plays in ``shard`` (all of them when unsharded)."""
    if shard is None:
        return protocol.n_games
    return 2 * len(shard_pairs(protocol.num_pairs, *shard))


# A factory receives the per-game seed the arena derives for that side.
PlayerFactory = Callable[[int | None], Any]


def _opponent(
    opponent_id: str,
    protocol: BenchmarkProtocol,
    opponent_models: Mapping[str, Any],
) -> tuple[PlayerFactory | None, Any]:
    """Resolve an opponent id to ``(player_factory, model)`` for the arena.

    Baselines come back as factories (they are not networks); a checkpoint
    opponent comes back as the model, searched with the protocol's own
    ``mcts_type`` and budget.
    """
    if opponent_id == OPPONENT_RANDOM:
        from alpha_zero.evaluation.baseline_agents import RandomAgent

        return (lambda s: RandomAgent(seed=s)), None
    if opponent_id == OPPONENT_HEURISTIC:
        from alpha_zero.evaluation.baseline_agents import GreedyHeuristicAgent

        return (lambda s: GreedyHeuristicAgent(seed=s)), None
    if opponent_id == OPPONENT_SIMPLE_MCTS:
        from alpha_zero.evaluation.baselines import SimpleMCTSPlayer

        sims = protocol.num_simulations
        return (lambda s: SimpleMCTSPlayer(num_simulations=sims, seed=s)), None
    if opponent_id.startswith(CHECKPOINT_OPPONENT_PREFIX):
        if opponent_id not in opponent_models:
            raise ValueError(
                f"opponent {opponent_id!r} has no model; pass it in opponent_models "
                "(keyed by checkpoint_opponent_id(path))"
            )
        return None, opponent_models[opponent_id]
    raise ValueError(
        f"unknown opponent {opponent_id!r}; expected one of {DEFAULT_OPPONENTS} "
        f"or '{CHECKPOINT_OPPONENT_PREFIX}<sha256>'"
    )


def _result_record(
    opponent_id: str, games: Sequence[Mapping[str, Any]], generated_utc: str
) -> dict[str, Any]:
    """One opponent's row from its per-game records: the storage record plus statistics.

    Every number is recomputed from ``games``, which is what lets
    :func:`merge_shards` rebuild a row from shard files exactly.
    """
    games = sorted((dict(g) for g in games), key=lambda g: (g["pair"], g["seat"]))
    results = [g["result"] for g in games]
    wins, draws, losses = (results.count(r) for r in ("win", "draw", "loss"))
    n = len(games)
    summary = summarize(wins, draws, losses, BENCHMARK_CONFIDENCE)
    record = EvaluationResult(
        iteration=0,
        candidate_version=0,
        champion_version=0,
        num_games=n,
        candidate_wins=wins,
        champion_wins=losses,
        draws=draws,
        win_rate=legacy_win_rate(wins, draws, losses),
        promoted=False,
        # The envelope's generation time, so one scrub covers the whole file.
        timestamp=generated_utc,
        score_rate=summary.score_rate,
        ci_low=summary.ci_low,
        ci_high=summary.ci_high,
    )
    elo_low, elo_high = elo_interval(wins, draws, losses, BENCHMARK_CONFIDENCE)
    intervals = {str(c): interval_summary(games, c) for c in REPORTED_CONFIDENCE_LEVELS}
    hull_ratio = facing_hull_ratio(
        summary.score_rate, intervals["0.95"]["iid"], intervals["0.95"]["hull"]
    )
    mde_025 = minimum_detectable_effect(n, summary.draw_rate, alpha=0.025)
    row = record.to_dict()
    row.update(
        {
            "opponent_id": opponent_id,
            "draw_rate": summary.draw_rate,
            "decisive_win_rate": summary.decisive_win_rate,
            "confidence": BENCHMARK_CONFIDENCE,
            # The edge over 0.5 this many games can see at 80% power, given
            # the draw rate actually observed against this opponent.
            "mde_80pct": minimum_detectable_effect(n, summary.draw_rate) if n else None,
            "elo_estimate": elo_from_score_rate(summary.score_rate),
            "elo_ci_low": elo_low,
            "elo_ci_high": elo_high,
            "games": games,
            "pairs": len(pair_scores(games)),
            "intervals": intervals,
            # The A1 INCONCLUSIVE report: one-sided 2.5% MDE at the observed draw
            # rate, and the same widened by how far the 95% hull reaches past the
            # iid interval on the side facing 0.5 (see facing_hull_ratio).
            "mde_alpha025_80pct": mde_025,
            "hull_ratio": hull_ratio,
            "mde_alpha025_80pct_hull": mde_025 * hull_ratio,
        }
    )
    return row


def _candidate_kind(candidate_id: str) -> str:
    if candidate_id in CANDIDATE_KINDS:
        return candidate_id
    return CANDIDATE_CHECKPOINT if candidate_id.startswith(CHECKPOINT_OPPONENT_PREFIX) else "other"


def run_benchmark(
    protocol: BenchmarkProtocol,
    candidate: Any,
    opponents: Sequence[str],
    seed: int | None,
    *,
    candidate_factory: PlayerFactory | None = None,
    opponent_models: Mapping[str, Any] | None = None,
    device: Any = None,
    command: str | None = None,
    shard: tuple[int, int] | None = None,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    """Play the candidate against each opponent under ``protocol``; return the envelope.

    Args:
        protocol: The frozen recipe.
        candidate: The candidate network (searched with ``protocol.mcts_type``),
            or ``None`` when ``candidate_factory`` supplies the player.
        opponents: Opponent ids -- ``random``, ``heuristic``, ``simple_mcts``
            or ``checkpoint:<sha256>`` (with the model in ``opponent_models``).
        seed: Base seed. Maps come from ``protocol.env_seeds(seed)``; searches
            and stochastic baselines are seeded per game and side by the arena.
            ``None`` is an unseeded, unreproducible run and is recorded as such.
        candidate_factory: Builds a non-network candidate (a baseline measuring
            another baseline, as the strength self-test does).
        opponent_models: Models for ``checkpoint:`` opponents.
        device: Torch device for the searches; the arena defaults to CPU.
        command: Recorded verbatim in ``evidence``; defaults to the running CLI,
            ``shlex``-joined.
        shard: ``(i, N)``: play only the ``i``-th of ``N`` contiguous blocks of
            pairs (1-based), each game with the seeds the unsharded run gives it.
        candidate_id: How the envelope names the candidate (``checkpoint:<sha256>``,
            ``heuristic``, ...). Defaults to ``network`` or ``factory``.

    Returns:
        The envelope, already valid under :func:`validate_envelope`.
    """
    from alpha_zero.cloud.gcp.arena import run_arena

    if candidate is None and candidate_factory is None:
        raise ValueError("run_benchmark needs a candidate network or a candidate_factory")
    if not opponents:
        raise ValueError("run_benchmark needs at least one opponent")

    models = dict(opponent_models or {})
    shard_block: dict[str, Any] | None = None
    game_indices: list[int] | None = None
    if shard is not None:
        pairs = shard_pairs(protocol.num_pairs, *shard)
        shard_block = {"index": shard[0], "count": shard[1], "pairs": [pairs.start, pairs.stop]}
        game_indices = [g for p in pairs for g in (2 * p, 2 * p + 1)]
    n_played = _shard_games(protocol, shard)
    if candidate_id is None:
        candidate_id = "network" if candidate_factory is None else "factory"
    claims, non_claims = benchmark_claims(candidate_id, opponents, n_played, shard_block)
    evidence = collect_evidence(
        seed,
        n_played * len(opponents),
        command,
        claims=claims,
        non_claims=non_claims,
        device=str(device) if device is not None else "cpu",
        search={
            "mcts_type": protocol.mcts_type,
            "num_simulations": protocol.num_simulations,
            "gumbel_k": protocol.gumbel_k,
            "search_semantics": SEARCH_SEMANTICS,
        },
    )
    env_seeds = protocol.env_seeds(seed)
    results: list[dict[str, Any]] = []
    for opponent_id in opponents:
        factory, model_b = _opponent(opponent_id, protocol, models)
        logger.info(
            "Benchmark: candidate vs %s, %d games (%d maps, both seats), %s x%d, board %d / %d turns",
            opponent_id,
            n_played,
            n_played // 2,
            protocol.mcts_type,
            protocol.num_simulations,
            protocol.board_size,
            protocol.max_turns,
        )
        arena = run_arena(
            model_a=candidate,
            model_b=model_b,
            num_games=protocol.n_games,
            mcts_type=protocol.mcts_type,
            mcts_sims=protocol.num_simulations,
            gumbel_num_actions=protocol.gumbel_k,
            board_size=protocol.board_size,
            max_turns=protocol.max_turns,
            device=device,
            gameplay_preset=protocol.gameplay_preset,
            seed=seed,
            player_a_factory=candidate_factory,
            player_b_factory=factory,
            env_seeds=env_seeds,
            game_indices=game_indices,
        )
        results.append(_result_record(opponent_id, arena.games, evidence["generated_utc"]))

    envelope = {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "protocol": protocol.to_dict(),
        "candidate": {"kind": _candidate_kind(candidate_id), "id": candidate_id},
        "shard": shard_block,
        "evidence": evidence,
        "results": results,
    }
    validate_envelope(envelope)
    return envelope


def _row_problems(i: int, row: Mapping[str, Any], expected_pairs: range | None) -> list[str]:
    """Contract violations in one result row (counts, games, pairs, intervals)."""
    required = ("candidate_wins", "champion_wins", "draws", "num_games", "score_rate")
    problems = [
        f"results[{i}] missing {key!r}" for key in required + _RESULT_EXTRA_KEYS if key not in row
    ]
    if problems:
        return problems
    total = row["candidate_wins"] + row["champion_wins"] + row["draws"]
    if total != row["num_games"]:
        problems.append(f"results[{i}] counts sum to {total} != num_games {row['num_games']}")
    outcomes = [g.get("result") for g in row["games"]]
    counts = (outcomes.count("win"), outcomes.count("draw"), outcomes.count("loss"))
    expected = (row["candidate_wins"], row["draws"], row["champion_wins"])
    if counts != expected or len(outcomes) != row["num_games"]:
        problems.append(f"results[{i}] counts disagree with its per-game records")
    try:
        pair_scores(row["games"])
    except (KeyError, ValueError) as exc:
        problems.append(f"results[{i}] games: {exc}")
    if expected_pairs is not None:
        seats = [(g.get("pair"), g.get("seat")) for g in row["games"]]
        if sorted(seats) != [(p, seat) for p in expected_pairs for seat in (0, 1)]:
            problems.append(
                f"results[{i}] games are not exactly pairs "
                f"[{expected_pairs.start}, {expected_pairs.stop}) from both seats"
            )
    for level in REPORTED_CONFIDENCE_LEVELS:
        if set(row["intervals"].get(str(level), {})) != {"iid", "pair", "hull"}:
            problems.append(f"results[{i}] intervals missing the {level} iid/pair/hull block")
    return problems


def validate_envelope(envelope: Mapping[str, Any]) -> None:
    """Raise :class:`ValueError` unless ``envelope`` has the current schema's shape.

    Checks keys, the schema version, that the protocol round-trips through
    :class:`BenchmarkProtocol`, the evidence block's keys and epoch, and for each
    result: the storage record's counts plus the benchmark's extra keys, that the
    counts add up and agree with the per-game records, and that those records are
    exactly the whole pairs this run (or shard) was meant to play.
    """
    problems: list[str] = []
    for key in _TOP_KEYS:
        if key not in envelope:
            problems.append(f"missing top-level key {key!r}")
    if problems:
        raise ValueError("invalid benchmark envelope: " + "; ".join(problems))

    if envelope["benchmark_schema_version"] != BENCHMARK_SCHEMA_VERSION:
        problems.append(
            f"benchmark_schema_version {envelope['benchmark_schema_version']!r} != "
            f"{BENCHMARK_SCHEMA_VERSION}"
        )
    expected_pairs: range | None = None
    try:
        protocol = BenchmarkProtocol.from_dict(envelope["protocol"])
        shard = envelope["shard"]
        if shard is None:
            expected_pairs = range(protocol.num_pairs)
        else:
            expected_pairs = shard_pairs(protocol.num_pairs, shard["index"], shard["count"])
            if [expected_pairs.start, expected_pairs.stop] != shard["pairs"]:
                problems.append(f"shard pairs {shard['pairs']} do not match shard {shard}")
    except (TypeError, ValueError, KeyError) as exc:
        problems.append(f"protocol or shard does not parse: {exc}")

    evidence = envelope["evidence"]
    for key in _EVIDENCE_KEYS:
        if key not in evidence:
            problems.append(f"evidence missing {key!r}")
    if evidence.get("search_epoch") != SEARCH_EPOCH:
        problems.append(f"evidence.search_epoch {evidence.get('search_epoch')!r} != {SEARCH_EPOCH}")
    hardware = evidence.get("hardware", {})
    for key in _HARDWARE_KEYS:
        if key not in hardware:
            problems.append(f"evidence.hardware missing {key!r}")

    results = envelope["results"]
    if not isinstance(results, list) or not results:
        problems.append("results must be a non-empty list")
        results = []
    for i, row in enumerate(results):
        problems.extend(_row_problems(i, row, expected_pairs))
    if problems:
        raise ValueError("invalid benchmark envelope: " + "; ".join(problems))


def merge_shards(
    envelopes: Sequence[Mapping[str, Any]], command: str | None = None
) -> dict[str, Any]:
    """Recombine the shard envelopes of one run into the unsharded envelope.

    Every shard must come from the same protocol, candidate, seed, commit and
    opponent list, and the set must be exactly shards ``1..N`` of one ``N``.
    Rows are rebuilt from the union of the per-game records, so every statistic
    is recomputed, not averaged; the result validates as an unsharded run.
    """
    if not envelopes:
        raise ValueError("merge_shards needs at least one shard envelope")
    for envelope in envelopes:
        validate_envelope(envelope)
    if any(e["shard"] is None for e in envelopes):
        raise ValueError("merge_shards takes shard envelopes; one is already unsharded")
    shards = sorted(envelopes, key=lambda e: e["shard"]["index"])
    count = shards[0]["shard"]["count"]
    found = [(e["shard"]["index"], e["shard"]["count"]) for e in shards]
    if found != [(i, count) for i in range(1, count + 1)]:
        raise ValueError(f"need exactly shards 1..{count} of {count}, got {found}")
    first = shards[0]
    checks: dict[str, Callable[[Mapping[str, Any]], Any]] = {
        "protocol": lambda e: e["protocol"],
        "candidate": lambda e: e["candidate"],
        "seed": lambda e: e["evidence"]["seed"],
        "git_sha": lambda e: e["evidence"]["git_sha"],
        "opponents": lambda e: [r["opponent_id"] for r in e["results"]],
        "device": lambda e: e["evidence"].get("device"),
        "torch": lambda e: e["evidence"].get("torch"),
        "numpy": lambda e: e["evidence"].get("numpy"),
        "search": lambda e: e["evidence"].get("search"),
    }
    for name, get in checks.items():
        if any(get(e) != get(first) for e in shards[1:]):
            raise ValueError(f"shards disagree on {name}: {[get(e) for e in shards]}")

    protocol = BenchmarkProtocol.from_dict(first["protocol"])
    opponents = [r["opponent_id"] for r in first["results"]]
    claims, non_claims = benchmark_claims(
        first["candidate"]["id"], opponents, protocol.n_games, None
    )
    evidence = collect_evidence(
        first["evidence"]["seed"],
        protocol.n_games * len(opponents),
        command,
        claims=claims,
        non_claims=non_claims,
    )
    # The games were played by the shards' commit and stack, not by whoever merges them.
    # That includes the hardware: ``gpu_model`` names the GPU ``device`` ran on, and the
    # merging host's GPU 0 is not it.
    for key in ("git_sha", "device", "search", "torch", "numpy", "hardware"):
        if key in first["evidence"]:
            evidence[key] = first["evidence"][key]
    evidence.pop("git_sha_null_reason", None)
    if evidence["git_sha"] is None:
        evidence["git_sha_null_reason"] = first["evidence"].get("git_sha_null_reason")
    evidence["shards"] = [
        {
            "index": e["shard"]["index"],
            "count": e["shard"]["count"],
            "pairs": e["shard"]["pairs"],
            "generated_utc": e["evidence"]["generated_utc"],
            "command": e["evidence"]["command"],
            "hardware": e["evidence"]["hardware"],
        }
        for e in shards
    ]
    results = [
        _result_record(
            opponent_id,
            [g for e in shards for g in e["results"][i]["games"]],
            evidence["generated_utc"],
        )
        for i, opponent_id in enumerate(opponents)
    ]
    merged = {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "protocol": first["protocol"],
        "candidate": first["candidate"],
        "shard": None,
        "evidence": evidence,
        "results": results,
    }
    validate_envelope(merged)
    return merged


def _shard_note(envelope: Mapping[str, Any]) -> str:
    shard = envelope["shard"]
    if shard is not None:
        return (
            f"**Shard {shard['index']}/{shard['count']}** (pairs {shard['pairs'][0]} to "
            f"{shard['pairs'][1] - 1}): a partial result; merge every shard before quoting it."
        )
    merged = envelope["evidence"].get("shards")
    return f"Merged from {len(merged)} shards." if merged else "Unsharded run."


def render_report(envelope: Mapping[str, Any]) -> str:
    """Markdown report for ``report.md``; the JSON stays the record."""
    protocol = envelope["protocol"]
    evidence = envelope["evidence"]
    hw = evidence["hardware"]
    lines = [
        "# Benchmark report",
        "",
        f"Generated {evidence['generated_utc']} at commit `{evidence.get('git_sha') or 'unknown'}` "
        f"(search epoch `{evidence['search_epoch']}`), seed `{evidence['seed']}`.",
        "",
        f"Candidate: `{envelope['candidate']['id']}`. " + _shard_note(envelope),
        "",
        "## Protocol",
        "",
        "| Setting | Value |",
        "|---|---|",
    ]
    for key in sorted(protocol):
        lines.append(f"| `{key}` | {protocol[key]} |")
    lines += [
        "",
        "## Results (candidate's perspective)",
        "",
        "| Opponent | W / D / L | Win rate (legacy) | Score rate | 95% CI | Draw rate "
        "| Elo (95% CI) | MDE @80% |",
        "|---|---|---:|---:|---|---:|---|---:|",
    ]
    for row in envelope["results"]:
        mde = row["mde_80pct"]
        lines.append(
            f"| `{row['opponent_id']}` "
            f"| {row['candidate_wins']} / {row['draws']} / {row['champion_wins']} "
            f"| {row['win_rate']:.3f} "
            f"| {row['score_rate']:.3f} "
            f"| [{row['ci_low']:.3f}, {row['ci_high']:.3f}] "
            f"| {row['draw_rate']:.1%} "
            f"| {row['elo_estimate']:+.0f} [{row['elo_ci_low']:+.0f}, {row['elo_ci_high']:+.0f}] "
            f"| {'-' if mde is None else f'{mde:.3f}'} |"
        )
    lines += [
        "",
        "`Win rate (legacy)` counts a draw as a loss and is kept for artifact "
        "compatibility only; `Score rate` is the gating statistic "
        "(`alpha_zero/evaluation/scoring.py`). `MDE @80%` is the smallest true "
        "edge over 0.5 this many games can detect at 80% power, one-sided alpha 0.05, "
        "at the observed draw rate.",
        "",
        "## Intervals: iid, pair-clustered and hull",
        "",
        "| Opponent | Pairs | Level | iid | pair | hull | MDE a=0.025 (x hull ratio) |",
        "|---|---:|---:|---|---|---|---:|",
    ]
    for row in envelope["results"]:
        for level, block in row["intervals"].items():
            iid, pair, hull = block["iid"], block["pair"], block["hull"]
            lines.append(
                f"| `{row['opponent_id']}` | {row['pairs']} | {float(level):.0%} "
                f"| [{iid[0]:.3f}, {iid[1]:.3f}] | [{pair[0]:.3f}, {pair[1]:.3f}] "
                f"| [{hull[0]:.3f}, {hull[1]:.3f}] "
                f"| {row['mde_alpha025_80pct']:.3f} ({row['mde_alpha025_80pct_hull']:.3f}) |"
            )
    lines += [
        "",
        "`iid` treats games as independent (ternary variance); `pair` takes each "
        "seat-swapped map pair as the unit; `hull` is `[min(lows), max(highs)]`, the "
        "bounds a pre-registered decision reads.",
        "",
        "## Evidence",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| command | `{evidence['command']}` |",
        f"| python / torch / numpy | {evidence['python']} / {evidence['torch']} / {evidence['numpy']} |",
        f"| cpu | {hw['cpu_model']} ({hw['logical_cores']} logical cores, "
        f"OMP_NUM_THREADS={hw['omp_num_threads']}) |",
        f"| gpu | {hw['gpu_model'] or 'none'} |",
        f"| games total | {evidence['games_total']} |",
        "",
    ]
    return "\n".join(lines)


def _load_network(path: Path, device: str, expected_max_turns: int | None = None) -> Any:
    """Load a ``MicroCivNetwork`` from a training checkpoint for the CLI.

    ``expected_max_turns`` is the horizon the benchmark will play at; a
    checkpoint recorded at another one raises ``HorizonMismatch`` (a legacy file
    with no provenance only warns).
    """
    from alpha_zero.core.checkpoint_utils import safe_torch_load
    from alpha_zero.utils.checkpoint import (
        build_micro_network_from_checkpoint,
        check_provenance,
    )

    checkpoint = safe_torch_load(path, map_location=device)
    check_provenance(
        checkpoint.get("metadata") if isinstance(checkpoint, dict) else None,
        expected_max_turns=expected_max_turns,
        source=path,
    )
    network = build_micro_network_from_checkpoint(
        checkpoint, default_board_size=8, default_action_size=None
    )
    state = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
    network.load_state_dict(state)
    network.eval()
    return network


def _shard_arg(spec: str) -> tuple[int, int]:
    try:
        return parse_shard(spec)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="benchmark_suite",
        description="Run the fixed benchmark protocol and write result.json + report.md.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Candidate checkpoint. Omitted: a freshly initialised network (smoke mode).",
    )
    parser.add_argument(
        "--candidate",
        choices=CANDIDATE_KINDS,
        default=CANDIDATE_CHECKPOINT,
        help="checkpoint (default: --checkpoint, or an untrained network), or a baseline "
        "candidate -- heuristic or random -- for a positive control. Baselines take no "
        "--checkpoint.",
    )
    parser.add_argument(
        "--opponents",
        type=str,
        default=",".join(DEFAULT_OPPONENTS),
        help="Comma-separated opponent ids (random, heuristic, simple_mcts); "
        "--opponents= for none (checkpoint opponents only).",
    )
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        action="append",
        default=[],
        help="Add a checkpoint opponent (repeatable); its id is checkpoint:<sha256>.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Base seed (None = unseeded).")
    parser.add_argument("--n-games", type=int, default=BenchmarkProtocol.n_games)
    parser.add_argument("--board-size", type=int, default=BenchmarkProtocol.board_size)
    parser.add_argument("--max-turns", type=int, default=BenchmarkProtocol.max_turns)
    parser.add_argument("--sims", type=int, default=BenchmarkProtocol.num_simulations)
    parser.add_argument("--gumbel-k", type=int, default=BenchmarkProtocol.gumbel_k)
    parser.add_argument(
        "--mcts-type", type=str, default=BenchmarkProtocol.mcts_type, choices=["gumbel", "civ6"]
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--shard",
        type=_shard_arg,
        default=None,
        metavar="i/N",
        help="Play only the i-th (1-based) of N contiguous blocks of seat-swapped pairs, "
        "with the seeds the unsharded run would use. Combine with --merge.",
    )
    parser.add_argument(
        "--merge",
        type=Path,
        nargs="+",
        default=None,
        metavar="SHARD",
        help="Merge shard result.json files (or directories holding one) into --out, "
        "recomputing every statistic. Plays no games.",
    )
    parser.add_argument("--out", type=Path, default=Path("benchmark_out"))
    add_budget_arguments(parser)
    args = parser.parse_args(argv)
    if args.candidate != CANDIDATE_CHECKPOINT and args.checkpoint is not None:
        parser.error(f"--candidate {args.candidate} plays a baseline; drop --checkpoint")
    if args.merge is not None and args.shard is not None:
        parser.error("--merge combines shards; it does not take --shard")
    if args.shard is not None and args.merge is None:
        try:
            shard_pairs(args.n_games // 2, *args.shard)
        except ValueError as exc:
            parser.error(f"--shard with --n-games {args.n_games}: {exc}")
    return args


def benchmark_budget_inputs(args: argparse.Namespace) -> dict[str, int]:
    """Resolved ``benchmark`` cost-model inputs for parsed CLI ``args``.

    ``n_games`` is what one opponent plays in *this* invocation: the shard's
    whole pairs under ``--shard``, and 0 under ``--merge``.
    """
    n_games = 0 if args.merge is not None else args.n_games
    if args.merge is None and args.shard is not None:
        n_games = 2 * len(shard_pairs(args.n_games // 2, *args.shard))
    return {
        "n_games": n_games,
        "num_opponents": len([o for o in args.opponents.split(",") if o.strip()]),
        "num_opponent_checkpoints": len(args.opponent_checkpoint),
    }


def _read_envelope(path: Path) -> dict[str, Any]:
    target = path / "result.json" if path.is_dir() else path
    data: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    return data


def _candidate(
    args: argparse.Namespace, protocol: BenchmarkProtocol
) -> tuple[Any, PlayerFactory | None, str]:
    """``(network, factory, id)`` for the candidate the CLI asked for."""
    if args.candidate == "heuristic":
        from alpha_zero.evaluation.baseline_agents import GreedyHeuristicAgent

        return None, (lambda s: GreedyHeuristicAgent(seed=s)), "heuristic"
    if args.candidate == "random":
        from alpha_zero.evaluation.baseline_agents import RandomAgent

        return None, (lambda s: RandomAgent(seed=s)), "random"
    if args.checkpoint is not None:
        network = _load_network(args.checkpoint, args.device, protocol.max_turns)
        return network, None, checkpoint_opponent_id(args.checkpoint)

    import torch

    from alpha_zero.core.civ6_network import MicroCivNetwork
    from alpha_zero.envs.microciv import MicroCivEnv

    logger.warning("No --checkpoint given: benchmarking a freshly initialised network.")
    torch.manual_seed(args.seed or 0)
    network = MicroCivNetwork(
        board_size=protocol.board_size,
        action_space_size=int(MicroCivEnv(board_size=protocol.board_size).action_space.n),
    )
    network.eval()
    return network, None, f"untrained-network:torch_seed={args.seed or 0}"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: run the protocol (or merge shards) and write ``result.json`` + ``report.md``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    # Refuse before loading any network: games = n_games per opponent.
    enforce_cli("benchmark", benchmark_budget_inputs(args), args)
    # Never overwrite a result: each run (arm, shard, merge) needs its own --out.
    target = args.out / "result.json"
    if target.exists():
        print(
            f"REFUSED: {target} already exists; give this run its own --out "
            "(results are never overwritten).",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_CODE_REFUSED)
    command = cli_command(argv)

    if args.merge is not None:
        envelope = merge_shards([_read_envelope(p) for p in args.merge], command=command)
    else:
        protocol = BenchmarkProtocol(
            board_size=args.board_size,
            max_turns=args.max_turns,
            num_simulations=args.sims,
            gumbel_k=args.gumbel_k,
            n_games=args.n_games,
            mcts_type=args.mcts_type,
        )
        import torch

        if args.seed is not None:
            # Before any network is built or loaded: a seeded GPU arena must repeat.
            enforce_deterministic_backend(args.device)
        candidate, candidate_factory, candidate_id = _candidate(args, protocol)
        opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]
        opponent_models: dict[str, Any] = {}
        for path in args.opponent_checkpoint:
            opponent_id = checkpoint_opponent_id(path)
            opponent_models[opponent_id] = _load_network(path, args.device, protocol.max_turns)
            opponents.append(opponent_id)

        envelope = run_benchmark(
            protocol,
            candidate,
            opponents,
            args.seed,
            candidate_factory=candidate_factory,
            opponent_models=opponent_models,
            device=torch.device(args.device),
            command=command,
            shard=args.shard,
            candidate_id=candidate_id,
        )
    args.out.mkdir(parents=True, exist_ok=True)
    with open(target, "x", encoding="utf-8") as fh:  # "x": refuse a file created meanwhile
        fh.write(json.dumps(envelope, indent=2))
    (args.out / "report.md").write_text(render_report(envelope), encoding="utf-8")
    logger.info("Wrote %s and %s", args.out / "result.json", args.out / "report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
