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

Usage::

    python -m alpha_zero.evaluation.benchmark --checkpoint ckpt.pt \\
        --opponents random,heuristic,simple_mcts --seed 42 --out bench/

The ``scripts/benchmark_suite.py`` wrapper is the same CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alpha_zero.evaluation.scoring import (
    elo_from_score_rate,
    elo_interval,
    minimum_detectable_effect,
    summarize,
)
from alpha_zero.storage.models import EvaluationResult
from alpha_zero.utils.game_budget import add_budget_arguments, enforce_cli
from alpha_zero.utils.seeding import SeedStream, derive_seed

logger = logging.getLogger(__name__)

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "CHECKPOINT_OPPONENT_PREFIX",
    "DEFAULT_OPPONENTS",
    "SEARCH_EPOCH",
    "BenchmarkProtocol",
    "checkpoint_opponent_id",
    "collect_evidence",
    "hardware_summary",
    "main",
    "render_report",
    "run_benchmark",
    "validate_envelope",
]

#: Bump when the envelope's shape changes incompatibly.
BENCHMARK_SCHEMA_VERSION = 1

#: Identifies the search semantics a result was produced under. Results from
#: different epochs are not comparable: the 2026-09-04 fixes changed the value
#: backup sign rule and the policy target, which changes how strong a given
#: checkpoint *plays*, not just how it was trained. Update when the search
#: changes in a way that alters play at fixed weights.
SEARCH_EPOCH = "2026-09-04.to_play-backup+completed-q"

OPPONENT_RANDOM = "random"
OPPONENT_HEURISTIC = "heuristic"
OPPONENT_SIMPLE_MCTS = "simple_mcts"
CHECKPOINT_OPPONENT_PREFIX = "checkpoint:"

DEFAULT_OPPONENTS: tuple[str, ...] = (OPPONENT_RANDOM, OPPONENT_HEURISTIC, OPPONENT_SIMPLE_MCTS)

#: Confidence level every benchmark interval is reported at.
BENCHMARK_CONFIDENCE = 0.95

# Envelope keys, named once so the validator and the writer cannot drift.
_TOP_KEYS = ("benchmark_schema_version", "protocol", "evidence", "results")
_EVIDENCE_KEYS = (
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


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    fallback = platform.processor() or None
    if not cpuinfo.exists():
        return fallback
    try:
        text = cpuinfo.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return fallback
    for line in text.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return fallback


def hardware_summary() -> dict[str, Any]:
    """CPU, GPU, core count and thread cap -- the things that move timings."""
    gpu_model: str | None = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu_model = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover - torch missing or CUDA probe failing
        gpu_model = None
    return {
        "cpu_model": _cpu_model(),
        "gpu_model": gpu_model,
        "logical_cores": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def _git_sha(start: Path | None = None) -> str | None:
    """Commit at ``HEAD`` by reading ``.git`` directly -- no subprocess.

    Walks up from ``start`` (default: the working directory) to the nearest
    ``.git``; follows a symbolic ``HEAD`` through loose and packed refs. Returns
    ``None`` outside a repository or when the ref cannot be resolved, which
    the envelope records as ``git_sha: null`` rather than guessing.
    """
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        git_dir = candidate / ".git"
        if git_dir.is_file():  # worktree: "gitdir: <path>"
            try:
                pointer = git_dir.read_text(encoding="utf-8").strip()
            except OSError:
                return None
            git_dir = Path(pointer.removeprefix("gitdir:").strip())
        if not git_dir.is_dir():
            continue
        try:
            head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not head.startswith("ref:"):
            return head or None
        ref = head.removeprefix("ref:").strip()
        loose = git_dir / ref
        if loose.is_file():
            try:
                return loose.read_text(encoding="utf-8").strip() or None
            except OSError:
                return None
        packed = git_dir / "packed-refs"
        if packed.is_file():
            try:
                for line in packed.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) == 2 and parts[1] == ref:
                        return parts[0]
            except OSError:
                return None
        return None
    return None


def _version_of(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else None


def collect_evidence(
    seed: int | None, games_total: int, command: str | None = None
) -> dict[str, Any]:
    """Everything needed to reproduce or dismiss the number in ``results``."""
    return {
        "git_sha": _git_sha(),
        "generated_utc": datetime.now(UTC).isoformat(),
        "search_epoch": SEARCH_EPOCH,
        "seed": seed,
        "command": command if command is not None else " ".join(sys.argv),
        "hardware": hardware_summary(),
        "python": platform.python_version(),
        "torch": _version_of("torch"),
        "numpy": _version_of("numpy"),
        "games_total": games_total,
    }


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
    opponent_id: str,
    wins: int,
    draws: int,
    losses: int,
    legacy_win_rate: float,
    generated_utc: str,
) -> dict[str, Any]:
    """One opponent's row: the storage record plus the draw-aware statistics."""
    n = wins + draws + losses
    summary = summarize(wins, draws, losses, BENCHMARK_CONFIDENCE)
    record = EvaluationResult(
        iteration=0,
        candidate_version=0,
        champion_version=0,
        num_games=n,
        candidate_wins=wins,
        champion_wins=losses,
        draws=draws,
        win_rate=legacy_win_rate,
        promoted=False,
        # The envelope's generation time, so one scrub covers the whole file.
        timestamp=generated_utc,
        score_rate=summary.score_rate,
        ci_low=summary.ci_low,
        ci_high=summary.ci_high,
    )
    elo_low, elo_high = elo_interval(wins, draws, losses, BENCHMARK_CONFIDENCE)
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
        }
    )
    return row


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
        command: Recorded verbatim in ``evidence``; defaults to ``sys.argv``.

    Returns:
        The envelope, already valid under :func:`validate_envelope`.
    """
    from alpha_zero.cloud.gcp.arena import run_arena

    if candidate is None and candidate_factory is None:
        raise ValueError("run_benchmark needs a candidate network or a candidate_factory")
    if not opponents:
        raise ValueError("run_benchmark needs at least one opponent")

    models = dict(opponent_models or {})
    evidence = collect_evidence(seed, protocol.n_games * len(opponents), command)
    env_seeds = protocol.env_seeds(seed)
    results: list[dict[str, Any]] = []
    for opponent_id in opponents:
        factory, model_b = _opponent(opponent_id, protocol, models)
        logger.info(
            "Benchmark: candidate vs %s, %d games (%d maps, both seats), %s x%d, board %d / %d turns",
            opponent_id,
            protocol.n_games,
            protocol.num_pairs,
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
        )
        results.append(
            _result_record(
                opponent_id,
                arena.wins,
                arena.draws,
                arena.losses,
                arena.win_rate,
                evidence["generated_utc"],
            )
        )

    envelope = {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "protocol": protocol.to_dict(),
        "evidence": evidence,
        "results": results,
    }
    validate_envelope(envelope)
    return envelope


def validate_envelope(envelope: Mapping[str, Any]) -> None:
    """Raise :class:`ValueError` unless ``envelope`` has the schema-1 shape.

    Checks keys, the schema version, that the protocol round-trips through
    :class:`BenchmarkProtocol`, that every result carries the storage record's
    counts plus the benchmark's extra keys, and that counts add up.
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
    try:
        protocol = BenchmarkProtocol.from_dict(envelope["protocol"])
    except (TypeError, ValueError) as exc:
        problems.append(f"protocol does not parse: {exc}")
        protocol = None

    evidence = envelope["evidence"]
    for key in _EVIDENCE_KEYS:
        if key not in evidence:
            problems.append(f"evidence missing {key!r}")
    hardware = evidence.get("hardware", {})
    for key in _HARDWARE_KEYS:
        if key not in hardware:
            problems.append(f"evidence.hardware missing {key!r}")

    results = envelope["results"]
    if not isinstance(results, list) or not results:
        problems.append("results must be a non-empty list")
        results = []
    for i, row in enumerate(results):
        for key in ("candidate_wins", "champion_wins", "draws", "num_games", "score_rate"):
            if key not in row:
                problems.append(f"results[{i}] missing {key!r}")
        for key in _RESULT_EXTRA_KEYS:
            if key not in row:
                problems.append(f"results[{i}] missing {key!r}")
        if all(k in row for k in ("candidate_wins", "champion_wins", "draws", "num_games")):
            total = row["candidate_wins"] + row["champion_wins"] + row["draws"]
            if total != row["num_games"]:
                problems.append(
                    f"results[{i}] counts sum to {total} != num_games {row['num_games']}"
                )
            if protocol is not None and row["num_games"] != protocol.n_games:
                problems.append(
                    f"results[{i}] num_games {row['num_games']} != protocol n_games "
                    f"{protocol.n_games}"
                )
    if problems:
        raise ValueError("invalid benchmark envelope: " + "; ".join(problems))


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


def _load_network(path: Path, device: str) -> Any:
    """Load a ``MicroCivNetwork`` from a training checkpoint for the CLI."""
    from alpha_zero.core.checkpoint_utils import safe_torch_load
    from alpha_zero.core.civ6_network import MicroCivNetwork

    checkpoint = safe_torch_load(path, map_location=device)
    network_config = dict(checkpoint.get("network_config", {}))
    board_size = network_config.pop("board_size", 8)
    action_size = network_config.pop("action_size", None) or network_config.pop(
        "action_space_size", None
    )
    if action_size is None:
        from alpha_zero.envs.microciv import MicroCivEnv

        action_size = MicroCivEnv(board_size=board_size).action_space.n
    network = MicroCivNetwork(
        board_size=board_size, action_space_size=action_size, **network_config
    )
    state = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
    network.load_state_dict(state)
    network.eval()
    return network


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
        "--opponents",
        type=str,
        default=",".join(DEFAULT_OPPONENTS),
        help="Comma-separated opponent ids (random, heuristic, simple_mcts).",
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
    parser.add_argument("--out", type=Path, default=Path("benchmark_out"))
    add_budget_arguments(parser)
    return parser.parse_args(argv)


def benchmark_budget_inputs(args: argparse.Namespace) -> dict[str, int]:
    """Resolved ``benchmark`` cost-model inputs for parsed CLI ``args``."""
    return {
        "n_games": args.n_games,
        "num_opponents": len([o for o in args.opponents.split(",") if o.strip()]),
        "num_opponent_checkpoints": len(args.opponent_checkpoint),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: run the protocol and write ``result.json`` + ``report.md`` to ``--out``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    # Refuse before loading any network: games = n_games per opponent.
    enforce_cli("benchmark", benchmark_budget_inputs(args), args)
    protocol = BenchmarkProtocol(
        board_size=args.board_size,
        max_turns=args.max_turns,
        num_simulations=args.sims,
        gumbel_k=args.gumbel_k,
        n_games=args.n_games,
        mcts_type=args.mcts_type,
    )

    import torch

    if args.checkpoint is not None:
        candidate = _load_network(args.checkpoint, args.device)
    else:
        from alpha_zero.core.civ6_network import MicroCivNetwork
        from alpha_zero.envs.microciv import MicroCivEnv

        logger.warning("No --checkpoint given: benchmarking a freshly initialised network.")
        torch.manual_seed(args.seed or 0)
        candidate = MicroCivNetwork(
            board_size=protocol.board_size,
            action_space_size=int(MicroCivEnv(board_size=protocol.board_size).action_space.n),
        )
        candidate.eval()

    opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]
    opponent_models: dict[str, Any] = {}
    for path in args.opponent_checkpoint:
        opponent_id = checkpoint_opponent_id(path)
        opponent_models[opponent_id] = _load_network(path, args.device)
        opponents.append(opponent_id)

    envelope = run_benchmark(
        protocol,
        candidate,
        opponents,
        args.seed,
        opponent_models=opponent_models,
        device=torch.device(args.device),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    (args.out / "report.md").write_text(render_report(envelope), encoding="utf-8")
    logger.info("Wrote %s and %s", args.out / "result.json", args.out / "report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
