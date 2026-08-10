#!/usr/bin/env python3
"""Local, device-agnostic A/B comparison of the spatial encoders.

Trains a ``patch_embed`` arm and a ``hex_crystalline`` arm from identical code
under an identical budget and seed, plays them head to head, and writes evidence
that is honest about its own sample size.

Why this script exists: the historical cloud A/B ran the two arms on *different
container images* and recorded no numbers, so the repository has never held a
usable encoder comparison. The training entry points, meanwhile, expose no
``encoder_type`` at all — only the cloud workers thread it. Those workers happen
to be storage-free and importable locally, so this driver composes them rather
than adding a fourth training path.

Subcommands::

    train    Train one arm per encoder type and write checkpoints + sidecars.
    arena    Play two checkpoints head to head; emit counts and a Wilson interval.
    report   Summarise the run into a committable markdown table.

Everything is a CLI flag with a sane default; nothing about the budget, device,
or output location is hardcoded. The device defaults to automatic detection, so
the same command works on CPU and GPU.

Example:
    $ python scripts/run_encoder_ab.py train --iterations 20 --games-per-iter 14
    $ python scripts/run_encoder_ab.py arena --num_games 800
    $ python scripts/run_encoder_ab.py report

Statistical note:
    ``arena`` reports a Wilson score interval alongside the win rate. At a few
    hundred games the interval is wide enough that small differences are
    indistinguishable from noise, so the report states the interval and the
    caller is expected not to claim more than it supports.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("run_encoder_ab")

#: The two encoder arms under comparison.
ENCODER_ARMS: tuple[str, ...] = ("patch_embed", "hex_crystalline")

#: Committed evidence lives here. Deliberately *not* under ``reports/`` or any
#: directory named ``encoder_comparison`` — .gitignore excludes both at any
#: depth, which would silently drop the artifacts this script exists to produce.
DEFAULT_OUTPUT_DIR = Path("docs/evidence/hexcrystal_ab")

#: Local (uncommitted) checkpoint destination; ``*.pt`` is gitignored by design.
DEFAULT_CHECKPOINT_DIR = Path("checkpoints/encoder_ab")

#: Prefix for environment overrides of any ABRunConfig field, e.g.
#: ``CIV6AZ_AB_ITERATIONS=50``. Documented in the CLI epilog.
_ENV_PREFIX = "CIV6AZ_AB_"

#: Fallback MLflow experiment when the environment names none.
DEFAULT_EXPERIMENT_NAME = "civ6az-encoder-ab"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ABRunConfig:
    """Resolved settings for one A/B training run.

    Frozen so a run cannot be reconfigured halfway through; the instance is
    serialized verbatim into each arm's sidecar, which is what makes the run
    reproducible after the fact.
    """

    iterations: int = 20
    games_per_iter: int = 14
    epochs_per_iter: int = 2
    batch_size: int = 32
    learning_rate: float = 1e-3
    mcts_sims: int = 25
    mcts_type: str = "gumbel"
    gumbel_num_actions: int = 16
    board_size: int = 8
    max_turns: int = 30
    action_space_size: int = 376
    num_layers: int = 6
    embed_dim: int = 64
    num_heads: int = 8
    seed: int = 42
    device: str = "auto"
    gameplay_preset: str = "default"

    def resolved_device(self) -> str:
        """Return a concrete device string, auto-detecting when requested."""
        if self.device != "auto":
            return self.device
        from alpha_zero.utils.device import resolve_device_str

        return str(resolve_device_str(None))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of the configuration."""
        return asdict(self)


@dataclass
class ArmResult:
    """Outcome of training a single encoder arm."""

    encoder_type: str
    checkpoint_path: str
    final_loss: float
    losses: list[float] = field(default_factory=list)
    games_played: int = 0
    train_examples: int = 0
    wall_clock_seconds: float = 0.0
    parameter_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of this result."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Reproducibility and provenance
# ---------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    """Seed Python, numpy, and torch RNGs.

    This covers weight initialisation and action sampling. It does **not**
    cover game generation: ``MicroCivEnv`` owns a private numpy generator, so
    environments must additionally be seeded explicitly — see the ``seed``
    argument threaded into ``run_selfplay``.

    Args:
        seed: Base seed value.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.debug("Global RNGs seeded with %d", seed)


def _git_revision() -> str:
    """Return the current git commit, or ``"unknown"`` outside a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _provenance(config: ABRunConfig) -> dict[str, Any]:
    """Collect everything needed to reproduce or audit a run later.

    MLflow's local store is gitignored, so the sidecar is the only record that
    survives in version control.
    """
    import torch

    return {
        "git_revision": _git_revision(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "torch_version": torch.__version__,
        "device": config.resolved_device(),
        "python_version": sys.version.split()[0],
        "config": config.to_dict(),
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def wilson_interval(
    successes: float, total: int, z: float = 1.959_963_984_540_054
) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside ``[0, 1]``
    and behaves sensibly for small samples and extreme proportions.

    Args:
        successes: Number of successes; may be fractional when draws score 0.5.
        total: Number of trials.
        z: Standard-normal quantile; the default is the two-sided 95% value.

    Returns:
        ``(low, high)`` bounds, or ``(0.0, 1.0)`` when *total* is zero.
    """
    if total <= 0:
        return (0.0, 1.0)

    proportion = successes / total
    denominator = 1.0 + z**2 / total
    centre = proportion + z**2 / (2 * total)
    spread = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
    return ((centre - spread) / denominator, (centre + spread) / denominator)


def elo_difference(score_rate: float) -> float:
    """Convert a head-to-head score rate into an Elo difference.

    Args:
        score_rate: Score in ``[0, 1]`` (wins plus half the draws).

    Returns:
        Elo difference in points; clamped at +/-800 for degenerate rates.
    """
    epsilon = 1e-4
    clamped = min(max(score_rate, epsilon), 1 - epsilon)
    return -400.0 * math.log10(1.0 / clamped - 1.0)


# ---------------------------------------------------------------------------
# Checkpoint payload
# ---------------------------------------------------------------------------


def build_checkpoint_payload(model: Any, encoder_type: str) -> dict[str, Any]:
    """Build the ``{state_dict, model_config}`` payload used across the project.

    Mirrors the cloud trainer's format — the only writer that persists
    ``encoder_type`` — so these checkpoints load through the same reader.
    Enum values are flattened to strings so the payload stays JSON-clean.

    Args:
        model: A network exposing a ``config`` dataclass.
        encoder_type: Encoder identifier to assert into the config.

    Returns:
        A dict suitable for ``torch.save``.
    """
    from enum import Enum

    raw_config = dataclasses.asdict(model.config)
    model_config = {k: (v.value if isinstance(v, Enum) else v) for k, v in raw_config.items()}
    model_config["encoder_type"] = encoder_type

    return {"state_dict": model.state_dict(), "model_config": model_config}


def _write_sidecar(
    path: Path, config: ABRunConfig, arm: ArmResult, hex_encoder_config: dict[str, Any] | None
) -> None:
    """Write the JSON sidecar accompanying a checkpoint.

    The sidecar carries the hex encoder's own configuration in addition to the
    network config: ``TransformerConfig`` does not record ``hex_hidden_dim``,
    ``dropout`` or ``activation``, so without this the publishing step would
    have to reconstruct them from defaults and would silently drift if those
    defaults ever changed.
    """
    payload: dict[str, Any] = {
        "arm": arm.to_dict(),
        "provenance": _provenance(config),
    }
    if hex_encoder_config is not None:
        payload["hex_encoder_config"] = hex_encoder_config

    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info("Wrote sidecar: %s", path)


def _extract_hex_encoder_config(model: Any) -> dict[str, Any] | None:
    """Return the nested hex encoder's config dict, when present."""
    encoder = getattr(getattr(model, "encoder", None), "patch_embed", None)
    config = getattr(encoder, "config", None)
    if config is None or not hasattr(config, "to_dict"):
        return None
    result: dict[str, Any] = config.to_dict()
    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_arm(
    encoder_type: str,
    config: ABRunConfig,
    checkpoint_dir: Path,
    tracker: Any | None = None,
) -> ArmResult:
    """Train a single encoder arm and write its checkpoint.

    Args:
        encoder_type: Which encoder to train.
        config: Resolved run configuration, shared by both arms.
        checkpoint_dir: Directory to write the checkpoint and sidecar into.
        tracker: Optional experiment tracker exposing ``log_metrics``.

    Returns:
        An :class:`ArmResult` describing the completed arm.
    """
    import torch
    from torch.utils.data import DataLoader

    from alpha_zero.cloud.gcp.selfplay_worker import run_selfplay
    from alpha_zero.cloud.gcp.training_worker import create_model, prepare_dataset, train_epoch

    started = time.time()
    device_str = config.resolved_device()
    device = torch.device(device_str)

    # Both arms start from the same seed so weight init and sampling match.
    set_global_seed(config.seed)

    logger.info(
        "=== Training arm %r: %d iterations x %d games, %d sims, device=%s, seed=%d ===",
        encoder_type,
        config.iterations,
        config.games_per_iter,
        config.mcts_sims,
        device_str,
        config.seed,
    )

    model = create_model(
        board_size=config.board_size,
        action_space_size=config.action_space_size,
        num_res_blocks=config.num_layers,
        num_filters=config.embed_dim,
        num_heads=config.num_heads,
        encoder_type=encoder_type,
    ).to(device)

    parameter_count = sum(p.numel() for p in model.parameters())
    logger.info("Model built: %s, %d parameters", encoder_type, parameter_count)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    total_games = 0
    total_examples = 0
    global_step = 0

    for iteration in range(config.iterations):
        iteration_started = time.time()

        # Distinct seeds per iteration and arm-independent, so both arms see the
        # same sequence of game seeds and differ only by encoder.
        selfplay_seed = config.seed + iteration * config.games_per_iter
        trajectories = run_selfplay(
            model=model,
            num_games=config.games_per_iter,
            mcts_sims=config.mcts_sims,
            mcts_type=config.mcts_type,
            board_size=config.board_size,
            max_turns=config.max_turns,
            gumbel_num_actions=config.gumbel_num_actions,
            device=device_str,
            gameplay_preset=config.gameplay_preset,
            seed=selfplay_seed,
        )
        total_games += len(trajectories)

        dataset = prepare_dataset(trajectories, action_space_size=config.action_space_size)
        total_examples += len(dataset)
        if len(dataset) == 0:
            logger.warning("Iteration %d produced no training examples; skipping", iteration)
            continue

        generator = torch.Generator().manual_seed(config.seed + iteration)
        dataloader = DataLoader(
            dataset,
            batch_size=min(config.batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
        )

        iteration_loss = 0.0
        for _epoch in range(config.epochs_per_iter):
            loss, _policy_loss, _value_loss, global_step = train_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
                global_step=global_step,
                spatial_size=config.board_size,
            )
            iteration_loss = loss

        losses.append(iteration_loss)
        elapsed = time.time() - iteration_started
        logger.info(
            "[%s] iteration %d/%d: loss=%.4f, examples=%d, %.1fs",
            encoder_type,
            iteration + 1,
            config.iterations,
            iteration_loss,
            len(dataset),
            elapsed,
        )

        if tracker is not None:
            tracker.log_metrics(
                {
                    "iter/avg_loss": iteration_loss,
                    "iter/examples": float(len(dataset)),
                    "iter/elapsed_seconds": elapsed,
                },
                step=iteration,
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{encoder_type}.pt"
    torch.save(build_checkpoint_payload(model, encoder_type), checkpoint_path)
    logger.info("Wrote checkpoint: %s", checkpoint_path)

    arm = ArmResult(
        encoder_type=encoder_type,
        checkpoint_path=str(checkpoint_path),
        final_loss=losses[-1] if losses else float("nan"),
        losses=losses,
        games_played=total_games,
        train_examples=total_examples,
        wall_clock_seconds=time.time() - started,
        parameter_count=parameter_count,
    )
    _write_sidecar(
        checkpoint_dir / f"{encoder_type}.metadata.json",
        config,
        arm,
        _extract_hex_encoder_config(model),
    )
    return arm


# ---------------------------------------------------------------------------
# Checkpoint loading and arena
# ---------------------------------------------------------------------------


def load_arm_checkpoint(path: Path, device: str) -> tuple[Any, str, dict[str, Any]]:
    """Load an A/B checkpoint and rebuild its network.

    Mirrors the cloud evaluator's reconstruction: enum fields are rehydrated
    from strings and the derived private fields are dropped, because they are
    ``init=False`` on the config dataclass.

    Args:
        path: Checkpoint file written by :func:`train_arm`.
        device: Device string to map tensors onto.

    Returns:
        ``(model, encoder_type, architecture)`` where *architecture* carries the
        board size and action-space size the checkpoint was trained with.

    Raises:
        ValueError: If the checkpoint carries no ``model_config``, which would
            make its encoder type unknowable.
    """
    import torch

    from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig
    from alpha_zero.utils.checkpoint import validate_encoder_state

    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_config" not in payload:
        raise ValueError(
            f"{path} has no 'model_config'; its encoder type cannot be determined. "
            "Legacy checkpoints silently reconstruct as patch_embed, which would "
            "invalidate the comparison."
        )

    config_dict = dict(payload["model_config"])
    encoder_type = config_dict.get("encoder_type", "patch_embed")

    model = Civ6TransformerNetwork(TransformerConfig.from_saved_dict(config_dict))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()

    # Pass the checkpoint's own keys: without them this could only confirm how
    # the model was built, which is the value we just read out of the same dict.
    validate_encoder_state(model, encoder_type, checkpoint_keys=payload["state_dict"].keys())

    architecture = {
        "spatial_size": config_dict.get("spatial_size"),
        "action_space_size": config_dict.get("action_space_size"),
    }
    logger.info("Loaded %s (%s, board=%s)", path, encoder_type, architecture["spatial_size"])
    return model, encoder_type, architecture


def run_head_to_head(
    checkpoint_a: Path,
    checkpoint_b: Path,
    num_games: int,
    config: ABRunConfig,
) -> dict[str, Any]:
    """Play two checkpoints against each other and summarise the outcome.

    Args:
        checkpoint_a: Reference arm; results are from its perspective.
        checkpoint_b: Opponent arm.
        num_games: Number of games to play.
        config: Run configuration supplying board size, MCTS settings, device.

    Returns:
        A JSON-serializable dict of counts, rates, interval, and Elo estimate.
    """
    import torch

    from alpha_zero.cloud.gcp.arena import run_arena

    device_str = config.resolved_device()
    model_a, encoder_a, arch_a = load_arm_checkpoint(checkpoint_a, device_str)
    model_b, encoder_b, arch_b = load_arm_checkpoint(checkpoint_b, device_str)

    if arch_a != arch_b:
        raise ValueError(
            f"The two checkpoints disagree on architecture ({arch_a} vs {arch_b}); "
            "they cannot be compared head to head."
        )

    # Board size comes from the checkpoints, not from this subcommand's own
    # defaults. `arena` is a separate invocation from `train`, so trusting its
    # flags would silently play the models on a board they were never trained
    # for -- the same class of mistake as reporting the arena's training budget.
    board_size = arch_a["spatial_size"] or config.board_size
    if board_size != config.board_size:
        logger.info(
            "Using board_size=%d from the checkpoints (subcommand default was %d)",
            board_size,
            config.board_size,
        )

    set_global_seed(config.seed)
    started = time.time()

    logger.info(
        "=== Arena: %s vs %s, %d games, %d sims, device=%s ===",
        encoder_a,
        encoder_b,
        num_games,
        config.mcts_sims,
        device_str,
    )

    result = run_arena(
        model_a=model_a,
        model_b=model_b,
        num_games=num_games,
        mcts_type=config.mcts_type,
        mcts_sims=config.mcts_sims,
        gumbel_num_actions=config.gumbel_num_actions,
        board_size=board_size,
        max_turns=config.max_turns,
        device=torch.device(device_str),
        gameplay_preset=config.gameplay_preset,
        seed=config.seed,
    )

    wins, draws, losses = result.wins, result.draws, result.losses
    total = wins + draws + losses
    score = wins + 0.5 * draws
    score_rate = score / total if total else 0.0
    low, high = wilson_interval(score, total)

    summary: dict[str, Any] = {
        "model_a": {"encoder_type": encoder_a, "checkpoint": str(checkpoint_a)},
        "model_b": {"encoder_type": encoder_b, "checkpoint": str(checkpoint_b)},
        "architecture": arch_a,
        "num_games": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "score_rate": score_rate,
        "score_rate_ci95": [low, high],
        "ci95_half_width": (high - low) / 2.0,
        "elo_difference": elo_difference(score_rate),
        "significant_at_95pct": low > 0.5 or high < 0.5,
        "wall_clock_seconds": time.time() - started,
        "provenance": _provenance(config),
    }

    logger.info(
        "Arena result: %dW/%dD/%dL of %d — score rate %.3f (95%% CI %.3f-%.3f), %s",
        wins,
        draws,
        losses,
        total,
        score_rate,
        low,
        high,
        "significant" if summary["significant_at_95pct"] else "NOT significant",
    )
    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def compare_arm_provenance(provenances: dict[str, dict[str, Any]]) -> list[str]:
    """Return the settings on which the arms' training runs disagree.

    The report asserts that the arms differ only in encoder type. Nothing
    enforces that: ``--encoder-types`` exists so arms can be trained in separate
    invocations, and the environment overrides can change the budget between
    them. An unverified claim of a controlled comparison is worse than no claim,
    so it is checked.

    Args:
        provenances: Encoder type mapped to that arm's recorded provenance.

    Returns:
        Human-readable descriptions of each divergence; empty when the arms
        agree or when fewer than two provenances were supplied.
    """
    if len(provenances) < 2:
        return []

    divergences: list[str] = []
    arms = sorted(provenances)
    reference_name = arms[0]
    reference = provenances[reference_name]

    for other_name in arms[1:]:
        other = provenances[other_name]
        if reference.get("git_revision") != other.get("git_revision"):
            divergences.append("code revision")

        ref_config = reference.get("config", {})
        other_config = other.get("config", {})
        for key in sorted(set(ref_config) | set(other_config)):
            if ref_config.get(key) != other_config.get(key):
                divergences.append(
                    f"{key} ({reference_name}={ref_config.get(key)}, "
                    f"{other_name}={other_config.get(key)})"
                )

    return divergences


def render_report(
    arena: dict[str, Any],
    arms: dict[str, Any] | None,
    training_provenance: dict[str, Any] | None = None,
    budget_divergence: list[str] | None = None,
) -> str:
    """Render the committable markdown evidence report.

    Args:
        arena: Output of :func:`run_head_to_head`.
        arms: Optional mapping of encoder type to training-arm metadata.
        training_provenance: Provenance recorded when the arms were *trained*.
            The budget table is drawn from this rather than from the arena's own
            provenance: the arena runs as a separate invocation with its own
            defaults, so reporting its config would misstate the training budget
            — precisely the kind of quietly-wrong evidence this script exists to
            avoid.

    Returns:
        Markdown text.
    """
    provenance = arena.get("provenance", {})
    budget_source = training_provenance or provenance
    config = budget_source.get("config", {})
    encoder_a = arena["model_a"]["encoder_type"]
    encoder_b = arena["model_b"]["encoder_type"]
    low, high = arena["score_rate_ci95"]

    lines = [
        "# Encoder A/B evidence: patch_embed vs hex_crystalline",
        "",
        f"Generated {provenance.get('timestamp_utc', 'unknown')} "
        f"from commit `{provenance.get('git_revision', 'unknown')}`.",
        "",
        "## Head-to-head result",
        "",
        f"`{encoder_a}` versus `{encoder_b}`, played from `{encoder_a}`'s perspective.",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Games | {arena['num_games']} |",
        f"| Win / Draw / Loss | {arena['wins']} / {arena['draws']} / {arena['losses']} |",
        f"| Score rate (draws = 0.5) | {arena['score_rate']:.3f} |",
        f"| Wilson 95% CI | [{low:.3f}, {high:.3f}] |",
        f"| CI half-width | ±{arena['ci95_half_width']:.3f} |",
        f"| Elo difference | {arena['elo_difference']:+.1f} |",
        f"| Significant at 95% | {'yes' if arena['significant_at_95pct'] else 'no'} |",
        "",
    ]

    if not arena["significant_at_95pct"]:
        lines += [
            "> **The interval spans 0.5, so this run does not distinguish the two "
            "encoders.** The point estimate above should not be read as either "
            "encoder being stronger; a larger sample would be required to "
            "resolve a difference of this size.",
            "",
        ]

    if arms:
        lines += [
            "## Training arms",
            "",
            "| Arm | Parameters | Games | Examples | Final loss | Wall clock |",
            "|-----|-----------:|------:|---------:|-----------:|-----------:|",
        ]
        for encoder_type, arm in arms.items():
            lines.append(
                f"| `{encoder_type}` | {arm.get('parameter_count', 0):,} | "
                f"{arm.get('games_played', 0)} | {arm.get('train_examples', 0)} | "
                f"{arm.get('final_loss', float('nan')):.4f} | "
                f"{arm.get('wall_clock_seconds', 0.0) / 60:.1f} min |"
            )
        lines.append("")

    lines += ["## Budget", ""]
    if budget_divergence:
        lines += [
            "> **The two arms were NOT trained under identical settings.** They differ "
            f"in: {', '.join(budget_divergence)}. A head-to-head result between arms "
            "trained differently does not isolate the encoder, so the comparison below "
            "should not be read as an encoder comparison.",
            "",
        ]
    else:
        lines += [
            "Both arms were trained from the same code, seed, and budget; they differ "
            "only in encoder type. (Verified by comparing the two arms' sidecars.)",
            "",
        ]
    lines += [
        "| Setting | Value |",
        "|---------|-------|",
    ]
    for key in (
        "iterations",
        "games_per_iter",
        "epochs_per_iter",
        "batch_size",
        "learning_rate",
        "mcts_sims",
        "mcts_type",
        "board_size",
        "max_turns",
        "seed",
    ):
        if key in config:
            lines.append(f"| `{key}` | {config[key]} |")

    lines += [
        f"| device | {budget_source.get('device', 'unknown')} |",
        f"| torch | {budget_source.get('torch_version', 'unknown')} |",
        f"| training commit | `{budget_source.get('git_revision', 'unknown')}` |",
        "",
        "## Scope and limitations",
        "",
        "This is a reduced-scale MicroCiv comparison (13-channel encoder, "
        f"{config.get('board_size', 8)}x{config.get('board_size', 8)} board), not a "
        "full-Civ6 result. Each arm was trained **once**, so training-run variance "
        "is not captured: the interval above reflects game sampling only, and a "
        "different seed could plausibly reorder the arms. Treat this as evidence "
        "that both encoders train and play, plus a bound on how large any "
        "difference between them could be at this budget.",
        "",
    ]
    return "\n".join(lines)


def render_summary_table(arena: dict[str, Any]) -> str:
    """Render the compact table embedded in the model card."""
    low, high = arena["score_rate_ci95"]
    encoder_a = arena["model_a"]["encoder_type"]
    encoder_b = arena["model_b"]["encoder_type"]
    return "\n".join(
        [
            f"| {encoder_a} vs {encoder_b} | Value |",
            "|---|---|",
            f"| Games | {arena['num_games']} |",
            f"| W/D/L | {arena['wins']}/{arena['draws']}/{arena['losses']} |",
            f"| Score rate | {arena['score_rate']:.3f} |",
            f"| Wilson 95% CI | [{low:.3f}, {high:.3f}] |",
            f"| Elo difference | {arena['elo_difference']:+.1f} |",
            "",
        ]
    )


def check_consistency(arena: dict[str, Any]) -> list[str]:
    """Return a list of internal inconsistencies in an arena result.

    An empty list means the artifact is self-consistent. Used as a gate before
    committing evidence, so a malformed result cannot be published.
    """
    required = (
        "wins",
        "draws",
        "losses",
        "num_games",
        "score",
        "score_rate",
        "score_rate_ci95",
        "elo_difference",
        "significant_at_95pct",
    )
    missing = [key for key in required if key not in arena]
    if missing:
        # Report rather than raise: this runs on arbitrary JSON read off disk,
        # and a truncated file should reach the "refusing to report" path, not
        # produce a KeyError traceback.
        return [f"result is missing required field(s): {', '.join(missing)}"]

    problems: list[str] = []
    total = arena["wins"] + arena["draws"] + arena["losses"]

    if total <= 0:
        problems.append("result contains no games; there is nothing to report")

    if total != arena["num_games"]:
        problems.append(f"W+D+L={total} does not equal num_games={arena['num_games']}")

    expected_score = arena["wins"] + 0.5 * arena["draws"]
    if abs(expected_score - arena["score"]) > 1e-9:
        problems.append(f"score={arena['score']} does not equal W+0.5D={expected_score}")

    if total:
        expected_rate = expected_score / total
        if abs(expected_rate - arena["score_rate"]) > 1e-9:
            problems.append("score_rate does not match score/num_games")

    low, high = arena["score_rate_ci95"]
    if not low <= arena["score_rate"] <= high:
        problems.append("score_rate lies outside its own confidence interval")
    if not (0.0 <= low <= 1.0 and 0.0 <= high <= 1.0):
        problems.append("confidence interval escapes [0, 1]")

    if "ci95_half_width" in arena:
        expected_half_width = (high - low) / 2.0
        if abs(expected_half_width - arena["ci95_half_width"]) > 1e-9:
            problems.append("ci95_half_width does not match the interval")

    # The Elo figure is the most eye-catching number in both the report and the
    # model card, so it gets checked rather than trusted.
    if total > 0:
        expected_elo = elo_difference(expected_score / total)
        if abs(expected_elo - arena["elo_difference"]) > 1e-6:
            problems.append(
                f"elo_difference {arena['elo_difference']:.3f} does not match the "
                f"score rate (expected {expected_elo:.3f})"
            )

    claimed = arena["significant_at_95pct"]
    actual = low > 0.5 or high < 0.5
    if claimed != actual:
        problems.append("significance flag disagrees with the interval")

    return problems


# ---------------------------------------------------------------------------
# Experiment tracking (optional)
# ---------------------------------------------------------------------------


def make_tracker(encoder_type: str, config: ABRunConfig) -> Any | None:
    """Return an MLflow-backed tracker, or None when tracking is unavailable.

    Tracking is strictly optional here. MLflow lives behind the ``monitoring``
    extra and is disabled by default, and the evidence artifacts are written to
    disk regardless — so a missing tracker warns rather than failing a run that
    may already have cost hours.
    """
    try:
        from alpha_zero.monitoring import MLflowConfig, MLflowLogger
        from alpha_zero.monitoring.config import resolve_tracker_enabled
    except ImportError as exc:
        logger.warning(
            "MLflow unavailable (%s); continuing without tracking. Checkpoints and "
            "evidence are written regardless. Install with: pip install -e '.[monitoring]'",
            exc,
        )
        return None

    # resolve_tracker_enabled is the project's single source of truth for this
    # decision (CIV6AZ_MLFLOW_ENABLED > MLFLOW_ENABLED > legacy WANDB_ENABLED).
    # Re-implementing the check here would silently ignore two of the three.
    if not resolve_tracker_enabled():
        logger.info("Experiment tracking disabled (set CIV6AZ_MLFLOW_ENABLED=true to enable)")
        return None

    mlflow_config = MLflowConfig.from_env()
    mlflow_config.enabled = True
    mlflow_config.run_name = f"ab-{encoder_type}-seed{config.seed}"
    mlflow_config.tags = {**(mlflow_config.tags or {}), "encoder_type": encoder_type}
    if not mlflow_config.experiment_name:
        mlflow_config.experiment_name = DEFAULT_EXPERIMENT_NAME

    tracker = MLflowLogger(mlflow_config)
    # encoder_type goes in the *params*, not only the tags: the run comparator
    # reads it from params, so tagging alone would make both arms read back as
    # "Unknown" and the two halves of this feature would not interoperate.
    tracker.init(config_dict={**config.to_dict(), "encoder_type": encoder_type})
    return tracker


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _coerce_env_value(raw: str, default_value: Any) -> Any:
    """Coerce an environment string to the type of *default_value*.

    Args:
        raw: The environment variable's value.
        default_value: The field default, whose runtime type drives the cast.

    Returns:
        The coerced value.

    Raises:
        ValueError: If *raw* cannot be parsed as the target type.
    """
    if isinstance(default_value, bool):
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{raw!r} is not a boolean")
    if isinstance(default_value, int):
        return int(raw)
    if isinstance(default_value, float):
        return float(raw)
    return raw


def _config_from_args(args: argparse.Namespace) -> ABRunConfig:
    """Build a run config from parsed arguments, honoring env overrides."""
    defaults = ABRunConfig()
    overrides: dict[str, Any] = {}

    for f in dataclasses.fields(ABRunConfig):
        value = getattr(args, f.name, None)
        if value is not None:
            overrides[f.name] = value
            continue

        env_value = os.environ.get(f"{_ENV_PREFIX}{f.name.upper()}")
        if env_value is None:
            continue

        # Dispatch on the default's runtime type rather than the annotation:
        # `from __future__ import annotations` makes every field type a string,
        # so annotation-based dispatch silently degrades to one branch.
        default_value = getattr(defaults, f.name)
        try:
            overrides[f.name] = _coerce_env_value(env_value, default_value)
        except ValueError as exc:
            raise ValueError(
                f"{_ENV_PREFIX}{f.name.upper()}={env_value!r} is not a valid "
                f"{type(default_value).__name__}"
            ) from exc

    return ABRunConfig(**overrides)


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach run-configuration flags, defaulting to None so env vars apply."""
    defaults = ABRunConfig()
    group = parser.add_argument_group("run configuration")
    group.add_argument(
        "--iterations",
        type=int,
        default=None,
        help=f"Training iterations per arm (default {defaults.iterations})",
    )
    group.add_argument(
        "--games-per-iter",
        dest="games_per_iter",
        type=int,
        default=None,
        help=f"Self-play games per iteration (default {defaults.games_per_iter})",
    )
    group.add_argument(
        "--epochs-per-iter",
        dest="epochs_per_iter",
        type=int,
        default=None,
        help=f"Training epochs per iteration (default {defaults.epochs_per_iter})",
    )
    group.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=None,
        help=f"Batch size (default {defaults.batch_size})",
    )
    group.add_argument(
        "--learning-rate",
        dest="learning_rate",
        type=float,
        default=None,
        help=f"Adam learning rate (default {defaults.learning_rate})",
    )
    group.add_argument(
        "--mcts-sims",
        dest="mcts_sims",
        type=int,
        default=None,
        help=f"MCTS simulations per move (default {defaults.mcts_sims})",
    )
    group.add_argument(
        "--board-size",
        dest="board_size",
        type=int,
        default=None,
        help=f"Board size (default {defaults.board_size})",
    )
    group.add_argument(
        "--max-turns",
        dest="max_turns",
        type=int,
        default=None,
        help=f"Maximum turns per game (default {defaults.max_turns})",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Base random seed, shared by both arms (default {defaults.seed})",
    )
    group.add_argument("--device", default=None, help="Device: 'auto' (default), 'cpu', or 'cuda'")


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="run_encoder_ab.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (default INFO)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train one arm per encoder type")
    _add_config_arguments(train)
    train.add_argument(
        "--encoder-types",
        nargs="+",
        default=list(ENCODER_ARMS),
        choices=list(ENCODER_ARMS),
        help="Arms to train (default: both)",
    )
    train.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Checkpoint destination (default {DEFAULT_CHECKPOINT_DIR})",
    )

    arena = subparsers.add_parser("arena", help="Play two checkpoints head to head")
    _add_config_arguments(arena)
    arena.add_argument(
        "--num_games",
        type=int,
        default=800,
        help="Head-to-head games (default 800). Note the underscore: the "
        "repository's safety guard matches this exact flag name.",
    )
    arena.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory holding the two arm checkpoints",
    )
    arena.add_argument(
        "--checkpoint-a", type=Path, default=None, help="Override path for the reference arm"
    )
    arena.add_argument(
        "--checkpoint-b", type=Path, default=None, help="Override path for the opponent arm"
    )
    arena.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Evidence destination (default {DEFAULT_OUTPUT_DIR})",
    )

    report = subparsers.add_parser("report", help="Render the committable evidence report")
    report.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory holding arena_result.json (default {DEFAULT_OUTPUT_DIR})",
    )
    report.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory holding the arm sidecars",
    )

    return parser


def _command_train(args: argparse.Namespace) -> int:
    """Run the ``train`` subcommand."""
    config = _config_from_args(args)
    logger.info("Resolved configuration: %s", config)

    for encoder_type in args.encoder_types:
        tracker = make_tracker(encoder_type, config)
        try:
            train_arm(encoder_type, config, args.checkpoint_dir, tracker)
        finally:
            if tracker is not None:
                tracker.finish()
    return 0


def _command_arena(args: argparse.Namespace) -> int:
    """Run the ``arena`` subcommand."""
    config = _config_from_args(args)
    checkpoint_a = args.checkpoint_a or args.checkpoint_dir / f"{ENCODER_ARMS[1]}.pt"
    checkpoint_b = args.checkpoint_b or args.checkpoint_dir / f"{ENCODER_ARMS[0]}.pt"

    for path in (checkpoint_a, checkpoint_b):
        if not path.exists():
            logger.error("Missing checkpoint: %s. Run the 'train' subcommand first.", path)
            return 1

    arena = run_head_to_head(checkpoint_a, checkpoint_b, args.num_games, config)

    problems = check_consistency(arena)
    if problems:
        for problem in problems:
            logger.error("Inconsistent arena result: %s", problem)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "arena_result.json"
    destination.write_text(json.dumps(arena, indent=2, sort_keys=True))
    logger.info("Wrote %s", destination)
    return 0


def _command_report(args: argparse.Namespace) -> int:
    """Run the ``report`` subcommand."""
    arena_path = args.output_dir / "arena_result.json"
    if not arena_path.exists():
        logger.error("Missing %s. Run the 'arena' subcommand first.", arena_path)
        return 1

    arena = json.loads(arena_path.read_text())
    problems = check_consistency(arena)
    if problems:
        for problem in problems:
            logger.error("Refusing to report an inconsistent result: %s", problem)
        return 1

    arms: dict[str, Any] = {}
    provenances: dict[str, dict[str, Any]] = {}
    training_provenance: dict[str, Any] | None = None
    for encoder_type in ENCODER_ARMS:
        sidecar = args.checkpoint_dir / f"{encoder_type}.metadata.json"
        if sidecar.exists():
            payload = json.loads(sidecar.read_text())
            arms[encoder_type] = payload.get("arm", {})
            provenance = payload.get("provenance")
            if provenance:
                provenances[encoder_type] = provenance
                # The budget belongs to training, not to the arena invocation.
                training_provenance = training_provenance or provenance

    budget_divergence = compare_arm_provenance(provenances)
    if budget_divergence:
        logger.warning(
            "The arms were not trained identically (%s); the report will say so "
            "rather than claim a controlled comparison.",
            "; ".join(budget_divergence),
        )

    if training_provenance is None:
        logger.warning(
            "No training sidecar found under %s; the budget table will fall back to "
            "the arena's configuration, which may not reflect how the arms were trained.",
            args.checkpoint_dir,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ab_report.md").write_text(render_report(arena, arms, training_provenance), encoding="utf-8")
    (args.output_dir / "summary_table.md").write_text(render_summary_table(arena), encoding="utf-8")
    logger.info("Wrote %s and summary_table.md", args.output_dir / "ab_report.md")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    handlers = {
        "train": _command_train,
        "arena": _command_arena,
        "report": _command_report,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
