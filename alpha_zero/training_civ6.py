"""Training pipeline for Civ6 AlphaZero.

Implements the AlphaZero self-play training loop:
1. Self-play: Generate games using MCTS + neural network
2. Training: Update network on generated data
3. Evaluation: Compare new network against previous version
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import shlex
import sys
import time
import warnings
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, TypedDict, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from alpha_zero.core.civ6_mcts import (
    Civ6MCTS,
    GumbelMCTS,
    GumbelMCTSConfig,
    MCTSConfig,
    SimpleMCTS,
)
from alpha_zero.core.civ6_network import MicroCivNetwork, compute_loss
from alpha_zero.core.mcts.gumbel import POLICY_TARGET_COMPLETED_Q, VALID_POLICY_TARGETS
from alpha_zero.core.mcts_constants import GREEDY_THRESHOLD
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.evaluation.benchmark_evidence import hardware_summary
from alpha_zero.evaluation.scoring import ScoreSummary, summarize
from alpha_zero.evaluation.scoring import win_rate as legacy_win_rate
from alpha_zero.multiplayer.network_heads import compute_nplayer_value_loss
from alpha_zero.multiplayer.scoring import (
    ScoringMethod,
    compute_player_values,
    compute_rankings,
)
from alpha_zero.training.diagnostics import StepRecord, prequential_value_metrics
from alpha_zero.training.run_record import IterationLog
from alpha_zero.utils.checkpoint import (
    GIT_SHA_ENV,
    build_provenance,
    check_provenance,
    read_git_sha,
    safe_load_checkpoint,
)
from alpha_zero.utils.constants import MCTS
from alpha_zero.utils.data_root import (
    assert_safe_data_root,
    default_checkpoint_dir,
    default_run_id,
    resolve_data_root,
    validate_run_id,
)
from alpha_zero.utils.device import inference_thread_limit, resolve_device_str
from alpha_zero.utils.game_budget import (
    DEFAULT_TRAINING_EVAL_GAMES,
    add_budget_arguments,
    enforce_cli,
)
from alpha_zero.utils.seeding import SeedStream, derive_seed, enforce_deterministic_backend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Environment
    board_size: int = 16
    max_turns: int = 200
    # Number of players. 2 keeps the classic scalar-value pipeline; > 2 enables
    # the N-player value pipeline (per-player value head + per-player targets).
    num_players: int = 2
    # How terminal game outcomes are converted to per-player training values.
    # MicroCiv only exposes a winner (no per-player scores), so the supported
    # methods are the winner-derivable ones in SUPPORTED_SCORING_METHODS:
    # "winner_take_all" (classic +1 winner / -1 loser / 0 draw) and
    # "linear_ranking" (zero-sum softer penalty: +1 winner, -1/(N-1) per loser,
    # 0 draw). "proportional" needs a per-player score signal this pipeline does
    # not provide, so it is not offered here.
    scoring_method: str = "winner_take_all"

    # Base RNG seed for the whole run. ``None`` means unseeded, which is what
    # every existing caller gets. Game ``i`` of a self-play round is seeded with
    # ``seed + i``, matching the convention documented in
    # ``cloud/gcp/selfplay_worker.py``, so games differ from one another while
    # the round as a whole replays exactly.
    seed: int | None = None

    # Network
    num_res_blocks: int = 6
    num_filters: int = 64

    # MCTS
    num_simulations: int = 50
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_threshold: int = 30

    # Gumbel MuZero MCTS (alternative to standard MCTS). Every field below
    # reaches BOTH the self-play and the evaluation search (gumbel_search_config).
    use_gumbel_mcts: bool = False
    gumbel_num_simulations: int = 16
    gumbel_num_sampled_actions: int = 16
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = MCTS.GUMBEL_C_SCALE
    gumbel_use_sequential_halving: bool = True
    gumbel_policy_target: str = POLICY_TARGET_COMPLETED_Q
    gumbel_greedy_threshold: float = GREEDY_THRESHOLD

    # Training
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    num_epochs: int = 10

    # Self-play
    num_games_per_iteration: int = 100
    num_iterations: int = 100
    replay_buffer_size: int = 50000
    min_replay_size: int = 1000

    # Checkpointing
    checkpoint_dir: str = "checkpoints/civ6"
    log_dir: str = "logs/civ6"
    save_interval: int = 10
    # In-loop evaluation against a random opponent, decoupled from checkpointing
    # (LF-A14). ``eval_interval=None`` keeps the historical coupling: evaluate on
    # the checkpoint cadence, so a run that sets neither flag is unchanged.
    # ``eval_games=0`` turns in-loop evaluation off.
    eval_interval: int | None = None
    eval_games: int = DEFAULT_TRAINING_EVAL_GAMES
    # Resume. `resume_from` restores network weights only -- see
    # `resume_network_weights` for what is deliberately not restored.
    # `start_iteration` is the loop index to begin at; 0 means "from the top".
    resume_from: str | None = None
    start_iteration: int = 0

    # Iteration-0 data source. "simple_mcts" (the historical behaviour) plays
    # iteration 0 with random-rollout SimpleMCTS because the network is untrained;
    # "none" plays it with the configured search on the freshly initialised
    # network, so the first training set already comes from the search under study.
    bootstrap: str = "simple_mcts"

    # Run identity stamped on every iterations.jsonl line. ``None`` falls back to
    # CIV6AZ_RUN_ID (set by the run-card driver), then to a timestamped default.
    run_id: str | None = None

    # Device. "auto" is resolved to a concrete device in __post_init__ rather
    # than probed at import time, so constructing a TrainingConfig on a GPU box
    # inside a test no longer silently selects cuda.
    device: str = "auto"

    def __post_init__(self) -> None:
        # Resolve rather than merely warn: an unresolved "cuda" on a CPU-only
        # host would otherwise crash later inside .to().
        self.device = resolve_device_str(self.device)
        if self.eval_interval is not None and self.eval_interval < 1:
            raise ValueError(f"eval_interval must be >= 1 or None, got {self.eval_interval}")
        if self.eval_games < 0:
            raise ValueError(f"eval_games must be >= 0, got {self.eval_games}")

    @property
    def resolved_eval_interval(self) -> int:
        """Iterations between in-loop evaluations: ``eval_interval``, else ``save_interval``."""
        return self.eval_interval if self.eval_interval is not None else self.save_interval


#: ``checkpoint_<N>.pt`` is written *after* the iteration at loop index
#: ``N - 1`` completes, so resuming from it means starting at loop index ``N``.
#: That off-by-one is the whole reason this is a named pattern rather than an
#: inline regex: guessing it wrong silently overwrites earlier checkpoints.
_CHECKPOINT_ITERATION_RE = re.compile(r"^checkpoint_(\d+)\.pt$")


def infer_start_iteration(resume_path: str) -> int | None:
    """Return the loop index to resume at, or ``None`` if the name says nothing.

    Args:
        resume_path: Path to a checkpoint, e.g. ``.../checkpoint_7.pt``.

    Returns:
        The loop index to resume at (``7`` for ``checkpoint_7.pt``), or ``None``
        for any other name -- ``latest.pt`` carries no iteration number, so the
        caller must be told rather than given a guess.
    """
    match = _CHECKPOINT_ITERATION_RE.match(os.path.basename(resume_path))
    return int(match.group(1)) if match else None


def resume_network_weights(
    network: nn.Module,
    resume_path: str,
    *,
    device: str = "cpu",
    max_turns: int | None = None,
    board_size: int | None = None,
) -> None:
    """Load weights from a checkpoint into an already-constructed network.

    The network is built from the *environment's* action space and spatial
    channel count, never from the checkpoint's own saved config, so a checkpoint
    trained at a different board size or player count is rejected here instead
    of being silently cross-loaded. The compatibility check is
    ``load_state_dict`` itself rather than a hand-written comparison of config
    keys, so it cannot drift out of step with the real architecture.

    Restores **weights only**. Optimizer moments, the replay buffer and its
    accumulated games are not restored; the caller is expected to say so.

    Args:
        network: The network to load into, already on ``device``.
        resume_path: Path to a checkpoint written by this script.
        device: Device to map the checkpoint onto.
        max_turns: The horizon this run plays at. When given, a checkpoint
            recorded at a different ``max_turns`` raises ``HorizonMismatch``; a
            legacy file with no provenance only warns.
        board_size: The board edge this run plays at. A recorded mismatch is
            reported by name; without provenance ``load_state_dict`` still
            refuses the shape.

    Raises:
        HorizonMismatch: ``max_turns`` disagrees with the checkpoint's.
        ValueError: If the checkpoint has no ``state_dict``, or its parameter
            shapes do not match ``network``.
        FileNotFoundError: If ``resume_path`` does not exist. I/O failures are
            re-raised as-is by the loader rather than being reported as a
            checkpoint problem, so a typo'd path says so plainly.
        UnsafeCheckpointError: If the tensor-only unpickler rejects the payload
            and unsafe loading has not been authorised (via
            ``CIV6AZ_ALLOW_UNSAFE_CHECKPOINT=1``). This is deliberate: a failed
            safe load never downgrades on its own.
    """
    result = safe_load_checkpoint(resume_path, map_location=device)
    checkpoint = result.data
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            f"{resume_path} is not a checkpoint written by this trainer: "
            "expected a dict with a 'state_dict' key."
        )
    metadata = checkpoint.get("metadata")
    check_provenance(metadata, expected_max_turns=max_turns, source=resume_path)
    if (
        board_size is not None
        and isinstance(metadata, dict)
        and metadata.get("board_size") not in (None, board_size)
    ):
        raise ValueError(
            f"{resume_path} was trained at board_size={metadata['board_size']}, "
            f"but this run uses {board_size}."
        )
    try:
        network.load_state_dict(checkpoint["state_dict"])
    except RuntimeError as exc:
        raise ValueError(
            f"{resume_path} does not match the network this run builds. The "
            "usual cause is a different --board_size, --num_players or action "
            "space than the checkpoint was trained with."
        ) from exc
    if not result.loaded_safely:
        logger.warning(
            "Checkpoint %s was loaded with weights_only=False. Only do this for "
            "checkpoints you trust.",
            resume_path,
        )


#: Evaluation games played per checkpoint on a normal (non-dry) run. Owned by
#: ``game_budget`` so the cost model and the trainer cannot drift apart.
DEFAULT_EVAL_GAMES = DEFAULT_TRAINING_EVAL_GAMES

# Superseded by `alpha_zero.utils.seeding.derive_seed`, which separates RNG
# streams by NAME rather than by an offset that had to be re-reasoned whenever a
# call site moved. Retained because it is part of this module's public surface;
# nothing in the package reads it any more.
#
# .. deprecated:: 1.9
#    Use :class:`~alpha_zero.utils.seeding.SeedStream` and
#    :func:`~alpha_zero.utils.seeding.derive_seed` instead.
EVAL_SEED_OFFSET = 1_000_000

#: Config fields that dominate dry-run cost. These are the *inner* dimensions:
#: total work scales roughly as
#: ``games * max_turns * actions_per_turn * num_simulations`` against a network
#: whose forward cost grows with ``board_size ** 2``.
_INNER_DIMENSION_FIELDS: tuple[str, ...] = ("board_size", "max_turns", "num_simulations")

#: Inner dimensions bounded only when the run uses the Gumbel search; under PUCT
#: the field is inert, so a PUCT config is left exactly as given.
_GUMBEL_INNER_DIMENSION_FIELDS: tuple[str, ...] = ("gumbel_num_simulations",)

#: The CLI flag that sets each bounded field, for the "pass --X to override" hint.
_FIELD_FLAGS: dict[str, str] = {"gumbel_num_simulations": "gumbel_sims"}


@dataclass(frozen=True)
class DryRunProfile:
    """The bounded shape of a ``--dry_run`` smoke test.

    ``--dry_run`` used to shrink only the *outer* loop -- iterations, games per
    iteration, epochs, evaluation games -- while leaving ``board_size``,
    ``max_turns`` and ``num_simulations`` at their production defaults (16, 200,
    50). Since cost is dominated by those inner dimensions, the "quick test run"
    was really 24 full-size games with full-depth search: measured at over 3.5
    hours of CPU without finishing, against a documented success criterion that
    says it completes.

    Every dry-run magic number now lives here, in one place, so the smoke test's
    shape can be read and changed without hunting through :func:`train`.
    """

    # Outer loop: always applied when dry_run is set.
    num_iterations: int = 2
    num_games_per_iteration: int = 2
    num_epochs: int = 1
    num_eval_games: int = 2

    # Inner dimensions: applied only to fields left at their TrainingConfig
    # default, so an explicit --board_size 12 still wins. See apply_to.
    board_size: int = 6
    max_turns: int = 8
    num_simulations: int = 4
    # Bounded only under --mcts gumbel (see _GUMBEL_INNER_DIMENSION_FIELDS).
    gumbel_num_simulations: int = 4

    def apply_to(self, config: TrainingConfig) -> TrainingConfig:
        """Return *config* with unset inner dimensions bounded for a smoke test.

        A field is overridden only when it still holds its :class:`TrainingConfig`
        default, so a caller who deliberately asked for a larger board keeps it.
        The trade-off: a caller who explicitly passes the default value is
        indistinguishable from one who omitted the flag, and gets the smoke-test
        value. That is the right default for a flag whose entire purpose is to
        finish quickly.

        The Gumbel budget is bounded only when the run uses the Gumbel search,
        and the budget the Gumbel search will actually use is always logged, so
        a dry run never reports one budget while searching with another.

        The original config is not mutated; a copy is returned.
        """
        defaults = TrainingConfig()
        fields = _INNER_DIMENSION_FIELDS + (
            _GUMBEL_INNER_DIMENSION_FIELDS if config.use_gumbel_mcts else ()
        )
        overrides = {
            field: getattr(self, field)
            for field in fields
            if getattr(config, field) == getattr(defaults, field)
        }
        for field, value in overrides.items():
            logger.info(
                "dry-run: bounding %s %s -> %s (pass --%s to override)",
                field,
                getattr(config, field),
                value,
                _FIELD_FLAGS.get(field, field),
            )
        bounded = replace(config, **overrides) if overrides else config
        if not overrides:
            logger.debug("dry-run: all inner dimensions set explicitly; none overridden")
        if bounded.use_gumbel_mcts:
            logger.info(
                "dry-run: Gumbel search uses %d simulations (k=%d) in self-play, %d in evaluation",
                bounded.gumbel_num_simulations,
                bounded.gumbel_num_sampled_actions,
                max(1, bounded.gumbel_num_simulations // 2),
            )
        return bounded


#: The profile used by every ``dry_run=True`` code path.
DRY_RUN_PROFILE = DryRunProfile()


@dataclass
class TrainingExample:
    """Single training example.

    ``value`` is the scalar outcome from the acting player's perspective
    (kept for backward compatibility and 2-player training). ``value_vector``
    is the optional per-player outcome vector in *relative* player order
    (index 0 = acting player), populated only for N-player (> 2) games so the
    N-player value head can be trained. When ``value_vector`` is ``None`` the
    pipeline uses the scalar ``value`` exactly as before.
    """

    spatial: np.ndarray
    vector: np.ndarray
    action_mask: np.ndarray
    policy: np.ndarray
    value: float
    value_vector: np.ndarray | None = None


class GameHistoryItem(TypedDict):
    """Type for game history items."""

    spatial: np.ndarray
    vector: np.ndarray
    action_mask: np.ndarray
    policy: np.ndarray
    player: int


class ReplayBuffer:
    """Replay buffer for storing training examples."""

    def __init__(self, max_size: int = 50000, seed: int | None = None) -> None:
        """Initialize replay buffer.

        Args:
            max_size: Maximum number of examples to store.
            seed: Seed for the sampling RNG. ``None`` means unseeded, matching
                the previous behaviour of drawing from the global numpy state.
        """
        self.buffer: deque[TrainingExample] = deque(maxlen=max_size)
        # Which examples a batch contains determines the gradient, so an
        # unseeded sampler makes the training half of a run unrepeatable even
        # when self-play is fully determined. Measured: with only self-play
        # seeded, two `--seed 42` runs agreed on all 525 lines of self-play
        # output and differed on the very first epoch loss.
        self._rng = np.random.default_rng(seed)

    def add(self, example: TrainingExample) -> None:
        """Add example to buffer."""
        self.buffer.append(example)

    def add_game(self, examples: list[TrainingExample]) -> None:
        """Add all examples from a game."""
        for example in examples:
            self.add(example)

    def sample(self, batch_size: int) -> list[TrainingExample]:
        """Sample batch from buffer.

        Args:
            batch_size: Number of examples to sample.

        Returns:
            List of sampled examples.
        """
        indices = self._rng.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)

    def mean_value(self) -> float:
        """Mean scalar value target over the buffer; ``0.0`` when it is empty."""
        if not self.buffer:
            return 0.0
        return float(np.mean([example.value for example in self.buffer]))


class ReplayDataset(Dataset):
    """PyTorch dataset for replay buffer."""

    def __init__(self, examples: list[TrainingExample]) -> None:
        """Initialize dataset.

        Args:
            examples: List of training examples.
        """
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        # Per-player value vector (N-player) when present; scalar otherwise.
        # Both collate cleanly: scalars stack to (B,), vectors to (B, P).
        if ex.value_vector is not None:
            value = torch.from_numpy(np.asarray(ex.value_vector)).float()
        else:
            value = torch.tensor(ex.value).float()
        return {
            "spatial": torch.from_numpy(ex.spatial).float(),
            "vector": torch.from_numpy(ex.vector).float(),
            "action_mask": torch.from_numpy(ex.action_mask).float(),
            "policy": torch.from_numpy(ex.policy).float(),
            "value": value,
        }


# Scoring methods this pipeline can honour. MicroCiv exposes only a winner
# (no per-player scores), so PROPORTIONAL — which needs a score signal — is
# excluded: it would silently fall back to LINEAR_RANKING inside
# multiplayer.scoring. Wire a real per-player score signal to add it.
SUPPORTED_SCORING_METHODS: tuple[ScoringMethod, ...] = (
    ScoringMethod.WINNER_TAKE_ALL,
    ScoringMethod.LINEAR_RANKING,
)


def _scoring_method_from_str(name: str) -> ScoringMethod:
    """Resolve a scoring-method name to a supported :class:`ScoringMethod`.

    Names that are unknown *or* unsupported by this winner-only pipeline (e.g.
    ``proportional``) fall back to ``winner_take_all`` (the classic AlphaZero
    +1/-1/0 targets) with a warning, so a bad config never crashes training and
    never silently trains on a degenerate score-less proportional signal.
    """
    try:
        method = ScoringMethod(name)
    except ValueError:
        logger.warning("Unknown scoring_method %r; falling back to 'winner_take_all'", name)
        return ScoringMethod.WINNER_TAKE_ALL
    if method not in SUPPORTED_SCORING_METHODS:
        logger.warning(
            "scoring_method %r needs a per-player score signal this pipeline does "
            "not provide; falling back to 'winner_take_all'",
            name,
        )
        return ScoringMethod.WINNER_TAKE_ALL
    return method


def compute_absolute_player_values(
    winner: int | None,
    num_players: int,
    method: ScoringMethod = ScoringMethod.WINNER_TAKE_ALL,
) -> np.ndarray:
    """Per-player training values in ABSOLUTE player order (index p = player p).

    Reuses :mod:`alpha_zero.multiplayer.scoring`. With ``winner_take_all`` this
    reproduces the classic +1 (winner) / -1 (loser) / 0 (draw) targets exactly,
    so the 2-player pipeline is unchanged.

    MicroCiv exposes only a *winner* (no per-player scores or full rankings), so
    ``compute_rankings`` ties every non-winner at rank 1. Feeding that to
    ``linear_ranking`` would (for ``num_players >= 4``) assign losers a
    *positive* value — making a loss score better than a draw. To keep a sane,
    zero-sum signal we special-case ``linear_ranking`` here to a softer penalty:
    winner ``+1``, each of the ``N - 1`` losers ``-1 / (N - 1)`` (draw = 0). For
    2 players this coincides with ``winner_take_all`` (``+1 / -1``).

    Args:
        winner: Winning player index, or ``None`` for a draw.
        num_players: Total number of players (>= 2).
        method: Outcome -> value conversion strategy.

    Returns:
        Array of shape ``(num_players,)`` with each value in ``[-1, 1]``.
    """
    if winner is None:
        # Draw: every scoring method agrees on all-zero targets.
        return np.zeros(num_players, dtype=np.float32)

    if method == ScoringMethod.LINEAR_RANKING:
        # Winner-only zero-sum mapping (see docstring): losers must stay
        # negative, so we cannot delegate to compute_rankings here.
        loser_value = -1.0 / (num_players - 1)
        values_arr = np.full(num_players, loser_value, dtype=np.float32)
        values_arr[winner] = 1.0
        return values_arr

    rankings = compute_rankings(winner=winner, num_players=num_players)
    values = compute_player_values(rankings, num_players=num_players, method=method)
    return np.array([values[p] for p in range(num_players)], dtype=np.float32)


def relative_value_vector(
    absolute_values: np.ndarray, acting_player: int, num_players: int
) -> np.ndarray:
    """Reorder absolute per-player values into the acting player's frame.

    The MicroCiv encoder is player-relative (index 0 == current player), so the
    N-player value head predicts values in relative order. This maps absolute
    outcomes into that frame: ``out[r] = absolute_values[(acting_player + r) % N]``.
    ``out[0]`` therefore equals the acting player's own outcome.
    """
    return np.array(
        [absolute_values[(acting_player + r) % num_players] for r in range(num_players)],
        dtype=np.float32,
    )


class _PredictNetwork(Protocol):
    """Structural type for a network exposing MCTS's ``predict`` interface."""

    def predict(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class CurrentPlayerValueView(nn.Module):
    """Adapt an N-player network to scalar-value MCTS.

    N-player networks emit a per-player value vector in *relative* order
    (index 0 = acting player). Scalar MCTS variants (``Civ6MCTS`` /
    ``GumbelMCTS``) expect one value per state, so this thin wrapper reduces the
    vector to its current-player (index 0) component. It holds a reference to
    the wrapped network as a submodule, so parameters are shared and self-play
    runs on the exact weights being trained. An already-scalar value passes
    through unchanged, keeping the wrapper safe for 2-player networks too.
    """

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    @staticmethod
    def _current_player_value(value: torch.Tensor) -> torch.Tensor:
        if value.dim() >= 2 and value.shape[-1] > 1:
            return value[..., 0]
        return value.reshape(value.shape[0]) if value.dim() >= 1 else value

    def forward(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        policy_logits, value = self.network(spatial, vector, action_mask)
        return policy_logits, self._current_player_value(value)

    def predict(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # nn.Module has no typed ``predict``; the wrapped network does.
        network = cast("_PredictNetwork", self.network)
        policy_probs, value = network.predict(spatial, vector, action_mask)
        return policy_probs, self._current_player_value(value)


def _mcts_network(network: torch.nn.Module, config: TrainingConfig) -> torch.nn.Module:
    """Return the network view MCTS should use for self-play / evaluation.

    For N-player training the network emits a value vector, so it is wrapped in
    :class:`CurrentPlayerValueView`; for 2-player training the network is
    returned unchanged (byte-identical to the legacy path).
    """
    if getattr(config, "num_players", 2) > 2:
        view = CurrentPlayerValueView(network)
        return view.to(config.device)
    return network


def gumbel_search_config(
    config: TrainingConfig,
    *,
    num_simulations: int,
    seed: int | None,
    **extra: Any,
) -> GumbelMCTSConfig:
    """The :class:`GumbelMCTSConfig` a run's self-play and evaluation searches use.

    One builder for both, so a ``gumbel_*`` field cannot reach one search and
    silently miss the other. ``max_considered_actions`` is raised to ``k`` when
    ``k`` exceeds its default, since the config rejects ``k`` above it.

    Args:
        config: The run's training configuration.
        num_simulations: Simulation budget for this search.
        seed: Seed for this search's own RNG.
        **extra: Further :class:`GumbelMCTSConfig` fields (e.g. temperature).
    """
    k = config.gumbel_num_sampled_actions
    return GumbelMCTSConfig(
        num_simulations=num_simulations,
        num_sampled_actions=k,
        max_considered_actions=max(MCTS.GUMBEL_MAX_CONSIDERED_ACTIONS, k),
        c_visit=config.gumbel_c_visit,
        c_scale=config.gumbel_c_scale,
        use_sequential_halving=config.gumbel_use_sequential_halving,
        policy_target=config.gumbel_policy_target,
        greedy_threshold=config.gumbel_greedy_threshold,
        seed=seed,
        **extra,
    )


def self_play_game(
    env: MicroCivEnv,
    mcts: Civ6MCTS | SimpleMCTS | GumbelMCTS,
    config: TrainingConfig,
    game_seed: int | None = None,
    *,
    step_sink: list[StepRecord] | None = None,
    game_idx: int = 0,
) -> list[TrainingExample]:
    """Play a single self-play game.

    Args:
        env: Game environment.
        mcts: MCTS instance.
        config: Training configuration.
        game_seed: Seed for this game's environment. ``None`` leaves the
            environment's existing RNG alone, which is the historical behaviour.
        step_sink: When given, one :class:`StepRecord` is appended per step
            actually played on ``env``. Search rollouts step *clones* of ``env``
            inside ``mcts.search`` and are never recorded.
        game_idx: Index of this game in its round, stamped on each record.

    Returns:
        List of training examples from the game.
    """
    examples = []
    obs, _ = env.reset(seed=game_seed)

    game_history: list[GameHistoryItem] = []

    while True:
        # Get MCTS policy
        action_probs, _ = mcts.search(env)  # type: ignore[arg-type]

        # Store state and policy
        game_history.append(
            {
                "spatial": obs["spatial"].copy(),
                "vector": obs["vector"].copy(),
                "action_mask": obs["action_mask"].copy(),
                "policy": action_probs.copy(),
                "player": env.current_player,
            },
        )

        # Select action
        temperature = config.temperature if env.turn < config.temperature_threshold else 0.0
        action = mcts.select_action(action_probs, temperature)

        # Take action
        obs, _, terminated, truncated, info = env.step(action)
        if step_sink is not None:
            step_sink.append(
                StepRecord(
                    game_idx=game_idx,
                    action=int(action),
                    action_type=str(env.action_space_handler.decode_action(int(action))["type"]),
                    success=bool(info["action_success"]),
                )
            )

        if terminated or truncated:
            break

    # Assign values based on game outcome. Derive player count from the env
    # (authoritative for the game just played) so this works when called
    # directly on an N-player env, not only via train().
    winner = info.get("winner")
    num_players = int(getattr(env, "num_players", 2))
    if num_players < 2:
        # A real MicroCivEnv always enforces num_players >= 2 (see
        # MicroCivEnv.__init__), so this branch is unreachable with a real
        # env. It only triggers for a non-conforming env double (e.g. a test
        # mock that never set `num_players`, whose auto-vivified attribute
        # resolves via int() to 1). Fall back to the historical 2-player
        # default rather than crashing self-play.
        logger.warning(
            "self_play_game: env.num_players resolved to %d (<2); defaulting "
            "to 2-player scoring. This usually means `env` is missing a real "
            "`num_players` attribute (e.g. an incompletely configured mock).",
            num_players,
        )
        num_players = 2
    method = _scoring_method_from_str(getattr(config, "scoring_method", "winner_take_all"))
    absolute_values = compute_absolute_player_values(winner, num_players, method)

    for hist in game_history:
        player = int(hist["player"])
        # Scalar target from the acting player's perspective. For 2-player
        # winner-take-all this is exactly the legacy +1 / -1 / 0 signal.
        value = float(absolute_values[player])
        # Per-player relative target vector only for N-player games.
        value_vector = (
            relative_value_vector(absolute_values, player, num_players) if num_players > 2 else None
        )

        examples.append(
            TrainingExample(
                spatial=hist["spatial"],
                vector=hist["vector"],
                action_mask=hist["action_mask"],
                policy=hist["policy"],
                value=value,
                value_vector=value_vector,
            ),
        )

    return examples


def run_self_play(
    network: torch.nn.Module | None,
    config: TrainingConfig,
    num_games: int,
    use_simple_mcts: bool = False,
    iteration: int = 0,
    step_sink: list[StepRecord] | None = None,
) -> list[TrainingExample]:
    """Run multiple self-play games.

    Args:
        network: Neural network guiding MCTS. May be ``None`` only when
            ``use_simple_mcts=True`` (SimpleMCTS uses random rollouts and never
            touches the network); required otherwise.
        config: Training configuration.
        num_games: Number of games to play.
        use_simple_mcts: If True, use SimpleMCTS instead of network-guided MCTS.
        iteration: Loop index; varies every seed stream (see ``derive_seed``).
        step_sink: Optional list receiving one :class:`StepRecord` per played
            step of every game (see :func:`self_play_game`).

    Returns:
        List of training examples from all games.

    Raises:
        ValueError: If ``network`` is ``None`` and ``use_simple_mcts`` is False.
    """
    if network is None and not use_simple_mcts:
        raise ValueError(
            "run_self_play requires a network for network-guided MCTS; "
            "pass use_simple_mcts=True to run without one."
        )

    examples = []
    env = MicroCivEnv(
        board_size=config.board_size,
        max_turns=config.max_turns,
        num_players=config.num_players,
        seed=derive_seed(config.seed, SeedStream.SELF_PLAY_ENV, iteration),
    )

    # Every stream is derived from a NAME, so none can collide with another, and
    # every one varies with `iteration`. Before this, self-play seeds carried no
    # iteration term: a seeded run regenerated the same games every iteration.
    search_seed = derive_seed(config.seed, SeedStream.SELF_PLAY_SEARCH, iteration)

    if use_simple_mcts:
        # SimpleMCTS ignores the network entirely (random rollouts), so skip
        # building the scalar-view wrapper.
        mcts: Civ6MCTS | SimpleMCTS | GumbelMCTS = SimpleMCTS(
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            seed=search_seed,
        )
    elif config.use_gumbel_mcts:
        # Use Gumbel MuZero MCTS for efficient search with few simulations.
        # N-player networks emit value vectors; wrap so scalar MCTS gets a scalar.
        gumbel_config = gumbel_search_config(
            config,
            num_simulations=config.gumbel_num_simulations,
            seed=search_seed,
            temperature=config.temperature,
            temperature_threshold=config.temperature_threshold,
        )
        mcts = GumbelMCTS(_mcts_network(network, config), gumbel_config, device=config.device)  # type: ignore[arg-type]
        logger.info(f"Using Gumbel MuZero MCTS with {config.gumbel_num_simulations} simulations")
    else:
        # N-player networks emit value vectors; wrap so scalar MCTS gets a scalar.
        mcts_config = MCTSConfig(
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
            temperature=config.temperature,
            temperature_threshold=config.temperature_threshold,
            seed=search_seed,
        )
        mcts = Civ6MCTS(_mcts_network(network, config), mcts_config, device=config.device)  # type: ignore[arg-type]

    # MCTS evaluates one position at a time, so this whole phase is batch-of-1
    # inference. See alpha_zero.utils.device for why that needs a thread cap.
    with inference_thread_limit():
        for game_idx in tqdm(range(num_games), desc="Self-play"):
            game_examples = self_play_game(
                env,
                mcts,
                config,
                game_seed=derive_seed(config.seed, SeedStream.SELF_PLAY_ENV, iteration, game_idx),
                step_sink=step_sink,
                game_idx=game_idx,
            )
            examples.extend(game_examples)

    return examples


def _masked_policy_loss(
    policy_logits: torch.Tensor,
    target_policy: torch.Tensor,
    action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked cross-entropy policy loss.

    Identical to the policy term in :func:`compute_loss`; factored out so the
    N-player loss path can reuse it without the deprecated scalar-value bundle.
    """
    if action_mask is not None:
        target_policy = target_policy * action_mask
        target_policy = target_policy / (target_policy.sum(dim=-1, keepdim=True) + 1e-8)
    log_probs = F.log_softmax(policy_logits, dim=-1)
    return -torch.sum(target_policy * log_probs, dim=-1).mean()


def train_epoch(
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    config: TrainingConfig,
) -> dict[str, float]:
    """Train network for one epoch.

    Args:
        network: Neural network.
        optimizer: Optimizer.
        dataloader: Training data loader.
        config: Training configuration.

    Returns:
        Dictionary of loss values.
    """
    network.train()
    total_loss = 0.0
    policy_loss = 0.0
    value_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        spatial = batch["spatial"].to(config.device)
        vector = batch["vector"].to(config.device)
        action_mask = batch["action_mask"].to(config.device)
        target_policy = batch["policy"].to(config.device)
        target_value = batch["value"].to(config.device)

        optimizer.zero_grad()

        # Forward pass
        policy_logits, value = network(spatial, vector, action_mask)

        # Compute loss. For N-player training the value tensor is per-player
        # (B, P), so use the N-player value loss; the 2-player path is
        # byte-identical to the legacy scalar loss.
        if getattr(config, "num_players", 2) > 2:
            # Guard against a scalar value head slipping through with a
            # per-player config (e.g. an NPlayerValueHead import fallback):
            # compute_nplayer_value_loss would broadcast (B,1) vs (B,P)
            # silently and train on a wrong signal. Fail loudly instead.
            if value.ndim != 2 or value.shape[-1] != config.num_players:
                raise RuntimeError(
                    f"Expected per-player value of shape (B, {config.num_players}) for "
                    f"num_players={config.num_players}, got {tuple(value.shape)}. "
                    "The network is not emitting per-player values (scalar value head?)."
                )
            # Mirror the guard on the targets: scalar targets (e.g. stale replay
            # data or a miswired value_vector) would be silently broadcast from
            # (B,) to (B, P) by compute_nplayer_value_loss and train on
            # duplicated values. Fail loudly instead.
            if target_value.ndim != 2 or target_value.shape[-1] != config.num_players:
                raise RuntimeError(
                    f"Expected per-player value targets of shape (B, {config.num_players}) "
                    f"for num_players={config.num_players}, got {tuple(target_value.shape)}. "
                    "Self-play must populate TrainingExample.value_vector for N-player runs."
                )
            p_loss = _masked_policy_loss(policy_logits, target_policy, action_mask)
            v_loss = compute_nplayer_value_loss(value, target_value, config.num_players)
            loss = p_loss + v_loss
        else:
            loss, p_loss, v_loss = compute_loss(
                policy_logits,
                value,
                target_policy,
                target_value,
                action_mask,
            )

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(network.parameters(), config.gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        policy_loss += p_loss.item()
        value_loss += v_loss.item()
        num_batches += 1

    return {
        "total_loss": total_loss / num_batches,
        "policy_loss": policy_loss / num_batches,
        "value_loss": value_loss / num_batches,
    }


def evaluate_against_random(
    network: torch.nn.Module,
    config: TrainingConfig,
    num_games: int = DEFAULT_TRAINING_EVAL_GAMES,
    iteration: int = 0,
) -> ScoreSummary:
    """Evaluate network against random player.

    Args:
        network: Neural network.
        config: Training configuration.
        num_games: Number of games to play.
        iteration: Loop index, for the caller's bookkeeping. Deliberately unused
            by the seeds (see the comment below).

    Returns:
        The full :class:`~alpha_zero.evaluation.scoring.ScoreSummary`: counts,
        draw-aware score rate with its ternary-variance interval, draw rate.

    .. versionchanged:: 2.0
       Returned ``wins / num_games`` as a float before LF-A3. That statistic
       scores every draw as a loss: the final checkpoint of the 2026-08-28
       validation run scored 0 W / 10 D / 10 L and was reported as ``0.0``,
       which reads as "lost every game" but did not. The float is still
       available, with a deprecation warning, from
       :func:`evaluate_win_rate_against_random`.
    """
    # Named streams keep the evaluation's three sources (environment, search,
    # random opponent) distinct from each other and from self-play's, so a
    # seeded run replays exactly without any two sharing a sequence.
    #
    # Deliberately NOT varying with `iteration`, unlike self-play. Evaluation is
    # a HELD-OUT set: comparing iteration N against N+1 is only meaningful if
    # both were scored on the same positions. Re-drawing the maps each iteration
    # would add map-difficulty variance to the very curve the learning-trend
    # check reads, on top of the sampling noise a 20-game probe already carries
    # (docs/evidence/validation_iteration_zero/ measured one checkpoint spanning
    # 0.25-0.40 across five probes). Self-play varies so the learner sees new
    # positions; evaluation is fixed so the score is comparable.
    env = MicroCivEnv(
        board_size=config.board_size,
        max_turns=config.max_turns,
        num_players=config.num_players,
        seed=derive_seed(config.seed, SeedStream.EVAL_ENV),
    )
    search_seed = derive_seed(config.seed, SeedStream.EVAL_SEARCH)
    # Evaluate with the search the run trains with. Before this the eval path
    # was hard-coded to PUCT, so a `use_gumbel_mcts` run was generated by one
    # search and scored by another. The default (PUCT at half the budget) is
    # unchanged.
    mcts: Civ6MCTS | GumbelMCTS
    if config.use_gumbel_mcts:
        gumbel_config = gumbel_search_config(
            config,
            num_simulations=max(1, config.gumbel_num_simulations // 2),
            seed=search_seed,
        )
        mcts = GumbelMCTS(
            cast(Any, _mcts_network(network, config)), gumbel_config, device=config.device
        )
    else:
        mcts_config = MCTSConfig(
            num_simulations=config.num_simulations // 2,
            seed=search_seed,
        )
        # N-player networks emit value vectors; wrap so scalar MCTS gets a scalar.
        mcts = Civ6MCTS(_mcts_network(network, config), mcts_config, device=config.device)  # type: ignore[arg-type]

    wins = 0
    draws = 0
    # The random opponent. Unseeded, this alone was enough to make a 20-game
    # probe unrepeatable -- five probes of one checkpoint spanned 0.25 to 0.40
    # (docs/evidence/validation_iteration_zero/).
    rng = np.random.default_rng(derive_seed(config.seed, SeedStream.EVAL_OPPONENT))

    # Batch-of-1 inference, same as self-play; see run_self_play.
    with inference_thread_limit():
        for game_idx in range(num_games):
            _obs, _info = env.reset(seed=derive_seed(config.seed, SeedStream.EVAL_ENV, game_idx))
            # Rotate which seat the network occupies across all players.
            network_player = game_idx % config.num_players

            while True:
                if env.current_player == network_player:
                    # Network's turn
                    action_probs, _ = mcts.search(env, add_noise=False)  # type: ignore[arg-type]
                    action = mcts.select_action(action_probs, temperature=0.0)
                else:
                    # Random player's turn
                    mask = env.get_valid_actions()
                    valid = np.where(mask > 0)[0]
                    action = rng.choice(valid) if len(valid) > 0 else 0

                _obs, _, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    winner = info.get("winner")
                    if winner == network_player:
                        wins += 1
                    elif winner is None:
                        draws += 1
                    break

    losses = num_games - wins - draws
    # Every rate from the one SSOT, so this line, the arena and the promotion
    # gate cannot disagree about what a draw is worth.
    win_rate = legacy_win_rate(wins, draws, losses)
    summary = summarize(wins, draws, losses)
    logger.info(
        "Eval vs random: %d W / %d D / %d L over %d games "
        "(win rate %.1f%%, draw-aware score rate %.3f, draw rate %.1f%%)",
        wins,
        draws,
        losses,
        num_games,
        win_rate * 100,
        summary.score_rate,
        summary.draw_rate * 100,
    )
    # Machine-parseable twin of the line above (scripts/summarize_training_log.py
    # reads it): the draw-aware statistic with its interval, beside the legacy one.
    logger.info(
        "Eval score rate vs random: %.4f (95%% CI [%.4f, %.4f], draw rate %.4f, "
        "win rate %.4f, n=%d)",
        summary.score_rate,
        summary.ci_low,
        summary.ci_high,
        summary.draw_rate,
        win_rate,
        num_games,
    )
    return summary


def evaluate_win_rate_against_random(
    network: torch.nn.Module,
    config: TrainingConfig,
    num_games: int = DEFAULT_TRAINING_EVAL_GAMES,
    iteration: int = 0,
) -> float:
    """Deprecated float shim: ``wins / num_games`` from :func:`evaluate_against_random`.

    .. deprecated:: 2.0
       Draws count as losses in this number. Use :func:`evaluate_against_random`
       and read ``score_rate`` (draw-aware) or the counts from its summary.
    """
    warnings.warn(
        "evaluate_win_rate_against_random is deprecated: it scores draws as losses. "
        "Use evaluate_against_random, which returns a ScoreSummary.",
        DeprecationWarning,
        stacklevel=2,
    )
    summary = evaluate_against_random(network, config, num_games=num_games, iteration=iteration)
    return legacy_win_rate(summary.wins, summary.draws, summary.losses)


#: Version of the ``eval_<iter>.json`` layout written by :func:`write_eval_json`.
EVAL_JSON_SCHEMA_VERSION = 1

#: ``docs/evidence/SCHEMA.md`` epoch for anything this module produces: the local
#: trainer post-dates the 2026-09-04 search fixes by construction.
EVAL_SEARCH_EPOCH = "post-2026-09-04"


def write_eval_json(
    path: str | os.PathLike[str], summary: ScoreSummary, meta: dict[str, Any]
) -> dict[str, Any]:
    """Write one in-loop evaluation as JSON, atomically, and return its ``result``.

    The file is ``{"eval_schema_version", **meta, "result"}``: ``result`` is the
    summary's counts, ``score_rate``, ``draw_rate`` and interval, plus the legacy
    ``win_rate`` beside it (never instead of it). ``meta`` carries the caller's
    context -- for :func:`train` the iteration, the opponent, the checkpoint
    provenance and an ``evidence`` block shaped like ``docs/evidence/SCHEMA.md``.
    It is a run artifact under the checkpoint directory, not committed evidence.
    """
    result: dict[str, Any] = dict(summary.to_dict())
    result["win_rate"] = legacy_win_rate(summary.wins, summary.draws, summary.losses)
    payload = {"eval_schema_version": EVAL_JSON_SCHEMA_VERSION, **meta, "result": result}
    target = Path(path)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return result


def eval_evidence_block(
    config: TrainingConfig,
    provenance: dict[str, Any],
    games_total: int,
    *,
    iteration: int | None = None,
) -> dict[str, Any]:
    """An ``evidence`` block (``docs/evidence/SCHEMA.md``) for one in-loop evaluation.

    ``iteration`` is 1-based (the network after that many iterations). The weights
    scored may never be saved -- evaluation and checkpoint cadences are independent
    -- so the claim names the iteration, and ``eval_<iter>.json`` names the
    checkpoint or ``null``.
    """
    subject = "this network" if iteration is None else f"the network after iteration {iteration}"
    block: dict[str, Any] = {
        "schema_version": 1,
        "git_sha": provenance.get("git_sha"),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "search_epoch": EVAL_SEARCH_EPOCH,
        "seed": config.seed,
        "command": shlex.join(["python", "-m", "alpha_zero.training_civ6", *sys.argv[1:]]),
        "hardware": hardware_summary(config.device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "games_total": games_total,
        "claims": [
            f"score rate of {subject} against a uniform-random opponent over "
            f"{games_total} held-out games, with its 95% ternary-variance interval"
        ],
        "non_claims": [
            "run artifact, not committed evidence",
            "says nothing about strength against any non-random opponent",
            "in-loop evaluation searches at half the self-play budget",
        ],
    }
    if config.seed is None:
        block["seed_null_reason"] = "the run was unseeded"
    if block["git_sha"] is None:
        block["git_sha_null_reason"] = (
            f"no valid {GIT_SHA_ENV} and no resolvable .git at the package's source root"
        )
    return block


def in_loop_eval_games(config: TrainingConfig, dry_run: bool) -> int:
    """Games per in-loop evaluation: 0 when ``eval_games`` is 0, else the dry-run or configured count."""
    if config.eval_games == 0:
        return 0
    return DRY_RUN_PROFILE.num_eval_games if dry_run else config.eval_games


def _evaluate_iteration(
    network: torch.nn.Module,
    config: TrainingConfig,
    iteration: int,
    dry_run: bool,
    provenance: dict[str, Any],
    checkpoint: str | None,
) -> dict[str, Any] | None:
    """Run the scheduled in-loop evaluation, if any, and write ``eval_<iter>.json``.

    Scheduled on ``config.resolved_eval_interval`` -- not on the checkpoint
    cadence -- or every iteration under ``--dry_run``; never when
    ``eval_games`` is 0. ``checkpoint`` is the file (relative to the checkpoint
    directory) this iteration's weights are saved to, or ``None`` when they are
    not saved. Returns the written ``result`` (summary plus legacy ``win_rate``),
    or ``None`` when no evaluation ran.
    """
    num_games = in_loop_eval_games(config, dry_run)
    if num_games == 0:
        return None
    if not dry_run and (iteration + 1) % config.resolved_eval_interval != 0:
        return None
    logger.info("Evaluating against random...")
    summary = evaluate_against_random(network, config, num_games=num_games, iteration=iteration)
    logger.info(
        f"Win rate vs random: {legacy_win_rate(summary.wins, summary.draws, summary.losses):.1%}"
    )
    path = os.path.join(config.checkpoint_dir, f"eval_{iteration + 1}.json")
    meta = {
        "iteration": iteration + 1,
        "opponent": "random",
        "checkpoint": checkpoint,
        "provenance": provenance,
        "evidence": eval_evidence_block(
            config, provenance, summary.total_games, iteration=iteration + 1
        ),
    }
    result = write_eval_json(path, summary, meta)
    logger.info(f"Wrote evaluation: {path}")
    return result


def _prequential_diagnostics(
    network: torch.nn.Module,
    config: TrainingConfig,
    examples: list[TrainingExample],
    step_records: list[StepRecord],
    buffer_mean: float,
    *,
    generator: str,
    prior_failures: int,
) -> dict[str, Any]:
    """LF-A13 diagnostics for one iteration; recorded, never gated.

    ``generator`` names what played the games (``"simple_mcts"`` for the
    bootstrap iteration, else ``"network"``); the scoring network is always the
    one as it stood before this iteration's training. A failure is recorded as
    ``{"error": ...}`` rather than raised -- a diagnostic must not end a
    multi-hour run -- and logged with its traceback: at ERROR the first time
    (``prior_failures == 0``), at WARNING after that. :func:`train` logs a final
    ``prequential failures: k/N`` count.
    """
    try:
        metrics = prequential_value_metrics(
            _mcts_network(network, config),
            examples,
            step_records,
            buffer_mean,
            greedy_threshold=config.gumbel_greedy_threshold,
        )
    except Exception as exc:
        level = logging.ERROR if prior_failures == 0 else logging.WARNING
        logger.log(level, "Prequential diagnostics failed; continuing", exc_info=True)
        return {"error": repr(exc), "generator": generator}
    metrics["generator"] = generator
    logger.info("Prequential diagnostics: %s", json.dumps(metrics, sort_keys=True))
    return metrics


def _open_iteration_log(
    config: TrainingConfig, provenance: dict[str, Any], start_iteration: int
) -> IterationLog:
    """The run's ``iterations.jsonl`` writer, stamped with the run's identity.

    A fresh start (no ``--resume_from``, loop index 0) rotates an earlier run's
    record and ``eval_*.json`` aside into ``previous_<UTC stamp>/``; a resume
    appends and marks re-run iterations. See :mod:`alpha_zero.training.run_record`.
    """
    run_id = config.run_id or os.environ.get("CIV6AZ_RUN_ID") or default_run_id(config.seed)
    identity = {
        "run_id": run_id,
        "seed": config.seed,
        "git_sha": provenance.get("git_sha"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resumed_from": config.resume_from,
        "start_iteration": start_iteration,
    }
    fresh = not config.resume_from and start_iteration == 0
    return IterationLog(config.checkpoint_dir, identity=identity, fresh=fresh)


def train(config: TrainingConfig, dry_run: bool = False) -> None:
    """Main training loop.

    Args:
        config: Training configuration.
        dry_run: If True, bound the run to a smoke test via DRY_RUN_PROFILE.
    """
    # Bound the *inner* dimensions before anything reads them -- the env below
    # is built from config.board_size / max_turns, so this has to come first.
    if dry_run:
        config = DRY_RUN_PROFILE.apply_to(config)

    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # Initialize environment to get action space size and (crucially) the
    # spatial-channel count, which is dynamic for N-player games
    # (3 + 4 * num_players + 2) — never hardcode it here.
    # Match the env config used in self-play/eval (incl. max_turns) so the
    # inferred action space and spatial-channel count can never desync.
    env = MicroCivEnv(
        board_size=config.board_size,
        max_turns=config.max_turns,
        num_players=config.num_players,
        seed=config.seed,
    )
    action_space_size = env.action_space.n
    sample_obs, _ = env.reset(seed=config.seed)
    spatial_channels = int(sample_obs["spatial"].shape[0])

    # Seed torch before constructing the network: weight initialisation is a
    # draw like any other, so an unseeded one makes every downstream number
    # unrepeatable even when the environments and searches are seeded. This is
    # process-global by necessity -- torch has no per-module generator for
    # parameter init -- and is therefore done once, here, rather than threaded.
    if config.seed is not None:
        torch.manual_seed(config.seed)
        # A seeded GPU run must repeat too: seeding alone does not make CUDA
        # kernels deterministic. No-op on CPU (utils.seeding).
        enforce_deterministic_backend(config.device)

    # Initialize network. When num_players > 2, MicroCivNetwork swaps in an
    # NPlayerValueHead that outputs a per-player value vector.
    network = MicroCivNetwork(
        board_size=config.board_size,
        action_space_size=int(action_space_size),
        num_res_blocks=config.num_res_blocks,
        num_filters=config.num_filters,
        spatial_channels=spatial_channels,
        num_players=config.num_players,
    ).to(config.device)

    logger.info(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}")
    logger.info(f"Action space size: {action_space_size}")
    logger.info(
        "Players: %d | spatial channels: %d | value head: %s | scoring: %s",
        config.num_players,
        spatial_channels,
        "per-player" if config.num_players > 2 else "scalar",
        config.scoring_method,
    )
    logger.info(f"Device: {config.device}")

    # Resume. Done after construction so the network is built from the
    # environment, and a mismatched checkpoint is refused rather than dictating
    # the architecture.
    start_iteration = config.start_iteration
    if start_iteration < 0:
        # Caught before the resume block so no misleading "Resuming at
        # iteration -4" line is logged first. A negative index would also name
        # checkpoints `checkpoint_-4.pt` and skip the iteration-0 SimpleMCTS
        # bootstrap, neither of which fails loudly on its own.
        raise ValueError(f"--start_iteration must be >= 0, got {start_iteration}.")
    if config.resume_from:
        resume_network_weights(
            network,
            config.resume_from,
            device=config.device,
            max_turns=config.max_turns,
            board_size=config.board_size,
        )
        if start_iteration == 0:
            inferred = infer_start_iteration(config.resume_from)
            if inferred is None:
                logger.warning(
                    "--resume_from %s: the file name carries no iteration number "
                    "and --start_iteration was not given, so numbering restarts "
                    "at 1 and existing checkpoints in %s will be overwritten.",
                    config.resume_from,
                    config.checkpoint_dir,
                )
            else:
                start_iteration = inferred
        logger.info(
            "Resuming from %s at iteration %d",
            config.resume_from,
            start_iteration + 1,
        )
        logger.warning(
            "Resume restores network weights only: optimizer moments and the "
            "replay buffer are not restored, so the first resumed iteration "
            "trains on freshly generated games alone.",
        )

    # Provenance is resolved once per run (no subprocess, no per-save lookup) from
    # the RESOLVED config -- after the dry-run profile -- so it records what the
    # run used, not what the caller typed.
    provenance = build_provenance(config, git_sha=read_git_sha())

    # checkpoint_0 is the pinned random-init the A1 protocol compares against, so
    # it is written before any training. Not when resuming: `checkpoint_<N>.pt`
    # resumes at index N, so `checkpoint_0.pt` means "from the top", and writing
    # here would overwrite the very file a resume just read (or a trained
    # network's weights under the random-init name).
    if not config.resume_from and start_iteration == 0:
        zero_path = os.path.join(config.checkpoint_dir, "checkpoint_0.pt")
        if os.path.exists(zero_path):
            logger.warning(
                "Overwriting existing %s: a fresh run replaces the pinned random-init, "
                "and any checkpoint_N.pt from the earlier run stays beside the new one. "
                "Use a new --checkpoint_dir/--run_id to keep both.",
                zero_path,
            )
        network.save(zero_path, metadata=provenance)
        logger.info(f"Saved pre-training checkpoint: {zero_path}")

    iteration_log = _open_iteration_log(config, provenance, start_iteration)

    # Initialize optimizer
    optimizer = optim.Adam(
        network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(
        max_size=config.replay_buffer_size,
        # A named stream, so it cannot share a sequence with self-play or
        # evaluation. The buffer spans iterations, so it takes no iteration
        # coordinate: re-seeding it per iteration would restart its sampling.
        seed=derive_seed(config.seed, SeedStream.REPLAY_SAMPLER),
    )

    # Training loop
    num_iterations = DRY_RUN_PROFILE.num_iterations if dry_run else config.num_iterations
    num_games = (
        DRY_RUN_PROFILE.num_games_per_iteration if dry_run else config.num_games_per_iteration
    )

    if start_iteration >= num_iterations:
        raise ValueError(
            f"Nothing to do: --start_iteration {start_iteration} is not less "
            f"than the total --num_iterations {num_iterations}. The total is "
            "the end of the run, not a count of additional iterations."
        )

    for iteration in range(start_iteration, num_iterations):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Iteration {iteration + 1}/{num_iterations}")
        logger.info(f"{'=' * 50}")

        # Self-play
        logger.info("Running self-play...")
        start_time = time.time()

        # Use simple MCTS for first iteration (no trained network) unless the
        # run asked for the configured search from the start (--bootstrap none).
        use_simple = iteration == 0 and config.bootstrap == "simple_mcts"
        step_records: list[StepRecord] = []
        examples = run_self_play(
            network,
            config,
            num_games,
            use_simple_mcts=use_simple,
            iteration=iteration,
            step_sink=step_records,
        )
        self_play_time = time.time() - start_time
        logger.info(
            f"Self-play: {len(examples)} examples from {num_games} games ({self_play_time:.1f}s)",
        )

        # Scored before insertion, by the network that played the games, so the
        # positions are out of sample (prequential). The baseline is the buffer's
        # mean BEFORE these targets join it: 0.0 at iteration 0.
        prequential = _prequential_diagnostics(
            network,
            config,
            examples,
            step_records,
            replay_buffer.mean_value(),
            generator="simple_mcts" if use_simple else "network",
            prior_failures=iteration_log.prequential_failures,
        )
        replay_buffer.add_game(examples)
        losses: dict[str, float] | None = None

        # Training
        if len(replay_buffer) >= config.min_replay_size or dry_run:
            logger.info("Training network...")
            start_time = time.time()

            # Sample from replay buffer
            sample_size = min(len(replay_buffer), config.batch_size * 100)
            train_examples = replay_buffer.sample(sample_size)

            # Create dataloader
            dataset = ReplayDataset(train_examples)
            # An explicit generator rather than the global torch RNG: batch
            # order changes the gradient, and relying on global state makes the
            # shuffle depend on however many draws happened to precede it.
            # Re-derived per iteration so successive iterations shuffle
            # differently while the run as a whole replays.
            loader_generator: torch.Generator | None = None
            loader_seed = derive_seed(config.seed, SeedStream.DATA_LOADER, iteration)
            if loader_seed is not None:
                loader_generator = torch.Generator()
                loader_generator.manual_seed(loader_seed)
            dataloader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=0,
                generator=loader_generator,
            )

            # Train for multiple epochs
            num_epochs = DRY_RUN_PROFILE.num_epochs if dry_run else config.num_epochs
            for epoch in range(num_epochs):
                losses = train_epoch(network, optimizer, dataloader, config)
                logger.info(
                    f"Epoch {epoch + 1}: loss={losses['total_loss']:.4f}, "
                    f"policy={losses['policy_loss']:.4f}, value={losses['value_loss']:.4f}",
                )

            train_time = time.time() - start_time
            logger.info(f"Training complete ({train_time:.1f}s)")

        # Evaluation, on its own cadence (LF-A14).
        # Checkpoint on the save cadence; its name is known before evaluating so
        # eval_<iter>.json can say which file (if any) holds the weights it scored.
        checkpoint_name: str | None = None
        if (iteration + 1) % config.save_interval == 0 or dry_run:
            checkpoint_name = f"checkpoint_{iteration + 1}.pt"
        evaluation = _evaluate_iteration(
            network, config, iteration, dry_run, provenance, checkpoint_name
        )
        if checkpoint_name is not None:
            checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_name)
            network.save(checkpoint_path, metadata=provenance)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        iteration_log.append(
            {
                "iteration": iteration + 1,
                "self_play": {
                    "games": num_games,
                    "examples": len(examples),
                    "seconds": round(self_play_time, 3),
                    "simple_mcts": use_simple,
                },
                "train": losses,
                "prequential": prequential,
                "eval": evaluation,
                "checkpoint": checkpoint_name,
            },
        )

    failures = iteration_log.prequential_failures
    logger.log(
        logging.WARNING if failures else logging.INFO,
        "prequential failures: %d/%d",
        failures,
        iteration_log.records_written,
    )

    # Save final model
    final_path = os.path.join(config.checkpoint_dir, "latest.pt")
    network.save(final_path, metadata=provenance)
    logger.info(f"Training complete. Final model: {final_path}")


def training_budget_inputs(config: TrainingConfig, dry_run: bool) -> dict[str, Any]:
    """Resolved values the game-budget cost model needs, as :func:`train` will use them.

    Mirrors the loop bounds in :func:`train` -- the dry-run profile replaces the
    iteration, game and evaluation counts, evaluation happens every iteration
    under ``--dry_run``, it runs on ``--eval_interval`` (defaulting to
    ``--save_interval``) with ``--eval_games`` games (none when 0), and
    ``--resume_from`` names the start iteration -- so the price is of the run that
    will actually happen.
    """
    start = config.start_iteration
    if config.resume_from and start == 0:
        start = infer_start_iteration(config.resume_from) or 0
    if dry_run:
        return {
            "num_games": DRY_RUN_PROFILE.num_games_per_iteration,
            "num_iterations": DRY_RUN_PROFILE.num_iterations,
            "start_iteration": start,
            "eval_games": in_loop_eval_games(config, dry_run=True),
            "eval_always": True,
        }
    return {
        "num_games": config.num_games_per_iteration,
        "num_iterations": config.num_iterations,
        "start_iteration": start,
        "save_interval": config.save_interval,
        "eval_interval": config.resolved_eval_interval,
        "eval_games": in_loop_eval_games(config, dry_run=False),
    }


def resolve_checkpoint_dir(explicit: str | None, run_id: str | None, seed: int | None) -> str:
    """Pick the checkpoint directory.

    An explicit ``--checkpoint_dir`` always wins. Otherwise, with
    ``CIV6AZ_DATA_ROOT`` unset the historical ``checkpoints/civ6`` is kept; with
    it set, the directory is ``<root>/runs/<run_id>/checkpoints`` after the root
    has passed :func:`assert_safe_data_root`.
    """
    if run_id is not None:
        validate_run_id(run_id)
    if explicit is not None:
        return explicit
    root = resolve_data_root()
    if root is None:
        return "checkpoints/civ6"
    assert_safe_data_root(root, Path(__file__).resolve().parents[1])
    return str(default_checkpoint_dir(root, run_id or default_run_id(seed)))


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1; got {value}")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0; got {value}")
    return value


def _non_negative_float(text: str) -> float:
    value = float(text)
    if not value >= 0.0:
        raise argparse.ArgumentTypeError(f"must be >= 0; got {value}")
    return value


def _unit_interval(text: str) -> float:
    value = float(text)
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be in (0, 1]; got {value}")
    return value


#: Flags that configure only the Gumbel search, as (argparse dest, flag). Given
#: without ``--mcts gumbel`` they would be silently ignored, so the parser refuses.
#: ``--policy_target`` is among them: the PUCT and Simple searches have no such field.
_GUMBEL_ONLY_FLAGS: tuple[tuple[str, str], ...] = (
    ("gumbel_k", "--gumbel_k"),
    ("gumbel_sims", "--gumbel_sims"),
    ("c_scale", "--c_scale"),
    ("greedy_threshold", "--greedy_threshold"),
    ("policy_target", "--policy_target"),
)


class _TrainingArgumentParser(argparse.ArgumentParser):
    """Refuses Gumbel-only flags unless ``--mcts gumbel`` selects that search.

    The check runs inside parsing so every caller of :func:`build_parser` --
    :func:`main` and the run-card pricer alike -- gets it.
    """

    def parse_known_args(  # type: ignore[override]
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> tuple[argparse.Namespace, list[str]]:
        parsed, extras = super().parse_known_args(args, namespace)
        if getattr(parsed, "mcts", "gumbel") != "gumbel":
            given = [
                flag for dest, flag in _GUMBEL_ONLY_FLAGS if getattr(parsed, dest, None) is not None
            ]
            if given:
                noun, pronoun = ("flag", "it") if len(given) == 1 else ("flags", "them")
                self.error(
                    f"{', '.join(given)}: Gumbel-only {noun}; "
                    f"pass --mcts gumbel or remove {pronoun}"
                )
        return parsed, extras


def build_parser() -> argparse.ArgumentParser:
    """The trainer's CLI parser. Also used by the run-card driver to price an argv."""
    parser = _TrainingArgumentParser(description="Train Civ6 AlphaZero")
    parser.add_argument("--board_size", type=int, default=16)
    parser.add_argument(
        "--max_turns",
        type=int,
        default=200,
        help=(
            "Turn limit per self-play game. MicroCiv only advances `turn` when the "
            "player index wraps, so the default 200 needs 400+ end-turn actions plus "
            "every unit ply — use a small value (e.g. 20) for a quick run."
        ),
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help=(
            "Checkpoint every N iterations; also the evaluation cadence unless "
            "--eval_interval is given."
        ),
    )
    parser.add_argument(
        "--eval_interval",
        type=_positive_int,
        default=None,
        help=(
            "Evaluate against a random opponent every N iterations, independently "
            "of checkpointing. Default: --save_interval's value."
        ),
    )
    parser.add_argument(
        "--eval_games",
        type=_non_negative_int,
        default=DEFAULT_EVAL_GAMES,
        help=(
            f"Games per in-loop evaluation (default {DEFAULT_EVAL_GAMES}). 0 turns "
            "in-loop evaluation off, including under --dry_run."
        ),
    )
    parser.add_argument(
        "--num_players",
        type=int,
        default=2,
        help="Number of players (2 = classic scalar pipeline; >2 = N-player "
        "per-player value pipeline).",
    )
    parser.add_argument(
        "--scoring_method",
        type=str,
        default="winner_take_all",
        choices=[m.value for m in SUPPORTED_SCORING_METHODS],
        help="How terminal outcomes become per-player training values "
        "(winner-derivable methods only; 'proportional' needs per-player scores).",
    )
    parser.add_argument(
        "--num_simulations",
        type=_positive_int,
        default=None,
        help=(
            "Search budget per move (default 50 for PUCT). With --mcts gumbel it is "
            "also the Gumbel budget unless --gumbel_sims is given."
        ),
    )
    parser.add_argument("--num_games", type=int, default=100)
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_res_blocks", type=int, default=6)
    parser.add_argument("--num_filters", type=int, default=64)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help=(
            "Where checkpoints are written. Default: 'checkpoints/civ6', or "
            "<CIV6AZ_DATA_ROOT>/runs/<run_id>/checkpoints when that variable is set."
        ),
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help=(
            "Run identifier (^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$). Only used to place the "
            "default checkpoint directory under CIV6AZ_DATA_ROOT. Default: "
            "train-<seed or 'unseeded'>-<UTC timestamp>."
        ),
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help=(
            "Checkpoint to start from, e.g. checkpoints/civ6/checkpoint_7.pt. "
            "Restores network weights ONLY -- optimizer moments and the replay "
            "buffer are not restored. The starting iteration is read from the "
            "file name unless --start_iteration says otherwise."
        ),
    )
    parser.add_argument(
        "--start_iteration",
        type=int,
        default=0,
        help=(
            "Loop index to begin at, overriding what --resume_from's file name "
            "implies. --num_iterations remains the END of the run, not a count "
            "of additional iterations."
        ),
    )
    parser.add_argument(
        "--bootstrap",
        type=str,
        default="simple_mcts",
        choices=["simple_mcts", "none"],
        help=(
            "Iteration-0 data source: 'simple_mcts' (default, unchanged) plays "
            "iteration 0 with random-rollout SimpleMCTS; 'none' plays it with the "
            "configured --mcts search on the freshly initialised network."
        ),
    )
    parser.add_argument("--dry_run", action="store_true", help="Quick test run")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--mcts",
        type=str,
        default="civ6",
        choices=["civ6", "gumbel"],
        help=(
            "Search used for self-play AND evaluation: 'civ6' (PUCT, the default, "
            "byte-identical to runs made before this flag existed) or 'gumbel' "
            "(sets use_gumbel_mcts). The Gumbel budget is --gumbel_sims, else "
            "--num_simulations, else 16; evaluation searches at half of it."
        ),
    )
    parser.add_argument(
        "--gumbel_sims",
        type=_positive_int,
        default=None,
        help=(
            "Gumbel simulation budget per move. Spent exactly whenever two or more "
            "candidates are legal (gumbel_ops.halving_schedule). Overrides "
            "--num_simulations for the Gumbel search."
        ),
    )
    parser.add_argument(
        "--gumbel_k",
        type=_positive_int,
        default=None,
        help="Gumbel-Top-k candidates sampled at the root (default 16).",
    )
    parser.add_argument(
        "--c_scale",
        type=_non_negative_float,
        default=None,
        help=f"Gumbel sigma scale c_scale (default {MCTS.GUMBEL_C_SCALE}).",
    )
    parser.add_argument(
        "--policy_target",
        type=str,
        default=None,
        choices=sorted(VALID_POLICY_TARGETS),
        help=f"Gumbel policy training target (default {POLICY_TARGET_COMPLETED_Q}).",
    )
    parser.add_argument(
        "--greedy_threshold",
        type=_unit_interval,
        default=None,
        help=(
            "Gumbel select_action plays greedily once one action holds at least this "
            f"probability mass, in (0, 1] (default {GREEDY_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Base RNG seed. Determines the environments, every search, the "
            "random evaluation opponent and network initialisation, so two runs "
            "at the same seed replay exactly. Omitted means unseeded."
        ),
    )
    add_budget_arguments(parser)
    return parser


def config_from_args(args: argparse.Namespace, checkpoint_dir: str) -> TrainingConfig:
    """The :class:`TrainingConfig` that :func:`main` trains with for parsed ``args``.

    Search flags left unset keep the :class:`TrainingConfig` defaults, so a run
    that passes none of them is unchanged. The one deliberate exception is the
    LF-A7 fix: ``--mcts gumbel`` used to ignore ``--num_simulations``; it is now
    the Gumbel budget too, unless ``--gumbel_sims`` names one explicitly.
    """
    search: dict[str, Any] = {}
    if args.num_simulations is not None:
        search["num_simulations"] = args.num_simulations
    gumbel_sims = args.gumbel_sims if args.gumbel_sims is not None else args.num_simulations
    if gumbel_sims is not None:
        search["gumbel_num_simulations"] = gumbel_sims
    optional = {
        "gumbel_num_sampled_actions": args.gumbel_k,
        "gumbel_c_scale": args.c_scale,
        "gumbel_policy_target": args.policy_target,
        "gumbel_greedy_threshold": args.greedy_threshold,
    }
    search.update({field: value for field, value in optional.items() if value is not None})
    return TrainingConfig(
        board_size=args.board_size,
        num_players=args.num_players,
        scoring_method=args.scoring_method,
        num_games_per_iteration=args.num_games,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        num_res_blocks=args.num_res_blocks,
        num_filters=args.num_filters,
        checkpoint_dir=checkpoint_dir,
        resume_from=args.resume_from,
        start_iteration=args.start_iteration,
        bootstrap=args.bootstrap,
        run_id=args.run_id,
        max_turns=args.max_turns,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        device=args.device or "auto",
        seed=args.seed,
        use_gumbel_mcts=args.mcts == "gumbel",
        **search,
    )


def main() -> None:
    """Main entry point."""
    args = build_parser().parse_args()
    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir, args.run_id, args.seed)
    config = config_from_args(args, checkpoint_dir)

    enforce_cli("training_civ6", training_budget_inputs(config, args.dry_run), args)

    train(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
