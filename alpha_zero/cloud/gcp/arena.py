"""
Shared Arena Module for Model Evaluation.

Provides reusable arena logic for running head-to-head matches between two models.
Used by both evaluation_worker.py and post-training tournament scripts.

Usage:
    from alpha_zero.cloud.gcp.arena import ArenaResult, run_arena

    result = run_arena(
        model_a=new_model,
        model_b=old_model,
        num_games=20,
        mcts_type="gumbel",
        mcts_sims=25,
    )
    print(f"Win rate: {result.win_rate:.2%}")
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from alpha_zero.cloud.gcp.config import BATCH_MCTS_TYPE
from alpha_zero.utils.seeding import SeedStream, derive_seed

# Variants that accept the Gumbel-family `num_sampled_actions` parameter.
_GUMBEL_VARIANTS: frozenset[str] = frozenset({"gumbel", BATCH_MCTS_TYPE})

logger = logging.getLogger(__name__)


class ArenaPlayer(Protocol):
    """Anything that can pick an action for the side it plays.

    Structurally identical to ``alpha_zero.evaluation.baseline_agents.BaseAgent``
    and ``alpha_zero.evaluation.baselines.BaselinePlayer`` so that the random,
    heuristic and SimpleMCTS baselines plug straight into the arena without an
    adapter.
    """

    def select_action(self, env: Any, observation: Mapping[str, Any]) -> int:
        """Return an action index for ``env``'s current player."""
        ...


#: Builds one side's player for one game. Receives the per-game seed derived
#: for that side (``None`` when the arena is unseeded) so a stochastic baseline
#: is reproducible game by game without sharing a stream with the other side.
PlayerFactory = Callable[[int | None], ArenaPlayer]


class _SearchPlayer:
    """Drive a network-guided search as an :class:`ArenaPlayer`.

    Greedy play for evaluation: ``temperature=0`` and no exploration noise.
    """

    def __init__(self, mcts: Any, is_muzero: bool) -> None:
        self._mcts = mcts
        self._is_muzero = is_muzero

    def select_action(self, env: Any, observation: Mapping[str, Any]) -> int:
        if self._is_muzero:
            # MuZeroMCTS.search() uses the learned dynamics model internally;
            # it requires observation dict + action_mask, not env.
            obs = env.get_observation()
            action_probs, _ = self._mcts.search(
                obs, add_noise=False, action_mask=obs["action_mask"]
            )
        else:
            # Standard MCTS variants (civ6/gumbel/simple) traverse the env directly
            action_probs, _ = self._mcts.search(env, add_noise=False)
        action: int = self._mcts.select_action(action_probs, temperature=0.0)
        return action


@dataclass
class ArenaResult:
    """Result of an arena evaluation between two models."""

    wins: int
    losses: int
    draws: int
    total_games: int
    #: ``{"pair", "seat", "result"}`` per game, from model_a's side, in play order.
    #: Additive: :meth:`to_dict` does not carry it, so persisted artifacts are unchanged.
    games: list[dict[str, Any]] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        """Wins over all games, counting a draw as a loss.

        .. deprecated:: 2026-11
            Kept **verbatim** for artifact compatibility only: this is the
            statistic the deployed promotion gate has always compared against
            its threshold, every ``results.json`` under ``evaluations/`` carries
            it, and changing it here would silently change which models get
            promoted. Its ceiling is ``1 - draw_rate`` (73.5% draws at the
            shipped defaults cap it at 0.265), so it cannot rank candidates.
            New consumers read :attr:`score_rate` and :attr:`draw_rate`; the
            division below is allow-listed in
            ``tests/regression/draw_aware_allowlist.toml`` for that reason.
        """
        return self.wins / max(self.total_games, 1)

    @property
    def score_rate(self) -> float:
        """Points taken over points available, scoring a draw as half a point.

        Delegates to :mod:`alpha_zero.evaluation.scoring` so the arena, the
        promotion gate, and the offline A/B tooling all agree on one
        definition.
        """
        from alpha_zero.evaluation.scoring import score_rate

        return score_rate(self.wins, self.draws, self.losses)

    @property
    def draw_rate(self) -> float:
        return self.draws / max(self.total_games, 1)

    @property
    def loss_rate(self) -> float:
        return self.losses / max(self.total_games, 1)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "win_rate": self.win_rate,
            "draw_rate": self.draw_rate,
            "loss_rate": self.loss_rate,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "total_games": self.total_games,
            # Added so the persisted artifact carries a draw-aware statistic.
            # save_results() splats this mapping, so the key reaches GCS with
            # no schema change, and readers that do not know it ignore it.
            "score_rate": self.score_rate,
        }


def run_arena(
    model_a: torch.nn.Module | None,
    model_b: torch.nn.Module | None,
    num_games: int,
    mcts_type: str = "gumbel",
    mcts_sims: int = 25,
    gumbel_num_actions: int = 16,
    board_size: int = 8,
    max_turns: int = 30,
    device: torch.device | None = None,
    num_parallel_games: int = 1,
    gameplay_preset: str = "default",
    seed: int | None = None,
    *,
    player_a_factory: PlayerFactory | None = None,
    player_b_factory: PlayerFactory | None = None,
    env_seeds: Sequence[int | None] | None = None,
    game_indices: Sequence[int] | None = None,
) -> ArenaResult:
    """Play model_a vs. model_b in an arena.

    **Paired seats.** Games come in pairs: game ``2p`` and game ``2p + 1`` are
    played on the *same* map (same environment seed) with the seats swapped,
    so a first-mover or map advantage cancels within each pair instead of
    landing on whichever side drew the easier boards. ``num_games`` should be
    even; an odd count leaves the last map played from one seat only, which
    is logged. Uses greedy play (temperature=0, no noise) for evaluation.

    Args:
        model_a: First model (results are from this model's perspective).
            May be ``None`` when ``player_a_factory`` supplies the player.
        model_b: Second model (opponent). May be ``None`` when
            ``player_b_factory`` supplies the player.
        num_games: Number of games to play.
        mcts_type: MCTS algorithm variant ("civ6", "gumbel", "simple" or "muzero").
        mcts_sims: Number of MCTS simulations per move.
        gumbel_num_actions: Number of sampled actions for Gumbel MCTS.
        board_size: Size of game board.
        max_turns: Maximum game turns before truncation.
        device: PyTorch device to use. If None, defaults to CPU.
        num_parallel_games: Number of parallel games (accepted for API consistency
            with selfplay_worker; arena currently runs games sequentially).
        gameplay_preset: Environment preset ("default" or "enhanced").
        seed: Base seed. The map for pair ``p`` is seeded with ``seed + p``
            (so pair ``p`` plays one map from both seats), and each side's
            search for game ``i`` draws from
            ``derive_seed(seed, SeedStream.ARENA_SEARCH, i, side)`` -- a
            seeded arena is therefore fully reproducible, not just its
            environments. ``None`` (the default) leaves environments **and**
            searches unseeded, preserving historical behavior. Seeding the
            global RNGs does not cover this: the environment and every search
            own private generators.
        player_a_factory: Optional builder for side A's player, called once per
            game with ``derive_seed(seed, SeedStream.ARENA_OPPONENT, i, "a")``.
            When given, ``model_a`` is not searched. Used by the benchmark
            suite to field baselines (random, heuristic, SimpleMCTS).
        player_b_factory: Same for side B.
        env_seeds: Explicit per-pair environment seeds, overriding the
            ``seed + p`` rule; must hold at least ``ceil(num_games / 2)``
            entries. The benchmark protocol uses this to name its maps via
            ``derive_seed`` rather than arithmetic.
        game_indices: Play only these games of the ``num_games`` run (default:
            all). Game ``g`` keeps its map (pair ``g // 2``), seat and seeds, so
            a subset replays exactly those games of the full run (benchmark shards).

    Returns:
        ArenaResult with wins/losses/draws from model_a's perspective.
    """
    from alpha_zero.core.mcts_factory import create_mcts
    from alpha_zero.envs.env_factory import create_game_env

    if device is None:
        device = torch.device("cpu")

    if num_parallel_games > 1:
        logger.info(
            "Arena: num_parallel_games=%d accepted for API consistency "
            "(arena runs games sequentially for player alternation)",
            num_parallel_games,
        )
    if num_games % 2:
        logger.warning(
            "Arena: num_games=%d is odd; the last map is played from one seat only, "
            "so seat advantage does not fully cancel. Prefer an even count.",
            num_games,
        )
    if model_a is None and player_a_factory is None:
        raise ValueError("run_arena needs model_a or player_a_factory")
    if model_b is None and player_b_factory is None:
        raise ValueError("run_arena needs model_b or player_b_factory")
    num_pairs = (num_games + 1) // 2
    indices = range(num_games) if game_indices is None else list(game_indices)
    if any(not 0 <= g < num_games for g in indices):
        raise ValueError(f"game_indices must lie in [0, {num_games}), got {list(indices)}")
    if env_seeds is not None and len(env_seeds) < num_pairs:
        raise ValueError(
            f"env_seeds holds {len(env_seeds)} entries but {num_games} games need {num_pairs}"
        )

    # MuZero has a different search API: (observation, action_mask) vs (env)
    is_muzero = mcts_type == "muzero"
    if is_muzero:
        logger.debug("Arena: using MuZero search path (observation + action_mask)")

    def _search_player(model: torch.nn.Module | None, game_idx: int, side: str) -> _SearchPlayer:
        # num_sampled_actions is a Gumbel-family parameter; create_mcts rejects
        # unknown kwargs, so passing it unconditionally made mcts_type="civ6"
        # -- an option this function's own docstring advertises -- raise
        # TypeError on the first game of every arena run.
        mcts_kwargs: dict[str, Any] = {"num_simulations": mcts_sims}
        if mcts_type in _GUMBEL_VARIANTS:
            mcts_kwargs["num_sampled_actions"] = gumbel_num_actions
        # Only thread a seed when the arena is seeded, so an unseeded call
        # builds exactly the search it always did.
        search_seed = derive_seed(seed, SeedStream.ARENA_SEARCH, game_idx, side)
        if search_seed is not None:
            mcts_kwargs["seed"] = search_seed
        mcts = create_mcts(variant=mcts_type, network=model, device=str(device), **mcts_kwargs)
        return _SearchPlayer(mcts, is_muzero)

    def _player(
        factory: PlayerFactory | None, model: torch.nn.Module | None, game_idx: int, side: str
    ) -> ArenaPlayer:
        if factory is not None:
            return factory(derive_seed(seed, SeedStream.ARENA_OPPONENT, game_idx, side))
        return _search_player(model, game_idx, side)

    start_time = time.time()
    wins = 0
    losses = 0
    draws = 0
    games: list[dict[str, Any]] = []

    for game_idx in indices:
        game_start_time = time.time()

        # Alternate which player model_a controls; the pair index selects the map.
        a_is_player_0 = game_idx % 2 == 0
        pair = game_idx // 2
        if env_seeds is not None:
            env_seed = env_seeds[pair]
        else:
            env_seed = None if seed is None else seed + pair

        env = create_game_env(
            board_size=board_size,
            max_turns=max_turns,
            gameplay_preset=gameplay_preset,
            seed=env_seed,
        )
        player_a = _player(player_a_factory, model_a, game_idx, "a")
        player_b = _player(player_b_factory, model_b, game_idx, "b")

        _obs, _info = env.reset()
        done = False
        move_count = 0

        while not done:
            current_is_a = (env.current_player == 0 and a_is_player_0) or (
                env.current_player == 1 and not a_is_player_0
            )
            player = player_a if current_is_a else player_b

            # A search player reads the env itself; a baseline needs the
            # observation (for its action mask), fetched only when asked for.
            observation: Mapping[str, Any] = (
                {} if isinstance(player, _SearchPlayer) else env.get_observation()
            )
            action = player.select_action(env, observation)

            _obs, _reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            move_count += 1

        # Determine result from model_a's perspective
        if env.winner is None:
            draws += 1
            result = "draw"
        elif (env.winner == 0 and a_is_player_0) or (env.winner == 1 and not a_is_player_0):
            wins += 1
            result = "win"
        else:
            losses += 1
            result = "loss"
        games.append({"pair": pair, "seat": 0 if a_is_player_0 else 1, "result": result})

        elapsed = time.time() - game_start_time
        player_str = "P0" if a_is_player_0 else "P1"
        logger.info(
            "Game %d/%d (map %d): model_a=%s, result=%s, moves=%d (%.1fs)",
            game_idx + 1,
            num_games,
            pair,
            player_str,
            result,
            move_count,
            elapsed,
        )

    arena_result = ArenaResult(
        wins=wins, losses=losses, draws=draws, total_games=len(games), games=games
    )

    # Metrics route to MLflow through WandbCompatAdapter despite the historical
    # naming; see alpha_zero/utils/experiment_tracker.py. Rates are read off the
    # result object rather than recomputed, so a logged metric cannot drift from
    # the value the promotion gate sees.
    total_elapsed = time.time() - start_time
    try:
        from alpha_zero.utils.experiment_tracker import log_metrics

        log_metrics(
            {
                "arena/total_games": arena_result.total_games,
                "arena/wins": wins,
                "arena/losses": losses,
                "arena/draws": draws,
                "arena/win_rate": arena_result.win_rate,
                "arena/score_rate": arena_result.score_rate,
                "arena/draw_rate": arena_result.draw_rate,
                "arena/games_per_minute": arena_result.total_games / max(total_elapsed / 60, 0.001),
            }
        )
    except ImportError:
        logger.debug("experiment tracker not available, skipping arena metrics logging")

    return arena_result
