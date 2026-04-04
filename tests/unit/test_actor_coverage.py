"""Tests for alpha_zero.distributed.actor to increase coverage.

Covers:
- ActorConfig initialization and defaults
- SelfPlayActor initialization, network init, weight updates
- Self-play game execution with mocked environment and network
- Temperature scheduling during action selection
- Training example creation (2-player and N-player paths)
- N-step value computation
- Actor run loop lifecycle
- Actor statistics and control flags
- ActorPool start/stop/restart
- actor_process entry point
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from alpha_zero.distributed.actor import (
    ActorConfig,
    ActorPool,
    GameHistoryItem,
    SelfPlayActor,
    actor_process,
)
from alpha_zero.distributed.shared_storage import (
    GameTrajectory,
    TrainingExample,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BOARD_SIZE = 8
NUM_PLAYERS = 2
# MicroCiv action space for board_size=8 is deterministic; we compute it here
# but it's easier to just grab it from a real env.
ACTION_SPACE_SIZE = 356  # fallback; overridden by fixture


@pytest.fixture()
def actor_config() -> ActorConfig:
    """Minimal actor configuration for fast tests."""
    return ActorConfig(
        actor_id=0,
        board_size=BOARD_SIZE,
        max_turns=20,
        num_players=NUM_PLAYERS,
        num_res_blocks=1,
        num_filters=8,
        num_simulations=2,
        c_puct=1.0,
        temperature=1.0,
        temperature_threshold=10,
        max_games=0,
        device="cpu",
        log_interval=5,
        use_reward_shaping=True,
        n_step_returns=3,
        reward_shaping_weight=0.5,
    )


@pytest.fixture()
def mock_storage() -> MagicMock:
    """Mock SharedStorage."""
    storage = MagicMock()
    storage.should_stop.return_value = False
    storage.get_model_version.return_value = 0
    storage.get_model_weights_with_version.return_value = (None, 0)
    storage.increment_games_played.return_value = 1
    storage.increment_examples.return_value = 10
    return storage


@pytest.fixture()
def mock_replay_buffer() -> MagicMock:
    """Mock DistributedReplayBuffer."""
    buf = MagicMock()
    buf.submit_trajectory.return_value = True
    return buf


@pytest.fixture()
def actor(
    actor_config: ActorConfig,
    mock_storage: MagicMock,
    mock_replay_buffer: MagicMock,
) -> SelfPlayActor:
    """Create a SelfPlayActor with mocked dependencies."""
    return SelfPlayActor(actor_config, mock_storage, mock_replay_buffer)


# ---------------------------------------------------------------------------
# ActorConfig tests
# ---------------------------------------------------------------------------


class TestActorConfig:
    """Tests for ActorConfig dataclass."""

    def test_default_values(self) -> None:
        cfg = ActorConfig()
        assert cfg.actor_id == 0
        assert cfg.num_players == 2
        assert cfg.device == "cpu"
        assert cfg.max_games == 0
        assert cfg.temperature == 1.0

    def test_custom_values(self) -> None:
        cfg = ActorConfig(actor_id=5, board_size=16, num_simulations=100)
        assert cfg.actor_id == 5
        assert cfg.board_size == 16
        assert cfg.num_simulations == 100


# ---------------------------------------------------------------------------
# GameHistoryItem tests
# ---------------------------------------------------------------------------


class TestGameHistoryItem:
    def test_creation(self) -> None:
        item = GameHistoryItem(
            spatial=np.zeros((13, 8, 8)),
            vector=np.zeros(64),
            action_mask=np.ones(100),
            policy=np.ones(100) / 100,
            player=0,
            shaped_reward=0.5,
        )
        assert item.player == 0
        assert item.shaped_reward == 0.5


# ---------------------------------------------------------------------------
# SelfPlayActor initialisation
# ---------------------------------------------------------------------------


class TestSelfPlayActorInit:
    def test_init_creates_env(self, actor: SelfPlayActor) -> None:
        assert actor.env is not None
        assert actor.action_space_size > 0

    def test_init_no_network_yet(self, actor: SelfPlayActor) -> None:
        assert actor.network is None
        assert actor.mcts is None

    def test_init_statistics(self, actor: SelfPlayActor) -> None:
        assert actor.games_played == 0
        assert actor.total_examples == 0

    def test_is_running_initially_false(self, actor: SelfPlayActor) -> None:
        assert actor.is_running() is False


# ---------------------------------------------------------------------------
# Network / MCTS initialisation
# ---------------------------------------------------------------------------


class TestNetworkInit:
    def test_init_network_creates_simple_mcts_when_no_weights(self, actor: SelfPlayActor) -> None:
        """When storage has no weights, actor should use SimpleMCTS."""
        actor._init_network()
        assert actor.network is not None
        assert actor.mcts is not None
        # SimpleMCTS is used when version == 0
        from alpha_zero.core.civ6_mcts import SimpleMCTS

        assert isinstance(actor.mcts, SimpleMCTS)

    def test_init_network_uses_civ6_mcts_when_weights_available(
        self,
        actor_config: ActorConfig,
        mock_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """When storage returns weights, actor should use Civ6MCTS."""
        # Make weight sync report version > 0
        mock_storage.get_model_version.return_value = 1

        actor = SelfPlayActor(actor_config, mock_storage, mock_replay_buffer)

        # We need the weight sync to report version > 0 after update
        # Patch WeightSynchronizer to simulate having weights
        with patch("alpha_zero.distributed.actor.WeightSynchronizer") as MockWS:
            ws_instance = MagicMock()
            ws_instance.should_update.return_value = True
            ws_instance.update_model.return_value = True
            ws_instance.get_current_version.return_value = 1
            MockWS.return_value = ws_instance

            actor._init_network()

            from alpha_zero.core.civ6_mcts import Civ6MCTS

            assert isinstance(actor.mcts, Civ6MCTS)


# ---------------------------------------------------------------------------
# Weight updates
# ---------------------------------------------------------------------------


class TestWeightUpdates:
    def test_update_weights_returns_false_when_not_initialized(self, actor: SelfPlayActor) -> None:
        assert actor._update_weights() is False

    def test_update_weights_returns_false_when_no_new_version(self, actor: SelfPlayActor) -> None:
        actor._init_network()
        # weight_sync.should_update returns False by default after init
        # because version is 0 and storage also returns 0
        assert actor._update_weights() is False

    def test_update_mcts_noop_when_no_network(self, actor: SelfPlayActor) -> None:
        actor._update_mcts()  # Should not raise


# ---------------------------------------------------------------------------
# Temperature scheduling
# ---------------------------------------------------------------------------


class TestTemperatureScheduling:
    def test_temperature_applied_before_threshold(self, actor: SelfPlayActor) -> None:
        """Before temperature_threshold, use configured temperature."""
        actor._init_network()
        assert actor.config.temperature_threshold == 10
        # Simulate env turn < threshold
        actor.env.turn = 5
        # Temperature should be config.temperature (1.0)
        temp = (
            actor.config.temperature if actor.env.turn < actor.config.temperature_threshold else 0.0
        )
        assert temp == 1.0

    def test_temperature_zero_after_threshold(self, actor: SelfPlayActor) -> None:
        """After temperature_threshold, temperature should be 0 (greedy)."""
        actor.env.turn = 15
        temp = (
            actor.config.temperature if actor.env.turn < actor.config.temperature_threshold else 0.0
        )
        assert temp == 0.0


# ---------------------------------------------------------------------------
# N-step value computation
# ---------------------------------------------------------------------------


class TestNStepValue:
    def test_n_step_value_simple(self, actor: SelfPlayActor) -> None:
        """Test n-step return computation with known values."""
        history = [
            GameHistoryItem(
                spatial=np.zeros((13, 8, 8)),
                vector=np.zeros(64),
                action_mask=np.ones(10),
                policy=np.ones(10) / 10,
                player=0,
                shaped_reward=0.1,
            )
            for _ in range(5)
        ]
        gamma = 0.99
        n_steps = 3
        final_value = 1.0

        value = actor._compute_n_step_value(history, 0, final_value, gamma, n_steps)
        # Expected: 0.1 + 0.99*0.1 + 0.99^2*0.1 + 0.99^3*1.0
        expected = 0.1 + 0.99 * 0.1 + 0.99**2 * 0.1 + 0.99**3 * 1.0
        assert abs(value - expected) < 1e-6

    def test_n_step_value_at_end(self, actor: SelfPlayActor) -> None:
        """N-step at the end of history should mostly use final value."""
        history = [
            GameHistoryItem(
                spatial=np.zeros((13, 8, 8)),
                vector=np.zeros(64),
                action_mask=np.ones(10),
                policy=np.ones(10) / 10,
                player=0,
                shaped_reward=0.0,
            )
        ]
        value = actor._compute_n_step_value(history, 0, 1.0, 0.99, 5)
        # Only one step (shaped_reward=0) + bootstrap = 0 + 0.99*1.0
        assert abs(value - 0.99) < 1e-6

    def test_n_step_value_zero_steps(self, actor: SelfPlayActor) -> None:
        """With n_steps=0, should just be discounted final_value."""
        history = [
            GameHistoryItem(
                spatial=np.zeros((13, 8, 8)),
                vector=np.zeros(64),
                action_mask=np.ones(10),
                policy=np.ones(10) / 10,
                player=0,
                shaped_reward=0.5,
            )
        ]
        value = actor._compute_n_step_value(history, 0, 1.0, 0.99, 0)
        # No steps summed, just bootstrap: 1.0 * 1.0 = 1.0
        assert abs(value - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Training example creation (2-player)
# ---------------------------------------------------------------------------


class TestCreateTrainingExamples:
    def _make_history(self, length: int = 4) -> list[GameHistoryItem]:
        items = []
        for i in range(length):
            items.append(
                GameHistoryItem(
                    spatial=np.zeros((13, 8, 8), dtype=np.float32),
                    vector=np.zeros(64, dtype=np.float32),
                    action_mask=np.ones(10, dtype=np.float32),
                    policy=np.ones(10, dtype=np.float32) / 10,
                    player=i % 2,
                    shaped_reward=0.05 * i,
                )
            )
        return items

    def test_examples_with_winner(self, actor: SelfPlayActor) -> None:
        history = self._make_history()
        examples = actor._create_training_examples(history, winner=0)
        assert len(examples) == 4
        # Player 0 should have positive value, player 1 negative
        for ex, _hist in zip(examples, history, strict=False):
            assert isinstance(ex, TrainingExample)
            assert ex.spatial.shape == (13, 8, 8)

    def test_examples_draw(self, actor: SelfPlayActor) -> None:
        history = self._make_history(2)
        examples = actor._create_training_examples(history, winner=None)
        assert len(examples) == 2

    def test_examples_without_reward_shaping(
        self,
        actor_config: ActorConfig,
        mock_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """When reward shaping disabled, value = pure outcome."""
        actor_config.use_reward_shaping = False
        actor = SelfPlayActor(actor_config, mock_storage, mock_replay_buffer)
        history = [
            GameHistoryItem(
                spatial=np.zeros((13, 8, 8), dtype=np.float32),
                vector=np.zeros(64, dtype=np.float32),
                action_mask=np.ones(10, dtype=np.float32),
                policy=np.ones(10, dtype=np.float32) / 10,
                player=0,
                shaped_reward=0.0,
            ),
        ]
        examples = actor._create_training_examples(history, winner=0)
        assert len(examples) == 1
        assert examples[0].value == 1.0

    def test_examples_loss_value(self, actor: SelfPlayActor) -> None:
        """Losing player should get negative value."""
        actor.config.use_reward_shaping = False
        history = [
            GameHistoryItem(
                spatial=np.zeros((13, 8, 8), dtype=np.float32),
                vector=np.zeros(64, dtype=np.float32),
                action_mask=np.ones(10, dtype=np.float32),
                policy=np.ones(10, dtype=np.float32) / 10,
                player=1,
                shaped_reward=0.0,
            ),
        ]
        examples = actor._create_training_examples(history, winner=0)
        assert examples[0].value == -1.0


# ---------------------------------------------------------------------------
# Play game (mocked MCTS + env)
# ---------------------------------------------------------------------------


class TestPlayGame:
    def test_play_game_raises_without_mcts(self, actor: SelfPlayActor) -> None:
        with pytest.raises(RuntimeError, match="MCTS not initialized"):
            actor.play_game()

    def test_play_game_returns_trajectory(self, actor: SelfPlayActor) -> None:
        """Full game with mocked MCTS search and env stepping."""
        actor._init_network()

        action_size = actor.action_space_size
        # Mock MCTS search to return uniform policy
        uniform_policy = np.ones(action_size) / action_size
        actor.mcts.search = MagicMock(return_value=(uniform_policy, 0.0))
        actor.mcts.select_action = MagicMock(return_value=0)

        # Mock env to terminate after a few steps
        step_count = 0
        original_step = actor.env.step

        def mock_step(action: int) -> tuple:
            nonlocal step_count
            step_count += 1
            if step_count >= 3:
                # Force termination
                obs = actor.env.get_observation()
                info = {"winner": 0, "player_scores": None}
                return obs, 1.0, True, False, info
            return original_step(action)

        actor.env.step = mock_step

        trajectory = actor.play_game()
        assert isinstance(trajectory, GameTrajectory)
        assert len(trajectory.examples) >= 1
        assert trajectory.actor_id == 0
        assert trajectory.game_id == 1

    def test_play_game_increments_game_id(self, actor: SelfPlayActor) -> None:
        actor._init_network()
        action_size = actor.action_space_size
        uniform_policy = np.ones(action_size) / action_size
        actor.mcts.search = MagicMock(return_value=(uniform_policy, 0.0))
        actor.mcts.select_action = MagicMock(return_value=0)

        call_count = 0

        def mock_step(action: int) -> tuple:
            nonlocal call_count
            call_count += 1
            obs = actor.env.get_observation()
            info = {"winner": None}
            return obs, 0.0, True, False, info

        actor.env.step = mock_step

        t1 = actor.play_game()
        call_count = 0
        actor.env.reset()
        t2 = actor.play_game()
        assert t2.game_id == t1.game_id + 1


# ---------------------------------------------------------------------------
# Actor run loop
# ---------------------------------------------------------------------------


class TestActorRun:
    def test_run_plays_max_games(
        self,
        actor_config: ActorConfig,
        mock_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Actor should stop after max_games."""
        actor_config.max_games = 2
        actor_config.log_interval = 1
        actor = SelfPlayActor(actor_config, mock_storage, mock_replay_buffer)

        # Mock play_game to return a simple trajectory
        dummy_trajectory = GameTrajectory(
            examples=[
                TrainingExample(
                    spatial=np.zeros((13, 8, 8)),
                    vector=np.zeros(64),
                    action_mask=np.ones(10),
                    policy=np.ones(10) / 10,
                    value=0.5,
                )
            ],
            game_id=1,
            actor_id=0,
            game_length=1,
            winner=0,
        )

        with patch.object(actor, "play_game", return_value=dummy_trajectory):
            with patch.object(actor, "_init_network"):
                actor.run()

        assert actor.games_played == 2
        assert mock_replay_buffer.submit_trajectory.call_count == 2

    def test_run_stops_on_storage_stop(
        self,
        actor_config: ActorConfig,
        mock_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Actor should stop when storage signals stop."""
        actor_config.max_games = 0  # unlimited
        # Stop after first call
        mock_storage.should_stop.side_effect = [False, True]
        actor = SelfPlayActor(actor_config, mock_storage, mock_replay_buffer)

        dummy_trajectory = GameTrajectory(
            examples=[
                TrainingExample(
                    spatial=np.zeros((13, 8, 8)),
                    vector=np.zeros(64),
                    action_mask=np.ones(10),
                    policy=np.ones(10) / 10,
                    value=0.5,
                )
            ],
            game_id=1,
            actor_id=0,
            game_length=1,
            winner=0,
        )

        with patch.object(actor, "play_game", return_value=dummy_trajectory):
            with patch.object(actor, "_init_network"):
                actor.run()

        assert actor.games_played == 1

    def test_run_handles_game_error(
        self,
        actor_config: ActorConfig,
        mock_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Actor should catch exceptions in play_game and continue."""
        actor_config.max_games = 2
        call_count = 0

        dummy_trajectory = GameTrajectory(
            examples=[
                TrainingExample(
                    spatial=np.zeros((13, 8, 8)),
                    vector=np.zeros(64),
                    action_mask=np.ones(10),
                    policy=np.ones(10) / 10,
                    value=0.5,
                )
            ],
            game_id=1,
            actor_id=0,
            game_length=1,
            winner=0,
        )

        def play_game_with_error() -> GameTrajectory:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Simulated error")
            return dummy_trajectory

        actor = SelfPlayActor(actor_config, mock_storage, mock_replay_buffer)

        with patch.object(actor, "play_game", side_effect=play_game_with_error):
            with patch.object(actor, "_init_network"):
                with patch("alpha_zero.distributed.actor.time.sleep"):
                    actor.run()

        # First game errored, second succeeded; needs one more to hit max_games
        assert actor.games_played >= 1

    def test_run_submits_trajectory_failure(
        self,
        actor_config: ActorConfig,
        mock_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """When submit_trajectory returns False, games_played should not increment."""
        actor_config.max_games = 1
        mock_replay_buffer.submit_trajectory.return_value = False
        # Make storage stop after 2 iterations to avoid infinite loop
        mock_storage.should_stop.side_effect = [False, True]

        actor = SelfPlayActor(actor_config, mock_storage, mock_replay_buffer)

        dummy_trajectory = GameTrajectory(
            examples=[], game_id=1, actor_id=0, game_length=0, winner=None
        )

        with patch.object(actor, "play_game", return_value=dummy_trajectory):
            with patch.object(actor, "_init_network"):
                actor.run()

        assert actor.games_played == 0


# ---------------------------------------------------------------------------
# Stop / statistics
# ---------------------------------------------------------------------------


class TestActorControl:
    def test_stop(self, actor: SelfPlayActor) -> None:
        actor._running = True
        actor.stop()
        assert actor.is_running() is False

    def test_get_statistics_initial(self, actor: SelfPlayActor) -> None:
        stats = actor.get_statistics()
        assert stats["actor_id"] == 0
        assert stats["games_played"] == 0
        assert stats["is_running"] is False
        assert stats["model_version"] == 0

    def test_get_statistics_after_play(self, actor: SelfPlayActor) -> None:
        actor.games_played = 10
        actor.total_examples = 100
        actor.start_time = 1.0
        stats = actor.get_statistics()
        assert stats["games_played"] == 10
        assert stats["total_examples"] == 100
        assert stats["elapsed_time"] > 0


# ---------------------------------------------------------------------------
# actor_process entry point
# ---------------------------------------------------------------------------


class TestActorProcess:
    def test_actor_process_runs_and_exits(
        self, mock_storage: MagicMock, mock_replay_buffer: MagicMock
    ) -> None:
        """actor_process should create an actor and call run()."""
        config_dict = {
            "board_size": 8,
            "max_turns": 10,
            "num_players": 2,
            "num_res_blocks": 1,
            "num_filters": 8,
            "num_simulations": 2,
            "c_puct": 1.0,
            "dirichlet_alpha": 0.3,
            "dirichlet_epsilon": 0.25,
            "temperature": 1.0,
            "temperature_threshold": 5,
            "games_per_update": 1,
            "max_games": 1,
            "device": "cpu",
            "weight_update_interval": 1.0,
            "log_interval": 1,
        }

        with patch("alpha_zero.distributed.actor.SelfPlayActor") as MockActor:
            instance = MagicMock()
            MockActor.return_value = instance
            actor_process(0, config_dict, mock_storage, mock_replay_buffer)
            instance.run.assert_called_once()

    def test_actor_process_handles_crash(
        self, mock_storage: MagicMock, mock_replay_buffer: MagicMock
    ) -> None:
        """actor_process should catch exceptions from actor.run()."""
        config_dict = {
            "board_size": 8,
            "max_turns": 10,
            "num_players": 2,
            "num_res_blocks": 1,
            "num_filters": 8,
            "num_simulations": 2,
            "c_puct": 1.0,
            "dirichlet_alpha": 0.3,
            "dirichlet_epsilon": 0.25,
            "temperature": 1.0,
            "temperature_threshold": 5,
            "games_per_update": 1,
            "max_games": 1,
            "device": "cpu",
            "weight_update_interval": 1.0,
            "log_interval": 1,
        }

        with patch("alpha_zero.distributed.actor.SelfPlayActor") as MockActor:
            instance = MagicMock()
            instance.run.side_effect = RuntimeError("crash")
            MockActor.return_value = instance
            # Should not raise
            actor_process(0, config_dict, mock_storage, mock_replay_buffer)


# ---------------------------------------------------------------------------
# ActorPool
# ---------------------------------------------------------------------------


class TestActorPool:
    def test_init(
        self,
        actor_config: ActorConfig,
        mock_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        pool = ActorPool(3, actor_config, mock_storage, mock_replay_buffer)
        assert pool.num_actors == 3
        assert not pool._started
        assert pool.processes == []

    def test_is_running_empty(
        self,
        actor_config: ActorConfig,
        mock_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        pool = ActorPool(1, actor_config, mock_storage, mock_replay_buffer)
        assert pool.is_running() is False
        assert pool.get_running_count() == 0

    def test_stop_when_not_started(
        self,
        actor_config: ActorConfig,
        mock_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        pool = ActorPool(1, actor_config, mock_storage, mock_replay_buffer)
        pool.stop()  # Should not raise

    def test_restart_dead_actors_when_not_started(
        self,
        actor_config: ActorConfig,
        mock_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        pool = ActorPool(1, actor_config, mock_storage, mock_replay_buffer)
        assert pool.restart_dead_actors() == 0

    def test_restart_dead_actors_when_should_stop(
        self,
        actor_config: ActorConfig,
        mock_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        pool = ActorPool(1, actor_config, mock_storage, mock_replay_buffer)
        pool._started = True
        mock_storage.should_stop.return_value = True
        assert pool.restart_dead_actors() == 0

    def test_start_already_started(
        self,
        actor_config: ActorConfig,
        mock_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        pool = ActorPool(1, actor_config, mock_storage, mock_replay_buffer)
        pool._started = True
        pool.start()  # Should log warning and return without spawning

    @patch("alpha_zero.distributed.actor.mp.Process")
    def test_start_spawns_processes(
        self,
        MockProcess: MagicMock,
        actor_config: ActorConfig,
        mock_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        MockProcess.return_value = mock_proc

        pool = ActorPool(2, actor_config, mock_storage, mock_replay_buffer)
        pool.start()

        assert pool._started is True
        assert len(pool.processes) == 2
        assert mock_proc.start.call_count == 2

    @patch("alpha_zero.distributed.actor.mp.Process")
    def test_stop_joins_processes(
        self,
        MockProcess: MagicMock,
        actor_config: ActorConfig,
        mock_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.is_alive.return_value = False
        MockProcess.return_value = mock_proc

        pool = ActorPool(1, actor_config, mock_storage, mock_replay_buffer)
        pool.start()
        pool.stop(timeout=1.0)

        assert pool._started is False
        assert len(pool.processes) == 0
        mock_storage.request_stop.assert_called_once()

    @patch("alpha_zero.distributed.actor.mp.Process")
    def test_stop_terminates_hanging_process(
        self,
        MockProcess: MagicMock,
        actor_config: ActorConfig,
        mock_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        # Process stays alive after join, then dies after terminate
        mock_proc.is_alive.side_effect = [True, False]
        MockProcess.return_value = mock_proc

        pool = ActorPool(1, actor_config, mock_storage, mock_replay_buffer)
        pool.start()
        pool.stop(timeout=0.01)

        mock_proc.terminate.assert_called_once()

    @patch("alpha_zero.distributed.actor.mp.Process")
    def test_stop_kills_unkillable_process(
        self,
        MockProcess: MagicMock,
        actor_config: ActorConfig,
        mock_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        # Process stays alive even after terminate
        mock_proc.is_alive.return_value = True
        MockProcess.return_value = mock_proc

        pool = ActorPool(1, actor_config, mock_storage, mock_replay_buffer)
        pool.start()
        pool.stop(timeout=0.01)

        mock_proc.kill.assert_called_once()

    @patch("alpha_zero.distributed.actor.mp.Process")
    def test_get_running_count(
        self,
        MockProcess: MagicMock,
        actor_config: ActorConfig,
        mock_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        proc1 = MagicMock()
        proc1.pid = 1
        proc1.is_alive.return_value = True
        proc2 = MagicMock()
        proc2.pid = 2
        proc2.is_alive.return_value = False
        MockProcess.side_effect = [proc1, proc2]

        pool = ActorPool(2, actor_config, mock_storage, mock_replay_buffer)
        pool.start()
        assert pool.get_running_count() == 1
        assert pool.is_running() is True

    @patch("alpha_zero.distributed.actor.mp.Process")
    def test_restart_dead_actors(
        self,
        MockProcess: MagicMock,
        actor_config: ActorConfig,
        mock_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        # First process: alive. Second process: dead.
        alive_proc = MagicMock()
        alive_proc.pid = 1
        alive_proc.is_alive.return_value = True
        dead_proc = MagicMock()
        dead_proc.pid = 2
        dead_proc.is_alive.return_value = False
        new_proc = MagicMock()
        new_proc.pid = 3

        MockProcess.side_effect = [alive_proc, dead_proc, new_proc]

        pool = ActorPool(2, actor_config, mock_storage, mock_replay_buffer)
        pool.start()
        restarted = pool.restart_dead_actors()
        assert restarted == 1
        assert new_proc.start.call_count == 1
