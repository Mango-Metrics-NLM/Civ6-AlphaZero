"""Comprehensive tests for alpha_zero.training_civ6 module.

Covers: TrainingConfig, TrainingExample, ReplayBuffer, ReplayDataset,
self_play_game, run_self_play, train_epoch, evaluate_against_random,
train, main (CLI parsing).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from alpha_zero.training_civ6 import (
    ReplayBuffer,
    ReplayDataset,
    TrainingConfig,
    TrainingExample,
    evaluate_against_random,
    main,
    run_self_play,
    self_play_game,
    train,
    train_epoch,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_example(
    board_size: int = 16, action_size: int = 356, value: float = 1.0
) -> TrainingExample:
    return TrainingExample(
        spatial=np.random.randn(13, board_size, board_size).astype(np.float32),
        vector=np.random.randn(64).astype(np.float32),
        action_mask=np.ones(action_size, dtype=np.float32),
        policy=np.ones(action_size, dtype=np.float32) / action_size,
        value=value,
    )


def _mock_reset_return(config: TrainingConfig) -> tuple[dict, dict]:
    """Build a realistic ``MicroCivEnv.reset()`` return for a mocked env.

    ``train()`` calls ``env.reset()`` to discover the *dynamic* spatial
    channel count (``3 + 4 * num_players + 2``) before constructing the
    network. A real Gym-style env always returns ``(obs, info)``; a bare
    ``MagicMock`` returns an empty auto-iterator instead, so tests that
    replace ``MicroCivEnv`` with a ``MagicMock`` must configure
    ``reset.return_value`` explicitly or unpacking crashes with
    ``ValueError: not enough values to unpack``.
    """
    spatial_channels = 3 + 4 * config.num_players + 2
    obs = {
        "spatial": np.zeros(
            (spatial_channels, config.board_size, config.board_size), dtype=np.float32
        ),
        "vector": np.zeros(64, dtype=np.float32),
        "action_mask": np.ones(1, dtype=np.float32),
    }
    return obs, {}


@pytest.fixture
def config():
    return TrainingConfig(
        board_size=8,
        max_turns=20,
        num_res_blocks=1,
        num_filters=16,
        num_simulations=2,
        batch_size=4,
        learning_rate=0.001,
        num_epochs=1,
        num_games_per_iteration=2,
        num_iterations=2,
        replay_buffer_size=200,
        min_replay_size=2,
        save_interval=1,
        device="cpu",
    )


@pytest.fixture
def examples():
    return [_make_example() for _ in range(10)]


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.board_size == 16
        assert cfg.num_iterations == 100
        assert cfg.replay_buffer_size == 50000
        assert cfg.gradient_clip == 1.0

    def test_custom_values(self):
        cfg = TrainingConfig(board_size=8, num_simulations=10, batch_size=32)
        assert cfg.board_size == 8
        assert cfg.num_simulations == 10
        assert cfg.batch_size == 32

    def test_gumbel_defaults(self):
        cfg = TrainingConfig()
        assert cfg.use_gumbel_mcts is False
        assert cfg.gumbel_num_simulations == 16
        assert cfg.gumbel_num_sampled_actions == 16
        assert cfg.gumbel_use_sequential_halving is True

    def test_device_field(self):
        cfg = TrainingConfig(device="cpu")
        assert cfg.device == "cpu"


# ---------------------------------------------------------------------------
# TrainingExample
# ---------------------------------------------------------------------------


class TestTrainingExample:
    def test_creation(self):
        ex = _make_example(value=-1.0)
        assert ex.value == -1.0
        assert ex.spatial.shape == (13, 16, 16)
        assert ex.vector.shape == (64,)


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------


class TestReplayBuffer:
    def test_init(self):
        buf = ReplayBuffer(max_size=100)
        assert len(buf) == 0

    def test_add_single(self):
        buf = ReplayBuffer()
        buf.add(_make_example())
        assert len(buf) == 1

    def test_add_game(self, examples):
        buf = ReplayBuffer()
        buf.add_game(examples)
        assert len(buf) == 10

    def test_max_size(self):
        buf = ReplayBuffer(max_size=5)
        for _ in range(10):
            buf.add(_make_example())
        assert len(buf) == 5

    def test_sample(self, examples):
        buf = ReplayBuffer()
        buf.add_game(examples)
        sampled = buf.sample(3)
        assert len(sampled) == 3
        assert all(isinstance(s, TrainingExample) for s in sampled)

    def test_sample_returns_correct_batch_size(self, examples):
        buf = ReplayBuffer()
        buf.add_game(examples)
        sampled = buf.sample(5)
        assert len(sampled) == 5


# ---------------------------------------------------------------------------
# ReplayDataset
# ---------------------------------------------------------------------------


class TestReplayDataset:
    def test_len(self, examples):
        ds = ReplayDataset(examples)
        assert len(ds) == 10

    def test_getitem(self, examples):
        ds = ReplayDataset(examples)
        item = ds[0]
        assert isinstance(item, dict)
        assert set(item.keys()) == {"spatial", "vector", "action_mask", "policy", "value"}
        assert item["spatial"].dtype == torch.float32
        assert item["value"].dtype == torch.float32

    def test_getitem_shapes(self, examples):
        ds = ReplayDataset(examples)
        item = ds[0]
        assert item["spatial"].shape == (13, 16, 16)
        assert item["vector"].shape == (64,)

    def test_dataloader(self, examples):
        ds = ReplayDataset(examples)
        dl = DataLoader(ds, batch_size=4, shuffle=False)
        batch = next(iter(dl))
        assert batch["spatial"].shape[0] == 4
        assert batch["value"].shape == (4,)


# ---------------------------------------------------------------------------
# self_play_game
# ---------------------------------------------------------------------------


class TestSelfPlayGame:
    def _make_mock_env(self, num_steps: int = 3):
        """Create a mock env that terminates after num_steps."""
        env = MagicMock()
        spatial = np.zeros((13, 8, 8), dtype=np.float32)
        vector = np.zeros(64, dtype=np.float32)
        action_mask = np.ones(100, dtype=np.float32)
        obs = {"spatial": spatial, "vector": vector, "action_mask": action_mask}

        env.reset.return_value = (obs, {})
        env.current_player = 0
        # A real MicroCivEnv always sets num_players (enforced >= 2 in its
        # constructor). Without this, a bare MagicMock auto-vivifies a child
        # attribute whose int() resolves to 1, which previously crashed
        # self_play_game's N-player scoring with "num_players must be >= 2".
        env.num_players = 2

        step_returns = []
        for _i in range(num_steps - 1):
            step_returns.append((obs, 0.0, False, False, {}))
        # Final step: terminated
        step_returns.append((obs, 0.0, True, False, {"winner": 0}))
        env.step.side_effect = step_returns
        env.turn = 5
        return env

    def _make_mock_mcts(self, action_size: int = 100):
        mcts = MagicMock()
        policy = np.ones(action_size, dtype=np.float32) / action_size
        mcts.search.return_value = (policy, None)
        mcts.select_action.return_value = 0
        return mcts

    def test_returns_examples(self):
        env = self._make_mock_env(num_steps=2)
        mcts = self._make_mock_mcts()
        cfg = TrainingConfig(temperature_threshold=30)

        result = self_play_game(env, mcts, cfg)
        assert len(result) == 2
        assert all(isinstance(e, TrainingExample) for e in result)

    def test_winner_values(self):
        env = self._make_mock_env(num_steps=2)
        # Player 0 wins
        mcts = self._make_mock_mcts()
        cfg = TrainingConfig()

        result = self_play_game(env, mcts, cfg)
        # All examples are for player 0 (env.current_player=0), winner=0
        assert all(e.value == 1.0 for e in result)

    def test_draw_values(self):
        env = self._make_mock_env(num_steps=1)
        # Override final step to have no winner
        env.step.side_effect = [
            (
                {
                    "spatial": np.zeros((13, 8, 8), dtype=np.float32),
                    "vector": np.zeros(64, dtype=np.float32),
                    "action_mask": np.ones(100, dtype=np.float32),
                },
                0.0,
                True,
                False,
                {"winner": None},
            )
        ]
        mcts = self._make_mock_mcts()
        cfg = TrainingConfig()

        result = self_play_game(env, mcts, cfg)
        assert all(e.value == 0.0 for e in result)

    def test_loser_values(self):
        env = self._make_mock_env(num_steps=1)
        # Player 0 plays, but player 1 wins
        env.step.side_effect = [
            (
                {
                    "spatial": np.zeros((13, 8, 8), dtype=np.float32),
                    "vector": np.zeros(64, dtype=np.float32),
                    "action_mask": np.ones(100, dtype=np.float32),
                },
                0.0,
                True,
                False,
                {"winner": 1},
            )
        ]
        mcts = self._make_mock_mcts()
        cfg = TrainingConfig()

        result = self_play_game(env, mcts, cfg)
        assert all(e.value == -1.0 for e in result)

    def test_truncated_game(self):
        env = MagicMock()
        obs = {
            "spatial": np.zeros((13, 8, 8), dtype=np.float32),
            "vector": np.zeros(64, dtype=np.float32),
            "action_mask": np.ones(100, dtype=np.float32),
        }
        env.reset.return_value = (obs, {})
        env.current_player = 0
        env.turn = 5
        env.num_players = 2
        env.step.side_effect = [(obs, 0.0, False, True, {"winner": None})]

        mcts = self._make_mock_mcts()
        cfg = TrainingConfig()

        result = self_play_game(env, mcts, cfg)
        assert len(result) == 1
        assert result[0].value == 0.0

    def test_temperature_selection(self):
        """Verify temperature is used for early turns, 0 for late."""
        env = self._make_mock_env(num_steps=1)
        env.turn = 50  # past threshold
        mcts = self._make_mock_mcts()
        cfg = TrainingConfig(temperature=1.0, temperature_threshold=30)

        self_play_game(env, mcts, cfg)
        # select_action should be called with temperature=0.0 (past threshold)
        call_args = mcts.select_action.call_args
        assert call_args[0][1] == 0.0

    def test_missing_num_players_falls_back_to_two_player_default(self, caplog):
        """A non-conforming env double without a real ``num_players`` must
        not crash self-play.

        A real ``MicroCivEnv`` always sets ``num_players`` (enforced >= 2 by
        its constructor), so this scenario is unreachable with a real env. It
        only arises from an incompletely configured test double: a bare
        ``MagicMock`` auto-vivifies a child attribute for ``env.num_players``,
        and ``int()`` of that attribute resolves to ``1`` -- which used to
        crash ``self_play_game`` deep inside ``compute_rankings`` with
        "num_players must be >= 2, got 1". ``self_play_game`` must instead log
        a warning and fall back to the historical 2-player default.
        """
        env = MagicMock()
        spatial = np.zeros((13, 8, 8), dtype=np.float32)
        vector = np.zeros(64, dtype=np.float32)
        action_mask = np.ones(100, dtype=np.float32)
        obs = {"spatial": spatial, "vector": vector, "action_mask": action_mask}
        env.reset.return_value = (obs, {})
        env.current_player = 0
        env.turn = 5
        # Intentionally NOT setting env.num_players -- reproduces the bug.
        env.step.side_effect = [(obs, 0.0, True, False, {"winner": 0})]

        mcts = self._make_mock_mcts()
        cfg = TrainingConfig()

        with caplog.at_level(logging.WARNING, logger="alpha_zero.training_civ6"):
            result = self_play_game(env, mcts, cfg)

        assert len(result) == 1
        # Effective 2-player winner-take-all: player 0 (the only recorded
        # actor) won, so the classic +1 signal applies.
        assert result[0].value == 1.0
        assert result[0].value_vector is None
        assert any("num_players" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# run_self_play
# ---------------------------------------------------------------------------


class TestRunSelfPlay:
    @patch("alpha_zero.training_civ6.self_play_game")
    @patch("alpha_zero.training_civ6.MicroCivEnv")
    def test_simple_mcts(self, mock_env_cls, mock_spg):
        mock_spg.return_value = [_make_example()]
        network = MagicMock()
        cfg = TrainingConfig(board_size=8, num_simulations=2)

        result = run_self_play(network, cfg, num_games=2, use_simple_mcts=True)
        assert len(result) == 2
        # SimpleMCTS should be used, so network is not passed to MCTS
        call_args = mock_spg.call_args_list
        for call in call_args:
            # We can't check type because it's the real SimpleMCTS, but we can
            # verify the function was called with the env
            assert call[0][2] == cfg

    @patch("alpha_zero.training_civ6.self_play_game")
    @patch("alpha_zero.training_civ6.MicroCivEnv")
    @patch("alpha_zero.training_civ6.Civ6MCTS")
    def test_civ6_mcts(self, mock_mcts_cls, mock_env_cls, mock_spg):
        mock_spg.return_value = [_make_example()]
        network = MagicMock()
        cfg = TrainingConfig(board_size=8, num_simulations=2, use_gumbel_mcts=False)

        result = run_self_play(network, cfg, num_games=1, use_simple_mcts=False)
        assert len(result) == 1
        mock_mcts_cls.assert_called_once()

    @patch("alpha_zero.training_civ6.self_play_game")
    @patch("alpha_zero.training_civ6.MicroCivEnv")
    @patch("alpha_zero.training_civ6.GumbelMCTS")
    def test_gumbel_mcts(self, mock_gumbel_cls, mock_env_cls, mock_spg):
        mock_spg.return_value = [_make_example()]
        network = MagicMock()
        cfg = TrainingConfig(board_size=8, use_gumbel_mcts=True)

        result = run_self_play(network, cfg, num_games=1, use_simple_mcts=False)
        assert len(result) == 1
        mock_gumbel_cls.assert_called_once()

    @patch("alpha_zero.training_civ6.self_play_game")
    @patch("alpha_zero.training_civ6.MicroCivEnv")
    def test_accumulates_examples(self, mock_env_cls, mock_spg):
        mock_spg.return_value = [_make_example(), _make_example()]
        network = MagicMock()
        cfg = TrainingConfig(board_size=8, num_simulations=2)

        result = run_self_play(network, cfg, num_games=3, use_simple_mcts=True)
        assert len(result) == 6  # 2 examples * 3 games


# ---------------------------------------------------------------------------
# train_epoch
# ---------------------------------------------------------------------------


class TestTrainEpoch:
    def test_returns_loss_dict(self, config):
        action_size = 100
        examples = [
            _make_example(board_size=config.board_size, action_size=action_size) for _ in range(8)
        ]
        ds = ReplayDataset(examples)
        dl = DataLoader(ds, batch_size=config.batch_size, shuffle=False)

        from alpha_zero.core.civ6_network import MicroCivNetwork

        network = MicroCivNetwork(
            board_size=config.board_size,
            action_space_size=action_size,
            num_res_blocks=1,
            num_filters=16,
        )
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            losses = train_epoch(network, optimizer, dl, config)

        assert "total_loss" in losses
        assert "policy_loss" in losses
        assert "value_loss" in losses
        assert all(isinstance(v, float) for v in losses.values())
        assert all(v >= 0 for v in losses.values())

    def test_network_in_train_mode(self, config):
        action_size = 100
        examples = [
            _make_example(board_size=config.board_size, action_size=action_size) for _ in range(4)
        ]
        ds = ReplayDataset(examples)
        dl = DataLoader(ds, batch_size=config.batch_size, shuffle=False)

        from alpha_zero.core.civ6_network import MicroCivNetwork

        network = MicroCivNetwork(
            board_size=config.board_size,
            action_space_size=action_size,
            num_res_blocks=1,
            num_filters=16,
        )
        network.eval()  # Start in eval mode
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            train_epoch(network, optimizer, dl, config)

        assert network.training  # Should be in train mode after

    def test_multiple_batches(self, config):
        action_size = 100
        # Enough examples for multiple batches
        examples = [
            _make_example(board_size=config.board_size, action_size=action_size) for _ in range(20)
        ]
        ds = ReplayDataset(examples)
        dl = DataLoader(ds, batch_size=4, shuffle=False)

        from alpha_zero.core.civ6_network import MicroCivNetwork

        network = MicroCivNetwork(
            board_size=config.board_size,
            action_space_size=action_size,
            num_res_blocks=1,
            num_filters=16,
        )
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            losses = train_epoch(network, optimizer, dl, config)

        # Should have averaged over all batches
        assert losses["total_loss"] > 0


# ---------------------------------------------------------------------------
# evaluate_against_random
# ---------------------------------------------------------------------------


class TestEvaluateAgainstRandom:
    @patch("alpha_zero.training_civ6.Civ6MCTS")
    @patch("alpha_zero.training_civ6.MicroCivEnv")
    def test_returns_win_rate(self, mock_env_cls, mock_mcts_cls):
        mock_env = MagicMock()
        mock_env_cls.return_value = mock_env

        obs = {
            "spatial": np.zeros((13, 8, 8), dtype=np.float32),
            "vector": np.zeros(64, dtype=np.float32),
            "action_mask": np.ones(100, dtype=np.float32),
        }
        mock_env.reset.return_value = (obs, {})
        mock_env.current_player = 0
        mock_env.get_valid_actions.return_value = np.ones(100, dtype=np.float32)

        # All games terminate immediately with player 0 winning
        mock_env.step.return_value = (obs, 0.0, True, False, {"winner": 0})

        mock_mcts = MagicMock()
        mock_mcts_cls.return_value = mock_mcts
        policy = np.ones(100, dtype=np.float32) / 100
        mock_mcts.search.return_value = (policy, None)
        mock_mcts.select_action.return_value = 0

        network = MagicMock()
        cfg = TrainingConfig(board_size=8, num_simulations=2, device="cpu")

        win_rate = evaluate_against_random(network, cfg, num_games=4)
        assert isinstance(win_rate, float)
        assert 0.0 <= win_rate <= 1.0

    @patch("alpha_zero.training_civ6.Civ6MCTS")
    @patch("alpha_zero.training_civ6.MicroCivEnv")
    def test_alternates_first_player(self, mock_env_cls, mock_mcts_cls):
        """Network player alternates between 0 and 1 across games."""
        mock_env = MagicMock()
        mock_env_cls.return_value = mock_env

        obs = {
            "spatial": np.zeros((13, 8, 8), dtype=np.float32),
            "vector": np.zeros(64, dtype=np.float32),
            "action_mask": np.ones(100, dtype=np.float32),
        }
        mock_env.reset.return_value = (obs, {})
        mock_env.get_valid_actions.return_value = np.ones(100, dtype=np.float32)

        # Track which player the network is for each game
        def step_side_effect(action):
            # Record what network_player was for this game
            return (obs, 0.0, True, False, {"winner": None})

        mock_env.step.side_effect = step_side_effect
        mock_env.current_player = 0

        mock_mcts = MagicMock()
        mock_mcts_cls.return_value = mock_mcts
        policy = np.ones(100, dtype=np.float32) / 100
        mock_mcts.search.return_value = (policy, None)
        mock_mcts.select_action.return_value = 0

        network = MagicMock()
        cfg = TrainingConfig(board_size=8, num_simulations=2, device="cpu")

        win_rate = evaluate_against_random(network, cfg, num_games=4)
        # No winners so 0 wins
        assert win_rate == 0.0

    @patch("alpha_zero.training_civ6.Civ6MCTS")
    @patch("alpha_zero.training_civ6.MicroCivEnv")
    def test_perfect_win_rate(self, mock_env_cls, mock_mcts_cls):
        mock_env = MagicMock()
        mock_env_cls.return_value = mock_env

        obs = {
            "spatial": np.zeros((13, 8, 8), dtype=np.float32),
            "vector": np.zeros(64, dtype=np.float32),
            "action_mask": np.ones(100, dtype=np.float32),
        }
        mock_env.reset.return_value = (obs, {})
        mock_env.get_valid_actions.return_value = np.ones(100, dtype=np.float32)

        game_idx_counter = [0]

        def step_fn(action):
            network_player = game_idx_counter[0] % 2
            game_idx_counter[0] += 1
            return (obs, 0.0, True, False, {"winner": network_player})

        mock_env.step.side_effect = step_fn
        mock_env.current_player = 0

        mock_mcts = MagicMock()
        mock_mcts_cls.return_value = mock_mcts
        policy = np.ones(100, dtype=np.float32) / 100
        mock_mcts.search.return_value = (policy, None)
        mock_mcts.select_action.return_value = 0

        network = MagicMock()
        cfg = TrainingConfig(board_size=8, num_simulations=2, device="cpu")

        win_rate = evaluate_against_random(network, cfg, num_games=4)
        assert win_rate == 1.0


# ---------------------------------------------------------------------------
# train (main training loop)
# ---------------------------------------------------------------------------


class TestTrain:
    @patch("alpha_zero.training_civ6.evaluate_against_random")
    @patch("alpha_zero.training_civ6.run_self_play")
    def test_dry_run(self, mock_self_play, mock_eval, config, tmp_path):
        """Dry run completes with mocked self-play."""
        config.checkpoint_dir = str(tmp_path / "checkpoints")
        config.log_dir = str(tmp_path / "logs")

        action_size = 100
        # Return enough examples for training
        mock_self_play.return_value = [
            _make_example(board_size=config.board_size, action_size=action_size) for _ in range(20)
        ]
        mock_eval.return_value = 0.5

        # We need to mock the MicroCivEnv to control action_space_size
        with patch("alpha_zero.training_civ6.MicroCivEnv") as mock_env_cls:
            mock_env = MagicMock()
            mock_env.action_space.n = action_size
            mock_env.reset.return_value = _mock_reset_return(config)
            mock_env_cls.return_value = mock_env

            train(config, dry_run=True)

        # Should have saved checkpoints
        assert mock_eval.called
        assert mock_self_play.called

    @patch("alpha_zero.training_civ6.evaluate_against_random")
    @patch("alpha_zero.training_civ6.run_self_play")
    def test_creates_directories(self, mock_self_play, mock_eval, config, tmp_path):
        config.checkpoint_dir = str(tmp_path / "new_ckpt_dir")
        config.log_dir = str(tmp_path / "new_log_dir")

        action_size = 100
        mock_self_play.return_value = [
            _make_example(board_size=config.board_size, action_size=action_size) for _ in range(20)
        ]
        mock_eval.return_value = 0.5

        with patch("alpha_zero.training_civ6.MicroCivEnv") as mock_env_cls:
            mock_env = MagicMock()
            mock_env.action_space.n = action_size
            mock_env.reset.return_value = _mock_reset_return(config)
            mock_env_cls.return_value = mock_env

            train(config, dry_run=True)

        import os

        assert os.path.isdir(config.checkpoint_dir)
        assert os.path.isdir(config.log_dir)

    @patch("alpha_zero.training_civ6.evaluate_against_random")
    @patch("alpha_zero.training_civ6.run_self_play")
    def test_uses_simple_mcts_first_iteration(self, mock_self_play, mock_eval, config, tmp_path):
        config.checkpoint_dir = str(tmp_path / "ckpt")
        config.log_dir = str(tmp_path / "log")

        action_size = 100
        mock_self_play.return_value = [
            _make_example(board_size=config.board_size, action_size=action_size) for _ in range(20)
        ]
        mock_eval.return_value = 0.5

        with patch("alpha_zero.training_civ6.MicroCivEnv") as mock_env_cls:
            mock_env = MagicMock()
            mock_env.action_space.n = action_size
            mock_env.reset.return_value = _mock_reset_return(config)
            mock_env_cls.return_value = mock_env

            train(config, dry_run=True)

        # First call should use simple MCTS
        first_call = mock_self_play.call_args_list[0]
        assert (
            first_call[1].get(
                "use_simple_mcts", first_call[0][3] if len(first_call[0]) > 3 else None
            )
            is True
            or first_call.kwargs.get("use_simple_mcts") is True
        )

    @patch("alpha_zero.training_civ6.evaluate_against_random")
    @patch("alpha_zero.training_civ6.run_self_play")
    def test_skips_training_when_buffer_small(self, mock_self_play, mock_eval, config, tmp_path):
        config.checkpoint_dir = str(tmp_path / "ckpt")
        config.log_dir = str(tmp_path / "log")
        config.min_replay_size = 99999  # Very large threshold

        action_size = 100
        mock_self_play.return_value = [
            _make_example(board_size=config.board_size, action_size=action_size) for _ in range(5)
        ]
        mock_eval.return_value = 0.5

        with patch("alpha_zero.training_civ6.MicroCivEnv") as mock_env_cls:
            mock_env = MagicMock()
            mock_env.action_space.n = action_size
            mock_env.reset.return_value = _mock_reset_return(config)
            mock_env_cls.return_value = mock_env

            with patch("alpha_zero.training_civ6.train_epoch") as mock_train:
                # dry_run=False so the min_replay_size threshold is not bypassed
                # But we set num_iterations=2 via config
                train(config, dry_run=False)
                # train_epoch should NOT have been called since buffer is too small
                mock_train.assert_not_called()

    @patch("alpha_zero.training_civ6.optim.Adam")
    @patch("alpha_zero.training_civ6.evaluate_against_random")
    @patch("alpha_zero.training_civ6.run_self_play")
    def test_saves_final_model(self, mock_self_play, mock_eval, mock_adam, config, tmp_path):
        config.checkpoint_dir = str(tmp_path / "ckpt")
        config.log_dir = str(tmp_path / "log")

        action_size = 100
        mock_self_play.return_value = [
            _make_example(board_size=config.board_size, action_size=action_size) for _ in range(20)
        ]
        mock_eval.return_value = 0.5

        with patch("alpha_zero.training_civ6.MicroCivEnv") as mock_env_cls:
            mock_env = MagicMock()
            mock_env.action_space.n = action_size
            mock_env.reset.return_value = _mock_reset_return(config)
            mock_env_cls.return_value = mock_env

            with patch("alpha_zero.training_civ6.MicroCivNetwork") as mock_net_cls:
                mock_net = MagicMock()
                mock_net.to.return_value = mock_net  # .to(device) returns self
                mock_net_cls.return_value = mock_net

                with patch(
                    "alpha_zero.training_civ6.train_epoch",
                    return_value={
                        "total_loss": 0.1,
                        "policy_loss": 0.05,
                        "value_loss": 0.05,
                    },
                ):
                    train(config, dry_run=True)

                # Should save final model as "latest.pt"
                save_calls = mock_net.save.call_args_list
                final_paths = [c[0][0] for c in save_calls]
                assert any("latest.pt" in p for p in final_paths)

    @patch("alpha_zero.training_civ6.optim.Adam")
    @patch("alpha_zero.training_civ6.evaluate_against_random")
    @patch("alpha_zero.training_civ6.run_self_play")
    def test_saves_checkpoint_at_interval(
        self, mock_self_play, mock_eval, mock_adam, config, tmp_path
    ):
        config.checkpoint_dir = str(tmp_path / "ckpt")
        config.log_dir = str(tmp_path / "log")
        config.save_interval = 1  # Save every iteration

        action_size = 100
        mock_self_play.return_value = [
            _make_example(board_size=config.board_size, action_size=action_size) for _ in range(20)
        ]
        mock_eval.return_value = 0.5

        with patch("alpha_zero.training_civ6.MicroCivEnv") as mock_env_cls:
            mock_env = MagicMock()
            mock_env.action_space.n = action_size
            mock_env.reset.return_value = _mock_reset_return(config)
            mock_env_cls.return_value = mock_env

            with patch("alpha_zero.training_civ6.MicroCivNetwork") as mock_net_cls:
                mock_net = MagicMock()
                mock_net.to.return_value = mock_net  # .to(device) returns self
                mock_net_cls.return_value = mock_net

                with patch(
                    "alpha_zero.training_civ6.train_epoch",
                    return_value={
                        "total_loss": 0.1,
                        "policy_loss": 0.05,
                        "value_loss": 0.05,
                    },
                ):
                    train(config, dry_run=True)

                save_calls = mock_net.save.call_args_list
                # Should have checkpoint saves + final save
                assert len(save_calls) >= 2


# ---------------------------------------------------------------------------
# main (CLI)
# ---------------------------------------------------------------------------


#: ``train`` is mocked in ``TestMain``, so these tests play no games; the reference
#: only satisfies the 1,000-game cap so they can keep asserting on the shipped defaults.
_TEST_APPROVAL = "unit-test-train-is-mocked"


class TestMain:
    @patch("alpha_zero.training_civ6.train")
    def test_dry_run_flag(self, mock_train):
        with patch("sys.argv", ["training_civ6", "--dry_run", "--board_size", "8"]):
            main()
        mock_train.assert_called_once()
        call_args = mock_train.call_args
        cfg = call_args[0][0]
        dry_run = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("dry_run")
        assert dry_run is True
        assert cfg.board_size == 8

    @patch("alpha_zero.training_civ6.train")
    def test_default_args(self, mock_train):
        with patch("sys.argv", ["training_civ6", "--approval_ref", _TEST_APPROVAL]):
            main()
        mock_train.assert_called_once()
        cfg = mock_train.call_args[0][0]
        assert cfg.board_size == 16
        assert cfg.num_simulations == 50
        assert cfg.batch_size == 128

    @patch("alpha_zero.training_civ6.train")
    def test_custom_args(self, mock_train):
        with patch(
            "sys.argv",
            [
                "training_civ6",
                "--board_size",
                "12",
                "--num_simulations",
                "100",
                "--num_games",
                "50",
                "--num_iterations",
                "200",
                "--batch_size",
                "64",
                "--learning_rate",
                "0.01",
                "--num_epochs",
                "5",
                "--num_res_blocks",
                "4",
                "--num_filters",
                "32",
                "--checkpoint_dir",
                "/tmp/test_ckpt",
                "--approval_ref",
                _TEST_APPROVAL,
            ],
        ):
            main()
        cfg = mock_train.call_args[0][0]
        assert cfg.board_size == 12
        assert cfg.num_simulations == 100
        assert cfg.num_games_per_iteration == 50
        assert cfg.num_iterations == 200
        assert cfg.batch_size == 64
        assert cfg.learning_rate == 0.01
        assert cfg.num_epochs == 5
        assert cfg.num_res_blocks == 4
        assert cfg.num_filters == 32
        assert cfg.checkpoint_dir == "/tmp/test_ckpt"

    @patch("alpha_zero.training_civ6.train")
    def test_device_arg(self, mock_train):
        with patch(
            "sys.argv", ["training_civ6", "--device", "cpu", "--approval_ref", _TEST_APPROVAL]
        ):
            main()
        cfg = mock_train.call_args[0][0]
        assert cfg.device == "cpu"

    @patch("alpha_zero.training_civ6.train")
    def test_device_default_none(self, mock_train):
        with patch("sys.argv", ["training_civ6", "--approval_ref", _TEST_APPROVAL]):
            main()
        cfg = mock_train.call_args[0][0]
        # Should be "cuda" or "cpu" depending on availability, not None
        assert cfg.device in ("cuda", "cpu")


# ---------------------------------------------------------------------------
# Integration-style: run_self_play with SimpleMCTS (no mocking)
# ---------------------------------------------------------------------------


class TestRunSelfPlayIntegration:
    def test_simple_mcts_small_game(self):
        """Run a real (tiny) self-play game with SimpleMCTS."""
        network = MagicMock()
        cfg = TrainingConfig(board_size=8, max_turns=10, num_simulations=2)
        examples = run_self_play(network, cfg, num_games=1, use_simple_mcts=True)
        assert len(examples) > 0
        for ex in examples:
            assert ex.spatial.shape == (13, 8, 8)
            assert ex.value in (-1.0, 0.0, 1.0)
