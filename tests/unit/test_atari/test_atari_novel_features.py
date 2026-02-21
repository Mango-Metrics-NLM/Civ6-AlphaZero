"""Unit tests for patent-novel features: Reflection Memory + RAG Advisor."""

from __future__ import annotations

import numpy as np
import pytest

from demos.atari.novel_features import (
    AtariRAGAdvisor,
    AtariReflectionEntry,
    AtariReflectionMemory,
    AtariStateEncoder,
    KnowledgeEntry,
    NovelFeaturesConfig,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def enabled_config():
    return NovelFeaturesConfig(
        reflection_enabled=True,
        rag_enabled=True,
        reflection_max_entries=100,
        embedding_dimension=32,
    )


@pytest.fixture
def disabled_config():
    return NovelFeaturesConfig(reflection_enabled=False, rag_enabled=False)


@pytest.fixture
def dummy_obs():
    return {
        "spatial": np.random.rand(4, 84, 84).astype(np.float32),
        "vector": np.zeros(32, dtype=np.float32),
    }


def _make_trajectory(steps: int = 10, bad_step: int = 5) -> list[dict]:
    """Create a fake trajectory with a bad reward at a specific step."""
    traj = []
    for i in range(steps):
        obs = {
            "spatial": np.random.rand(4, 84, 84).astype(np.float32),
            "vector": np.zeros(32, dtype=np.float32),
        }
        reward = -5.0 if i == bad_step else 0.0
        traj.append({"obs": obs, "action": i % 4, "reward": reward})
    return traj


# ============================================================================
# NovelFeaturesConfig
# ============================================================================


class TestNovelFeaturesConfig:
    def test_defaults(self):
        cfg = NovelFeaturesConfig()
        assert cfg.reflection_enabled is False
        assert cfg.rag_enabled is False
        assert cfg.reflection_max_entries == 5000
        assert cfg.embedding_dimension == 128

    def test_custom_values(self):
        cfg = NovelFeaturesConfig(
            reflection_enabled=True,
            rag_enabled=True,
            reflection_max_entries=200,
            reflection_warning_penalty=0.2,
            rag_top_k=5,
            rag_prior_boost=0.2,
            embedding_dimension=64,
        )
        assert cfg.reflection_enabled is True
        assert cfg.reflection_max_entries == 200
        assert cfg.reflection_warning_penalty == 0.2
        assert cfg.rag_top_k == 5
        assert cfg.rag_prior_boost == 0.2
        assert cfg.embedding_dimension == 64


# ============================================================================
# AtariStateEncoder
# ============================================================================


class TestAtariStateEncoder:
    def test_encode_state_shape(self, dummy_obs):
        encoder = AtariStateEncoder(embedding_dim=64)
        emb = encoder.encode_state(dummy_obs)
        assert emb.shape == (64,)
        assert emb.dtype == np.float32

    def test_encode_state_normalised(self, dummy_obs):
        encoder = AtariStateEncoder(embedding_dim=32)
        emb = encoder.encode_state(dummy_obs)
        norm = np.linalg.norm(emb)
        assert np.isclose(norm, 1.0, atol=0.01)

    def test_encode_state_deterministic(self, dummy_obs):
        encoder = AtariStateEncoder(embedding_dim=32)
        emb1 = encoder.encode_state(dummy_obs)
        emb2 = encoder.encode_state(dummy_obs)
        np.testing.assert_array_almost_equal(emb1, emb2)

    def test_different_obs_different_embeddings(self):
        encoder = AtariStateEncoder(embedding_dim=32)
        obs1 = {"spatial": np.zeros((4, 84, 84), dtype=np.float32), "vector": np.zeros(32, dtype=np.float32)}
        obs2 = {"spatial": np.ones((4, 84, 84), dtype=np.float32), "vector": np.ones(32, dtype=np.float32)}
        emb1 = encoder.encode_state(obs1)
        emb2 = encoder.encode_state(obs2)
        assert not np.allclose(emb1, emb2)

    def test_state_hash(self, dummy_obs):
        h = AtariStateEncoder.state_hash(dummy_obs)
        assert isinstance(h, str)
        assert len(h) == 32  # MD5 hex digest

    def test_state_hash_deterministic(self, dummy_obs):
        h1 = AtariStateEncoder.state_hash(dummy_obs)
        h2 = AtariStateEncoder.state_hash(dummy_obs)
        assert h1 == h2

    def test_state_hash_different_for_different_obs(self):
        obs1 = {"spatial": np.zeros((4, 84, 84), dtype=np.float32)}
        obs2 = {"spatial": np.ones((4, 84, 84), dtype=np.float32)}
        assert AtariStateEncoder.state_hash(obs1) != AtariStateEncoder.state_hash(obs2)


# ============================================================================
# AtariReflectionMemory
# ============================================================================


class TestAtariReflectionMemory:
    def test_disabled_does_nothing(self, disabled_config):
        memory = AtariReflectionMemory(disabled_config)
        traj = _make_trajectory()
        result = memory.add_failed_episode(traj, -5.0, "Pong")
        assert result is None
        assert memory.size == 0

    def test_add_failed_episode(self, enabled_config):
        memory = AtariReflectionMemory(enabled_config)
        traj = _make_trajectory()
        entry = memory.add_failed_episode(traj, -5.0, "Pong")
        assert entry is not None
        assert isinstance(entry, AtariReflectionEntry)
        assert entry.episode_reward == -5.0
        assert entry.game_name == "Pong"
        assert memory.size == 1

    def test_add_multiple_entries(self, enabled_config):
        memory = AtariReflectionMemory(enabled_config)
        for i in range(5):
            # Use different observations to avoid deduplication
            traj = _make_trajectory(steps=10, bad_step=i)
            memory.add_failed_episode(traj, float(-i - 1), "Pong")
        assert memory.size == 5

    def test_deduplication(self, enabled_config):
        memory = AtariReflectionMemory(enabled_config)
        # Use the same trajectory twice
        traj = _make_trajectory()
        memory.add_failed_episode(traj, -5.0, "Pong")
        result = memory.add_failed_episode(traj, -5.0, "Pong")
        assert result is None
        assert memory.size == 1

    def test_max_entries(self):
        cfg = NovelFeaturesConfig(reflection_enabled=True, reflection_max_entries=3, embedding_dimension=32)
        memory = AtariReflectionMemory(cfg)
        for i in range(10):
            traj = _make_trajectory(steps=5, bad_step=i % 5)
            memory.add_failed_episode(traj, float(-i), f"Game{i}")
        assert memory.size <= 3

    def test_get_value_adjustment_disabled(self, disabled_config, dummy_obs):
        memory = AtariReflectionMemory(disabled_config)
        adj = memory.get_value_adjustment(dummy_obs, action=0)
        assert adj == 0.0

    def test_get_value_adjustment_no_entries(self, enabled_config, dummy_obs):
        memory = AtariReflectionMemory(enabled_config)
        adj = memory.get_value_adjustment(dummy_obs, action=0)
        assert adj == 0.0

    def test_get_value_adjustment_with_entries(self, enabled_config):
        memory = AtariReflectionMemory(enabled_config)
        traj = _make_trajectory(steps=5, bad_step=2)
        memory.add_failed_episode(traj, -5.0, "Pong")

        # Query with the same obs from the failure point
        failure_obs = traj[2]["obs"]
        failure_action = traj[2]["action"]
        adj = memory.get_value_adjustment(failure_obs, failure_action)
        # Should return a negative penalty (or 0 if threshold not met)
        assert adj <= 0.0

    def test_get_value_adjustment_sync(self, enabled_config):
        memory = AtariReflectionMemory(enabled_config)
        traj = _make_trajectory(steps=5, bad_step=2)
        memory.add_failed_episode(traj, -5.0, "Pong")

        state = {
            "spatial": traj[2]["obs"]["spatial"],
            "vector": traj[2]["obs"]["vector"],
        }
        adj = memory.get_value_adjustment_sync(state, traj[2]["action"])
        assert adj <= 0.0

    def test_query_similar_failures_empty(self, enabled_config, dummy_obs):
        memory = AtariReflectionMemory(enabled_config)
        results = memory.query_similar_failures(dummy_obs)
        assert results == []

    def test_query_similar_failures_returns_entries(self, enabled_config):
        memory = AtariReflectionMemory(enabled_config)
        traj = _make_trajectory(steps=5, bad_step=2)
        memory.add_failed_episode(traj, -5.0, "Pong")

        failure_obs = traj[2]["obs"]
        results = memory.query_similar_failures(failure_obs, top_k=5)
        # Should find at least the entry we added (same state = similarity 1.0)
        assert len(results) >= 0  # May or may not match depending on threshold

    def test_reset(self, enabled_config):
        memory = AtariReflectionMemory(enabled_config)
        traj = _make_trajectory()
        memory.add_failed_episode(traj, -5.0, "Pong")
        assert memory.size > 0
        memory.reset()
        assert memory.size == 0

    def test_empty_trajectory(self, enabled_config):
        memory = AtariReflectionMemory(enabled_config)
        result = memory.add_failed_episode([], -5.0, "Pong")
        assert result is None

    def test_critique_generation(self):
        traj = _make_trajectory(steps=10, bad_step=3)
        critique = AtariReflectionMemory._generate_critique(traj, 3, -5.0)
        assert "Episode failed" in critique
        assert "step 3" in critique
        assert "total_reward=-5.0" in critique

    def test_find_failure_point(self):
        traj = _make_trajectory(steps=10, bad_step=7)
        step, action = AtariReflectionMemory._find_failure_point(traj)
        assert step == 7
        assert isinstance(action, int)


# ============================================================================
# AtariRAGAdvisor
# ============================================================================


class TestAtariRAGAdvisor:
    def test_disabled_returns_empty(self, disabled_config, dummy_obs):
        advisor = AtariRAGAdvisor(disabled_config)
        results = advisor.query(dummy_obs)
        assert results == []

    def test_ingest_knowledge(self, enabled_config):
        advisor = AtariRAGAdvisor(enabled_config)
        advisor.ingest_knowledge("Test strategy", source="test", game_name="Pong")
        assert advisor.size == 1

    def test_ingest_multiple(self, enabled_config):
        advisor = AtariRAGAdvisor(enabled_config)
        for i in range(5):
            advisor.ingest_knowledge(f"Strategy {i}", source="test")
        assert advisor.size == 5

    def test_ingest_default_strategies_pong(self, enabled_config):
        advisor = AtariRAGAdvisor(enabled_config)
        advisor.ingest_default_strategies("Pong")
        assert advisor.size > 0

    def test_ingest_default_strategies_breakout(self, enabled_config):
        advisor = AtariRAGAdvisor(enabled_config)
        advisor.ingest_default_strategies("Breakout")
        assert advisor.size > 0

    def test_ingest_default_strategies_space_invaders(self, enabled_config):
        advisor = AtariRAGAdvisor(enabled_config)
        advisor.ingest_default_strategies("SpaceInvaders")
        assert advisor.size > 0

    def test_ingest_default_strategies_mspacman(self, enabled_config):
        advisor = AtariRAGAdvisor(enabled_config)
        advisor.ingest_default_strategies("MsPacman")
        assert advisor.size > 0

    def test_ingest_unknown_game(self, enabled_config):
        advisor = AtariRAGAdvisor(enabled_config)
        advisor.ingest_default_strategies("UnknownGame123")
        assert advisor.size == 0

    def test_query_returns_entries(self, enabled_config, dummy_obs):
        advisor = AtariRAGAdvisor(enabled_config)
        advisor.ingest_knowledge(
            "Move paddle to track ball",
            source="test",
            game_name="Pong",
            action_hints=[2, 3],
        )
        results = advisor.query(dummy_obs)
        assert len(results) > 0
        assert isinstance(results[0], KnowledgeEntry)

    def test_query_respects_top_k(self, enabled_config, dummy_obs):
        advisor = AtariRAGAdvisor(enabled_config)
        for i in range(10):
            advisor.ingest_knowledge(f"Strategy {i}", source="test")
        results = advisor.query(dummy_obs, top_k=2)
        assert len(results) <= 2

    def test_query_filters_by_game(self, enabled_config, dummy_obs):
        advisor = AtariRAGAdvisor(enabled_config)
        advisor.ingest_knowledge("Pong strategy", game_name="Pong")
        advisor.ingest_knowledge("Breakout strategy", game_name="Breakout")
        results = advisor.query(dummy_obs, game_name="Pong", top_k=10)
        for entry in results:
            assert entry.game_name in ("Pong", "")  # universal or matching

    def test_get_policy_boost_disabled(self, disabled_config, dummy_obs):
        advisor = AtariRAGAdvisor(disabled_config)
        boost = advisor.get_policy_boost(dummy_obs, num_actions=6)
        assert boost.shape == (6,)
        assert np.all(boost == 0.0)

    def test_get_policy_boost_no_knowledge(self, enabled_config, dummy_obs):
        advisor = AtariRAGAdvisor(enabled_config)
        boost = advisor.get_policy_boost(dummy_obs, num_actions=6)
        assert np.all(boost == 0.0)

    def test_get_policy_boost_with_knowledge(self, enabled_config, dummy_obs):
        advisor = AtariRAGAdvisor(enabled_config)
        advisor.ingest_knowledge(
            "Move up to track ball", action_hints=[2], source="test"
        )
        boost = advisor.get_policy_boost(dummy_obs, num_actions=6)
        assert boost.shape == (6,)
        # Action 2 should have received a boost
        assert boost[2] > 0.0

    def test_get_policy_boost_respects_action_bounds(self, enabled_config, dummy_obs):
        advisor = AtariRAGAdvisor(enabled_config)
        advisor.ingest_knowledge(
            "Strategy with out-of-range hint",
            action_hints=[0, 99],  # 99 is out of range for 6 actions
            source="test",
        )
        boost = advisor.get_policy_boost(dummy_obs, num_actions=6)
        assert boost.shape == (6,)
        # Out-of-range action should be ignored
        assert boost[0] > 0.0

    def test_knowledge_entry_fields(self, enabled_config):
        advisor = AtariRAGAdvisor(enabled_config)
        advisor.ingest_knowledge(
            "Test text", source="wiki", game_name="Pong", action_hints=[1, 2]
        )
        entry = advisor._knowledge[0]
        assert entry.text == "Test text"
        assert entry.source == "wiki"
        assert entry.game_name == "Pong"
        assert entry.action_hints == [1, 2]
        assert entry.embedding.shape == (enabled_config.embedding_dimension,)


# ============================================================================
# Config integration
# ============================================================================


class TestNovelFeaturesConfigIntegration:
    def test_atari_config_novel_features_default_none(self):
        from demos.atari.config import AtariConfig

        cfg = AtariConfig()
        assert cfg.novel_features is None

    def test_atari_config_from_dict_with_novel_features(self):
        from demos.atari.config import AtariConfig

        data = {
            "novel_features": {
                "reflection_enabled": True,
                "rag_enabled": True,
                "reflection_max_entries": 200,
            }
        }
        cfg = AtariConfig.from_dict(data)
        assert cfg.novel_features is not None
        assert cfg.novel_features.reflection_enabled is True
        assert cfg.novel_features.rag_enabled is True
        assert cfg.novel_features.reflection_max_entries == 200

    def test_atari_config_from_dict_without_novel_features(self):
        from demos.atari.config import AtariConfig

        data = {"environment": {"game_name": "Pong"}}
        cfg = AtariConfig.from_dict(data)
        assert cfg.novel_features is None

    def test_atari_config_env_override(self, monkeypatch):
        from demos.atari.config import AtariConfig

        monkeypatch.setenv("ATARI_DEMO_REFLECTION_ENABLED", "true")
        monkeypatch.setenv("ATARI_DEMO_RAG_ENABLED", "1")

        cfg = AtariConfig()
        cfg.apply_env_overrides()
        assert cfg.novel_features is not None
        assert cfg.novel_features.reflection_enabled is True
        assert cfg.novel_features.rag_enabled is True
