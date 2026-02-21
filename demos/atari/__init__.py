"""Atari demo for AlphaZero-style training.

This package demonstrates the AlphaZero training pipeline on Gymnasium
Atari environments. It reuses core components (ResidualBlock, MCTS,
replay buffers, schedulers) from the main alpha_zero project while
providing Atari-specific adapters for environments and networks.

Key components:
    - AtariEnvConfig / AtariEnv: Gymnasium Atari wrapper implementing
      the MCTSEnvironment protocol.
    - AtariNetworkConfig / AtariNetwork: CNN for pixel observations
      reusing ResidualBlock from the core project.
    - AtariTrainingConfig / AtariTrainer: Single-player training loop
      adapted for reward-based (non-adversarial) environments.
    - AtariReflectionMemory: LATS-style reflection memory (patent novel).
    - AtariRAGAdvisor: RAG-enhanced MCTS advisor (patent novel).

Usage:
    python -m demos.atari --game Pong --dry-run
    python -m demos.atari --config demos/atari/configs/breakout.yaml
    python -m demos.atari --game Pong --reflection --rag --dry-run
"""

from __future__ import annotations

from demos.atari.atari_env import AtariEnv
from demos.atari.atari_network import AtariNetwork
from demos.atari.config import (
    AtariConfig,
    AtariEnvConfig,
    AtariNetworkConfig,
    AtariTrainingConfig,
)
from demos.atari.novel_features import (
    AtariRAGAdvisor,
    AtariReflectionMemory,
    NovelFeaturesConfig,
)
from demos.atari.trainer import AtariTrainer

__all__ = [
    "AtariConfig",
    "AtariEnv",
    "AtariEnvConfig",
    "AtariNetwork",
    "AtariNetworkConfig",
    "AtariRAGAdvisor",
    "AtariReflectionMemory",
    "AtariTrainer",
    "AtariTrainingConfig",
    "NovelFeaturesConfig",
]
