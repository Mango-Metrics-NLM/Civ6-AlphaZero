"""Cloud-native components for Vertex AI deployment.

This module provides containerized components for running AlphaZero
training on Google Cloud Platform:

- SelfPlayWorker: Generates game trajectories using self-play
- Trainer: Trains neural network on collected trajectories
- Evaluator: Compares candidate models against champions

Each component is designed to run in a container on Vertex AI
with GCS-backed storage for persistence.

Usage:
    # Run self-play worker
    python -m alpha_zero.cloud.selfplay_worker --num-games=100

    # Run trainer
    python -m alpha_zero.cloud.trainer --training-steps=1000

    # Run evaluator
    python -m alpha_zero.cloud.evaluator --num-games=400
"""

from alpha_zero.cloud.config import CloudConfig, MachineProfile, get_machine_profiles
from alpha_zero.cloud.preemption import PreemptionHandler, register_preemption_handler

__all__ = [
    "CloudConfig",
    "MachineProfile",
    "get_machine_profiles",
    "PreemptionHandler",
    "register_preemption_handler",
]
