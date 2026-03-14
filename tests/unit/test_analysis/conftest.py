from __future__ import annotations

import pytest

from alpha_zero.analysis.data_model import (
    IterationData,
    ModelVersionData,
    RunData,
    RunIdentifier,
)


@pytest.fixture
def sample_run_data() -> RunData:
    """Create a realistic RunData for testing."""
    iterations = []
    for i in range(10):
        iterations.append(
            IterationData(
                iteration=i,
                policy_loss=1.0 - i * 0.05,
                value_loss=0.8 - i * 0.03,
                total_loss=1.8 - i * 0.08,
                eval_win_rate=0.4 + i * 0.02,
                model_version=100 + i,
                promoted=(i == 5 or i == 8),
                selfplay_cost=0.5,
                training_cost=0.3,
                eval_cost=0.1,
                cumulative_cost=0.9 * (i + 1),
                games_completed=200,
                trajectory_count=200,
                value_correlation=0.5 + i * 0.04,
                duration_minutes=7.5,
            )
        )
    models = [
        ModelVersionData(version=100 + i, iteration=i, elo=1500 + i * 3)
        for i in range(10)
    ]
    return RunData(
        identifier=RunIdentifier(
            wandb_run_id="test-run-123", wandb_group="test-group"
        ),
        iterations=iterations,
        models=models,
        config={"mcts_type": "gumbel", "learning_rate": 0.001},
        summary={"best_win_rate": 0.58},
        start_model_version=100,
        end_model_version=109,
    )
