"""PBT dashboard telemetry against the MLflow-backed tracker.

Ports the live behaviors of the retired ``test_pbt_dashboarding.py`` (W&B era)
onto ``WandbCompatAdapter``: population-stats payloads, per-agent ELO metrics,
and mutation-event tables emitted by ``PBTOrchestrator``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.cloud.gcp.config import GCPConfig
from alpha_zero.cloud.gcp.pbt_orchestrator import PBTOrchestrator
from alpha_zero.monitoring.mlflow_logger import WandbCompatAdapter
from alpha_zero.pbt.config import AgentState, PBTConfig

_ORCH = "alpha_zero.cloud.gcp.pbt_orchestrator"


def _mock_gcp_config() -> MagicMock:
    cfg = MagicMock(spec=GCPConfig)
    cfg.project_id = "test-project"
    cfg.region = "us-central1"
    cfg.bucket = "test-bucket"
    cfg.paths = MagicMock()
    cfg.paths.logs_prefix = "logs"
    cfg.service_account = "sa@example.com"
    return cfg


def _agents() -> list[AgentState]:
    return [
        AgentState(
            agent_id="agent_0",
            hyperparameters={"learning_rate": 0.001},
            elo_rating=1500.0,
            fitness_score=0.6,
        ),
        AgentState(
            agent_id="agent_1",
            hyperparameters={"learning_rate": 0.002},
            elo_rating=1420.0,
            fitness_score=0.4,
        ),
    ]


@pytest.fixture()
def orchestrator():
    with (
        patch(f"{_ORCH}.VertexAIClient"),
        patch(f"{_ORCH}.VertexTrainingJob"),
        patch(f"{_ORCH}.VertexSelfPlayJob"),
        patch(f"{_ORCH}.GCSBackend"),
        patch(f"{_ORCH}.GCSModelStorage"),
    ):
        tracker = MagicMock(spec=WandbCompatAdapter)
        orch = PBTOrchestrator(
            PBTConfig(population_size=4, num_generations=2, evaluation_games=10),
            _mock_gcp_config(),
            wandb_logger=tracker,
        )
        orch.manager.agents = {a.agent_id: a for a in _agents()}
        yield orch, tracker


def test_population_dashboard_logs_stats_and_per_agent_elo(orchestrator):
    orch, tracker = orchestrator
    orch._log_population_dashboard(gen=3, report=MagicMock(), elapsed=12.5)

    kwargs = tracker.log_population_stats.call_args.kwargs
    assert kwargs["generation"] == 3
    assert kwargs["population_size"] == 2
    assert kwargs["max_elo"] == 1500.0
    assert kwargs["min_elo"] == 1420.0
    assert kwargs["mean_elo"] == pytest.approx(1460.0)
    assert kwargs["max_fitness"] == 0.6
    assert kwargs["min_fitness"] == 0.4
    assert kwargs["elapsed_seconds"] == 12.5

    per_agent, per_agent_kwargs = tracker.log.call_args
    assert per_agent[0] == {
        "pbt/agent/agent_0/elo": 1500.0,
        "pbt/agent/agent_1/elo": 1420.0,
    }
    assert per_agent_kwargs["step"] == 3


def test_population_dashboard_noop_without_tracker(orchestrator):
    orch, tracker = orchestrator
    orch.wandb_logger = None
    orch._log_population_dashboard(gen=1, report=MagicMock(), elapsed=1.0)
    tracker.log_population_stats.assert_not_called()


def test_population_dashboard_noop_with_empty_population(orchestrator):
    orch, tracker = orchestrator
    orch.manager.agents = {}
    orch._log_population_dashboard(gen=1, report=MagicMock(), elapsed=1.0)
    tracker.log_population_stats.assert_not_called()


def test_mutation_events_log_count_and_table(orchestrator):
    orch, tracker = orchestrator
    changes = [
        ("agent_1", "agent_0", {"learning_rate": 0.003, "c_puct": 1.5}),
        ("agent_3", "agent_2", {"learning_rate": 0.004}),
    ]
    orch._log_mutation_events(gen=5, changes=changes)

    tracker.log.assert_called_once_with({"pbt/mutations_count": 2}, step=5)
    name, columns, rows = tracker.log_table.call_args.args
    assert name == "pbt/mutations_gen_5"
    assert columns == ["replaced_id", "source_id", "learning_rate", "c_puct"]
    assert rows == [
        ["agent_1", "agent_0", 0.003, 1.5],
        ["agent_3", "agent_2", 0.004, ""],  # missing c_puct -> empty cell
    ]


def test_mutation_events_no_table_when_no_changes(orchestrator):
    orch, tracker = orchestrator
    orch._log_mutation_events(gen=2, changes=[])
    tracker.log.assert_called_once_with({"pbt/mutations_count": 0}, step=2)
    tracker.log_table.assert_not_called()


def test_mutation_events_noop_without_tracker(orchestrator):
    orch, tracker = orchestrator
    orch.wandb_logger = None
    orch._log_mutation_events(gen=2, changes=[("a", "b", {})])
    tracker.log.assert_not_called()
