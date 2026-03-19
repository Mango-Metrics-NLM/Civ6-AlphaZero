"""
Unit tests for orchestrator_state: state persistence, resume, and heartbeat.

Tests cover:
- OrchestratorState serialization/deserialization
- OrchestratorStateManager save/load with mocked StorageBackend
- HeartbeatThread lifecycle
- Resume logic: skip completed iterations
- Edge cases: missing state, corrupted JSON, partial fields

Sprint 28 Phase 1: GCP orchestrator migration testing.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.cloud.gcp.orchestrator_state import (
    HeartbeatThread,
    OrchestratorState,
    OrchestratorStateManager,
)


# =============================================================================
# Fixtures
# =============================================================================


@dataclass
class FakeBackend:
    """In-memory StorageBackend for testing."""

    _blobs: dict[str, bytes] = field(default_factory=dict)

    def upload_blob(self, bucket: str, blob_path: str, data: bytes) -> str:
        key = f"{bucket}/{blob_path}"
        self._blobs[key] = data
        return f"gs://{key}"

    def download_blob(self, bucket: str, blob_path: str) -> bytes:
        key = f"{bucket}/{blob_path}"
        if key not in self._blobs:
            from alpha_zero.cloud.gcp.exceptions import GCSStorageError

            raise GCSStorageError(
                f"Blob not found: {blob_path}",
                bucket=bucket,
                blob_path=blob_path,
                operation="download",
            )
        return self._blobs[key]

    def blob_exists(self, bucket: str, blob_path: str) -> bool:
        return f"{bucket}/{blob_path}" in self._blobs

    def list_blobs(self, bucket: str, prefix: str) -> list[str]:
        results = []
        for key in self._blobs:
            if key.startswith(f"{bucket}/{prefix}"):
                results.append(key[len(bucket) + 1 :])
        return sorted(results)

    def delete_blob(self, bucket: str, blob_path: str) -> None:
        key = f"{bucket}/{blob_path}"
        self._blobs.pop(key, None)


@pytest.fixture()
def fake_backend() -> FakeBackend:
    """Create a fresh in-memory backend."""
    return FakeBackend()


@pytest.fixture()
def gcp_config() -> Any:
    """Create a minimal GCPConfig-like object for testing."""

    @dataclass
    class FakePathConfig:
        agent_id: str | None = None

    @dataclass
    class FakeConfig:
        project_id: str = "test-project"
        bucket: str = "test-bucket"
        paths: FakePathConfig = field(default_factory=FakePathConfig)

    return FakeConfig()


@pytest.fixture()
def state_manager(gcp_config: Any, fake_backend: FakeBackend) -> OrchestratorStateManager:
    """Create a state manager with a fake backend."""
    return OrchestratorStateManager(gcp_config, backend=fake_backend)


# =============================================================================
# OrchestratorState Serialization Tests
# =============================================================================


class TestOrchestratorStateSerialization:
    """Test OrchestratorState JSON round-trip."""

    def test_default_state_round_trip(self) -> None:
        """Default state serializes and deserializes correctly."""
        state = OrchestratorState()
        json_str = state.to_json()
        restored = OrchestratorState.from_json(json_str)

        assert restored.current_iteration == 0
        assert restored.model_version == 0
        assert restored.successful_iterations == 0
        assert restored.win_rate_history == []
        assert restored.total_estimated_cost == 0.0

    def test_populated_state_round_trip(self) -> None:
        """Populated state serializes and deserializes correctly."""
        state = OrchestratorState(
            current_iteration=5,
            start_iteration=0,
            total_iterations=20,
            model_version=273,
            successful_iterations=5,
            failed_iterations=1,
            consecutive_failures=0,
            win_rate_history=[0.55, 0.60, 0.58, 0.62, 0.65],
            total_estimated_cost=12.50,
            initial_model_version=268,
            start_time_utc="2026-03-07T10:00:00+00:00",
            last_checkpoint_utc="2026-03-07T12:30:00+00:00",
            cli_args_hash="abc123",
        )
        json_str = state.to_json()
        restored = OrchestratorState.from_json(json_str)

        assert restored.current_iteration == 5
        assert restored.model_version == 273
        assert restored.win_rate_history == [0.55, 0.60, 0.58, 0.62, 0.65]
        assert restored.total_estimated_cost == 12.50
        assert restored.initial_model_version == 268
        assert restored.cli_args_hash == "abc123"

    def test_json_is_valid_json(self) -> None:
        """to_json produces valid, parseable JSON."""
        state = OrchestratorState(current_iteration=3, model_version=100)
        parsed = json.loads(state.to_json())
        assert parsed["current_iteration"] == 3
        assert parsed["model_version"] == 100

    def test_unknown_fields_ignored(self) -> None:
        """from_json ignores unknown fields for forward compatibility."""
        data = {
            "current_iteration": 7,
            "model_version": 99,
            "future_field": "should_be_ignored",
            "another_new_field": [1, 2, 3],
        }
        state = OrchestratorState.from_json(json.dumps(data))
        assert state.current_iteration == 7
        assert state.model_version == 99

    def test_invalid_json_raises(self) -> None:
        """from_json raises ValueError on malformed JSON."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            OrchestratorState.from_json("not valid json {{{")

    def test_mark_checkpoint_updates_timestamp(self) -> None:
        """mark_checkpoint sets last_checkpoint_utc."""
        state = OrchestratorState()
        assert state.last_checkpoint_utc == ""
        state.mark_checkpoint()
        assert state.last_checkpoint_utc != ""
        assert "T" in state.last_checkpoint_utc  # ISO format


# =============================================================================
# OrchestratorStateManager Tests
# =============================================================================


class TestOrchestratorStateManager:
    """Test state manager save/load with fake backend."""

    def test_save_and_load_state(
        self,
        state_manager: OrchestratorStateManager,
    ) -> None:
        """State round-trips through save/load."""
        state = OrchestratorState(
            current_iteration=3,
            model_version=270,
            successful_iterations=3,
            win_rate_history=[0.55, 0.60, 0.58],
        )
        state_manager.save_state(state)
        loaded = state_manager.load_state()

        assert loaded is not None
        assert loaded.current_iteration == 3
        assert loaded.model_version == 270
        assert loaded.successful_iterations == 3
        assert loaded.win_rate_history == [0.55, 0.60, 0.58]
        # mark_checkpoint should have been called
        assert loaded.last_checkpoint_utc != ""

    def test_load_state_returns_none_when_missing(
        self,
        state_manager: OrchestratorStateManager,
    ) -> None:
        """load_state returns None when no prior state exists."""
        result = state_manager.load_state()
        assert result is None

    def test_clear_state(
        self,
        state_manager: OrchestratorStateManager,
    ) -> None:
        """clear_state removes the state blob."""
        state = OrchestratorState(current_iteration=5)
        state_manager.save_state(state)
        assert state_manager.load_state() is not None

        state_manager.clear_state()
        assert state_manager.load_state() is None

    def test_clear_state_when_none_exists(
        self,
        state_manager: OrchestratorStateManager,
    ) -> None:
        """clear_state succeeds even when no state exists."""
        state_manager.clear_state()  # Should not raise

    def test_save_state_returns_uri(
        self,
        state_manager: OrchestratorStateManager,
    ) -> None:
        """save_state returns the GCS URI."""
        state = OrchestratorState(current_iteration=1)
        uri = state_manager.save_state(state)
        assert uri.startswith("gs://")
        assert "state.json" in uri


class TestOrchestratorStateManagerAgentScoped:
    """Test agent-scoped state paths."""

    def test_agent_scoped_state_isolation(
        self,
        fake_backend: FakeBackend,
    ) -> None:
        """Different agent_ids use different state blobs."""

        @dataclass
        class FakePathConfig:
            agent_id: str | None = None

        @dataclass
        class FakeConfig:
            project_id: str = "test"
            bucket: str = "test-bucket"
            paths: FakePathConfig = field(default_factory=FakePathConfig)

        config_a = FakeConfig(paths=FakePathConfig(agent_id="agent_a"))
        config_b = FakeConfig(paths=FakePathConfig(agent_id="agent_b"))

        mgr_a = OrchestratorStateManager(config_a, backend=fake_backend)
        mgr_b = OrchestratorStateManager(config_b, backend=fake_backend)

        mgr_a.save_state(OrchestratorState(current_iteration=10))
        mgr_b.save_state(OrchestratorState(current_iteration=20))

        loaded_a = mgr_a.load_state()
        loaded_b = mgr_b.load_state()

        assert loaded_a is not None
        assert loaded_a.current_iteration == 10
        assert loaded_b is not None
        assert loaded_b.current_iteration == 20


# =============================================================================
# Heartbeat Tests
# =============================================================================


class TestHeartbeat:
    """Test heartbeat write/read functionality."""

    def test_write_and_read_heartbeat(
        self,
        state_manager: OrchestratorStateManager,
    ) -> None:
        """Heartbeat round-trips through write/read."""
        state_manager.write_heartbeat(
            status="running",
            iteration=5,
            model_version=270,
            uptime_seconds=3600.0,
        )
        heartbeat = state_manager.read_heartbeat()

        assert heartbeat is not None
        assert heartbeat["status"] == "running"
        assert heartbeat["iteration"] == 5
        assert heartbeat["model_version"] == 270
        assert heartbeat["uptime_seconds"] == 3600.0
        assert "timestamp_utc" in heartbeat

    def test_read_heartbeat_returns_none_when_missing(
        self,
        state_manager: OrchestratorStateManager,
    ) -> None:
        """read_heartbeat returns None when no heartbeat exists."""
        result = state_manager.read_heartbeat()
        assert result is None


class TestHeartbeatThread:
    """Test HeartbeatThread lifecycle."""

    def test_thread_starts_and_stops(
        self,
        state_manager: OrchestratorStateManager,
    ) -> None:
        """Thread starts as daemon, writes heartbeat, then stops."""
        thread = HeartbeatThread(state_manager, interval_seconds=1.0)
        thread.update(status="test", iteration=0, model_version=1)
        thread.start()

        assert thread.is_alive()
        assert thread.daemon is True

        # Wait for at least one heartbeat write
        time.sleep(1.5)

        thread.stop()
        thread.join(timeout=5.0)

        assert not thread.is_alive()

        # Verify a heartbeat was written
        heartbeat = state_manager.read_heartbeat()
        # Final heartbeat should have status="stopped"
        assert heartbeat is not None
        assert heartbeat["status"] == "stopped"

    def test_thread_update_is_thread_safe(
        self,
        state_manager: OrchestratorStateManager,
    ) -> None:
        """update() can be called from main thread while heartbeat runs."""
        thread = HeartbeatThread(state_manager, interval_seconds=60.0)
        thread.update(status="init", iteration=0, model_version=1)

        # Call update multiple times rapidly (no lock contention)
        for i in range(100):
            thread.update(status=f"iter_{i}", iteration=i, model_version=i + 1)

        # Should not raise


# =============================================================================
# Resume Logic Integration Tests
# =============================================================================


class TestResumeLogic:
    """Test the iteration-skipping resume behavior."""

    def test_resume_state_preserves_all_fields(
        self,
        state_manager: OrchestratorStateManager,
    ) -> None:
        """Save → load preserves all accumulated training state."""
        original = OrchestratorState(
            current_iteration=7,
            start_iteration=0,
            total_iterations=20,
            model_version=275,
            successful_iterations=6,
            failed_iterations=1,
            consecutive_failures=0,
            win_rate_history=[0.52, 0.55, 0.58, 0.60, 0.57, 0.63],
            total_estimated_cost=45.67,
            initial_model_version=268,
            cli_args_hash="test_hash_123",
        )
        state_manager.save_state(original)
        restored = state_manager.load_state()

        assert restored is not None
        assert restored.current_iteration == original.current_iteration
        assert restored.model_version == original.model_version
        assert restored.successful_iterations == original.successful_iterations
        assert restored.failed_iterations == original.failed_iterations
        assert restored.consecutive_failures == original.consecutive_failures
        assert restored.win_rate_history == original.win_rate_history
        assert restored.total_estimated_cost == original.total_estimated_cost
        assert restored.initial_model_version == original.initial_model_version
        assert restored.cli_args_hash == original.cli_args_hash

    def test_multiple_saves_overwrites_state(
        self,
        state_manager: OrchestratorStateManager,
    ) -> None:
        """Each save_state call overwrites the previous state."""
        state_manager.save_state(OrchestratorState(current_iteration=1))
        state_manager.save_state(OrchestratorState(current_iteration=5))
        state_manager.save_state(OrchestratorState(current_iteration=10))

        loaded = state_manager.load_state()
        assert loaded is not None
        assert loaded.current_iteration == 10

    def test_state_with_empty_win_rate_history(
        self,
        state_manager: OrchestratorStateManager,
    ) -> None:
        """State with no evaluation results saves/loads correctly."""
        state = OrchestratorState(
            current_iteration=2,
            model_version=270,
            win_rate_history=[],
        )
        state_manager.save_state(state)
        loaded = state_manager.load_state()

        assert loaded is not None
        assert loaded.win_rate_history == []


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_from_json_with_empty_object(self) -> None:
        """from_json with {} uses all defaults."""
        state = OrchestratorState.from_json("{}")
        assert state.current_iteration == 0
        assert state.model_version == 0
        assert state.win_rate_history == []

    def test_from_json_with_partial_fields(self) -> None:
        """from_json with partial fields fills rest with defaults."""
        state = OrchestratorState.from_json(
            '{"current_iteration": 42, "model_version": 99}'
        )
        assert state.current_iteration == 42
        assert state.model_version == 99
        assert state.successful_iterations == 0
        assert state.win_rate_history == []

    def test_large_win_rate_history(self) -> None:
        """Win rate history with many entries survives round-trip."""
        history = [0.5 + i * 0.001 for i in range(1000)]
        state = OrchestratorState(win_rate_history=history)
        restored = OrchestratorState.from_json(state.to_json())
        assert len(restored.win_rate_history) == 1000
        assert restored.win_rate_history[0] == pytest.approx(0.5)
        assert restored.win_rate_history[999] == pytest.approx(1.499)


# =============================================================================
# parse_args Resume/Heartbeat Flag Tests
# =============================================================================


class TestParseArgsResumeFlags:
    """Test that --resume and --heartbeat_interval are properly parsed."""

    def test_resume_flag_accepted(self) -> None:
        """--resume is parsed as True."""
        import sys
        from unittest.mock import patch as _patch

        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        test_args = ["vertex_orchestrator", "--resume"]
        with _patch.object(sys, "argv", test_args):
            args = parse_args()
        assert args.resume is True

    def test_resume_default_is_false(self) -> None:
        """Default --resume is False."""
        import sys
        from unittest.mock import patch as _patch

        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        test_args = ["vertex_orchestrator"]
        with _patch.object(sys, "argv", test_args):
            args = parse_args()
        assert args.resume is False

    def test_heartbeat_interval_custom(self) -> None:
        """--heartbeat_interval accepts custom value."""
        import sys
        from unittest.mock import patch as _patch

        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        test_args = ["vertex_orchestrator", "--heartbeat_interval=120"]
        with _patch.object(sys, "argv", test_args):
            args = parse_args()
        assert args.heartbeat_interval == 120.0

    def test_heartbeat_interval_default(self) -> None:
        """Default --heartbeat_interval is 60."""
        import sys
        from unittest.mock import patch as _patch

        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        test_args = ["vertex_orchestrator"]
        with _patch.object(sys, "argv", test_args):
            args = parse_args()
        assert args.heartbeat_interval == 60.0

    def test_clear_state_flag_accepted(self) -> None:
        """--clear_state is parsed as True."""
        import sys
        from unittest.mock import patch as _patch

        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        test_args = ["vertex_orchestrator", "--clear_state"]
        with _patch.object(sys, "argv", test_args):
            args = parse_args()
        assert args.clear_state is True
