"""End-to-end tests for LLM integration and full system.

These tests verify that all components work together correctly,
including LLM integration with the Civ6 AlphaZero system.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alpha_zero.core.civ6_network import MicroCivNetwork

# Import core components
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.utils.agent_tools import (
    CodeExplorer,
    DebugHelper,
    FileManager,
    TaskTracker,
)
from alpha_zero.utils.config import Civ6Config, ConfigManager, load_config

# Import new utilities
from alpha_zero.utils.test_runner import RunConfig, PytestRunner

# LLM imports (optional)
try:
    from alpha_zero.utils.llm_client import (  # noqa: F401
        LLMClient,
        LLMConfig,
        LLMMessage,
        LLMProvider,
        LLMResponse,
        create_llm_client,
    )

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class TestEnvironmentSetup:
    """Test that the environment is properly configured."""

    def test_env_file_exists(self):
        """Test that .env file exists."""
        project_root = Path(__file__).parent.parent.parent
        env_template = project_root / ".env.template"

        # At least the template should exist
        assert env_template.exists(), ".env.template should exist"

    def test_config_file_exists(self):
        """Test that config file exists."""
        project_root = Path(__file__).parent.parent.parent
        config_file = project_root / "configs" / "civ6_config.yaml"
        assert config_file.exists(), "civ6_config.yaml should exist"

    def test_config_loads(self):
        """Test that configuration loads successfully."""
        project_root = Path(__file__).parent.parent.parent
        config_file = project_root / "configs" / "civ6_config.yaml"

        config = load_config(config_file)
        assert isinstance(config, Civ6Config)
        assert config.environment.board_size == 16


class TestMicroCivIntegration:
    """Test MicroCiv environment integration."""

    def test_environment_creation(self):
        """Test creating MicroCiv environment."""
        env = MicroCivEnv(board_size=8)
        obs, info = env.reset()

        assert "spatial" in obs
        assert "vector" in obs
        assert "action_mask" in obs
        assert obs["spatial"].shape == (13, 8, 8)

    def test_environment_step(self):
        """Test stepping in MicroCiv environment."""
        env = MicroCivEnv(board_size=8, seed=42)
        obs, _ = env.reset()

        # Get valid actions
        valid_actions = obs["action_mask"]
        valid_indices = valid_actions.nonzero()[0]

        if len(valid_indices) > 0:
            action = valid_indices[0]
            next_obs, reward, terminated, truncated, info = env.step(action)

            assert "spatial" in next_obs
            assert isinstance(reward, (int, float))

    def test_network_forward_pass(self):
        """Test neural network forward pass."""
        import torch

        env = MicroCivEnv(board_size=8)
        obs, _ = env.reset()

        network = MicroCivNetwork(
            board_size=8,
            action_space_size=env.action_space.n,
        )

        # Prepare input
        spatial = torch.tensor(obs["spatial"]).unsqueeze(0).float()
        vector = torch.tensor(obs["vector"]).unsqueeze(0).float()

        # Forward pass
        with torch.no_grad():
            policy, value = network(spatial, vector)

        assert policy.shape == (1, env.action_space.n)
        # Value can be (1,) or (1, 1) depending on network implementation
        assert value.numel() == 1


class TestAgentToolsIntegration:
    """Test agent tools integration."""

    def test_code_explorer_finds_project_files(self):
        """Test that CodeExplorer finds project files."""
        project_root = Path(__file__).parent.parent.parent
        explorer = CodeExplorer(project_root)

        py_files = explorer.find_python_files()
        assert len(py_files) > 0
        assert any("microciv" in str(f) for f in py_files)

    def test_code_explorer_grep(self):
        """Test that CodeExplorer grep works."""
        project_root = Path(__file__).parent.parent.parent
        explorer = CodeExplorer(project_root)

        results = explorer.grep("class MicroCivEnv", "**/*.py")
        assert len(results) > 0

    def test_file_manager_reads_config(self):
        """Test that FileManager can read config files."""
        project_root = Path(__file__).parent.parent.parent
        manager = FileManager(project_root)

        content = manager.read_file("configs/civ6_config.yaml")
        assert "environment" in content
        assert "network" in content

    def test_task_tracker_workflow(self):
        """Test complete task tracker workflow."""
        tracker = TaskTracker()

        # Add tasks
        task1 = tracker.add("Setup environment")
        _task2 = tracker.add("Run tests")
        _task3 = tracker.add("Deploy")

        # Start first task
        tracker.start(task1.id)
        assert len(tracker.list_in_progress()) == 1

        # Complete first task
        tracker.complete(task1.id)
        assert len(tracker.list_completed()) == 1

        # Check summary
        summary = tracker.summary()
        assert "3 total" in summary

    def test_debug_helper_environment_check(self):
        """Test debug helper environment check."""
        env_info = DebugHelper.check_environment()

        assert "python_version" in env_info
        assert "platform" in env_info

    def test_debug_helper_dependency_validation(self):
        """Test debug helper validates dependencies."""
        deps = DebugHelper.validate_dependencies(
            [
                "torch",
                "numpy",
                "pytest",
            ]
        )

        assert deps["torch"] is True
        assert deps["numpy"] is True
        assert deps["pytest"] is True


class TestTestRunnerIntegration:
    """Test test runner integration."""

    def test_test_runner_initialization(self):
        """Test test runner initializes correctly."""
        project_root = Path(__file__).parent.parent.parent
        config = RunConfig(project_root=project_root)
        runner = PytestRunner(config)

        assert runner.config.project_root == project_root

    def test_build_pytest_args(self):
        """Test building pytest arguments."""
        config = RunConfig(
            parallel=False,
            verbose=True,
            coverage_enabled=True,
            coverage_source=["alpha_zero"],
            junit_xml=False,
        )
        runner = PytestRunner(config)
        args = runner._build_pytest_args(with_coverage=True)

        assert "-v" in args
        assert "--cov" in args


@pytest.mark.skipif(not LLM_AVAILABLE, reason="LLM packages not installed")
class TestLLMIntegration:
    """Test LLM integration with the system."""

    def test_llm_config_from_env(self):
        """Test LLM config loads from environment."""
        config = LLMConfig.from_env()

        # Config should load without error
        assert config.provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.AUTO]

    def test_llm_message_creation(self):
        """Test creating LLM messages."""
        msg = LLMMessage(
            role="user",
            content="Analyze this game state for optimal moves.",
        )

        assert msg.role == "user"
        assert "game state" in msg.content

    @pytest.mark.asyncio
    async def test_llm_client_with_mock(self):
        """Test LLM client with mocked response."""
        config = LLMConfig()
        config.openai_api_key = "test-key"  # Fake key for testing

        # Mock the OpenAI client
        with patch("alpha_zero.utils.llm_client.OPENAI_AVAILABLE", True):
            with patch("alpha_zero.utils.llm_client.AsyncOpenAI") as mock_openai:
                # Setup mock response
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = "Move warrior to (3, 4)"
                mock_response.choices[0].finish_reason = "stop"
                mock_response.usage = MagicMock()
                mock_response.usage.prompt_tokens = 50
                mock_response.usage.completion_tokens = 10

                mock_client = MagicMock()
                mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
                mock_openai.return_value = mock_client

                from alpha_zero.utils.llm_client import OpenAIClient

                client = OpenAIClient(config)
                response = await client.generate(
                    prompt="What's the best move?",
                    system_prompt="You are a Civ6 strategy expert.",
                )

                assert "warrior" in response.content.lower() or "move" in response.content.lower()

    def test_llm_response_properties(self):
        """Test LLM response properties."""
        response = LLMResponse(
            content="Build a warrior unit.",
            model="gpt-4",
            provider=LLMProvider.OPENAI,
            usage={"input_tokens": 100, "output_tokens": 50},
        )

        assert response.input_tokens == 100
        assert response.output_tokens == 50
        assert response.total_tokens == 150


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"),
    reason="No API keys configured for live test",
)
class TestLLMLiveIntegration:
    """Live LLM integration tests (require API keys)."""

    @pytest.mark.asyncio
    async def test_live_llm_generate(self):
        """Test live LLM generation."""
        client = create_llm_client()

        response = await client.generate(
            prompt="What is 2 + 2? Reply with just the number.",
            max_tokens=10,
        )

        assert "4" in response.content

    @pytest.mark.asyncio
    async def test_live_llm_game_analysis(self):
        """Test live LLM game state analysis."""
        client = create_llm_client()

        game_state = """
        Board: 8x8 grid
        Your units: Warrior at (2, 3), Settler at (4, 4)
        Enemy units: Warrior at (5, 3)
        Your cities: Capital at (4, 4)
        """

        response = await client.generate(
            prompt=f"Analyze this game state and suggest the best move:\n{game_state}",
            system_prompt="You are a Civ6 strategy expert. Be concise.",
            max_tokens=100,
        )

        assert len(response.content) > 0
        # Should mention some strategic element
        assert any(
            word in response.content.lower()
            for word in ["move", "attack", "warrior", "settler", "city", "defend"]
        )


class TestFullSystemIntegration:
    """Test full system integration."""

    def test_game_to_network_pipeline(self):
        """Test complete pipeline from game to network."""
        import torch

        # Create environment
        env = MicroCivEnv(board_size=8, seed=42)
        obs, _ = env.reset()

        # Create network
        network = MicroCivNetwork(
            board_size=8,
            action_space_size=env.action_space.n,
        )

        # Play a few steps
        total_reward = 0
        for _ in range(10):
            # Get network prediction
            spatial = torch.tensor(obs["spatial"]).unsqueeze(0).float()
            vector = torch.tensor(obs["vector"]).unsqueeze(0).float()

            with torch.no_grad():
                policy, value = network(spatial, vector)

            # Apply action mask
            action_mask = torch.tensor(obs["action_mask"])
            masked_policy = policy.squeeze() * action_mask
            masked_policy = masked_policy / (masked_policy.sum() + 1e-8)

            # Sample action
            valid_actions = action_mask.nonzero().squeeze(-1)
            if len(valid_actions) == 0:
                break

            action = valid_actions[torch.randint(len(valid_actions), (1,))].item()

            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        # Verify pipeline completed
        assert isinstance(total_reward, (int, float))

    def test_config_driven_setup(self):
        """Test setting up system from configuration."""
        project_root = Path(__file__).parent.parent.parent
        config_file = project_root / "configs" / "civ6_config.yaml"

        # Load configuration
        manager = ConfigManager(config_file)
        env_config = manager.get_environment_config()
        _network_config = manager.get_network_config()

        # Create environment from config
        env = MicroCivEnv(board_size=env_config.board_size)

        # Verify configuration applied
        assert env.board_size == env_config.board_size


class TestUtilsImports:
    """Test that all utils are properly importable."""

    def test_import_logging(self):
        """Test importing logging utilities."""
        from alpha_zero.utils import (
            GameLogger,
            get_logger,
        )

        assert GameLogger is not None
        assert get_logger is not None

    def test_import_config(self):
        """Test importing config utilities."""
        from alpha_zero.utils import (
            Civ6Config,
            ConfigManager,
        )

        assert Civ6Config is not None
        assert ConfigManager is not None

    def test_import_test_runner(self):
        """Test importing test runner utilities."""
        from alpha_zero.utils import (
            PytestRunner,
            RunStatus,
        )

        assert PytestRunner is not None
        assert RunStatus.PASSED.value == "passed"

    def test_import_agent_tools(self):
        """Test importing agent tools."""
        from alpha_zero.utils import (
            CodeExplorer,
            FileManager,
        )

        assert CodeExplorer is not None
        assert FileManager is not None

    @pytest.mark.skipif(not LLM_AVAILABLE, reason="LLM packages not installed")
    def test_import_llm_client(self):
        """Test importing LLM client."""
        from alpha_zero.utils import (
            LLMClient,
            create_llm_client,
        )

        assert LLMClient is not None
        assert create_llm_client is not None
