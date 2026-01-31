"""
Tests for configuration management.
"""

from agentic_framework.config import (
    AgenticConfig,
    ConfigLoader,
    load_config,
    merge_configs,
)


class TestAgenticConfig:
    """Tests for AgenticConfig dataclass."""

    def test_default_construction(self):
        """Test creating default config."""
        config = AgenticConfig()
        assert config.version == "1.0"
        assert config.objective.system_intent == ""
        assert config.constraints.hard == []

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "version": "2.0",
            "objective": {
                "system_intent": "Test system",
                "success_criteria": [{"criterion": "Tests pass"}],
            },
            "constraints": {
                "hard": [{"name": "python", "type": "language", "value": "3.10+"}],
            },
        }

        config = AgenticConfig.from_dict(data)

        assert config.version == "2.0"
        assert config.objective.system_intent == "Test system"
        assert len(config.objective.success_criteria) == 1
        assert len(config.constraints.hard) == 1

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = AgenticConfig()
        config.objective.system_intent = "Test"

        data = config.to_dict()

        assert isinstance(data, dict)
        assert data["objective"]["system_intent"] == "Test"


class TestConfigLoader:
    """Tests for ConfigLoader."""

    def test_load_yaml_simple(self, tmp_path):
        """Test loading a simple YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
version: "1.0"
objective:
  system_intent: "Test objective"
""")

        loader = ConfigLoader(config_dir=tmp_path)
        data = loader.load_yaml(config_file)

        assert data["version"] == "1.0"
        assert data["objective"]["system_intent"] == "Test objective"

    def test_env_var_substitution(self, tmp_path, monkeypatch):
        """Test environment variable substitution."""
        monkeypatch.setenv("TEST_VALUE", "substituted")

        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
test_key: "${TEST_VALUE}"
""")

        loader = ConfigLoader(config_dir=tmp_path)
        data = loader.load_yaml(config_file)

        assert data["test_key"] == "substituted"

    def test_env_var_default(self, tmp_path):
        """Test environment variable with default value."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
test_key: "${NONEXISTENT:-default_value}"
""")

        loader = ConfigLoader(config_dir=tmp_path)
        data = loader.load_yaml(config_file)

        assert data["test_key"] == "default_value"

    def test_load_with_project_config(self, tmp_path):
        """Test loading with project-specific config overlay."""
        # Create default config
        default_config = tmp_path / "default_config.yaml"
        default_config.write_text("""
version: "1.0"
objective:
  system_intent: "Default"
constraints:
  hard: []
  soft: []
  anti: []
permissions:
  scope:
    in_scope: ["**/*.py"]
    out_of_scope: []
  autonomy:
    autonomous: []
    confirm_first: []
    prohibited: []
  resource_budget:
    max_iterations: 5
    max_files_per_pass: 20
    exploration_timeout_seconds: 600
feedback:
  verification_commands: []
  error_handling:
    auto_fix_patterns: []
  success_verification: []
session:
  memory_file: "CLAUDE.md"
  sections: []
execution:
  initial_actions: []
  implementation_order: []
  completion_checklist: []
logging:
  level: INFO
  format: "%(message)s"
  handlers: []
""")

        # Create project config
        project_config = tmp_path / "project.yaml"
        project_config.write_text("""
objective:
  system_intent: "Project specific"
""")

        loader = ConfigLoader(config_dir=tmp_path)
        config = loader.load(project_config_path=project_config)

        assert config.objective.system_intent == "Project specific"

    def test_validate_missing_required(self, tmp_path):
        """Test validation catches missing required fields."""
        loader = ConfigLoader(config_dir=tmp_path)
        config = AgenticConfig()  # Empty config

        errors = loader.validate(config)

        assert len(errors) > 0
        assert any("system_intent" in e for e in errors)


class TestMergeConfigs:
    """Tests for configuration merging."""

    def test_merge_simple(self):
        """Test simple merge of two configs."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        result = merge_configs(base, override)

        assert result["a"] == 1
        assert result["b"] == 3
        assert result["c"] == 4

    def test_merge_nested(self):
        """Test deep merge of nested configs."""
        base = {"outer": {"a": 1, "b": 2}}
        override = {"outer": {"b": 3, "c": 4}}

        result = merge_configs(base, override)

        assert result["outer"]["a"] == 1
        assert result["outer"]["b"] == 3
        assert result["outer"]["c"] == 4

    def test_merge_list_replacement(self):
        """Test that lists are replaced, not merged."""
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}

        result = merge_configs(base, override)

        assert result["items"] == [4, 5]


class TestLoadConfig:
    """Tests for the convenience load_config function."""

    def test_load_config_default(self, tmp_path, monkeypatch):
        """Test loading default config."""
        # Create a minimal default config
        default_config = tmp_path / "default_config.yaml"
        default_config.write_text("""
version: "1.0"
objective:
  system_intent: ""
  success_criteria: []
  problem_description: {}
constraints:
  hard: []
  soft: []
  anti: []
permissions:
  scope:
    in_scope: []
    out_of_scope: []
  autonomy:
    autonomous: []
    confirm_first: []
    prohibited: []
  resource_budget:
    max_iterations: 5
    max_files_per_pass: 20
    exploration_timeout_seconds: 600
feedback:
  verification_commands: []
  error_handling:
    auto_fix_patterns: []
  success_verification: []
session:
  memory_file: "CLAUDE.md"
  sections: []
execution:
  initial_actions: []
  implementation_order: []
  completion_checklist: []
logging:
  level: INFO
  format: "%(message)s"
  handlers: []
""")

        config = load_config(config_dir=tmp_path)

        assert isinstance(config, AgenticConfig)

    def test_load_config_with_overrides(self, tmp_path):
        """Test loading config with runtime overrides."""
        default_config = tmp_path / "default_config.yaml"
        default_config.write_text("""
version: "1.0"
objective:
  system_intent: "Original"
""")

        config = load_config(
            config_dir=tmp_path,
            overrides={"objective": {"system_intent": "Overridden"}},
        )

        assert config.objective.system_intent == "Overridden"
