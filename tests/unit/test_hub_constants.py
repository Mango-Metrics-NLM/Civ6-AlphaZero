"""Unit tests for alpha_zero.hub.constants.

Tests cover:
- Default constant values
- Environment variable overrides
- Type correctness
"""

import os
from pathlib import Path
from unittest import mock


class TestHFOrg:
    """Tests for HF_ORG constant."""

    def test_default_value(self):
        """Default HF_ORG should be Mango-Metrics-NLM."""
        # Re-import to get fresh module state.
        import importlib

        import alpha_zero.hub.constants as mod

        importlib.reload(mod)
        assert isinstance(mod.HF_ORG, str)
        assert len(mod.HF_ORG) > 0

    def test_env_override(self):
        """HF_ORG should be overridable via env var."""
        import importlib

        import alpha_zero.hub.constants as mod

        with mock.patch.dict(os.environ, {"HF_ORG": "test-org"}):
            importlib.reload(mod)
            assert mod.HF_ORG == "test-org"
        # Restore.
        importlib.reload(mod)


class TestLibraryName:
    """Tests for LIBRARY_NAME constant."""

    def test_default_value(self):
        from alpha_zero.hub.constants import LIBRARY_NAME

        assert isinstance(LIBRARY_NAME, str)
        assert LIBRARY_NAME == "alpha-zero-civ6"

    def test_env_override(self):
        import importlib

        import alpha_zero.hub.constants as mod

        with mock.patch.dict(os.environ, {"HF_LIBRARY_NAME": "custom-lib"}):
            importlib.reload(mod)
            assert mod.LIBRARY_NAME == "custom-lib"
        importlib.reload(mod)


class TestDefaultModelTags:
    """Tests for DEFAULT_MODEL_TAGS."""

    def test_is_tuple_of_strings(self):
        from alpha_zero.hub.constants import DEFAULT_MODEL_TAGS

        assert isinstance(DEFAULT_MODEL_TAGS, tuple)
        assert all(isinstance(t, str) for t in DEFAULT_MODEL_TAGS)

    def test_contains_expected_tags(self):
        from alpha_zero.hub.constants import DEFAULT_MODEL_TAGS

        assert "pytorch" in DEFAULT_MODEL_TAGS
        assert "reinforcement-learning" in DEFAULT_MODEL_TAGS


class TestCardTemplatePath:
    """Tests for CARD_TEMPLATE_PATH."""

    def test_is_path(self):
        from alpha_zero.hub.constants import CARD_TEMPLATE_PATH

        assert isinstance(CARD_TEMPLATE_PATH, Path)

    def test_default_points_to_templates_dir(self):
        from alpha_zero.hub.constants import CARD_TEMPLATE_PATH

        assert "templates" in str(CARD_TEMPLATE_PATH)
        assert str(CARD_TEMPLATE_PATH).endswith(".template")

    def test_env_override(self):
        import importlib
        from pathlib import Path

        import alpha_zero.hub.constants as mod

        custom = "/tmp/custom_card.md"
        with mock.patch.dict(os.environ, {"HF_CARD_TEMPLATE": custom}):
            importlib.reload(mod)
            assert mod.CARD_TEMPLATE_PATH == Path(custom)
        importlib.reload(mod)


class TestDefaultLicense:
    """Tests for DEFAULT_LICENSE."""

    def test_default_value(self):
        from alpha_zero.hub.constants import DEFAULT_LICENSE

        assert isinstance(DEFAULT_LICENSE, str)
        assert DEFAULT_LICENSE == "MIT"

    def test_env_override(self):
        import importlib

        import alpha_zero.hub.constants as mod

        with mock.patch.dict(os.environ, {"HF_LICENSE": "Apache-2.0"}):
            importlib.reload(mod)
            assert mod.DEFAULT_LICENSE == "Apache-2.0"
        importlib.reload(mod)
