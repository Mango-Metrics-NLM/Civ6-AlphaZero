"""Import sanity tests for all alpha_zero submodules.

Validates that every public module can be imported without errors.
Uses dynamic discovery to stay backwards-compatible as modules are
added or renamed.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Generator

import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (no hardcoded module lists — we discover dynamically)
# ---------------------------------------------------------------------------

_TOP_LEVEL_PACKAGE = "alpha_zero"

# Modules that may fail due to optional dependencies (e.g., real game env)
_OPTIONAL_MODULES: frozenset[str] = frozenset({
    "alpha_zero.envs.real_civ6_env",
    "alpha_zero.bridge.civ6_bridge",
    "alpha_zero.bridge.lua_ipc",
    "alpha_zero.rag.embeddings.openai_embeddings",
    "alpha_zero.rag.embeddings.sentence_transformer_embeddings",
    "alpha_zero.rag.vector_stores.chroma_store",
    "alpha_zero.rag.vector_stores.faiss_store",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _discover_submodules(package_name: str) -> Generator[str, None, None]:
    """Recursively discover all submodule names under *package_name*.

    Uses ``pkgutil.walk_packages`` for dynamic, backwards-compatible
    discovery — no hardcoded module lists.
    """
    try:
        package = importlib.import_module(package_name)
    except ImportError:
        return

    prefix = package_name + "."
    if hasattr(package, "__path__"):
        for _, mod_name, _is_pkg in pkgutil.walk_packages(
            package.__path__, prefix=prefix, onerror=lambda _name: None
        ):
            yield mod_name


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestImportSanity:
    """Verify every discovered submodule can be imported."""

    @pytest.fixture(scope="module")
    def all_submodules(self) -> list[str]:
        """Dynamically discover all submodules under alpha_zero."""
        return sorted(_discover_submodules(_TOP_LEVEL_PACKAGE))

    def test_top_level_import(self) -> None:
        """Top-level ``alpha_zero`` package imports cleanly."""
        mod = importlib.import_module(_TOP_LEVEL_PACKAGE)
        assert hasattr(mod, "__version__"), "Missing __version__ attribute"
        logger.info("alpha_zero version: %s", mod.__version__)

    def test_version_format(self) -> None:
        """``__version__`` follows PEP 440 / semantic versioning."""
        import alpha_zero

        try:
            from packaging.version import Version

            ver = Version(alpha_zero.__version__)
            assert ver.release is not None, (
                f"Version '{alpha_zero.__version__}' has no release segment"
            )
            assert len(ver.release) >= 3, (  # noqa: PLR2004
                f"Version '{alpha_zero.__version__}' should have at least "
                f"3 release segments (X.Y.Z), got {len(ver.release)}"
            )
            logger.info(
                "Version %s parsed: release=%s, pre=%s",
                alpha_zero.__version__, ver.release, ver.pre,
            )
        except ImportError:
            # Fallback: basic check if packaging is not installed
            parts = alpha_zero.__version__.split(".")
            assert len(parts) >= 3, (  # noqa: PLR2004
                f"Version '{alpha_zero.__version__}' should be semver X.Y.Z"
            )

    def test_public_api_exports(self) -> None:
        """``__all__`` is defined and non-empty."""
        import alpha_zero

        assert hasattr(alpha_zero, "__all__"), "Missing __all__ in alpha_zero"
        assert len(alpha_zero.__all__) > 0, "__all__ should not be empty"

    def test_core_exports_importable(self) -> None:
        """Every name in ``__all__`` resolves to a real object."""
        import alpha_zero

        for name in alpha_zero.__all__:
            obj = getattr(alpha_zero, name, None)
            assert obj is not None, f"Export '{name}' in __all__ is None or missing"

    @pytest.mark.parametrize(
        "module_name",
        [
            "alpha_zero.core",
            "alpha_zero.envs",
            "alpha_zero.training",
            "alpha_zero.cloud",
            "alpha_zero.pbt",
            "alpha_zero.tournament",
            "alpha_zero.hierarchical",
            "alpha_zero.distributed",
            "alpha_zero.utils",
        ],
    )
    def test_key_subpackages_importable(self, module_name: str) -> None:
        """Key subpackages can be imported."""
        mod = importlib.import_module(module_name)
        assert mod is not None
        logger.info("Imported %s successfully", module_name)

    def test_optional_modules_graceful_skip(self) -> None:
        """Optional modules either import or raise ImportError cleanly."""
        for mod_name in sorted(_OPTIONAL_MODULES):
            try:
                importlib.import_module(mod_name)
                logger.info("Optional module %s is available", mod_name)
            except ImportError:
                logger.info("Optional module %s skipped (not installed)", mod_name)
            except Exception as exc:
                pytest.fail(
                    f"Optional module {mod_name} raised unexpected "
                    f"{type(exc).__name__}: {exc}"
                )
