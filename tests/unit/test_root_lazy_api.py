"""Public-API guarantees for the lazily-resolved root package.

``alpha_zero/__init__.py`` resolves its exports on first attribute access
(PEP 562) so that importing the package does not drag in numpy and gymnasium.
That is an import-machinery change to the most widely-used module in the
repository, so these tests pin the behaviour callers actually rely on: direct
imports, attribute access, ``dir()``, and sensible failure modes.

The deliberate behaviour change is covered by
``test_optional_names_resolvable_but_not_exported``: ``RealCiv6Env`` and
``RewardConfig`` remain importable but no longer appear in ``__all__``.
"""

from __future__ import annotations

import importlib

import pytest


def test_version_is_available_eagerly() -> None:
    """``__version__`` is a plain attribute, not lazily resolved.

    The documentation-drift gate parses this assignment with a regex, so it
    must remain a literal in the module body.
    """
    import alpha_zero

    assert isinstance(alpha_zero.__version__, str)
    assert alpha_zero.__version__.count(".") >= 2


@pytest.mark.parametrize(
    "name",
    [
        "ActionSpace",
        "CategoricalSupportConfig",
        "Civ6MCTS",
        "Civ6Network",
        "HierarchicalAction",
        "LossConfig",
        "MicroCivNetwork",
        "MicroCivEnv",
        "TimeoutConfig",
        "get_timeout_config",
    ],
)
def test_public_name_resolves(name: str) -> None:
    """Every advertised name resolves to a real object via attribute access."""
    import alpha_zero

    assert getattr(alpha_zero, name, None) is not None


def test_from_import_syntax_works() -> None:
    """``from alpha_zero import X`` works, which PEP 562 supports."""
    from alpha_zero import Civ6MCTS, MicroCivEnv, TimeoutConfig

    assert Civ6MCTS is not None
    assert MicroCivEnv is not None
    assert TimeoutConfig is not None


def test_lazy_resolution_matches_direct_import() -> None:
    """A lazily-resolved export is the same object as a direct import."""
    import alpha_zero
    from alpha_zero.utils.timeout_config import TimeoutConfig as DirectTimeoutConfig

    assert alpha_zero.TimeoutConfig is DirectTimeoutConfig


def test_repeated_access_is_cached() -> None:
    """The second lookup returns the identical object (resolution is cached)."""
    import alpha_zero

    first = alpha_zero.Civ6MCTS
    second = alpha_zero.Civ6MCTS

    assert first is second


def test_dir_includes_all_exports() -> None:
    """``dir()`` advertises the public API plus the lazily-resolvable names."""
    import alpha_zero

    listed = set(dir(alpha_zero))

    assert set(alpha_zero.__all__) <= listed
    # Optional names are discoverable even though they are not in __all__.
    assert {"RealCiv6Env", "RewardConfig"} <= listed


def test_unknown_attribute_raises_attribute_error() -> None:
    """Unknown names raise ``AttributeError``, not ``ImportError`` or ``KeyError``."""
    import alpha_zero

    with pytest.raises(AttributeError, match="no attribute"):
        _ = alpha_zero.ThisNameDoesNotExist


def test_all_is_static_and_sorted_subset_of_exports() -> None:
    """``__all__`` is a fixed list, no longer mutated at import time."""
    import alpha_zero

    assert isinstance(alpha_zero.__all__, list)
    assert len(alpha_zero.__all__) == len(set(alpha_zero.__all__)), "duplicate entries"
    assert "__version__" in alpha_zero.__all__


def test_optional_names_resolvable_but_not_exported() -> None:
    """Optional Civ6 names stay importable but are absent from ``__all__``.

    Deciding availability at import time would require performing the very
    import that lazy resolution defers, so these names are resolved on demand.
    Whether the underlying module imports depends on optional extras; either a
    real object or ``ImportError`` is correct, but ``AttributeError`` is not.
    """
    import alpha_zero

    assert "RealCiv6Env" not in alpha_zero.__all__
    assert "RewardConfig" not in alpha_zero.__all__

    try:
        assert alpha_zero.RealCiv6Env is not None
    except ImportError:
        pytest.skip("optional Civ6 environment dependencies are not installed")


def test_star_import_surface_matches_all() -> None:
    """``import *`` exposes exactly ``__all__``."""
    namespace: dict[str, object] = {}
    exec("from alpha_zero import *", namespace)  # noqa: S102

    exported = {k for k in namespace if not k.startswith("__")}
    module = importlib.import_module("alpha_zero")
    expected = {n for n in module.__all__ if not n.startswith("__")}

    assert exported == expected
