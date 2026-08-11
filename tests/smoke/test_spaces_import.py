"""Smoke tests: verify Gradio Space apps import and construct without errors.

Uses ``unittest.mock`` to block the actual ``gradio`` import so these tests
run in the base environment (no Gradio install required).
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

_SPACES_ROOT = Path(__file__).parents[2] / "spaces"


def _make_fake_gradio() -> types.ModuleType:
    """Build a minimal stub of the ``gradio`` package."""
    gr = types.ModuleType("gradio")

    # Stub out the pieces each app uses.
    class _FakeComponent:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

        def click(self, *a: object, **kw: object) -> None:
            pass

    class _FakeBlocks:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

        def __enter__(self) -> _FakeBlocks:
            return self

        def __exit__(self, *a: object) -> None:
            pass

        def launch(self, **kw: object) -> None:
            pass

    class _FakeRow(_FakeBlocks):
        pass

    class _FakeColumn(_FakeBlocks):
        pass

    class _FakeTab(_FakeBlocks):
        pass

    class _FakeSoft:
        pass

    class _FakeThemes:
        Soft = _FakeSoft

    gr.Blocks = _FakeBlocks  # type: ignore[attr-defined]
    gr.Row = _FakeRow  # type: ignore[attr-defined]
    gr.Column = _FakeColumn  # type: ignore[attr-defined]
    gr.Tab = _FakeTab  # type: ignore[attr-defined]
    gr.Markdown = _FakeComponent  # type: ignore[attr-defined]
    gr.Textbox = _FakeComponent  # type: ignore[attr-defined]
    gr.Number = _FakeComponent  # type: ignore[attr-defined]
    gr.Button = _FakeComponent  # type: ignore[attr-defined]
    gr.Dropdown = _FakeComponent  # type: ignore[attr-defined]
    gr.themes = _FakeThemes  # type: ignore[attr-defined]
    return gr


_SPACE_APPS = [
    ("microciv-explorer", "app"),
    ("elo-dashboard", "app"),
]


@pytest.fixture(autouse=True)
def _fake_gradio():
    """Inject a fake ``gradio`` module for the duration of each test."""
    fake = _make_fake_gradio()
    with mock.patch.dict(sys.modules, {"gradio": fake}):
        yield


@pytest.mark.parametrize("space_name,module_name", _SPACE_APPS)
def test_space_imports(space_name: str, module_name: str) -> None:
    """Space app must be importable without errors."""
    space_dir = _SPACES_ROOT / space_name
    assert space_dir.exists(), f"Space directory not found: {space_dir}"

    # Add space dir to sys.path temporarily.
    sys.path.insert(0, str(space_dir))
    try:
        # Force re-import to avoid cached failures from previous test.
        if module_name in sys.modules:
            del sys.modules[module_name]
        mod = importlib.import_module(module_name)
        assert hasattr(mod, "create_app"), (
            f"Space '{space_name}' missing create_app() function"
        )
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize("space_name,module_name", _SPACE_APPS)
def test_create_app_returns_blocks(space_name: str, module_name: str) -> None:
    """create_app() must return without raising."""
    space_dir = _SPACES_ROOT / space_name
    sys.path.insert(0, str(space_dir))
    try:
        if module_name in sys.modules:
            del sys.modules[module_name]
        mod = importlib.import_module(module_name)
        # Should not raise.
        app = mod.create_app()
        assert app is not None
    finally:
        sys.path.pop(0)
