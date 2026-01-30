"""Test fixtures for Civ6 AlphaZero tests."""

from pathlib import Path

# Fixtures directory path
FIXTURES_DIR = Path(__file__).parent
GAME_STATES_DIR = FIXTURES_DIR / "game_states"


def get_fixture_path(name: str) -> Path:
    """Get path to a fixture file.

    Args:
        name: Fixture filename (e.g., "turn_1_start.json")

    Returns:
        Path to the fixture file.
    """
    return GAME_STATES_DIR / name


def load_fixture(name: str) -> dict:
    """Load a JSON fixture file.

    Args:
        name: Fixture filename (e.g., "turn_1_start.json")

    Returns:
        Parsed JSON data.
    """
    import json

    path = get_fixture_path(name)
    with open(path) as f:
        return json.load(f)
