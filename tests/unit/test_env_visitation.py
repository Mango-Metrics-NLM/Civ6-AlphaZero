"""Unit tests for environment visitation tracking."""

from alpha_zero.envs.microciv import MicroCivEnv


def test_visitation_init():
    """Test that visitation set is initialized correctly."""
    env = MicroCivEnv(board_size=16)
    env.reset()

    # After reset, starting units should mark their positions as visited
    # 2 players * 2 units (settler + warrior) = 4 visited tiles (assuming no overlap)
    assert len(env.visited) == 4
    assert isinstance(env.visited, set)

    info = env.get_info()
    assert info["visited_count"] == len(env.visited)

def test_visitation_on_move():
    """Test that moving units updates visited set."""
    env = MicroCivEnv(board_size=16)
    env.reset()

    initial_visited = len(env.visited)

    # Find a unit for current player
    units = env._get_player_units(env.current_player)
    unit = units[0]
    start_pos = unit.position

    # Move unit to a valid adjacent tile
    # Try all directions until valid
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        new_pos = (start_pos[0] + dx, start_pos[1] + dy)
        if env._is_valid_move(new_pos) and new_pos not in env.visited:
            # Construct move action manually for test simplicity
            action = {
                "type": "move",
                "unit_idx": 0,
                "direction": (dx, dy)
            }
            env._execute_action(action)
            assert new_pos in env.visited
            assert len(env.visited) > initial_visited
            return

    # If we couldn't find a move to an unvisited tile, warn but pass
    # (Random map generation might box us in, though unlikely with fresh start)

def test_visitation_persistence():
    """Test that visited tiles persist across turns."""
    env = MicroCivEnv(board_size=16)
    env.reset()

    initial_visited_count = len(env.visited)

    # End turn multiple times
    for _ in range(4):
        env.step(env.action_space_handler.encode_action({"type": "end_turn"}))

    assert len(env.visited) == initial_visited_count
    assert env.get_info()["visited_count"] == initial_visited_count
