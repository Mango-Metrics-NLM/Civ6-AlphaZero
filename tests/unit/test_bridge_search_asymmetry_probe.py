"""GM-B8b three-arm asymmetry probe: does losing-only-endings make the search avoid contact?

Zero games, untrained networks. Three arms on the same roots, seeds and networks:

* (a) merged rule: the acting player cannot win by elimination but can lose (no patch);
* (b) raw simulator rule: ``BridgeMCTSEnv.step`` replaced by the simulator's own;
* (c) symmetric rule: the root player's loss of every simulator piece is also taken back.

27 roots (R1-R4 x 3 positions x with/without an own city, plus 3 R5 controls) x decision
seeds 1-10 x network-initialisation seeds 0-2 = 810 searches per arm, 2,430 in total, searched
with ``Civ6MctsPolicy(max_turns=30)`` and 24 simulations over at most 8 sampled actions. The
roots sit at live turn 3 and the simulator's limit is 30, so the turn limit is never reached
inside a 24-simulation tree: every terminal in the tree is an elimination.

The tests ASSERT that the arms are what they claim, that the probe can see an effect (positive
control), that two controls where the arms must agree do, and determinism. They do NOT assert
the verdict; it is printed and written to JSON (``run_probe`` returns the same dict).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest
import torch

from alpha_zero.bridge.mcts_policy import BridgeMCTSEnv, Civ6MctsPolicy
from alpha_zero.core.civ6_mcts import GumbelMCTSConfig
from alpha_zero.core.civ6_network import MicroCivNetwork
from tests.fixtures.bridge_asymmetry_analysis import analyze, run_probe
from tests.fixtures.bridge_asymmetry_probe import (
    ARMS,
    INIT_SEEDS,
    POSITIONS,
    Root,
    all_roots,
    arm_patched,
    build_state,
    measure_chunk,
)
from tests.fixtures.bridge_asymmetry_stats import (
    VERDICTS,
    classify,
    cluster_bootstrap,
    hex_distance,
    hex_neighbours,
    sign_test_p,
)
from tests.fixtures.bridge_search_states import make_state, unit
from tests.fixtures.bridge_search_translation import convert

__all__ = ["run_probe"]

END_TURN = 0

# ---------------------------------------------------------------------------------------
# Pure functions (fast)
# ---------------------------------------------------------------------------------------


class TestHexDistance:
    def test_known_distances_on_odd_rows(self) -> None:
        assert hex_distance((5, 5), (5, 5)) == 0
        assert hex_distance((5, 5), (7, 5)) == 2
        assert hex_distance((5, 5), (5, 7)) == 2
        assert hex_distance((5, 5), (8, 5)) == 3
        # Odd row 5 sits half a tile right: (6,6) is a neighbour of (5,5), (5,6) is not.
        assert hex_distance((5, 5), (6, 6)) == 1
        assert hex_distance((5, 5), (5, 6)) == 1
        assert hex_distance((6, 4), (5, 5)) == 1
        assert hex_distance((5, 4), (5, 5)) == 1
        assert hex_distance((4, 4), (5, 5)) == 2

    @pytest.mark.parametrize("pos", [(5, 5), (8, 3), (3, 9), (4, 4), (2, 8)])
    def test_the_six_neighbours_are_at_one_and_the_ring_at_two(self, pos: tuple[int, int]) -> None:
        near = hex_neighbours(*pos)
        assert len(set(near)) == 6
        assert all(hex_distance(pos, n) == 1 for n in near)
        for n in near:
            assert hex_distance(n, pos) == 1  # symmetric

    def test_every_offered_move_in_the_fixtures_is_a_neighbour(self) -> None:
        for root in all_roots():
            state = build_state(root)
            for action in state.legal_actions:
                if action.action_type in ("move", "attack"):
                    for target in action.targets:
                        assert hex_distance(root.pos, target) == 1, (root.key, target)


class TestSignTest:
    def test_known_values(self) -> None:
        assert sign_test_p(0, 0) == 1.0
        assert sign_test_p(5, 5) == 1.0
        assert sign_test_p(10, 0) == pytest.approx(2 / 1024)
        assert sign_test_p(0, 10) == pytest.approx(2 / 1024)
        assert sign_test_p(9, 1) == pytest.approx(22 / 1024)
        assert sign_test_p(3, 1) == pytest.approx(10 / 16)

    def test_symmetric_and_bounded(self) -> None:
        for a, b in [(7, 2), (1, 12), (20, 20)]:
            assert sign_test_p(a, b) == sign_test_p(b, a)
            assert 0.0 < sign_test_p(a, b) <= 1.0


class TestClusterBootstrap:
    CLUSTERS = {("r", 0): [1.0, 1.0, 0.0], ("r", 1): [0.0, 0.0, 0.0], ("r", 2): [1.0, 0.0, 0.0]}

    def test_seeded_and_deterministic(self) -> None:
        one = cluster_bootstrap(self.CLUSTERS, seed=3)
        assert one == cluster_bootstrap(self.CLUSTERS, seed=3)
        many = {k: [float(k % 3 == 0), float(k % 5 == 0)] for k in range(12)}
        assert cluster_bootstrap(many, seed=3) == cluster_bootstrap(many, seed=3)
        assert cluster_bootstrap(many, seed=3) != cluster_bootstrap(many, seed=4)

    def test_mean_and_an_interval_that_contains_it(self) -> None:
        mean, lo, hi = cluster_bootstrap(self.CLUSTERS, seed=3)
        assert mean == pytest.approx(3 / 9)
        assert lo <= mean <= hi and lo < hi

    def test_a_constant_has_a_degenerate_interval(self) -> None:
        mean, lo, hi = cluster_bootstrap({k: [0.0, 0.0] for k in range(5)}, seed=1)
        assert (mean, lo, hi) == (0.0, 0.0, 0.0)

    def test_resamples_whole_clusters(self) -> None:
        # Two clusters, all-ones and all-zeros: every resample mean is 0, 0.5 or 1.
        _, lo, hi = cluster_bootstrap({"x": [1.0, 1.0], "y": [0.0, 0.0]}, seed=1, n_boot=500)
        assert (lo, hi) == (0.0, 1.0)


class TestClassify:
    BASE: dict[str, Any] = {
        "diff": 0.2,
        "ci": (0.1, 0.3),
        "p": 0.001,
        "per_init_diffs": [0.1, 0.2, 0.3],
        "own_elimination_in_a": 50,
    }

    def test_each_of_the_four_verdicts(self) -> None:
        assert classify(**self.BASE) == "avoids contact"
        assert classify(**{**self.BASE, "diff": 0.01, "ci": (-0.05, 0.06), "p": 0.8}) == (
            "no measurable difference"
        )
        assert classify(**{**self.BASE, "diff": 0.05, "ci": (-0.2, 0.3), "p": 0.4}) == (
            "inconclusive"
        )
        assert classify(**{**self.BASE, "own_elimination_in_a": 0}) == "vacuous"

    def test_each_condition_of_avoidance_is_needed(self) -> None:
        for change in (
            {"diff": 0.09},
            {"p": 0.06},
            {"per_init_diffs": [0.1, -0.01, 0.3]},
        ):
            assert classify(**{**self.BASE, **change}) != "avoids contact", change

    def test_vacuous_is_never_reported_as_no_difference(self) -> None:
        flat = {"diff": 0.0, "ci": (0.0, 0.0), "p": 1.0, "per_init_diffs": [0.0, 0.0, 0.0]}
        assert classify(**{**self.BASE, **flat, "own_elimination_in_a": 0}) == "vacuous"
        assert classify(**{**self.BASE, **flat}) == "no measurable difference"


# ---------------------------------------------------------------------------------------
# The arms themselves (before anything is measured)
# ---------------------------------------------------------------------------------------


def _attack(env: BridgeMCTSEnv, direction: tuple[int, int]) -> int:
    handler = env.env.action_space_handler
    return int(handler.encode_action({"type": "attack", "unit_idx": 0, "direction": direction}))


def _kill_last_enemy(arm: str) -> tuple[bool, Any]:
    with arm_patched(arm):
        env = BridgeMCTSEnv(convert(build_state(Root("R1", (5, 5), False)), 16), max_turns=30)
        clone = env.clone()  # the arm must reach clones
        _, _, terminated, _, info = clone.step(_attack(clone, (1, 0)))
        assert [u.player for u in clone.env.units] == [0]  # the enemy really is gone
        return terminated, info["winner"]


def _enemy_kills_last_piece(arm: str) -> tuple[bool, Any]:
    state = make_state(
        size=16,
        units=[unit(2, "UNIT_WARRIOR", 0, 5, 5, health=1), unit(3, "UNIT_WARRIOR", 1, 6, 5)],
    )
    with arm_patched(arm):
        env = BridgeMCTSEnv(convert(state, 16), max_turns=30)
        child = env.clone()
        child.step(END_TURN)
        grandchild = child.clone()
        assert grandchild.env.current_player == 1
        _, _, terminated, _, info = grandchild.step(_attack(grandchild, (-1, 0)))
        return terminated, info["winner"]


class TestTheArmsAreWhatTheyClaim:
    @pytest.mark.parametrize(
        ("arm", "expected"),
        [("a", (False, None)), ("b", (True, 0)), ("c", (False, None))],
    )
    def test_root_kills_the_last_visible_enemy(self, arm: str, expected: tuple[bool, Any]) -> None:
        assert _kill_last_enemy(arm) == expected

    @pytest.mark.parametrize(
        ("arm", "expected"),
        [("a", (True, 1)), ("b", (True, 1)), ("c", (False, None))],
    )
    def test_enemy_kills_the_roots_last_piece(self, arm: str, expected: tuple[bool, Any]) -> None:
        assert _enemy_kills_last_piece(arm) == expected

    @pytest.mark.parametrize("arm", ARMS)
    def test_the_in_tree_counters_fire_on_hand_stepped_lines(self, arm: str) -> None:
        tally = {"enemy_eliminated": 0, "own_eliminated": 0}
        with arm_patched(arm, tally):
            env = BridgeMCTSEnv(convert(build_state(Root("R1", (5, 5), False)), 16), max_turns=30)
            env.clone().step(_attack(env, (1, 0)))
        assert tally == {"enemy_eliminated": 1, "own_eliminated": 0}
        state = make_state(
            size=16,
            units=[unit(2, "UNIT_WARRIOR", 0, 5, 5, health=1), unit(3, "UNIT_WARRIOR", 1, 6, 5)],
        )
        with arm_patched(arm, tally):
            doomed = BridgeMCTSEnv(convert(state, 16), max_turns=30)
            doomed.step(END_TURN)
            doomed.clone().step(_attack(doomed, (-1, 0)))
        assert tally == {"enemy_eliminated": 1, "own_eliminated": 1}

    def test_patches_are_undone_afterwards(self) -> None:
        step, init = BridgeMCTSEnv.step, BridgeMCTSEnv.__init__
        for arm in ARMS:
            with arm_patched(arm, {"enemy_eliminated": 0, "own_eliminated": 0}):
                pass
        assert BridgeMCTSEnv.step is step and BridgeMCTSEnv.__init__ is init

    def test_roots_are_27_with_the_designed_shape(self) -> None:
        roots = all_roots()
        assert len(roots) == 27 and len({r.key for r in roots}) == 27
        assert sorted({r.pos for r in roots}) == sorted(POSITIONS)
        assert sum(r.family == "R5" for r in roots) == 3


# ---------------------------------------------------------------------------------------
# The measurement (slow): 9 chunks of 270 searches, then the comparison
# ---------------------------------------------------------------------------------------

_ROWS: dict[tuple[str, int], list[dict[str, Any]]] = {}
_WALL: dict[tuple[str, int], float] = {}


def _chunk(arm: str, init: int) -> list[dict[str, Any]]:
    if (arm, init) not in _ROWS:
        start = time.perf_counter()
        _ROWS[(arm, init)] = measure_chunk(arm, init)
        _WALL[(arm, init)] = time.perf_counter() - start
    return _ROWS[(arm, init)]


@pytest.mark.slow
@pytest.mark.parametrize("init", INIT_SEEDS)
@pytest.mark.parametrize("arm", ARMS)
def test_measure_chunk(arm: str, init: int) -> None:
    rows = _chunk(arm, init)
    assert len(rows) == 27 * 10
    assert all(r["arm"] == arm and r["init"] == init for r in rows)


@pytest.mark.slow
def test_determinism_same_root_seeds_and_arm_twice() -> None:
    roots = [Root("R1", (5, 5), False), Root("R4", (8, 3), False), Root("R3", (3, 9), True)]
    for arm in ARMS:
        one = measure_chunk(arm, 1, roots, [2, 7])
        two = measure_chunk(arm, 1, roots, [2, 7])
        assert [(r["command"], r["record"]) for r in one] == [
            (r["command"], r["record"]) for r in two
        ], arm


@pytest.mark.slow
@pytest.mark.timeout(1500)
def test_the_probe_verdict(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    start = time.perf_counter()
    rows = [r for arm in ARMS for init in INIT_SEEDS for r in _chunk(arm, init)]
    measured = sum(_WALL.values())
    result = analyze(rows, measured + (time.perf_counter() - start))
    out = tmp_path / "asymmetry_probe.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    with capsys.disabled():
        print("\n[asymmetry probe] " + json.dumps(result, indent=2, sort_keys=True, default=str))

    assert len(rows) == 2430
    # Probe sensitivity (positive control): the raw rule must look better and attack more.
    pos = result["positive_control"]
    assert pos["root_value_b_minus_a"] > 0.2, pos
    assert pos["attacks_b"] > pos["attacks_a"], pos
    # Negative controls: the arms may differ only through the rule.
    neg = result["negative_controls"]
    assert neg["own_city_a_vs_c_agreement"] >= 0.95, neg
    assert neg["r5_a_vs_b_agreement"] >= 0.95, neg
    # The measurement itself is reported, never asserted.
    assert result["verdict"] in VERDICTS
    assert result["vacuous"] == (result["verdict"] == "vacuous")
    assert json.loads(out.read_text())["verdict"] == result["verdict"]


@pytest.mark.slow
def test_a_chosen_command_is_always_offered() -> None:
    # measure_chunk asserts ``is_offered`` and ``fallback is None`` per search; make that visible.
    network = MicroCivNetwork(board_size=16, action_space_size=376)
    config = GumbelMCTSConfig(num_simulations=24, num_sampled_actions=8)
    policy = Civ6MctsPolicy(network, config, device="cpu", max_turns=30)
    torch.manual_seed(0)
    for root in all_roots()[:6]:
        command, record = asyncio.run(
            policy.select_command(convert(build_state(root), 16), decision_seed=1)
        )
        assert record["fallback"] is None and command["action_type"] != "", root.key
