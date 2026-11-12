"""The Gumbel search spends exactly the budget it is configured with (LF-A11, A6, A7).

Why this test exists
--------------------
Sequential halving lived in three hand-written copies -- ``GumbelMCTS``,
``BatchGumbelMCTS`` and ``MuZeroMCTS`` -- each computing a per-round floor of
``max(1, (n // rounds) // survivors)`` and stopping once one survivor was left.
The realised number of simulations therefore depended on ``k``, ``n`` and the
legal-move count in ways nobody had written down, and differed from ``n``:
measured before the fix, ``(k=16, n=64)`` with 5 legal moves spent 40, with 3
legal moves 30, and ``(k=3, n=10)`` spent 3 of 10. The batch copy derived its
round count from the *largest* candidate set across environments, so the same
position spent 30 simulations searched alone and 15 searched beside a position
with more legal moves.

The schedule now has one definition, :func:`planned_halving_sims` /
:func:`halving_schedule` in ``gumbel_ops``, and every search reports what it
actually spent in a :class:`SearchStats`. The equality tests below compare the
two -- and also count the fake network's calls independently, so a search that
reported the *plan* instead of *measuring* itself would still fail.

Negative control: restoring the old per-round floor in ``batch_mcts`` alone
fails ``test_sim_evals_equal_plan[batch-*]`` at every point where the old floor
diverged from ``n``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from alpha_zero.core.batch_mcts import BatchGumbelMCTS, BatchMCTSConfig
from alpha_zero.core.mcts.gumbel import GumbelMCTS, GumbelMCTSConfig
from alpha_zero.core.mcts.gumbel_ops import (
    HalvingRound,
    SearchStats,
    halving_schedule,
    planned_halving_sims,
)
from alpha_zero.core.mcts.muzero import MuZeroMCTS, MuZeroMCTSConfig

ACTION_SPACE = 40
SPATIAL_SHAPE = (1, 2, 2)
VECTOR_SIZE = 2
AMPLE_LEGAL = 30

#: (k, n, num_legal). The first two are the registered budgets (LF-A11); the
#: rest cover k > num_legal, a non-power-of-two k, a truncated schedule (n is
#: smaller than one full pass of every round) and leftover budget.
BUDGET_POINTS: tuple[tuple[int, int, int], ...] = (
    (8, 24, AMPLE_LEGAL),
    (16, 64, AMPLE_LEGAL),
    (16, 64, 5),
    (16, 64, 3),
    (8, 24, 6),
    (3, 10, AMPLE_LEGAL),
    (16, 32, AMPLE_LEGAL),
    (16, 8, AMPLE_LEGAL),
    (8, 30, AMPLE_LEGAL),
    (16, 64, 2),
    # k = 1 is not a forced move: mctx spends all n on the lone candidate.
    (1, 24, AMPLE_LEGAL),
    (1, 24, 2),
)


# ---------------------------------------------------------------------------
# Deterministic fakes. The environment never terminates, so every simulation
# ends in exactly one network leaf evaluation.
# ---------------------------------------------------------------------------


class _Env:
    """Endless two-player game with ``legal`` legal actions at every state."""

    num_players = 2

    def __init__(self, legal: int) -> None:
        self.legal = legal
        self.action_space = SimpleNamespace(n=ACTION_SPACE)
        self.current_player = 0

    def clone(self) -> _Env:
        return _Env(self.legal)

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        return self.get_observation(), 0.0, False, False, {}

    def get_valid_actions(self) -> np.ndarray:
        mask = np.zeros(ACTION_SPACE, dtype=np.float32)
        mask[: self.legal] = 1.0
        return mask

    def get_observation(self) -> dict[str, np.ndarray]:
        return {
            "spatial": np.zeros(SPATIAL_SHAPE, dtype=np.float32),
            "vector": np.zeros(VECTOR_SIZE, dtype=np.float32),
            "action_mask": self.get_valid_actions(),
        }

    def get_info(self) -> dict[str, Any]:
        return {}


class _CountingNet:
    """Uniform policy, zero value; counts every position it evaluates."""

    def __init__(self) -> None:
        self.positions = 0
        # Rows per predict() call, in order. Every search evaluates the root(s)
        # in its first call, so calls[0] is the root count and the rest are
        # simulation leaves -- counted here, independently of the search.
        self.calls: list[int] = []

    def eval(self) -> None:
        return None

    def predict(
        self, spatial: torch.Tensor, vector: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = spatial.shape[0]
        self.positions += batch
        self.calls.append(batch)
        return torch.full((batch, ACTION_SPACE), 1.0 / ACTION_SPACE), torch.zeros(batch, 1)


class _CountingMuZeroNet:
    """MuZero double: counts root inferences and leaf value evaluations."""

    def __init__(self) -> None:
        self.initial_calls = 0
        self.value_calls = 0

    def eval(self) -> None:
        return None

    def initial_inference(self, spatial: Any, vector: Any, mask: Any) -> Any:
        self.initial_calls += 1
        return SimpleNamespace(
            hidden_state=torch.zeros(1, 1),
            policy_logits=torch.zeros(1, ACTION_SPACE),
            value=torch.zeros(1),
        )

    def recurrent_inference(self, hidden: Any, action: Any, mask: Any) -> Any:
        return torch.zeros(1, 1), torch.zeros(1), torch.zeros(1, ACTION_SPACE), torch.zeros(1)

    def prediction_function(self, hidden: Any, mask: Any) -> Any:
        self.value_calls += 1
        return torch.zeros(1, ACTION_SPACE), torch.zeros(1)


def _run(variant: str, k: int, n: int, legal: int) -> tuple[SearchStats, int, int]:
    """Search once; return (reported stats, counted root evals, counted sim evals)."""
    env = _Env(legal)
    if variant == "gumbel":
        net = _CountingNet()
        search = GumbelMCTS(
            net,
            GumbelMCTSConfig(
                num_simulations=n,
                num_sampled_actions=k,
                max_considered_actions=max(k, 32),
                seed=0,
            ),
        )
        search.search(env)
        return search.last_search_stats, net.calls[0], sum(net.calls[1:])
    if variant == "batch":
        net = _CountingNet()
        batch = BatchGumbelMCTS(
            net, BatchMCTSConfig(num_simulations=n, num_sampled_actions=k, seed=0)
        )
        batch.search(env)
        return batch.last_search_stats, net.calls[0], sum(net.calls[1:])
    if variant == "muzero":
        mz = _CountingMuZeroNet()
        muzero = MuZeroMCTS(
            mz,
            MuZeroMCTSConfig(
                num_simulations=n, num_sampled_actions=k, use_categorical_reward=False, seed=0
            ),
        )
        muzero.search(env.get_observation(), env.get_valid_actions())
        return muzero.last_search_stats, mz.initial_calls, mz.value_calls
    raise AssertionError(variant)


# ---------------------------------------------------------------------------
# The schedule itself
# ---------------------------------------------------------------------------


class TestPlannedHalvingSims:
    @pytest.mark.parametrize(("k", "n"), [(8, 24), (16, 64)])
    def test_registered_budgets_are_exact(self, k: int, n: int) -> None:
        assert planned_halving_sims(k, n, AMPLE_LEGAL) == n

    def test_registered_budget_round_structure(self) -> None:
        # Danihelka et al.: ceil(log2 k) rounds, floor(n / (rounds * m)) visits
        # per survivor. At (8, 24) every round spends n / rounds = 8.
        rounds = halving_schedule(8, 24, AMPLE_LEGAL)
        assert [(r.survivors, r.simulations) for r in rounds] == [(8, 8), (4, 8), (2, 8)]
        rounds = halving_schedule(16, 64, AMPLE_LEGAL)
        assert [(r.survivors, r.simulations) for r in rounds] == [
            (16, 16),
            (8, 16),
            (4, 16),
            (2, 16),
        ]

    @pytest.mark.parametrize(("k", "n", "legal"), BUDGET_POINTS)
    def test_spends_exactly_n_unless_forced(self, k: int, n: int, legal: int) -> None:
        assert planned_halving_sims(k, n, legal) == n

    @pytest.mark.parametrize("legal", [2, AMPLE_LEGAL])
    def test_lone_candidate_gets_the_whole_budget(self, legal: int) -> None:
        # --gumbel_k 1 with several legal moves is a search, not a forced move.
        assert halving_schedule(1, 24, legal) == (HalvingRound(survivors=1, simulations=24),)
        assert planned_halving_sims(1, 24, legal) == 24

    def test_k_is_capped_by_legal_moves(self) -> None:
        assert halving_schedule(16, 64, 5) == halving_schedule(5, 64, 5)
        assert halving_schedule(16, 64, 5)[0].survivors == 5

    @pytest.mark.parametrize("legal", [0, 1])
    def test_forced_or_empty_root_spends_nothing(self, legal: int) -> None:
        assert planned_halving_sims(16, 64, legal) == 0
        assert halving_schedule(16, 64, legal) == ()

    def test_leftover_goes_to_the_final_survivors(self) -> None:
        # (8, 30): 8 + 8 + 10 = 26 from the floors; the 4 left over are spent
        # on the final two survivors, as mctx's schedule does.
        rounds = halving_schedule(8, 30, AMPLE_LEGAL)
        assert [(r.survivors, r.simulations) for r in rounds] == [(8, 8), (4, 8), (2, 14)]

    def test_truncation_drops_later_rounds(self) -> None:
        # (16, 32): the floors would spend 16 + 8 + 8 + 8 = 40; the schedule stops at n.
        rounds = halving_schedule(16, 32, AMPLE_LEGAL)
        assert [(r.survivors, r.simulations) for r in rounds] == [(16, 16), (8, 8), (4, 8)]

    def test_partial_pass_favours_top_ranked_survivors(self) -> None:
        (only,) = halving_schedule(16, 8, AMPLE_LEGAL)
        assert only.survivors == 16
        assert only.visits() == [1] * 8 + [0] * 8
        assert list(only.order()) == list(range(8))

    def test_order_is_pass_major(self) -> None:
        rounds = halving_schedule(8, 30, AMPLE_LEGAL)
        assert list(rounds[-1].order()) == [0, 1] * 7

    def test_min_survivors_changes_the_shape_not_the_total(self) -> None:
        rounds = halving_schedule(16, 64, AMPLE_LEGAL, min_survivors=4)
        assert [r.survivors for r in rounds] == [16, 8, 4, 4]
        assert sum(r.simulations for r in rounds) == 64

    def test_deterministic(self) -> None:
        assert halving_schedule(16, 64, 7) == halving_schedule(16, 64, 7)

    @pytest.mark.parametrize(("k", "n", "legal"), [(0, 8, 4), (4, -1, 4), (4, 8, -1)])
    def test_rejects_invalid_arguments(self, k: int, n: int, legal: int) -> None:
        with pytest.raises(ValueError):
            planned_halving_sims(k, n, legal)


# ---------------------------------------------------------------------------
# Every search spends the plan, and measures itself
# ---------------------------------------------------------------------------

VARIANTS = ("gumbel", "batch", "muzero")


class TestSearchesSpendThePlan:
    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize(("k", "n", "legal"), BUDGET_POINTS)
    def test_sim_evals_equal_plan(self, variant: str, k: int, n: int, legal: int) -> None:
        stats, counted_root, counted_sims = _run(variant, k, n, legal)
        expected = planned_halving_sims(k, n, legal)
        assert stats == SearchStats(root_evals=1, sim_evals=expected)
        # Independent count from the fake network: the stats are measured, not
        # copied from the plan.
        assert counted_root == stats.root_evals
        assert counted_sims == stats.sim_evals

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("legal", [2, AMPLE_LEGAL])
    def test_lone_candidate_is_searched(self, variant: str, legal: int) -> None:
        # Literal, not the plan: `--gumbel_k 1` with several legal moves must
        # spend the whole budget in every search, even if the plan regresses.
        stats, _, counted_sims = _run(variant, 1, 24, legal)
        assert stats.sim_evals == 24
        assert counted_sims == 24

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_forced_move_is_not_searched(self, variant: str) -> None:
        stats, _, counted_sims = _run(variant, 16, 64, 1)
        assert stats == SearchStats(root_evals=1, sim_evals=0)
        assert counted_sims == 0

    def test_batch_envs_do_not_share_a_round_count(self) -> None:
        # Each environment spends its OWN plan, whatever it is searched beside.
        net = _CountingNet()
        batch = BatchGumbelMCTS(
            net, BatchMCTSConfig(num_simulations=64, num_sampled_actions=16, seed=0)
        )
        batch.batch_search([_Env(AMPLE_LEGAL), _Env(3)])
        per_env = batch.last_batch_search_stats
        assert per_env == [
            SearchStats(1, planned_halving_sims(16, 64, AMPLE_LEGAL)),
            SearchStats(1, planned_halving_sims(16, 64, 3)),
        ]
        assert batch.last_search_stats == SearchStats(2, 128)
        assert net.positions == 2 + 128

    def test_stats_are_frozen(self) -> None:
        stats = SearchStats(root_evals=1, sim_evals=2)
        with pytest.raises(AttributeError):
            stats.sim_evals = 3  # type: ignore[misc]


# ---------------------------------------------------------------------------
# training_civ6 CLI (LF-A6, A7)
# ---------------------------------------------------------------------------


def _config_from_argv(argv: list[str]) -> Any:
    from alpha_zero import training_civ6 as tc

    args = tc.build_parser().parse_args(argv)
    return tc.config_from_args(args, checkpoint_dir="unused")


class TestTrainingCli:
    def test_flags_round_trip(self) -> None:
        cfg = _config_from_argv(
            [
                "--mcts",
                "gumbel",
                "--gumbel_k",
                "8",
                "--gumbel_sims",
                "24",
                "--c_scale",
                "0.1",
                "--policy_target",
                "visit_counts",
                "--greedy_threshold",
                "0.9",
            ]
        )
        assert cfg.use_gumbel_mcts is True
        assert cfg.gumbel_num_sampled_actions == 8
        assert cfg.gumbel_num_simulations == 24
        assert cfg.gumbel_c_scale == pytest.approx(0.1)
        assert cfg.gumbel_policy_target == "visit_counts"
        assert cfg.gumbel_greedy_threshold == pytest.approx(0.9)

    def test_defaults_are_unchanged(self) -> None:
        from alpha_zero.training_civ6 import TrainingConfig

        defaults = TrainingConfig()
        for argv in ([], ["--mcts", "gumbel"]):
            cfg = _config_from_argv(argv)
            assert cfg.num_simulations == defaults.num_simulations
            assert cfg.gumbel_num_simulations == defaults.gumbel_num_simulations
            assert cfg.gumbel_num_sampled_actions == defaults.gumbel_num_sampled_actions
            assert cfg.gumbel_c_scale == defaults.gumbel_c_scale
            assert cfg.gumbel_policy_target == defaults.gumbel_policy_target
            assert cfg.gumbel_greedy_threshold == defaults.gumbel_greedy_threshold

    def test_default_gumbel_config_is_pinned(self) -> None:
        # Literals, not the same module's defaults: a changed default must fail here.
        from alpha_zero.training_civ6 import TrainingConfig, gumbel_search_config

        built = gumbel_search_config(
            TrainingConfig(),
            num_simulations=16,
            seed=None,
            temperature=1.0,
            temperature_threshold=30,
        )
        assert built == GumbelMCTSConfig(
            num_simulations=16,
            num_sampled_actions=16,
            max_considered_actions=32,
            c_visit=50.0,
            c_scale=1.0,
            use_sequential_halving=True,
            min_actions_after_halving=1,
            temperature=1.0,
            temperature_threshold=30,
            seed=None,
            policy_target="completed_q",
            greedy_threshold=0.99,
        )

    def test_gumbel_honours_num_simulations(self) -> None:
        # LF-A7: --mcts gumbel used to ignore --num_simulations entirely.
        cfg = _config_from_argv(["--mcts", "gumbel", "--num_simulations", "32"])
        assert cfg.gumbel_num_simulations == 32
        assert cfg.num_simulations == 32

    def test_gumbel_sims_wins_over_num_simulations(self) -> None:
        cfg = _config_from_argv(
            ["--mcts", "gumbel", "--num_simulations", "32", "--gumbel_sims", "24"]
        )
        assert cfg.gumbel_num_simulations == 24
        assert cfg.num_simulations == 32

    @pytest.mark.parametrize(
        "argv",
        [
            ["--gumbel_k", "0"],
            ["--gumbel_sims", "0"],
            ["--c_scale", "-1"],
            ["--greedy_threshold", "0"],
            ["--greedy_threshold", "1.5"],
            ["--policy_target", "argmax"],
        ],
    )
    def test_invalid_values_are_rejected(self, argv: list[str]) -> None:
        with pytest.raises(SystemExit):
            _config_from_argv(["--mcts", "gumbel", *argv])

    @pytest.mark.parametrize(
        "argv",
        [
            ["--gumbel_k", "8"],
            ["--gumbel_sims", "24"],
            ["--c_scale", "0.1"],
            ["--greedy_threshold", "0.9"],
            ["--policy_target", "visit_counts"],
        ],
    )
    @pytest.mark.parametrize("mcts", [[], ["--mcts", "civ6"]])
    def test_gumbel_only_flags_need_mcts_gumbel(
        self, argv: list[str], mcts: list[str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Silently ignoring them under PUCT would let a run believe it was
        # configured when it was not.
        with pytest.raises(SystemExit):
            _config_from_argv([*mcts, *argv])
        assert "--mcts gumbel" in capsys.readouterr().err

    def test_gumbel_only_flags_are_accepted_with_mcts_gumbel(self) -> None:
        cfg = _config_from_argv(["--mcts", "gumbel", "--policy_target", "visit_counts"])
        assert cfg.gumbel_policy_target == "visit_counts"

    def test_gumbel_k_1_is_accepted(self) -> None:
        assert (
            _config_from_argv(["--mcts", "gumbel", "--gumbel_k", "1"]).gumbel_num_sampled_actions
            == 1
        )

    def test_new_flags_do_not_change_the_price(self) -> None:
        from alpha_zero.training_civ6 import training_budget_inputs
        from alpha_zero.utils.game_budget import worst_case_games

        base = ["--num_games", "4", "--num_iterations", "3"]
        extra = ["--mcts", "gumbel", "--gumbel_k", "8", "--gumbel_sims", "24", "--c_scale", "0.1"]
        priced = [
            worst_case_games("training_civ6", training_budget_inputs(_config_from_argv(a), False))
            for a in (base, base + extra)
        ]
        assert priced[0] == priced[1]

    @staticmethod
    def _capture_gumbel_configs(fn: Any) -> list[GumbelMCTSConfig]:
        from alpha_zero import training_civ6 as tc

        instance = MagicMock()

        def search(env: Any, add_noise: bool = False) -> tuple[np.ndarray, float]:
            mask = env.get_valid_actions().astype(np.float32)
            return mask / max(mask.sum(), 1.0), 0.0

        instance.search.side_effect = search
        instance.select_action.side_effect = lambda probs, *a, **kw: int(np.argmax(probs))
        cls = MagicMock(return_value=instance)
        with patch.object(tc, "GumbelMCTS", cls):
            fn()
        return [c.args[1] for c in cls.call_args_list]

    def test_flags_reach_self_play_and_evaluation(self) -> None:
        from alpha_zero import training_civ6 as tc

        cfg = tc.TrainingConfig(
            board_size=6,
            max_turns=2,
            use_gumbel_mcts=True,
            gumbel_num_simulations=24,
            gumbel_num_sampled_actions=8,
            gumbel_c_scale=0.1,
            gumbel_policy_target="visit_counts",
            gumbel_greedy_threshold=0.9,
            device="cpu",
            seed=3,
        )
        (selfplay,) = self._capture_gumbel_configs(
            lambda: tc.run_self_play(MagicMock(), cfg, num_games=1)
        )
        (evaluation,) = self._capture_gumbel_configs(
            lambda: tc.evaluate_against_random(MagicMock(), cfg, num_games=1)
        )
        assert selfplay.num_simulations == 24
        # Evaluation keeps its historical half budget.
        assert evaluation.num_simulations == 12
        for gcfg in (selfplay, evaluation):
            assert gcfg.num_sampled_actions == 8
            assert gcfg.c_scale == pytest.approx(0.1)
            assert gcfg.policy_target == "visit_counts"
            assert gcfg.greedy_threshold == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# m-1: the chosen move is always one the search visited
# ---------------------------------------------------------------------------

ROOT_VALUE = 0.9
ADVERSARIAL_SEEDS = range(50)


class _AdversarialNet(_CountingNet):
    """Root value +0.9, every leaf ``leaf_value``.

    With ``leaf_value`` = -0.9 an unvisited candidate's completed Q (the root
    value) beats every visited one, which is what tempts a search that ranks
    unvisited candidates alongside visited ones into returning one of them.
    Both signs are swept so the invariant does not depend on a variant's sign rule.
    """

    def __init__(self, leaf_value: float) -> None:
        super().__init__()
        self.leaf_value = leaf_value

    def predict(
        self, spatial: torch.Tensor, vector: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        policy, _ = super().predict(spatial, vector, mask)
        value = ROOT_VALUE if len(self.calls) == 1 else self.leaf_value
        return policy, torch.full((spatial.shape[0], 1), value)


class _AdversarialMuZeroNet(_CountingMuZeroNet):
    def __init__(self, leaf_value: float) -> None:
        super().__init__()
        self.leaf_value = leaf_value

    def initial_inference(self, spatial: Any, vector: Any, mask: Any) -> Any:
        out = super().initial_inference(spatial, vector, mask)
        return SimpleNamespace(
            hidden_state=out.hidden_state,
            policy_logits=out.policy_logits,
            value=torch.full((1,), ROOT_VALUE),
        )

    def prediction_function(self, hidden: Any, mask: Any) -> Any:
        logits, _ = super().prediction_function(hidden, mask)
        return logits, torch.full((1,), self.leaf_value)


def _chosen_and_visited(variant: str, seed: int, leaf_value: float) -> tuple[int, set[int]]:
    """Search at (16, 8) -- the default evaluation shape -- with a one-hot target."""
    k, n = 16, 8
    env = _Env(AMPLE_LEGAL)
    visited: set[int] = set()
    if variant == "gumbel":
        gumbel = GumbelMCTS(
            _AdversarialNet(leaf_value),
            GumbelMCTSConfig(
                num_simulations=n, num_sampled_actions=k, seed=seed, policy_target="one_hot"
            ),
        )
        original = gumbel._run_simulation_from_action

        def record(e: Any, root: Any, action: int) -> None:
            visited.add(int(action))
            original(e, root, action)

        gumbel._run_simulation_from_action = record  # type: ignore[method-assign]
        probs, _ = gumbel.search(env)
    elif variant == "batch":
        batch = BatchGumbelMCTS(
            _AdversarialNet(leaf_value),
            BatchMCTSConfig(
                num_simulations=n, num_sampled_actions=k, seed=seed, policy_target="one_hot"
            ),
        )
        original_b = batch._run_simulation_for_env

        def record_b(i: int, e: Any, root: Any, candidates: np.ndarray) -> Any:
            visited.add(int(candidates[0]))
            return original_b(i, e, root, candidates)

        batch._run_simulation_for_env = record_b  # type: ignore[method-assign]
        probs, _ = batch.search(env)
    else:
        muzero = MuZeroMCTS(
            _AdversarialMuZeroNet(leaf_value),
            MuZeroMCTSConfig(
                num_simulations=n, num_sampled_actions=k, use_categorical_reward=False, seed=seed
            ),
        )
        original_m = muzero._run_muzero_simulation

        def record_m(root: Any, action: int, mask: np.ndarray) -> None:
            visited.add(int(action))
            original_m(root, action, mask)

        muzero._run_muzero_simulation = record_m  # type: ignore[method-assign]
        probs, _ = muzero.search(env.get_observation(), env.get_valid_actions())
    assert np.isclose(probs.sum(), 1.0) and probs.max() == 1.0, "expected a one-hot target"
    return int(np.argmax(probs)), visited


class TestChosenMoveWasVisited:
    """mctx's ``score_considered``: choose only among the most-visited candidates.

    At (16, 8) the first round is truncated, so half the candidates are never
    visited. Ranking them alongside the visited ones let an unsearched move win.
    """

    @pytest.mark.parametrize("leaf_value", [-0.9, 0.9])
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_no_unvisited_move_is_returned(self, variant: str, leaf_value: float) -> None:
        bad = []
        for seed in ADVERSARIAL_SEEDS:
            chosen, visited = _chosen_and_visited(variant, seed, leaf_value)
            assert len(visited) == 8
            if chosen not in visited:
                bad.append(seed)
        assert not bad, f"{variant} returned an unvisited move for seeds {bad}"


# ---------------------------------------------------------------------------
# m-3: the dry run bounds the Gumbel budget too, and says so
# ---------------------------------------------------------------------------


class TestDryRunBoundsGumbel:
    def test_default_gumbel_budget_is_bounded(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        from alpha_zero.training_civ6 import DRY_RUN_PROFILE, TrainingConfig

        with caplog.at_level(logging.INFO, logger="alpha_zero.training_civ6"):
            bounded = DRY_RUN_PROFILE.apply_to(TrainingConfig(use_gumbel_mcts=True))
        assert bounded.gumbel_num_simulations == DRY_RUN_PROFILE.gumbel_num_simulations
        assert bounded.gumbel_num_simulations < TrainingConfig().gumbel_num_simulations
        messages = [r.getMessage() for r in caplog.records]
        assert any(
            f"Gumbel search uses {bounded.gumbel_num_simulations} simulations" in m
            for m in messages
        ), messages

    def test_explicit_gumbel_budget_wins(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        from alpha_zero.training_civ6 import DRY_RUN_PROFILE

        cfg = _config_from_argv(["--mcts", "gumbel", "--gumbel_k", "8", "--gumbel_sims", "24"])
        with caplog.at_level(logging.INFO, logger="alpha_zero.training_civ6"):
            bounded = DRY_RUN_PROFILE.apply_to(cfg)
        assert bounded.gumbel_num_simulations == 24
        assert any("Gumbel search uses 24 simulations" in r.getMessage() for r in caplog.records)

    def test_puct_config_gumbel_field_is_left_alone(self) -> None:
        from alpha_zero.training_civ6 import DRY_RUN_PROFILE, TrainingConfig

        config = TrainingConfig(board_size=12, max_turns=40, num_simulations=25)
        assert DRY_RUN_PROFILE.apply_to(config) is config
