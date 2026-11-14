"""The A1 registration (``docs/EVALUATION.md`` section 7.1) is executable as written.

Section 7.1 is frozen by its sha256 before A1 runs, so anything wrong in it is
frozen too. These tests read the commands **out of the hashed region** and check:

* each parses with its runner's own parser and prices through the run-card
  driver: 480 + 500 = **980**, each invocation <= 1,000 -- for the base commands
  and for the 60-game positive-control branch;
* the training command is the plan's (§5.2) verbatim, and each arena command is
  the plan's common prefix plus its arm's flags plus its own ``--out``;
* ``configs/run_cards/a1.yaml`` and ``a1r.yaml`` equal those commands token for
  token (A1-R with the seed swapped), and ``run_card --price_only`` prints 980;
* the Gumbel settings 7.1 registers are the ones ``training_civ6`` resolves;
* the hash helper is stable, scoped to the region, and fails closed on bad markers.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from alpha_zero.evaluation.a1_protocol import (
    PROTOCOL_BEGIN,
    PROTOCOL_END,
    ProtocolMarkerError,
    iid_positive_control_void_probability,
    pairs_from_games,
    positive_control_void_probability,
    positive_control_void_probability_interval,
    protocol_sha256,
    read_registered_protocol,
    registered_protocol,
)
from alpha_zero.evaluation.a1_registration import (
    A1R_SEED as REGISTERED_A1R_SEED,
)
from alpha_zero.evaluation.a1_registration import (
    DATA_ROOT,
    RegistrationError,
    branch_in_force,
    command_block,
    registered_invocations,
    run_directory,
)
from alpha_zero.local import run_card as rc
from alpha_zero.utils import game_budget

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_MD = REPO_ROOT / "docs" / "EVALUATION.md"
CARDS = REPO_ROOT / "configs" / "run_cards"

A1_SEED, A1R_SEED = "20261111", "20261112"
TRAIN_PREFIX = ["python", "-m", "alpha_zero.training_civ6"]
BENCH_PREFIX = ["python", "scripts/benchmark_suite.py"]

#: Plan §5.2, copied by hand from the plan -- the anchor the section is checked against.
PLAN_TRAINING = (
    "python -m alpha_zero.training_civ6 --seed 20261111 --board_size 8 --max_turns 60 "
    "--num_iterations 20 --num_games 24 --mcts gumbel --gumbel_k 8 --gumbel_sims 24 "
    "--device cuda:0 --bootstrap none --save_interval 5 --eval_games 0 --checkpoint_dir <D>\\ckpt"
)
PLAN_ARENA_PREFIX = (
    "python scripts/benchmark_suite.py --opponents= --seed 20261111 --board-size 8 "
    "--max-turns 60 --sims 24 --gumbel-k 8 --mcts-type gumbel --device cuda:1"
)
#: Plan §5.2's arms table, with ``…<N>.pt`` written out and the final arm's candidate named.
PLAN_ARMS = {
    "primary": "--checkpoint <D>\\ckpt\\checkpoint_20.pt "
    "--opponent-checkpoint <D>\\ckpt\\checkpoint_0.pt --n-games 200",
    "symmetry": "--checkpoint <D>\\ckpt\\checkpoint_0.pt "
    "--opponent-checkpoint <D>\\ckpt\\checkpoint_0.pt --n-games 40",
    "positive-control": "--candidate heuristic --opponents=random --n-games 40",
    "secondary-5": "--checkpoint <D>\\ckpt\\checkpoint_5.pt "
    "--opponent-checkpoint <D>\\ckpt\\checkpoint_0.pt --n-games 60",
    "secondary-10": "--checkpoint <D>\\ckpt\\checkpoint_10.pt "
    "--opponent-checkpoint <D>\\ckpt\\checkpoint_0.pt --n-games 60",
    "secondary-15": "--checkpoint <D>\\ckpt\\checkpoint_15.pt "
    "--opponent-checkpoint <D>\\ckpt\\checkpoint_0.pt --n-games 60",
    "final-vs-random": "--checkpoint <D>\\ckpt\\checkpoint_20.pt --opponents=random --n-games 40",
}
BASE_GAMES = [480, 200, 40, 40, 60, 60, 60, 40]
BRANCH_GAMES = {"positive-control": 60, "final-vs-random": 20}


def _data_dir(card_id: str) -> str:
    return run_directory(card_id)


def parse_commands(section: str, block: str) -> list[tuple[str, str]]:
    """``[(name, command)]`` via the shared parser the release verifier also uses."""
    return [(name, " ".join(tokens)) for name, tokens in command_block(section, block)]


def to_invocation(name: str, command: str, card_id: str) -> rc.Invocation:
    """The run-card invocation a 7.1 command becomes under ``card_id``."""
    assert "'" not in command and '"' not in command, "registered commands carry no quoting"
    tokens = command.replace("<D>", _data_dir(card_id)).split()
    if tokens[:3] == TRAIN_PREFIX:
        return rc.Invocation(name, "training_civ6", tuple(tokens[3:]))
    assert tokens[:2] == BENCH_PREFIX, command
    return rc.Invocation(name, "benchmark", tuple(tokens[2:]))


def _with_seed(invocation: rc.Invocation, seed: str) -> rc.Invocation:
    argv = list(invocation.argv)
    for i, arg in enumerate(argv[:-1]):
        if arg == "--seed":
            argv[i + 1] = seed
    return rc.Invocation(invocation.name, invocation.runner, tuple(argv))


@pytest.fixture(scope="module")
def section() -> str:
    return read_registered_protocol(EVALUATION_MD)


@pytest.fixture(scope="module")
def base(section: str) -> list[tuple[str, str]]:
    return parse_commands(section, "base")


@pytest.fixture(scope="module")
def branch(section: str) -> list[tuple[str, str]]:
    return parse_commands(section, "branch-60")


def _price(invocations: list[rc.Invocation]) -> rc.PricedCard:
    card = rc.RunCard(
        card_id="a1-20261111", sha="0" * 40, max_games=980, invocations=tuple(invocations)
    )
    return rc.price_card(card)


# ---------------------------------------------------------------------------
# The commands as registered
# ---------------------------------------------------------------------------


def test_training_command_is_the_plan_verbatim(base: list[tuple[str, str]]) -> None:
    assert base[0] == ("train", PLAN_TRAINING)


def test_arena_commands_are_the_plan_prefix_plus_arm_flags_plus_own_out(
    base: list[tuple[str, str]], branch: list[tuple[str, str]]
) -> None:
    outs = []
    for name, command in base[1:]:
        assert command.startswith(PLAN_ARENA_PREFIX + " "), name
        rest = command[len(PLAN_ARENA_PREFIX) + 1 :]
        flags, out_flag, out = rest.rsplit(" ", 2)
        assert flags == PLAN_ARMS[name], name
        assert out_flag == "--out" and re.fullmatch(r"<D>\\arena\\[a-z0-9_]+", out), out
        outs.append(out)
    assert [name for name, _ in base[1:]] == list(PLAN_ARMS)
    assert len(set(outs)) == len(outs), "every arm needs its own --out (T11 review I1)"
    by_name = dict(base)
    for name, command in branch:
        assert command.startswith(PLAN_ARENA_PREFIX + " ")
        # The branch changes only the game count of an existing arm.
        expected = re.sub(r"--n-games \d+", f"--n-games {BRANCH_GAMES[name]}", by_name[name])
        assert command == expected, name


def test_base_commands_price_at_980(base: list[tuple[str, str]]) -> None:
    priced = _price([to_invocation(n, c, "a1-20261111") for n, c in base])
    assert [p.games for p in priced.invocations] == BASE_GAMES
    assert priced.total == 980
    assert all(p.games <= game_budget.DEFAULT_MAX_GAMES for p in priced.invocations)


def test_branch_60_keeps_980(base: list[tuple[str, str]], branch: list[tuple[str, str]]) -> None:
    replaced = dict(base) | dict(branch)
    priced = _price([to_invocation(n, replaced[n], "a1-20261111") for n, _ in base])
    games = {p.invocation.name: p.games for p in priced.invocations}
    assert games["positive-control"] == 60 and games["final-vs-random"] == 20
    assert priced.total == 980


def test_a_mispriced_command_would_be_caught(base: list[tuple[str, str]]) -> None:
    """Negative control: the pricing path sees an argv change, so the 980 check is not vacuous."""
    name, command = base[1]
    bumped = command.replace("--n-games 200", "--n-games 220")
    invocations = [to_invocation(n, c, "a1-20261111") for n, c in base]
    invocations[1] = to_invocation(name, bumped, "a1-20261111")
    card = rc.RunCard("a1-20261111", "0" * 40, 1000, tuple(invocations))
    assert rc.price_card(card).total == 1000


# ---------------------------------------------------------------------------
# The run cards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("file", "card_id", "seed"),
    [("a1.yaml", "a1-20261111", A1_SEED), ("a1r.yaml", "a1r-20261112", A1R_SEED)],
)
def test_card_equals_the_registered_commands_token_for_token(
    base: list[tuple[str, str]], file: str, card_id: str, seed: str
) -> None:
    card = rc.load_card(CARDS / file)
    assert card.card_id == card_id
    assert card.max_games == 980 and card.approval_ref is None
    expected = [_with_seed(to_invocation(n, c, card_id), seed) for n, c in base]
    assert list(card.invocations) == expected
    assert rc.price_card(card).total == 980


@pytest.mark.parametrize("file", ["a1.yaml", "a1r.yaml"])
def test_placeholder_card_is_refused_before_its_run_dir_exists(tmp_path: Path, file: str) -> None:
    """The shipped cards' all-zero sha is refused after pricing, before anything is created."""
    root = tmp_path / "root"
    with pytest.raises(rc.RunCardError, match="placeholder sha"):
        rc.run_card(CARDS / file, data_root=root, repo_root=REPO_ROOT)
    assert not root.exists()


def test_price_only_prints_980(capsys: pytest.CaptureFixture[str]) -> None:
    assert rc.main([str(CARDS / "a1.yaml"), "--price_only"]) == 0
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert re.match(r"total\s+980 \(max_games 980\)$", last), last


# ---------------------------------------------------------------------------
# The registered search settings are the ones training_civ6 resolves
# ---------------------------------------------------------------------------


def test_registered_gumbel_settings_match_training_civ6(
    base: list[tuple[str, str]], section: str
) -> None:
    from alpha_zero import training_civ6 as tc

    argv = list(to_invocation(*base[0], "a1-20261111").argv)
    config = tc.config_from_args(tc.build_parser().parse_args(argv), checkpoint_dir="unused")
    resolved = tc.gumbel_search_config(
        config,
        num_simulations=config.gumbel_num_simulations,
        seed=None,
        temperature=config.temperature,
        temperature_threshold=config.temperature_threshold,
    )
    registered = dict(
        re.findall(r"^- `(\w+)`(?: \(`--\w+`\))?: `?([\w.]+)`?$", section, re.MULTILINE)
    )
    fields = (
        "num_simulations",
        "num_sampled_actions",
        "c_visit",
        "c_scale",
        "policy_target",
        "greedy_threshold",
        "use_sequential_halving",
        "temperature",
        "temperature_threshold",
    )
    assert set(fields) <= set(registered), sorted(registered)
    for name in fields:
        assert registered[name] == str(getattr(resolved, name)), name
    assert config.use_gumbel_mcts


# ---------------------------------------------------------------------------
# The hash
# ---------------------------------------------------------------------------

_DOC = f"intro\n{PROTOCOL_BEGIN} x -->\nline one\nline two\n{PROTOCOL_END} -->\noutro\n"


def test_hash_covers_only_the_region() -> None:
    assert registered_protocol(_DOC) == "line one\nline two\n"
    assert protocol_sha256(_DOC) == protocol_sha256(_DOC.replace("outro", "changed outside"))
    assert protocol_sha256(_DOC) != protocol_sha256(_DOC.replace("line two", "line 2"))


def test_hash_is_line_ending_independent() -> None:
    assert protocol_sha256(_DOC) == protocol_sha256(_DOC.replace("\n", "\r\n"))


@pytest.mark.parametrize(
    "doc",
    [
        "no markers\n",
        _DOC + f"{PROTOCOL_BEGIN} again -->\n",
        f"{PROTOCOL_END} -->\nbody\n{PROTOCOL_BEGIN} -->\n",
    ],
)
def test_bad_markers_fail_closed(doc: str) -> None:
    with pytest.raises(ProtocolMarkerError):
        protocol_sha256(doc)


def test_the_hash_is_not_written_inside_the_region(section: str) -> None:
    digest = protocol_sha256(EVALUATION_MD.read_text(encoding="utf-8"))
    assert digest not in section
    assert "### 7.1" in section, "the section heading is inside the hashed region"


# ---------------------------------------------------------------------------
# Positive-control calibration (T11 review I2, T13 review I1): P(hull L <= 0.5)
# ---------------------------------------------------------------------------

#: Map-decided pairs: the map hands both games to one side. Score 0.75, no draws.
CLUSTERED = [("win", "win")] * 30 + [("loss", "loss")] * 10


def test_pair_bootstrap_sees_clustering_the_iid_model_cannot() -> None:
    """The registered method must not inherit the iid model's blindness to maps.

    On map-decided pairs the pair-clustered interval is wide, so the hull VOIDs
    the control far more often than independent games at the same score would.
    Measured: pair 0.37 / iid 0.10 at 40 games; pair 0.19 / iid 0.02 at 60.
    """
    pair_40 = positive_control_void_probability(CLUSTERED, 40, reps=4000)
    iid_40 = iid_positive_control_void_probability(0.75, 0.0, 40, reps=4000)
    pair_60 = positive_control_void_probability(CLUSTERED, 60, reps=4000)
    iid_60 = iid_positive_control_void_probability(0.75, 0.0, 60, reps=4000)
    assert pair_40 > iid_40 + 0.15, (pair_40, iid_40)
    assert pair_60 > iid_60 + 0.10, (pair_60, iid_60)
    # The rule's two answers diverge exactly where it matters: iid says 60 games
    # would do (< 5%), the pairs say they would not.
    assert iid_60 < 0.05 < pair_60


def test_pair_bootstrap_of_a_dominant_control_never_voids() -> None:
    assert positive_control_void_probability([("win", "win")] * 5, 40, reps=200) == 0.0


def test_pairs_from_games_keeps_seat_order_and_rejects_half_pairs() -> None:
    games = [
        {"pair": 1, "seat": 1, "result": "loss"},
        {"pair": 0, "seat": 0, "result": "win"},
        {"pair": 1, "seat": 0, "result": "draw"},
        {"pair": 0, "seat": 1, "result": "win"},
    ]
    assert pairs_from_games(games) == [("win", "win"), ("draw", "loss")]
    with pytest.raises(ValueError, match="incomplete"):
        pairs_from_games(games[:3])


def test_iid_fallback_reproduces_the_t11_figures_and_the_crossover() -> None:
    """Informational only: d ~ 0 at 40 games ~14%; the 5% crossover is near d = 0.23."""
    near_zero_draws_40 = iid_positive_control_void_probability(0.7375, 0.0, 40, reps=4000)
    assert 0.11 <= near_zero_draws_40 <= 0.18, near_zero_draws_40
    below = iid_positive_control_void_probability(0.7375 - 0.10, 0.20, 40, reps=8000)
    above = iid_positive_control_void_probability(0.7375 - 0.13, 0.26, 40, reps=8000)
    assert below > 0.05 > above, (below, above)


#: A strong control on a finite reference: few VOIDs at the point estimate, but
#: a reference this size cannot rule out more (T13 rereview N3).
STRONG = (
    [("win", "win")] * 80
    + [("win", "draw"), ("draw", "win")] * 20
    + [("draw", "draw")] * 15
    + [("loss", "loss")] * 10
    + [("win", "loss"), ("loss", "win")] * 10
)


def test_interval_rule_takes_60_where_the_point_estimate_would_take_40() -> None:
    """Measured: point 0.03, interval about [0.003, 0.10] -- the registered rule reads the upper bound."""
    point = positive_control_void_probability(STRONG, 40, reps=2000)
    low, high = positive_control_void_probability_interval(STRONG, 40, outer=60, inner=300, seed=3)
    assert point < 0.05 < high, (point, low, high)
    assert low <= point <= high


def test_interval_of_a_dominant_control_is_zero() -> None:
    assert positive_control_void_probability_interval(
        [("win", "win")] * 5, 40, outer=5, inner=50
    ) == (0.0, 0.0)


def test_interval_is_seeded() -> None:
    a = positive_control_void_probability_interval(STRONG, 40, outer=10, inner=100, seed=9)
    assert a == positive_control_void_probability_interval(STRONG, 40, outer=10, inner=100, seed=9)


@pytest.mark.parametrize(
    ("outer", "inner", "level"), [(0, 10, 0.95), (10, 0, 0.95), (10, 10, 1.0), (10, 10, 0.0)]
)
def test_interval_rejects_bad_arguments(outer: int, inner: int, level: float) -> None:
    with pytest.raises(ValueError):
        positive_control_void_probability_interval(
            STRONG, 40, outer=outer, inner=inner, level=level
        )


def test_calibration_is_seeded() -> None:
    a = positive_control_void_probability(CLUSTERED, 40, reps=300, seed=7)
    assert a == positive_control_void_probability(CLUSTERED, 40, reps=300, seed=7)


@pytest.mark.parametrize(
    ("pairs", "n_games", "reps"),
    [
        (CLUSTERED, 41, 10),
        (CLUSTERED, 0, 10),
        (CLUSTERED, 40, 0),
        ([("win", "win")], 40, 10),
        ([("win", "tie"), ("win", "win")], 40, 10),
    ],
)
def test_pair_calibration_rejects_bad_arguments(
    pairs: list[tuple[str, str]], n_games: int, reps: int
) -> None:
    with pytest.raises(ValueError):
        positive_control_void_probability(pairs, n_games, reps=reps)


@pytest.mark.parametrize(("p_win", "p_draw"), [(0.8, 0.3), (-0.1, 0.1)])
def test_iid_calibration_rejects_bad_probabilities(p_win: float, p_draw: float) -> None:
    with pytest.raises(ValueError):
        iid_positive_control_void_probability(p_win, p_draw, 40, reps=10)


# ---------------------------------------------------------------------------
# The shared registration parser (a1_registration; the release verifier uses it)
# ---------------------------------------------------------------------------


def _with_branch(section: str, value: str) -> str:
    old = "**Branch in force: pending T0b.**"
    assert section.count(old) == 1
    return section.replace(old, f"**Branch in force: {value}.**")


def test_the_branch_is_pending_until_t0b(section: str) -> None:
    with pytest.raises(RegistrationError, match="not registered yet"):
        branch_in_force(section)
    with pytest.raises(RegistrationError):
        registered_invocations(section, card_id="a1-20261111", seed=20261111)


@pytest.mark.parametrize("value", ["40", "60"])
def test_registered_invocations_follow_the_branch_in_force(section: str, value: str) -> None:
    filled = _with_branch(section, value)
    assert branch_in_force(filled) == int(value)
    got = registered_invocations(filled, card_id="a1-20261111", seed=20261111)
    base = [to_invocation(n, c, "a1-20261111") for n, c in parse_commands(section, "base")]
    assert list(got) == [i.name for i in base]
    games = {"40": (40, 40), "60": (60, 20)}[value]
    assert got["positive-control"][-3:-2] == [str(games[0])]
    assert got["final-vs-random"][-3:-2] == [str(games[1])]
    if value == "40":
        assert got == {i.name: list(i.argv) for i in base}


def test_registered_invocations_substitute_the_card_and_seed(section: str) -> None:
    got = registered_invocations(
        _with_branch(section, "40"), card_id="a1r-20261112", seed=REGISTERED_A1R_SEED
    )
    expected = [
        _with_seed(to_invocation(n, c, "a1r-20261112"), A1R_SEED)
        for n, c in parse_commands(section, "base")
    ]
    assert got == {i.name: list(i.argv) for i in expected}
    assert all("<D>" not in a for args in got.values() for a in args)
    assert got["train"][-1] == f"{DATA_ROOT}\\runs\\a1r-20261112\\ckpt"


def test_prose_constants_match_the_registered_text(section: str) -> None:
    assert f"`--seed {A1R_SEED}`" in section and str(REGISTERED_A1R_SEED) == A1R_SEED
    assert f"`{DATA_ROOT}\\runs\\<card>`" in section


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.replace("<!-- a1-commands:base -->", "", 1),
        lambda s: s.replace("# primary\n", "", 1),
        lambda s: s.replace("--n-games 200", "--n-games '200'", 1),
        lambda s: _with_branch(s, "50"),
    ],
    ids=["no-marker", "unpaired", "quoted", "bad-branch"],
)
def test_the_parser_fails_closed(section: str, mutate: object) -> None:
    with pytest.raises(RegistrationError):
        registered_invocations(mutate(section), card_id="x", seed=1)  # type: ignore[operator]
