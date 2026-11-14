"""T0b: the card that measures the positive-control reference, and the command that reads it.

Section 7.1 (``docs/EVALUATION.md``) leaves two slots "pending T0b": the measured
heuristic-vs-random reference (with P(40), its 95% interval, P(60) and the iid
P(40)) and the 40/60 branch in force. The final review (I-3) found no card that
runs T0b and no command that computes those numbers, so the slots would have been
filled from an ad-hoc snippet. These tests hold:

* ``configs/run_cards/t0b.yaml`` to section 7.1's registered positive-control
  command (only the seed, the game count and ``--out`` differ), at seeds
  1001-1003 x 200 games, plus one wall-clock calibration, priced under the cap;
* ``python -m alpha_zero.evaluation.a1_protocol calibrate`` to the registered
  method: pairs pooled in seed order, the branch read from the upper bound of
  P(40)'s interval, the exact slot text, and a non-zero exit for anything that is
  not the registered reference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from alpha_zero.evaluation import a1_protocol as ap
from alpha_zero.evaluation.a1_registration import command_block, flag_value
from alpha_zero.local import run_card as rc

REPO_ROOT = Path(__file__).resolve().parents[2]
CARD = REPO_ROOT / "configs" / "run_cards" / "t0b.yaml"
EVALUATION = (REPO_ROOT / "docs" / "EVALUATION.md").read_text(encoding="utf-8")
QUICK = {"reps": 200, "outer": 4, "inner": 50, "seed": 7}


def _games(pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    return [
        {"pair": p, "seat": seat, "result": result}
        for p, pair in enumerate(pairs)
        for seat, result in enumerate(pair)
    ]


def _envelope(seed: int, pairs: list[tuple[str, str]], *, board: int = 8) -> dict[str, Any]:
    return {
        "protocol": {"board_size": board, "max_turns": 60},
        "candidate": {"kind": "heuristic", "id": "heuristic"},
        "shard": None,
        "evidence": {"seed": seed},
        "results": [{"opponent_id": "random", "games": _games(pairs)}],
    }


STRONG = [("win", "win"), ("win", "draw"), ("win", "win"), ("draw", "win")] * 25  # 100 pairs
WEAK = [("win", "loss"), ("draw", "draw"), ("win", "draw"), ("loss", "win")] * 25


def _registered_files(pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    return [_envelope(s, pairs) for s in ap.T0B_SEEDS]


# ---------------------------------------------------------------------------
# The card
# ---------------------------------------------------------------------------


def test_the_slots_the_command_prints_are_verbatim_in_section_7_1() -> None:
    section = ap.registered_protocol(EVALUATION)
    assert section.count(ap.REFERENCE_SLOT) == 1
    assert section.count(ap.BRANCH_SLOT) == 1


def test_t0b_card_prices_under_the_cap_per_invocation_and_in_total() -> None:
    priced = rc.price_card(rc.load_card(CARD))
    games = {p.invocation.name: p.games for p in priced.invocations}
    assert games == {
        "throughput": 4,
        "throughput-cuda1": 4,
        "throughput-cpu": 4,
        "seed-1001": 200,
        "seed-1002": 200,
        "seed-1003": 200,
    }
    assert priced.total == 612 <= priced.card.max_games <= 1000


def test_t0b_arms_are_the_registered_positive_control_but_for_seed_games_and_out() -> None:
    """Same recipe as the A1 positive control, so the reference measures that arm."""
    section = ap.registered_protocol(EVALUATION)
    registered = dict(command_block(section, "base"))["positive-control"][2:]
    card = rc.load_card(CARD)
    seen = []
    for inv in card.invocations:
        if inv.runner != "benchmark":
            continue
        argv = list(inv.argv)
        seed = int(flag_value(argv, "--seed") or 0)
        seen.append(seed)
        assert flag_value(argv, "--n-games") == str(ap.T0B_GAMES)
        assert (
            flag_value(argv, "--out")
            == f"E:\\civ6az-data\\runs\\{card.card_id}\\arena\\seed_{seed}"
        )
        expected = [
            {"--seed": str(seed), "--n-games": str(ap.T0B_GAMES)}.get(prev, tok)
            for prev, tok in zip(["", *registered], registered, strict=False)
        ]
        expected[expected.index("--out") + 1] = flag_value(argv, "--out") or ""
        assert argv == expected
    assert tuple(seen) == ap.T0B_SEEDS


def test_t0b_wall_clock_calibration_is_a1s_search_on_a1s_training_device() -> None:
    """``throughput`` is the file ``calibrate --throughput`` reads, so it must time
    A1's search on the device A1 trains and self-plays on (section 7.1)."""
    section = ap.registered_protocol(EVALUATION)
    train_device = flag_value(dict(command_block(section, "base"))["train"], "--device")
    assert train_device == "cuda:0"
    card = rc.load_card(CARD)
    by_name = {i.name: list(i.argv) for i in card.invocations if i.runner == "calibrate_throughput"}
    argv = by_name["throughput"]
    for flag, value in (
        ("--board_size", "8"),
        ("--max_turns", "60"),
        ("--mcts", "gumbel"),
        ("--k", "8"),
        ("--n", "24"),
        ("--device", train_device),
    ):
        assert flag_value(argv, flag) == value, flag


def test_t0b_comparison_timings_differ_from_the_registered_one_only_in_device_and_out() -> None:
    """cuda:1 and cpu are timed with the same search, so the three numbers compare."""
    card = rc.load_card(CARD)
    by_name = {i.name: list(i.argv) for i in card.invocations if i.runner == "calibrate_throughput"}
    assert set(by_name) == {"throughput", "throughput-cuda1", "throughput-cpu"}
    base = by_name["throughput"]
    outs = {flag_value(base, "--out")}
    for name, device in (("throughput-cuda1", "cuda:1"), ("throughput-cpu", "cpu")):
        argv = by_name[name]
        assert flag_value(argv, "--device") == device
        out = flag_value(argv, "--out")
        assert out not in outs, "each timing needs its own file"
        outs.add(out)
        swapped = [
            {"--device": flag_value(base, "--device"), "--out": flag_value(base, "--out")}.get(
                prev, tok
            )
            for prev, tok in zip(["", *argv], argv, strict=False)
        ]
        assert swapped == base, name


def test_cards_pin_a_cuda_torch_and_both_gpus() -> None:
    """Every registered card names its torch build and makes cuda:0 and cuda:1 visible."""
    for path in (CARD, CARD.with_name("a1.yaml"), CARD.with_name("a1r.yaml")):
        card = rc.load_card(path)
        assert card.torch_pin is not None, path.name
        assert card.torch_pin.index_url.startswith("https://download.pytorch.org/whl/cu")
        assert card.cuda_visible_devices == "0,1", path.name
        devices = set(rc.requested_devices(card).values())
        assert {"cuda:0", "cuda:1"} <= devices, path.name


def test_calibrate_throughput_is_a_run_card_runner_priced_by_its_own_parser() -> None:
    inv = rc.Invocation("t", "calibrate_throughput", ("--games", "3"))
    assert rc.price_invocation(inv) == 3
    with pytest.raises(rc.game_budget.GameBudgetExceededError):
        rc.price_invocation(rc.Invocation("t", "calibrate_throughput", ("--games", "1001")))


# ---------------------------------------------------------------------------
# The calibration
# ---------------------------------------------------------------------------


def test_a_strong_reference_selects_branch_40() -> None:
    ref = ap.calibrate_t0b(_registered_files(STRONG), **QUICK)
    assert (ref.wins, ref.draws, ref.losses) == (450, 150, 0)
    assert ref.pairs == 300 and ref.seeds == [1001, 1002, 1003]
    assert ref.p40_interval[1] <= ap.BRANCH_VOID_RISK
    assert ref.branch == 40
    assert not ref.shape_problems
    assert not ref.registered  # QUICK is not the registered bootstrap


def test_a_weak_reference_selects_branch_60() -> None:
    ref = ap.calibrate_t0b(_registered_files(WEAK), **QUICK)
    assert ref.p40_interval[1] > ap.BRANCH_VOID_RISK
    assert ref.branch == 60


def test_pairs_are_pooled_in_seed_order_whatever_the_file_order() -> None:
    files = [_envelope(1003, WEAK), _envelope(1001, STRONG), _envelope(1002, WEAK)]
    forward = ap.calibrate_t0b(files, **QUICK)
    shuffled = ap.calibrate_t0b(list(reversed(files)), **QUICK)
    assert forward == shuffled
    assert forward.seeds == [1001, 1002, 1003]


def test_the_wrong_shape_is_computed_but_flagged() -> None:
    files = [_envelope(1001, STRONG), _envelope(1002, STRONG[:50], board=6)]
    ref = ap.calibrate_t0b(files, **QUICK)
    joined = "; ".join(ref.shape_problems)
    assert "seeds [1001, 1002]" in joined
    assert "not 200 each" in joined
    assert "board 6" in joined


def test_not_a_heuristic_vs_random_file_is_an_error() -> None:
    envelope = _envelope(1001, STRONG)
    envelope["candidate"]["id"] = "checkpoint:abc"
    with pytest.raises(ValueError, match="not a heuristic-vs-random"):
        ap.calibrate_t0b([envelope], **QUICK)
    with pytest.raises(ValueError, match="appears twice"):
        ap.calibrate_t0b([_envelope(1001, STRONG), _envelope(1001, STRONG)], **QUICK)


def test_slot_text_states_every_number_section_7_1_asks_for() -> None:
    ref = ap.calibrate_t0b(_registered_files(STRONG), **QUICK)
    reference, branch = ap.t0b_slot_text(ref)
    assert branch == "**Branch in force: 40.**"
    for needle in (
        "W/D/L 450/150/0",
        "score 0.8750",
        "95% hull [",
        "300 pooled pairs",
        f"P(40) = {ref.p40:.4f}",
        "95% interval [",
        f"P(60) = {ref.p60:.4f}",
        f"iid P(40) = {ref.iid_p40:.4f}",
        "seeds 1001, 1002, 1003 x 200 games",
    ):
        assert needle in reference, needle
    # The filled branch slot is what a1_registration reads.
    from alpha_zero.evaluation.a1_registration import branch_in_force

    filled = ap.registered_protocol(EVALUATION).replace(ap.BRANCH_SLOT, branch)
    assert branch_in_force(filled) == 40


def test_cli_prints_the_slots_and_refuses_a_quick_look(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dirs = []
    for envelope in _registered_files(STRONG):
        d = tmp_path / f"seed_{envelope['evidence']['seed']}"
        d.mkdir()
        (d / "result.json").write_text(json.dumps(envelope), encoding="utf-8")
        dirs.append(str(d))
    throughput = tmp_path / "throughput.json"
    throughput.write_text(json.dumps({"results": {"s_per_game": 36.0, "ms_per_eval": 5.0}}))
    quick = ["--reps", "200", "--outer", "4", "--inner", "50"]
    code = ap.main(["calibrate", *dirs, *quick, "--throughput", str(throughput)])
    out = capsys.readouterr().out
    assert code == 1, "non-registered bootstrap parameters must not exit 0"
    assert "NOT the registered T0b reference" in out
    assert ap.REFERENCE_SLOT in out and ap.BRANCH_SLOT in out
    assert "**Branch in force: 40.**" in out
    # 940 search games at 36 s.
    assert "9.4 h" in out
