"""Golden fingerprint of the default MicroCiv game (GM-S0a).

Later refactors of ``alpha_zero/envs/microciv.py`` must prove that the default
game did not move. Each case plays one seeded game on mask-legal actions only
and hashes the INTEGER game state (units, cities, turn, player, winner) at every
step, plus the full action sequence. No float tensor or observation enters the
digest, so the numbers do not depend on torch or on float formatting.

The fixture is named after ``ENV_VERSION``. A change of game rules must bump
``ENV_VERSION`` and regenerate the fixture under a NEW file name; a digest that
moves with the version unchanged is a behaviour change nobody declared.

Regenerate deliberately (never during a test run)::

    python tests/regression/test_default_env_golden_regression.py --regenerate

A fixture is written once per ``ENV_VERSION``: ``--regenerate`` refuses when the
file for the current version exists, unless numpy's Generator stream itself moved
(the stored canary differs from the recomputed one). Keep the old files.

Not covered, on purpose: rewards, observations and the encoder (the digest is
integer game state only), so a change to a reward constant moves nothing here.
Probability thresholds (terrain ratios, the settler auto-found chance) are
covered only by sampling: a change is seen when one of the seeded draws falls
between the old and the new threshold. The 0.10 -> 0.15 auto-found control below
records that these cases do see that one.

Every case is its own trajectory on its own map: no seed is shared between two
cases, because a 60-turn game replays the 30-turn game of the same seed for the
whole of the shorter game.

Field types (read from ``microciv.py``): every Unit field is int/str/tuple of
int, every City field likewise; ``Unit.health`` is an int (combat uses integer
damage). ``_snapshot`` asserts ``type(v) is int`` for each numeric field, so a
field that ever becomes a float fails loudly instead of being cast.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alpha_zero.envs import microciv
from alpha_zero.envs.microciv import ENV_VERSION, MicroCivEnv

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "golden"
FIXTURE_PATH = FIXTURE_DIR / f"default_env_v{ENV_VERSION}.json"

BOARD_SIZE = 8
# (max_turns, num_players) -> seeds. No seed appears twice, so the 20 cases are 20
# trajectories on 20 maps. Seeds 17, 36, 9, 20 (2 players) and 5, 26, 21, 31
# (3 players) are games won by a player other than player 0; 13, 21 and 34 end
# by elimination before the turn limit.
SEEDS: dict[tuple[int, int], tuple[int, ...]] = {
    (30, 2): (1, 2, 3, 17, 36),
    (30, 3): (5, 7, 8, 11, 26),
    (60, 2): (4, 6, 9, 13, 20),
    (60, 3): (10, 12, 21, 31, 34),
}

CASES = [(BOARD_SIZE, t, n, s) for (t, n), seeds in SEEDS.items() for s in seeds]

CANARY_SEED = 12345
CANARY_DRAWS = 64


def case_id(board: int, turns: int, players: int, seed: int) -> str:
    return f"b{board}_t{turns}_p{players}_s{seed}"


def _int(value: Any, what: str) -> int:
    if type(value) is not int:  # bool and float both rejected on purpose
        raise TypeError(f"{what} is {type(value).__name__}, not int: {value!r}")
    return value


def _snapshot(env: MicroCivEnv) -> dict[str, Any]:
    """Integer-only state. Raises if a numeric field is not an exact int."""
    units = []
    for u in sorted(env.units, key=lambda u: u.id):
        units.append(
            [
                _int(u.id, "unit.id"),
                _int(u.player, "unit.player"),
                u.unit_type,
                [_int(u.position[0], "unit.x"), _int(u.position[1], "unit.y")],
                _int(u.health, "unit.health"),
                _int(u.moves_remaining, "unit.moves_remaining"),
                _int(u.strength, "unit.strength"),
            ]
        )
    cities = []
    for c in sorted(env.cities, key=lambda c: c.id):
        cities.append(
            [
                _int(c.id, "city.id"),
                _int(c.player, "city.player"),
                [_int(c.position[0], "city.x"), _int(c.position[1], "city.y")],
                _int(c.population, "city.population"),
                _int(c.production, "city.production"),
                c.production_queue,
                _int(c.defense, "city.defense"),
            ]
        )
    return {
        "visited": len(env.visited),
        "turn": _int(env.turn, "turn"),
        "player": _int(env.current_player, "current_player"),
        "done": bool(env.done),
        "winner": None if env.winner is None else _int(env.winner, "winner"),
        "units": units,
        "cities": cities,
    }


def _terrain_string(env: MicroCivEnv) -> str:
    return "".join(
        env.terrain[(x, y)].terrain_type[0]
        for y in range(env.board_size)
        for x in range(env.board_size)
    )


def play_case(board: int, turns: int, players: int, seed: int) -> dict[str, Any]:
    """Play one seeded game; return the digest input and summary facts."""
    env = MicroCivEnv(board_size=board, max_turns=turns, num_players=players, seed=seed)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    states = [_snapshot(env)]
    actions: list[int] = []
    kinds: list[str] = []  # decoded action types, for the vacuity guards; not hashed
    rejections = 0
    done = False
    while not done:
        legal = np.flatnonzero(env.get_valid_actions() > 0)
        action = int(rng.choice(legal))
        _, _, terminated, truncated, info = env.step(action)
        actions.append(action)
        kinds.append(env.action_space_handler.decode_action(action)["type"])
        if not info["action_success"]:
            rejections += 1
        states.append(_snapshot(env))
        done = terminated or truncated
    return {
        "terrain": _terrain_string(env),
        "states": states,
        "actions": actions,
        "kinds": kinds,
        "rejections": rejections,
    }


def numpy_stream_canary() -> str:
    """Digest of the Generator draws the env and the driver use, from a fixed seed.

    The numpy version string differs between machines far more often than the
    stream does, so the diagnosis reads this, not the version.
    """
    rng = np.random.default_rng(CANARY_SEED)
    draws: list[Any] = [float(x).hex() for x in rng.random(CANARY_DRAWS)]
    draws += [int(x) for x in rng.integers(0, 3, CANARY_DRAWS)]
    pool = np.arange(376)
    draws += [int(rng.choice(pool)) for _ in range(CANARY_DRAWS)]
    return hashlib.sha256(json.dumps(draws).encode()).hexdigest()


def play_facts(record: dict[str, Any], max_turns: int) -> dict[str, Any]:
    """What happened in a game, derived from the recorded states and actions."""
    states, kinds = record["states"], record["kinds"]
    attacks = combat_kills = auto_founds = 0
    produced: set[str] = set()
    for before, after, kind in zip(states[:-1], states[1:], kinds, strict=True):
        if kind == "attack":
            attacks += 1
            combat_kills += max(0, len(before["units"]) - len(after["units"]))
        if len(after["cities"]) > len(before["cities"]) and kind != "found_city":
            auto_founds += len(after["cities"]) - len(before["cities"])
        known = {u[0] for u in before["units"]}
        produced |= {u[2] for u in after["units"] if u[0] not in known}
    return {
        "attacks": attacks,
        "combat_kills": combat_kills,
        "auto_founds": auto_founds,
        "produced": produced,
        "ended_early": states[-1]["turn"] < max_turns,
        "winner": states[-1]["winner"],
    }


def digest_of(record: dict[str, Any]) -> str:
    payload = {k: record[k] for k in ("terrain", "states", "actions")}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _summary(record: dict[str, Any], players: int) -> dict[str, Any]:
    final = record["states"][-1]
    cities = [sum(1 for c in final["cities"] if c[1] == p) for p in range(players)]
    return {
        "digest": digest_of(record),
        "steps": len(record["actions"]),
        "winner": final["winner"],
        "final_turn": final["turn"],
        "final_cities": cities,
    }


def build_fixture() -> dict[str, Any]:
    cases = {}
    for board, turns, players, seed in CASES:
        rec = play_case(board, turns, players, seed)
        cases[case_id(board, turns, players, seed)] = _summary(rec, players)
    return {
        "env_version": ENV_VERSION,
        "numpy_version": np.__version__,
        "numpy_stream_canary": numpy_stream_canary(),
        "cases": cases,
    }


def _load_fixture() -> dict[str, Any]:
    if not FIXTURE_PATH.exists():
        present = sorted(p.name for p in FIXTURE_DIR.glob("default_env_v*.json"))
        pytest.fail(
            f"ENV_VERSION is now {ENV_VERSION} and {FIXTURE_PATH.name} does not exist; "
            f"fixtures present: {present or 'none'}. "
            + ("ENV_VERSION moved: " if present else "")
            + "regenerate under the new name and keep the old file: python "
            "tests/regression/test_default_env_golden_regression.py --regenerate"
        )
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _walk_no_float(node: Any, path: str = "$") -> None:
    if isinstance(node, float):
        raise AssertionError(f"float at {path}: {node!r}")
    if isinstance(node, dict):
        for k, v in node.items():
            _walk_no_float(v, f"{path}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            _walk_no_float(v, f"{path}[{i}]")


@pytest.fixture(scope="module")
def played() -> dict[tuple[int, int, int, int], dict[str, Any]]:
    return {c: play_case(*c) for c in CASES}


def _explain(fixture: dict[str, Any], case: str) -> str:
    if fixture["env_version"] != ENV_VERSION:
        return (
            f"ENV_VERSION moved ({fixture['env_version']} -> {ENV_VERSION}); "
            "regenerate the fixture under a new file name"
        )
    was, now = fixture["numpy_version"], np.__version__
    if fixture["numpy_stream_canary"] != numpy_stream_canary():
        return (
            f"numpy's Generator stream moved (numpy {was} -> {now}) while ENV_VERSION did "
            "not: the seeded draws changed, so this is not evidence of an env change"
        )
    if was != now:
        return (
            f"ENV_VERSION did not move; the numpy version moved ({was} -> {now}) but its "
            "Generator stream did not (canary equal), so numpy is not the cause: the "
            f"environment's behaviour changed without a version bump (case {case}). "
            "Bump ENV_VERSION if intended."
        )
    return (
        "neither ENV_VERSION nor numpy moved: the environment's behaviour changed "
        f"without a version bump (case {case}). Bump ENV_VERSION if intended."
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: case_id(*c))
def test_digest_matches_fixture(case: tuple[int, int, int, int], played: dict) -> None:
    fixture = _load_fixture()
    cid = case_id(*case)
    assert cid in fixture["cases"], f"case {cid} missing from fixture"
    want = fixture["cases"][cid]
    got = _summary(played[case], case[2])
    assert got["digest"] == want["digest"], (
        f"golden digest differs for {cid}: {_explain(fixture, cid)}. "
        f"fixture facts={ {k: v for k, v in want.items() if k != 'digest'} } "
        f"now={ {k: v for k, v in got.items() if k != 'digest'} }"
    )
    assert got == want


def test_fixture_env_version_is_current() -> None:
    fixture = _load_fixture()
    assert fixture["env_version"] == ENV_VERSION, (
        f"fixture records ENV_VERSION {fixture['env_version']} but the env is {ENV_VERSION}: "
        f"regenerate the fixture under a NEW file name (default_env_v{ENV_VERSION}.json), "
        "do not edit the old one"
    )
    assert FIXTURE_PATH.name == f"default_env_v{fixture['env_version']}.json"


def test_fixture_covers_exactly_the_cases() -> None:
    assert set(_load_fixture()["cases"]) == {case_id(*c) for c in CASES}


def test_vacuity_guards(played: dict) -> None:
    fixture = _load_fixture()
    winners = []
    for case, rec in played.items():
        assert len(rec["actions"]) > 0, case
        assert rec["states"][-1]["done"] is True, f"{case} did not end"
        assert rec["rejections"] == 0, f"{case}: {rec['rejections']} executor rejections"
        winners.append(rec["states"][-1]["winner"])
    assert any(w is not None for w in winners), "every game is a draw: fingerprint is weak"
    assert any(w is None for w in winners), "no game is a draw"
    assert len({d["digest"] for d in fixture["cases"].values()}) == len(CASES)
    assert any(rec["states"][-1]["cities"] for rec in played.values())

    # A fingerprint that never sees combat, production, elimination, an auto-found
    # city or a win by a later player cannot see a change to any of them.
    facts = {case: play_facts(rec, case[1]) for case, rec in played.items()}
    assert sum(f["attacks"] for f in facts.values()) > 0, "no attack was ever played"
    assert sum(f["combat_kills"] for f in facts.values()) > 0, "no unit died in combat"
    assert sum(f["auto_founds"] for f in facts.values()) > 0, "no settler auto-founded"
    produced = set().union(*(f["produced"] for f in facts.values()))
    assert produced >= {"warrior", "settler", "archer"}, f"only {produced} were produced"
    assert any(f["ended_early"] for f in facts.values()), "no game ended before max_turns"
    for players in (2, 3):
        later = {f["winner"] for case, f in facts.items() if case[2] == players} - {None, 0}
        assert later, f"no {players}-player game is won by a player other than player 0"
    assert any(f["ended_early"] and f["winner"] not in (None, 0) for f in facts.values()), (
        "no elimination is won by a player other than player 0"
    )
    assert any(not f["ended_early"] and f["winner"] not in (None, 0) for f in facts.values()), (
        "no turn-limit game is won by a player other than player 0"
    )


def test_no_seed_is_shared_between_cases() -> None:
    seeds = [c[3] for c in CASES]
    assert len(set(seeds)) == len(seeds)


def test_the_diagnosis_names_what_moved(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _load_fixture()
    assert "neither ENV_VERSION nor numpy moved" in _explain(fixture, "c")
    bumped = {**fixture, "env_version": ENV_VERSION + 1}
    assert _explain(bumped, "c").startswith("ENV_VERSION moved")
    stream = {**fixture, "numpy_stream_canary": "0" * 64}
    assert "Generator stream moved" in _explain(stream, "c")
    assert "not evidence of an env change" in _explain(stream, "c")
    # A different numpy version with the same stream (CI, another workstation)
    # must not take the blame for an env change.
    monkeypatch.setattr(np, "__version__", "0.0.0-other")
    text = _explain(fixture, "c")
    assert "numpy version moved" in text and "numpy is not the cause" in text


def test_negative_control_a_perturbed_action_moves_the_digest(played: dict) -> None:
    fixture = _load_fixture()
    case = CASES[0]
    record = dict(played[case])
    record["actions"] = [*record["actions"][:-1], record["actions"][-1] + 1]
    assert digest_of(record) != fixture["cases"][case_id(*case)]["digest"]


def test_no_float_enters_the_digest_input(played: dict) -> None:
    for case, rec in played.items():
        _walk_no_float({k: rec[k] for k in ("terrain", "states", "actions")}, case_id(*case))


def test_the_float_walker_detects_a_float() -> None:
    with pytest.raises(AssertionError):
        _walk_no_float({"a": [1, {"b": 2.0}]})
    with pytest.raises(TypeError):
        _int(2.0, "x")


def test_same_game_twice_gives_same_digest() -> None:
    case = (BOARD_SIZE, 30, 2, 1)
    assert digest_of(play_case(*case)) == digest_of(play_case(*case))


def test_clone_step_matches_parent_step() -> None:
    env = MicroCivEnv(board_size=BOARD_SIZE, max_turns=30, num_players=3, seed=2)
    env.reset(seed=2)
    rng = np.random.default_rng(2)
    checked = 0
    for _ in range(80):
        if env.done:
            break
        action = int(rng.choice(np.flatnonzero(env.get_valid_actions() > 0)))
        twin = env.clone()
        twin.step(action)
        env.step(action)
        assert _snapshot(twin) == _snapshot(env)
        checked += 1
    assert checked > 0


def _moved_cases(fixture: dict[str, Any]) -> list[tuple[int, int, int, int]]:
    return [
        c
        for c in CASES
        if _summary(play_case(*c), c[2])["digest"] != fixture["cases"][case_id(*c)]["digest"]
    ]


def test_negative_control_a_changed_rule_moves_the_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Change a game rule in-process (no file under alpha_zero/ edited): the fixture must notice.

    A warrior costs one more than the shipped cost, so every game that
    completes a warrior does so later and diverges from there.
    """
    fixture = _load_fixture()
    costs = dict(MicroCivEnv.UNIT_COSTS)
    costs["warrior"] += 1
    monkeypatch.setattr(MicroCivEnv, "UNIT_COSTS", costs)
    moved = _moved_cases(fixture)
    assert moved, "a changed unit cost did not move any golden digest: the fingerprint is blind"


def test_negative_control_a_changed_probability_moves_the_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settler auto-found chance 0.10 -> 0.15: seen only if a seeded draw lands between.

    With one seed shared by the 30- and 60-turn cases and three seeds in all,
    this change moved 0 of 12 digests (review finding); these cases must see it.
    ``GAME`` is a frozen dataclass, so the module name is rebound, not the field.
    """
    fixture = _load_fixture()
    assert microciv.GAME.SETTLER_AUTO_FOUND_PROBABILITY == 0.10
    monkeypatch.setattr(
        microciv, "GAME", dataclasses.replace(microciv.GAME, SETTLER_AUTO_FOUND_PROBABILITY=0.15)
    )
    moved = _moved_cases(fixture)
    assert moved, "auto-found probability 0.10 -> 0.15 moved no golden digest"


if __name__ == "__main__":
    if "--regenerate" not in sys.argv:
        sys.exit("refusing to run without --regenerate (the fixture is never rewritten silently)")
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    if FIXTURE_PATH.exists():
        stored = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        if stored.get("numpy_stream_canary") == numpy_stream_canary():
            sys.exit(
                f"{FIXTURE_PATH.name} already exists and numpy's Generator stream has not "
                "moved: a fixture is written once per ENV_VERSION and never overwritten; "
                "bump ENV_VERSION for a rule change"
            )
        print(f"numpy's Generator stream moved: rewriting {FIXTURE_PATH.name}")
    with FIXTURE_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(build_fixture(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {FIXTURE_PATH}")
