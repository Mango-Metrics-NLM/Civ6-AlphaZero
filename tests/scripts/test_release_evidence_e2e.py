"""Real producers -> ``build_release_bundle`` -> ``verify_release_evidence`` (final review I-3).

Every other release-evidence test feeds the verifier hand-built records
(``release_evidence_fixture``). This one feeds it what the producers actually
write, once, at tiny scale:

1. ``alpha_zero.local.run_card.run_card`` runs a card through its real pricing,
   data-root handling, isolation (``enter_local_mode``) and ``run.json`` writer;
2. each invocation runs the real ``alpha_zero.training_civ6`` and
   ``alpha_zero.evaluation.benchmark`` ``main()``, so ``iterations.jsonl``,
   checkpoints (with provenance) and every arm's ``result.json`` (with its
   evidence block) are the producers' own;
3. ``scripts/build_release_bundle.py`` assembles the bundle;
4. ``scripts/verify_release_evidence.py`` must exit 0 on it.

What is replaced, and why (so it is not mistaken for coverage):

* **Pinning.** ``run_card`` normally builds a venv from ``git archive`` of the
  card's SHA and launches each child as a subprocess. That takes minutes, so
  ``prepare_pinned_run``/``_preflight``/``_pip_freeze`` are stubbed and
  ``_run_invocation`` runs the same ``python -m <module> <launched argv>`` in this
  process, under the child environment run_card built. ``tests/scripts/test_run_card.py``
  covers the real pinning and subprocess path.
* **The registration.** Section 7.1's card is 980 games at board 8, so the
  repository under test holds a *test-only* section 7.1 with the same structure
  (arms ``primary``/``symmetry``/``positive-control``/``final-vs-random``, the
  registered seed, ``Branch in force: 40``) at board 4 / 8 turns, 2 iterations,
  2 games per arm and 20 for the positive control (fewer cannot clear its 0.5
  bound, and a VOID-only bundle rightly fails). Its digest is therefore not the registration log's; the digest gate is
  ``tests/regression/test_a1_registration_digest_regression.py``.
* **The data root.** ``<D>`` is ``E:\\civ6az-data\\runs\\<card>`` in the
  registration; here ``a1_registration.run_directory`` points it at the test's
  temporary root, so the paths the verifier reconstructs are the ones that ran.
* **Check runs.** The rc's CI is read through the fake GitHub API the verifier's
  own tests use.

So what this proves: the verifier's schema, registration, recomputation, verdict,
identity and freshness checks all accept real producer output. What it does not:
the pinned-venv launch, a real ``E:`` data root, the real 980-game registration's
digest, and the GitHub API.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import shutil
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import pytest
import yaml

from alpha_zero.evaluation import a1_registration
from alpha_zero.evaluation.a1_protocol import PROTOCOL_BEGIN, PROTOCOL_END
from alpha_zero.local import run_card as rc
from alpha_zero.local.pinned_env import PinnedEnv
from scripts import build_release_bundle as builder
from scripts import verify_release_evidence as vre
from tests.scripts import release_evidence_fixture as fx
from tests.scripts.git_fixture import GitRepo

CARD_ID = "a1-tiny"
SEED = 20261111
BENCH = (
    "python scripts/benchmark_suite.py --opponents= --seed 20261111 --board-size 4 "
    "--max-turns 8 --sims 2 --gumbel-k 2 --mcts-type gumbel --device cpu"
)
COMMANDS = [
    (
        "train",
        "python -m alpha_zero.training_civ6 --seed 20261111 --board_size 4 --max_turns 8 "
        "--num_iterations 2 --num_games 2 --mcts gumbel --gumbel_k 2 --gumbel_sims 2 "
        "--device cpu --bootstrap none --save_interval 1 --eval_games 0 "
        "--checkpoint_dir <D>/ckpt",
    ),
    (
        "primary",
        f"{BENCH} --checkpoint <D>/ckpt/checkpoint_2.pt "
        "--opponent-checkpoint <D>/ckpt/checkpoint_0.pt --n-games 2 --out <D>/arena/primary",
    ),
    (
        "symmetry",
        f"{BENCH} --checkpoint <D>/ckpt/checkpoint_0.pt "
        "--opponent-checkpoint <D>/ckpt/checkpoint_0.pt --n-games 2 --out <D>/arena/symmetry",
    ),
    (
        "positive-control",
        f"{BENCH} --candidate heuristic --opponents=random --n-games 20 "
        "--out <D>/arena/positive_control",
    ),
    (
        "final-vs-random",
        f"{BENCH} --checkpoint <D>/ckpt/checkpoint_2.pt --opponents=random --n-games 2 "
        "--out <D>/arena/final_vs_random",
    ),
]


def tiny_evaluation_md() -> str:
    """A ``docs/EVALUATION.md`` whose section 7.1 registers the tiny card."""
    block = "\n".join(f"# {name}\n{command}" for name, command in COMMANDS)
    return (
        "# Evaluation (test-only registration)\n\n"
        f"{PROTOCOL_BEGIN} test -->\n"
        "### 7.1 A1 (tiny, test only)\n\n"
        "<!-- a1-commands:base -->\n"
        f"```text\n{block}\n```\n\n"
        "- **Branch in force: 40.**\n"
        f"{PROTOCOL_END} -->\n"
    )


@pytest.fixture(autouse=True)
def restore_process_state() -> Iterator[None]:
    """``enter_local_mode`` and the in-process children mutate the environment and argv."""
    environ, argv = os.environ.copy(), list(sys.argv)
    yield
    os.environ.clear()
    os.environ.update(environ)
    sys.argv[:] = argv


def _in_process(cmd: Sequence[str], *, cwd: Path, env: Mapping[str, str], log: Path) -> int:
    """``run_card._run_invocation``, run in this process: ``python -m MODULE ARGV...``."""
    assert list(cmd[1:2]) == ["-m"], cmd
    module, argv = cmd[2], list(cmd[3:])
    saved = os.environ.copy()
    os.environ.clear()
    os.environ.update(env)
    sys.argv[:] = [module, *argv]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            entry = importlib.import_module(module)
            try:
                code = entry.main()
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
        return int(code or 0)
    finally:
        log.write_text(out.getvalue(), encoding="utf-8")
        os.environ.clear()
        os.environ.update(saved)


@pytest.fixture
def released(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[GitRepo, Path]:
    """A repository at a tagged rc whose section 7.1 is the tiny card, and that card, run."""
    root = tmp_path / "root"
    monkeypatch.setattr(
        a1_registration,
        "run_directory",
        lambda card_id, data_root="": os.path.join(str(root.resolve()), "runs", card_id),
    )
    repo = GitRepo(tmp_path / "repo")
    evaluation = tiny_evaluation_md()
    run_sha = repo.commit(
        "code",
        {
            "pyproject.toml": fx.PYPROJECT,
            "docs/EVALUATION.md": evaluation,
            "docs/evidence/README.md": (
                "# Evidence index\n\n| Artifact | Status |\n|---|---|\n"
                "| [`older/`](older/report.md) | current |\n\nProse after the table.\n"
            ),
            "CHANGELOG.md": "# Changelog\n\n## [2.1.0]\n\n- Tiny.\n",
        },
    )
    repo.tag(fx.RC_TAG, run_sha)

    section = evaluation.split("\n", 2)[2]
    invocations = a1_registration.registered_invocations(section, card_id=CARD_ID, seed=SEED)
    card = tmp_path / "card.yaml"
    card.write_text(
        yaml.safe_dump(
            {
                "card_id": CARD_ID,
                "sha": run_sha,
                "max_games": 1000,
                "invocations": [
                    {
                        "name": name,
                        "runner": "training_civ6" if name == "train" else "benchmark",
                        "argv": argv,
                    }
                    for name, argv in invocations.items()
                ],
            }
        ),
        encoding="utf-8",
    )

    venv_python = tmp_path / "venv" / ("Scripts" if os.name == "nt" else "bin") / "python"
    pinned = PinnedEnv(python=venv_python, src=fx.REPO_ROOT, sha=run_sha, wheel_sha256="0" * 64)
    monkeypatch.setattr(rc, "prepare_pinned_run", lambda *a, **k: pinned)
    monkeypatch.setattr(rc, "_preflight", lambda env, child: {"exit_code": 0})
    monkeypatch.setattr(rc, "verify_preflight", lambda env, report: None)
    monkeypatch.setattr(rc, "_pip_freeze", lambda env, child, runner: [])
    monkeypatch.setattr(rc, "_run_invocation", _in_process)
    monkeypatch.setattr(
        rc,
        "harden_data_root",
        lambda r, **_: [(Path(r) / s).mkdir(parents=True, exist_ok=True) for s in rc.SUBDIRS],
    )

    result = rc.run_card(card, data_root=root, repo_root=repo.path)
    logs = {
        p.name: p.read_text(encoding="utf-8")[-2000:] for p in (result.run_dir / "logs").glob("*")
    }
    assert result.status == rc.STATUS_COMPLETED, logs
    return repo, result.run_dir


def test_real_producer_output_builds_and_verifies(
    released: tuple[GitRepo, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    repo, run_dir = released
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "COMPLETED"
    assert [i["name"] for i in record["invocations"]] == [n for n, _ in COMMANDS]

    code = builder.main(
        [
            "--release", fx.RELEASE, "--rc", fx.RC_TAG, "--run", f"A1={run_dir}",
            "--repo-root", str(repo.path), "--date", "2026-12-01",
        ]
    )  # fmt: skip
    out = capsys.readouterr()
    assert code == builder.EXIT_BUILT, out.out + out.err
    bundle_dir = repo.path / fx.BUNDLE_DIR
    bundle = json.loads((bundle_dir / vre.BUNDLE_FILE).read_text(encoding="utf-8"))
    (run,) = bundle["runs"]
    assert set(run["arms"]) == {"primary", "symmetry", "positive_control", "final_vs_random"}
    assert set(run["checkpoints"]) == {"ckpt/checkpoint_0.pt", "ckpt/checkpoint_2.pt"}
    assert bundle["learning_claimed"] is False  # no A1-R: the claim cannot be made
    index = (repo.path / "docs" / "evidence" / "README.md").read_text(encoding="utf-8")
    assert "](release_v2_1_0/report.md)" in index
    assert index.index("release_v2_1_0") < index.index("Prose after the table")

    repo.git("add", "-A")
    repo.git("commit", "-q", "-m", "evidence")
    code = vre.main(
        [str(bundle_dir / vre.BUNDLE_FILE), "--repo-root", str(repo.path), "--repo", fx.REPO],
        api=fx.fake_api(bundle["rc_sha"]),
        env={},
    )
    out = capsys.readouterr()
    assert code == vre.EXIT_PASS, out.out + out.err

    # Refuses to overwrite what it wrote.
    again = builder.main(
        ["--release", fx.RELEASE, "--rc", fx.RC_TAG, "--run", f"A1={run_dir}",
         "--repo-root", str(repo.path)]
    )  # fmt: skip
    assert again == builder.EXIT_REFUSED
    assert "never overwritten" in capsys.readouterr().err

    # Negative control: one tampered arm and the build fails and publishes nothing.
    shutil.rmtree(bundle_dir)
    result = run_dir / "arena" / "symmetry" / "result.json"
    envelope = json.loads(result.read_text(encoding="utf-8"))
    envelope["results"][0]["draws"] += 1
    result.write_text(json.dumps(envelope), encoding="utf-8")
    code = builder.main(
        ["--release", fx.RELEASE, "--rc", fx.RC_TAG, "--run", f"A1={run_dir}",
         "--repo-root", str(repo.path)]
    )  # fmt: skip
    assert code == builder.EXIT_FAILED
    assert "arm symmetry" in capsys.readouterr().err
    assert not bundle_dir.exists()
