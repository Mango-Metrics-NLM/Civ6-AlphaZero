"""``--checkpoint_dir`` / ``--run_id`` and ``CIV6AZ_DATA_ROOT`` in ``training_civ6`` (LF-E4).

Negative controls performed when this file was written (each reverted):

* the ``CIV6AZ_DATA_ROOT`` branch removed from ``resolve_checkpoint_dir`` ->
  ``test_data_root_sets_default_checkpoint_dir`` fails;
* ``validate_run_id`` call removed -> ``test_bad_run_id_is_rejected`` fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from alpha_zero import training_civ6
from alpha_zero.training_civ6 import TrainingConfig
from alpha_zero.utils import data_root as dr


@pytest.fixture(autouse=True)
def _no_real_root_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(dr.ENV_VAR, raising=False)
    # Safety validation is covered in test_data_root.py; never probe real volumes here.
    monkeypatch.setattr(training_civ6, "assert_safe_data_root", lambda root, repo: None)


def _run_main(argv: list[str]) -> TrainingConfig:
    captured: dict[str, Any] = {}

    def fake_train(config: TrainingConfig, dry_run: bool = False) -> None:
        captured["config"] = config

    with (
        patch.object(training_civ6, "train", fake_train),
        patch("sys.argv", ["training_civ6", "--approval_ref", "unit-test-train-is-faked", *argv]),
    ):
        training_civ6.main()
    return captured["config"]  # type: ignore[no-any-return]


def test_default_is_unchanged_without_data_root() -> None:
    assert _run_main([]).checkpoint_dir == "checkpoints/civ6"


def test_explicit_checkpoint_dir_wins_over_data_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(dr.ENV_VAR, str(tmp_path))
    assert _run_main(["--checkpoint_dir", "somewhere"]).checkpoint_dir == "somewhere"


def test_data_root_sets_default_checkpoint_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(dr.ENV_VAR, str(tmp_path))
    got = _run_main(["--run_id", "myrun"]).checkpoint_dir
    assert Path(got) == tmp_path.resolve() / "runs" / "myrun" / "checkpoints"


def test_generated_run_id_when_not_given(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(dr.ENV_VAR, str(tmp_path))
    got = Path(_run_main(["--seed", "7"]).checkpoint_dir)
    assert got.parent.parent == tmp_path.resolve() / "runs"
    assert got.parent.name.startswith("train-7-")


def test_main_runs_the_real_safety_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the autouse stub: a data root inside the repo must abort ``main``.

    Negative control: deleting the ``assert_safe_data_root`` call in
    ``resolve_checkpoint_dir`` makes this fail.
    """
    monkeypatch.undo()  # drop the autouse stub (and env cleanup, re-applied below)
    monkeypatch.delenv(dr.ENV_VAR, raising=False)
    repo = Path(training_civ6.__file__).resolve().parents[1]
    monkeypatch.setenv(dr.ENV_VAR, str(repo / "scratch-data-root"))
    monkeypatch.setattr(dr, "_is_windows", lambda: True)
    monkeypatch.setattr(dr, "_volume_filesystem", lambda path: "NTFS")
    monkeypatch.setattr(dr, "_is_removable_drive", lambda path: False)
    monkeypatch.setattr(dr, "_free_bytes", lambda path: 500 * 1024**3)
    with pytest.raises(dr.DataRootError, match="repo"):
        _run_main([])


def test_bad_run_id_is_rejected() -> None:
    with pytest.raises(dr.DataRootError):
        _run_main(["--run_id", "../escape"])
