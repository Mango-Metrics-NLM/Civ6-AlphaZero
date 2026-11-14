"""The release verifier holds a run to the registered devices (section 7.1 "Devices").

``run.json``'s ``device_plan`` records the torch build, the device order and each
GPU's name as the pinned environment reported them. Section 7.1 registers the same
three things, so a run on other hardware or another build is honest evidence that
does not count: VOID, the route an ``MCTS_*`` override takes. A ``run_card
--probe`` record launched nothing and is never an attempt.

Kept apart from ``test_verify_release_evidence.py`` so neither file grows past the
size the debt ratchet holds test modules to.

Negative controls (recorded when these were written, then reverted):
  * removing the ``check_registered_devices`` call from the verifier made every
    test in ``test_a_run_on_other_hardware_or_another_build_is_void`` fail -- a
    run on the wrong GPU, torch build or device order verified as a PASS;
  * making ``device_plan_problems`` return ``[]`` did the same;
  * disabling the ``probe`` check made ``test_a_probe_record_is_not_an_attempt`` fail.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import verify_release_evidence as vre
from tests.scripts import release_evidence_fixture as fx
from tests.scripts.test_verify_release_evidence import verify


def test_a_run_on_the_registered_devices_passes(tmp_path: Path) -> None:
    release = fx.make_release(tmp_path, [fx.RunSpec()])
    path = release.bundle_dir / release.bundle["runs"][0]["run_json"]["path"]
    plan = json.loads(path.read_text(encoding="utf-8"))["device_plan"]
    assert plan["torch_pin"]["spec"].startswith("torch=="), "the fixture wrote no device plan"
    code, out = verify(release)
    assert code == vre.EXIT_PASS, out


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda p: {**p, "torch": {**p["torch"], "devices": p["torch"]["devices"][::-1]}},
            "cuda:0 is",
        ),
        (
            lambda p: {**p, "torch": {**p["torch"], "version": "2.13.0+cu130"}},
            "device_plan.torch.version '2.13.0+cu130' is not the registered",
        ),
        (
            # The registered release from another CUDA index: same public version.
            lambda p: {
                **p,
                "torch": {**p["torch"], "version": p["torch"]["version"].split("+")[0] + "+cu130"},
            },
            "+cu130' is not the registered",
        ),
        (lambda p: {**p, "CUDA_DEVICE_ORDER": None}, "device_plan.CUDA_DEVICE_ORDER None"),
        (lambda p: {**p, "torch_pin": None}, "device_plan.torch_pin None"),
        (lambda p: None, "records no device_plan"),
    ],
)
def test_a_run_on_other_hardware_or_another_build_is_void(
    tmp_path: Path, edit: Any, message: str
) -> None:
    release = fx.make_release(tmp_path, [fx.RunSpec(edit_device_plan=edit)])
    code, out = verify(release)
    assert code == vre.EXIT_FAIL
    assert message in out


def test_a_probe_record_is_not_an_attempt(tmp_path: Path) -> None:
    release = fx.make_release(tmp_path, [fx.RunSpec(probe=True)])
    code, out = verify(release)
    assert code == vre.EXIT_FAIL
    assert "is a run_card --probe rehearsal, not an attempt" in out
