"""``alpha_zero.local.devices``: a pinned run gets the devices it names, or a reason why not.

These are pure functions over plain data, so every rule is tested without a GPU.
The one thing that needs the real library is :data:`TORCH_REPORT_CODE` -- a code
string run in the pinned child -- which is executed here against a faked two-GPU
``torch.cuda`` so a misspelt API name fails in CI and not on the GPU machine.

Negative controls (recorded when this file was written, then reverted):
  * making ``architecture_supported`` return ``True`` unconditionally made
    ``test_a_build_without_kernels_for_the_gpu_is_refused`` and three rows of
    ``test_architecture_supported`` fail;
  * replacing ``env.setdefault`` with plain assignment in
    ``apply_device_environment`` made
    ``test_an_operators_cublas_setting_survives`` fail;
  * removing the order ``setdefault`` for a CUDA request made
    ``test_a_cuda_request_pins_the_order_even_without_card_visibility`` fail;
  * making ``implicit_device_problem`` return ``None`` made
    ``test_a_pinned_card_may_not_leave_a_device_to_auto_or_a_default`` fail;
  * renaming ``get_arch_list`` in the report code made
    ``test_report_code_runs_against_the_real_torch_api`` fail (the report became
    an ``AttributeError`` record).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from alpha_zero.local import devices
from alpha_zero.local.pinned_env import TorchPin
from alpha_zero.utils.seeding import CUBLAS_WORKSPACE_DETERMINISTIC

_CU128 = "https://download.pytorch.org/whl/cu128"
#: What a cu128 build reports on the target machine: two Blackwell cards.
BLACKWELL = {
    "version": "2.11.0+cu128",
    "cuda": "12.8",
    "cuda_available": True,
    "device_count": 2,
    "devices": ["RTX 5060 Ti", "RTX 5060"],
    "capabilities": [[12, 0], [12, 0]],
    "arch_list": ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"],
}


# --------------------------------------------------------------------------
# A card that pins a torch build names every device
# --------------------------------------------------------------------------


def test_a_pinned_card_may_not_leave_a_device_to_auto_or_a_default() -> None:
    pin = TorchPin("torch==2.11.0", _CU128)
    resolved = {"train": "cuda:0", "eval": "auto", "sweep": None, "timing": "cpu"}
    problem = devices.implicit_device_problem(resolved, torch_pin=pin)
    assert problem is not None and problem.endswith("eval, sweep leave it to a default or 'auto'")
    assert devices.implicit_device_problem({"train": "cuda:0", "t": "cpu"}, torch_pin=pin) is None


def test_an_unpinned_card_is_not_held_to_explicit_devices() -> None:
    assert devices.implicit_device_problem({"smoke": None, "x": "auto"}, torch_pin=None) is None


# --------------------------------------------------------------------------
# The architecture rule
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("capability", "arch_list", "supported"),
    [
        ((12, 0), ["sm_80", "sm_90", "sm_120"], True),
        ((12, 0), ["sm_50", "sm_80", "sm_86", "sm_90"], False),  # Blackwell on a cu12x<8 build
        ((8, 9), ["sm_80", "sm_86", "sm_90"], True),  # an sm_89 card runs sm_80/sm_86 kernels
        ((8, 0), ["sm_86"], False),  # a newer-minor kernel does not run on an older card
        ((7, 5), ["sm_75"], True),
        ((12, 0), [], True),  # no list proves nothing
        ((12, 0), ["compute_120", "sm_x"], True),  # unparseable proves nothing
    ],
)
def test_architecture_supported(
    capability: tuple[int, int], arch_list: list[str], supported: bool
) -> None:
    assert devices.architecture_supported(capability, arch_list) is supported


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------


def test_two_supported_gpus_are_accepted() -> None:
    requested = {"train": "cuda:0", "primary": "cuda:1"}
    pin = TorchPin("torch==2.11.0", _CU128)
    assert devices.device_problem(requested, BLACKWELL, torch_pin=pin, visible="0,1") is None


def test_a_build_without_kernels_for_the_gpu_is_refused() -> None:
    """``is_available()`` is True here; only the architecture list shows the build cannot run."""
    old_build = {**BLACKWELL, "arch_list": ["sm_50", "sm_60", "sm_70", "sm_80", "sm_86", "sm_90"]}
    problem = devices.device_problem({"primary": "cuda:1"}, old_build)
    assert problem is not None
    assert "RTX 5060" in problem and "sm_120" in problem and "kernels only for" in problem


def test_missing_ordinal_message_quotes_the_visibility_in_force() -> None:
    one_gpu = {**BLACKWELL, "device_count": 1}
    problem = devices.device_problem({"primary": "cuda:1"}, one_gpu, visible="0")
    assert problem is not None and "CUDA_VISIBLE_DEVICES='0'" in problem


def test_unavailable_cuda_message_carries_the_childs_own_error_detail() -> None:
    report = {"error": "OSError", "detail": "c10_cuda.dll not found"}
    problem = devices.device_problem({"train": "cuda:0"}, report)
    assert problem is not None and "c10_cuda.dll not found" in problem


@pytest.mark.parametrize("report", [None, "not a mapping", 0, []])
def test_a_missing_or_malformed_report_refuses_cuda_and_allows_cpu(report: Any) -> None:
    assert devices.device_problem({"train": "cuda:0"}, report) is not None
    assert devices.device_problem({"train": "cpu", "eval": "auto"}, report) is None


def test_a_pin_without_a_build_tag_is_checked_on_the_public_version() -> None:
    pin = TorchPin("torch==2.11.0", _CU128)
    assert devices.device_problem({}, {"version": "2.11.0+cu128"}, torch_pin=pin) is None
    assert devices.device_problem({}, {"version": "2.11.0"}, torch_pin=pin) is None
    wrong = devices.device_problem({}, {"version": "2.11.1+cu128"}, torch_pin=pin)
    assert wrong is not None and "pins torch==2.11.0" in wrong


@pytest.mark.parametrize("installed", ["2.11.0+cu130", "2.11.0+cpu", "2.11.0", "2.11.1+cu128", ""])
def test_a_pin_that_names_its_build_refuses_any_other_build(installed: str) -> None:
    """A cached venv whose torch was swapped for another CUDA variant must not run.

    Negative control: comparing only the public version (the code before the
    Copilot review of PR #138) accepts the first three of these.
    """
    pin = TorchPin("torch==2.11.0+cu128", _CU128)
    assert devices.device_problem({}, {"version": "2.11.0+cu128"}, torch_pin=pin) is None
    wrong = devices.device_problem({}, {"version": installed}, torch_pin=pin)
    assert wrong is not None and "pins torch==2.11.0+cu128" in wrong
    assert repr(installed) in wrong or not installed


# --------------------------------------------------------------------------
# The child environment
# --------------------------------------------------------------------------


def test_card_visibility_overrides_inherited_and_fixes_the_order() -> None:
    env = {"CUDA_VISIBLE_DEVICES": "1"}
    plan = devices.apply_device_environment(
        env, cuda_visible_devices="0,1", requested={"train": "cuda:0"}
    )
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["CUDA_DEVICE_ORDER"] == devices.PCI_BUS_ORDER
    assert env["CUBLAS_WORKSPACE_CONFIG"] == CUBLAS_WORKSPACE_DETERMINISTIC
    assert plan["source"] == devices.SOURCE_CARD and plan["requested"] == {"train": "cuda:0"}


def test_inherited_visibility_is_kept_and_recorded_as_inherited() -> None:
    env = {"CUDA_VISIBLE_DEVICES": "1"}
    plan = devices.apply_device_environment(env, cuda_visible_devices=None, requested={})
    assert env == {"CUDA_VISIBLE_DEVICES": "1"}, "a CPU card adds no CUDA variable"
    assert plan["source"] == devices.SOURCE_INHERITED
    assert plan["CUDA_DEVICE_ORDER"] is None and plan["CUBLAS_WORKSPACE_CONFIG"] is None


def test_a_cuda_request_pins_the_order_even_without_card_visibility() -> None:
    """Otherwise torch orders GPUs fastest-first and cuda:1 is not nvidia-smi's GPU 1."""
    env = {"CUDA_VISIBLE_DEVICES": "0,1"}
    plan = devices.apply_device_environment(
        env, cuda_visible_devices=None, requested={"arena": "cuda:1"}
    )
    assert env["CUDA_DEVICE_ORDER"] == devices.PCI_BUS_ORDER
    assert plan["source"] == devices.SOURCE_INHERITED


def test_an_operators_device_order_survives_when_the_card_sets_no_visibility() -> None:
    env = {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}
    devices.apply_device_environment(env, cuda_visible_devices=None, requested={"a": "cuda:0"})
    assert env["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"


def test_a_cpu_card_with_nothing_inherited_is_unset() -> None:
    env: dict[str, str] = {}
    plan = devices.apply_device_environment(
        env, cuda_visible_devices=None, requested={"train": "cpu"}
    )
    assert env == {} and plan["source"] == devices.SOURCE_UNSET and plan["torch_pin"] is None


def test_an_operators_cublas_setting_survives() -> None:
    env = {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}
    devices.apply_device_environment(env, cuda_visible_devices="0", requested={"t": "cuda"})
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_the_plan_records_the_torch_pin() -> None:
    pin = TorchPin("torch==2.11.0", _CU128)
    plan = devices.apply_device_environment(
        {}, cuda_visible_devices=None, requested={}, torch_pin=pin
    )
    assert plan["torch_pin"] == {"spec": "torch==2.11.0", "index_url": _CU128}


# --------------------------------------------------------------------------
# The report code is real Python against the real torch API
# --------------------------------------------------------------------------


def _run_report() -> dict[str, Any]:
    scope: dict[str, Any] = {"info": {}}
    exec(compile(devices.TORCH_REPORT_CODE, "<torch-report>", "exec"), scope)  # noqa: S102
    return scope["info"]["torch"]


def _ones_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the report's one real kernel per device run here: place it on the CPU."""
    real_ones = torch.ones
    monkeypatch.setattr(torch, "ones", lambda *size, **_: real_ones(*size))


def test_report_code_runs_against_the_real_torch_api(monkeypatch: pytest.MonkeyPatch) -> None:
    _ones_on_cpu(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: f"GPU-{i}")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_90", "sm_120"])
    report = _run_report()
    assert "error" not in report, report
    assert report["cuda_available"] is True and report["device_count"] == 2
    assert report["devices"] == ["GPU-0", "GPU-1"]
    assert report["capabilities"] == [[12, 0], [12, 0]]
    assert report["arch_list"] == ["sm_90", "sm_120"]
    assert report["smoke"] == [None, None], "one real operation per device, both fine"
    assert report["version"] == str(torch.__version__)
    # ...and the verdict reads exactly these fields.
    assert devices.device_problem({"a": "cuda:1"}, report) is None


def test_report_code_without_cuda_reports_no_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report = _run_report()
    assert report["cuda_available"] is False and report["device_count"] == 0
    assert report["devices"] == [] and report["capabilities"] == [] and report["arch_list"] == []


def test_report_code_records_a_failure_instead_of_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> bool:
        raise OSError("driver went away")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    report = _run_report()
    assert report == {"error": "OSError", "detail": "driver went away"}


# --------------------------------------------------------------------------
# The GPU a card expects at each ordinal
#
# Negative control (recorded, then reverted): deleting the ``expected_gpus`` loop
# from ``device_problem`` made ``test_gpus_in_the_other_order_are_refused`` fail.
# --------------------------------------------------------------------------

_EXPECTED = {0: "RTX 5060 Ti", 1: "RTX 5060"}


def test_expected_gpus_in_place_are_accepted() -> None:
    requested = {"train": "cuda:0", "primary": "cuda:1"}
    assert devices.device_problem(requested, BLACKWELL, expected_gpus=_EXPECTED) is None


def test_gpus_in_the_other_order_are_refused() -> None:
    """Both cards are present and supported; only their order is wrong."""
    swapped = {**BLACKWELL, "devices": ["RTX 5060", "RTX 5060 Ti"]}
    problem = devices.device_problem(
        {"train": "cuda:0"}, swapped, visible="0,1", expected_gpus=_EXPECTED
    )
    assert problem is not None
    assert "cuda:0 is 'RTX 5060' but the card expects 'RTX 5060 Ti'" in problem


def test_an_expected_gpu_is_checked_even_if_no_invocation_names_it() -> None:
    one_gpu = {**BLACKWELL, "device_count": 1, "devices": ["RTX 5060 Ti"]}
    problem = devices.device_problem({"train": "cuda:0"}, one_gpu, expected_gpus=_EXPECTED)
    assert problem is not None and "cuda:1 is None" in problem


def test_the_plan_records_the_expected_gpus() -> None:
    plan = devices.apply_device_environment(
        {}, cuda_visible_devices="0,1", requested={}, expected_gpus=_EXPECTED
    )
    assert plan["expected_gpus"] == {"0": "RTX 5060 Ti", "1": "RTX 5060"}


def test_report_code_records_the_cudnn_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "GPU-0")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_120"])
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 91002)
    assert _run_report()["cudnn"] == 91002


def test_report_code_records_the_driver_listing_or_why_not() -> None:
    """``nvidia_smi`` is best effort: a list of lines, or an error record -- never a crash."""
    scope: dict[str, Any] = {"info": {}}
    exec(compile(devices.TORCH_REPORT_CODE, "<torch-report>", "exec"), scope)  # noqa: S102
    smi = scope["info"]["nvidia_smi"]
    assert isinstance(smi, list) or set(smi) == {"error"}


def test_report_code_records_a_kernel_that_does_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """``is_available()`` is True and the device is listed, yet no kernel runs on it."""

    def no_kernel(*_size: int, **_kwargs: Any) -> torch.Tensor:
        raise RuntimeError("CUDA error: no kernel image is available for execution on the device")

    monkeypatch.setattr(torch, "ones", no_kernel)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "GPU-0")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_120"])
    report = _run_report()
    assert report["smoke"] == [
        "RuntimeError: CUDA error: no kernel image is available for execution on the device"
    ]
    problem = devices.device_problem({"train": "cuda:0"}, report)
    assert problem is not None and "a CUDA kernel failed there" in problem
    assert "no kernel image" in problem


def test_a_failed_smoke_on_an_unrequested_device_does_not_refuse() -> None:
    report = {**BLACKWELL, "smoke": [None, "RuntimeError: no kernel image"]}
    assert devices.device_problem({"train": "cuda:0"}, report) is None
    refused = devices.device_problem({"primary": "cuda:1"}, report)
    assert refused is not None and "no kernel image" in refused
