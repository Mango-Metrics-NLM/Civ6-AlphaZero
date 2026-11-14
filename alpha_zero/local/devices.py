"""Devices for a pinned run: what it asks for, what its environment has, what its children see.

Why this module exists
----------------------
A run card names devices in its argv (``--device cuda:1``). Three things can make
that a lie without anything failing:

* the pinned environment's torch has no CUDA at all (PyPI's default wheel on
  Windows), or was built without kernels for this GPU's architecture -- in which
  case ``torch.cuda.is_available()`` is still ``True``;
* the device ordinal does not exist under the ``CUDA_VISIBLE_DEVICES`` in force;
* the runners resolve an unavailable device to CPU with a warning
  (``alpha_zero.utils.device``), so the run completes, exits 0 and records
  evidence under a device it never used.

The driver therefore asks the pinned child what it actually has
(:data:`TORCH_REPORT_CODE`) and refuses the card when a named device cannot be
honoured (:func:`device_problem`). It never substitutes another device.

Everything here is a pure function over plain data, so the same checks serve the
run-card driver today and any other local launcher later; nothing imports torch.
Which device an invocation uses is not decided here: the driver asks the runner's
own argument parser (``run_card.invocation_devices``), because only the parser knows
abbreviations and defaults.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from typing import Any

from alpha_zero.local.pinned_env import TorchPin
from alpha_zero.utils.device import cuda_ordinal, torch_build_matches
from alpha_zero.utils.seeding import CUBLAS_WORKSPACE_DETERMINISTIC, CUBLAS_WORKSPACE_ENV

logger = logging.getLogger(__name__)

__all__ = [
    "ENV_CUBLAS_WORKSPACE",
    "ENV_CUDA_DEVICE_ORDER",
    "ENV_CUDA_VISIBLE_DEVICES",
    "PCI_BUS_ORDER",
    "SOURCE_CARD",
    "SOURCE_INHERITED",
    "SOURCE_UNSET",
    "TORCH_REPORT_CODE",
    "apply_device_environment",
    "architecture_supported",
    "device_problem",
    "implicit_device_problem",
]

ENV_CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
ENV_CUDA_DEVICE_ORDER = "CUDA_DEVICE_ORDER"
ENV_CUBLAS_WORKSPACE = CUBLAS_WORKSPACE_ENV

#: torch's default device order is fastest-first, so on a machine with two different
#: GPUs ``cuda:0`` need not be the card ``nvidia-smi`` calls GPU 0. PCI order makes a
#: card's device names mean the same thing on every run.
PCI_BUS_ORDER = "PCI_BUS_ID"

#: Where ``CUDA_VISIBLE_DEVICES`` in the child came from (recorded in ``run.json``).
SOURCE_CARD = "card"
SOURCE_INHERITED = "inherited"
SOURCE_UNSET = "unset"

#: Run inside the pinned child, with an ``info`` dict in scope: adds ``info['torch']``.
#: It is the pinned environment's own account of itself -- the only one that counts,
#: because the driver's interpreter may carry a different torch. ``capabilities`` are
#: ``[major, minor]`` per device and ``arch_list`` is what the build has kernels for;
#: ``is_available()`` alone does not show that a build can run a given GPU.
TORCH_REPORT_CODE = (
    "try:\n"
    "    import torch\n"
    "    _cuda = bool(torch.cuda.is_available())\n"
    "    _n = int(torch.cuda.device_count()) if _cuda else 0\n"
    "    info['torch'] = {\n"
    "        'version': str(torch.__version__),\n"
    "        'cuda': torch.version.cuda,\n"
    "        'cuda_available': _cuda,\n"
    "        'device_count': _n,\n"
    "        'devices': [str(torch.cuda.get_device_name(i)) for i in range(_n)],\n"
    "        'capabilities': [list(torch.cuda.get_device_capability(i)) for i in range(_n)],\n"
    "        'arch_list': [str(a) for a in torch.cuda.get_arch_list()] if _cuda else [],\n"
    "        'cudnn': torch.backends.cudnn.version() if _cuda else None,\n"
    "    }\n"
    # One real kernel per device: the arch list says what a build was compiled for, this
    # says whether it runs. None = ok; otherwise the error, which names the reason.
    "    _smoke = []\n"
    "    for _i in range(_n):\n"
    "        try:\n"
    "            _ok = float((torch.ones(4, device='cuda:%d' % _i) * 2).sum().item()) == 8.0\n"
    "            _smoke.append(None if _ok else 'wrong result from a CUDA kernel')\n"
    "        except Exception as _exc:\n"
    "            _smoke.append('%s: %s' % (type(_exc).__name__, str(_exc)[:200]))\n"
    "    info['torch']['smoke'] = _smoke\n"
    "except Exception as exc:\n"
    "    info['torch'] = {'error': type(exc).__name__, 'detail': str(exc)[:300]}\n"
    # The driver is part of what a GPU run's bits depend on, and torch does not expose
    # it; nvidia-smi does. Best effort: absent on a machine without NVIDIA tooling.
    "try:\n"
    "    import subprocess\n"
    "    _smi = subprocess.run(['nvidia-smi', '--query-gpu=index,name,pci.bus_id,"
    "driver_version', '--format=csv,noheader'], capture_output=True, text=True, timeout=30)\n"
    "    info['nvidia_smi'] = [ln.strip() for ln in _smi.stdout.splitlines() if ln.strip()]\n"
    "except Exception as exc:\n"
    "    info['nvidia_smi'] = {'error': type(exc).__name__}\n"
)


def apply_device_environment(
    env: MutableMapping[str, str],
    *,
    cuda_visible_devices: str | None,
    requested: Mapping[str, str],
    torch_pin: TorchPin | None = None,
    expected_gpus: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Set the device variables in ``env`` (in place) and return the ``device_plan`` record.

    * A card-level ``cuda_visible_devices`` wins over an inherited value and fixes the
      device order (:data:`PCI_BUS_ORDER`): the card owns what its ``cuda:N`` means.
      An inherited visibility is left as the operator set it, and recorded as such.
    * Whenever an invocation names a CUDA device the order is pinned too, unless the
      operator's environment already sets one -- without it torch orders GPUs
      fastest-first and ``cuda:1`` need not be the card ``nvidia-smi`` calls GPU 1.
    * If any invocation names a CUDA device, ``CUBLAS_WORKSPACE_CONFIG`` is set: cuBLAS
      reads it when it starts, so it has to be in the child's environment rather than
      set by the runner after import (``utils.seeding.enforce_deterministic_backend``).
      An operator's own value is kept.
    """
    source = SOURCE_UNSET
    if cuda_visible_devices is not None:
        env[ENV_CUDA_VISIBLE_DEVICES] = cuda_visible_devices
        env[ENV_CUDA_DEVICE_ORDER] = PCI_BUS_ORDER
        source = SOURCE_CARD
    elif ENV_CUDA_VISIBLE_DEVICES in env:
        source = SOURCE_INHERITED
    if any(cuda_ordinal(device) is not None for device in requested.values()):
        env.setdefault(ENV_CUDA_DEVICE_ORDER, PCI_BUS_ORDER)
        env.setdefault(ENV_CUBLAS_WORKSPACE, CUBLAS_WORKSPACE_DETERMINISTIC)
    plan: dict[str, Any] = {
        ENV_CUDA_VISIBLE_DEVICES: env.get(ENV_CUDA_VISIBLE_DEVICES),
        ENV_CUDA_DEVICE_ORDER: env.get(ENV_CUDA_DEVICE_ORDER),
        ENV_CUBLAS_WORKSPACE: env.get(ENV_CUBLAS_WORKSPACE),
        "source": source,
        "requested": dict(requested),
        "torch_pin": None if torch_pin is None else torch_pin.as_dict(),
        "expected_gpus": {str(k): v for k, v in sorted((expected_gpus or {}).items())},
    }
    logger.info(
        "device plan: visible=%s (%s) order=%s requested=%s torch_pin=%s",
        plan[ENV_CUDA_VISIBLE_DEVICES],
        source,
        plan[ENV_CUDA_DEVICE_ORDER],
        plan["requested"],
        plan["torch_pin"],
    )
    return plan


def implicit_device_problem(
    resolved: Mapping[str, str | None], *, torch_pin: TorchPin | None
) -> str | None:
    """Why a card that pins a torch build may not leave a device to chance, or ``None``.

    ``resolved`` maps each invocation to the device its runner's own parser resolved
    (``None`` when the runner has no device argument or leaves it unset). A card with a
    torch pin is a statement about hardware, so every invocation must name its device:
    ``auto`` or an unset device would pick whatever is there, go unchecked by
    :func:`device_problem` and leave ``device_plan`` silent about what ran. A card
    without a pin is not held to this.
    """
    if torch_pin is None:
        return None
    implicit = sorted(
        name
        for name, device in resolved.items()
        if device is None or str(device).strip().lower() == "auto"
    )
    if not implicit:
        return None
    return (
        f"the card pins {torch_pin.spec}, so every invocation must name its device explicitly "
        f"(cpu, cuda or cuda:N); {', '.join(implicit)} leave it to a default or 'auto'"
    )


def architecture_supported(capability: Sequence[int], arch_list: Iterable[str]) -> bool:
    """Whether a torch build with ``arch_list`` has kernels a ``capability`` GPU can run.

    CUDA binaries run on a GPU of the same major compute capability and an equal or
    newer minor one: an ``sm_80`` kernel runs on an ``sm_89`` card, an ``sm_86`` kernel
    does not run on an ``sm_80`` card, and nothing built for ``sm_90`` runs on
    ``sm_120``. An empty or unparseable ``arch_list`` proves nothing, so it passes --
    the availability and ordinal checks still apply.
    """
    major, minor = int(capability[0]), int(capability[1])
    built = [int(a.split("_", 1)[1]) for a in arch_list if a.startswith("sm_") and a[3:].isdigit()]
    if not built:
        return True
    return any(sm // 10 == major and sm % 10 <= minor for sm in built)


def _torch_label(report: Mapping[str, Any]) -> str:
    return str(report.get("version", report.get("error", "unknown")))


def device_problem(
    requested: Mapping[str, str],
    torch_report: Any,
    *,
    torch_pin: TorchPin | None = None,
    visible: str | None = None,
    expected_gpus: Mapping[int, str] | None = None,
) -> str | None:
    """Why the pinned environment cannot honour the requested devices, or ``None``.

    ``torch_report`` is the pinned child's ``info['torch']`` (:data:`TORCH_REPORT_CODE`);
    anything that is not a mapping counts as "no report". Checked, in order:

    1. the torch pin, when there is one, is the torch actually installed (CPU cards too);
       a pin that names its build (``+cu128``) is compared in full;
    2. every explicit ``cuda`` / ``cuda:N`` exists: CUDA is available and ``N`` is
       below the visible device count;
    3. the build has kernels for that GPU's architecture
       (:func:`architecture_supported`), and one real CUDA operation ran there;
    4. each ordinal in ``expected_gpus`` holds the GPU the card expects there. A
       two-GPU machine can present its cards in either order; evidence that says
       ``cuda:1`` must mean one physical card every time.

    ``auto`` and ``cpu`` request nothing, so they are not checked. ``visible`` is the
    ``CUDA_VISIBLE_DEVICES`` in force, quoted in the message so the operator can see
    why an ordinal is missing.
    """
    report: Mapping[str, Any] = torch_report if isinstance(torch_report, Mapping) else {}
    if torch_pin is not None:
        if not torch_build_matches(report.get("version", ""), torch_pin.version):
            return (
                f"the card pins {torch_pin.spec} but the pinned environment has torch "
                f"{_torch_label(report)!r}"
            )
    wanted = {
        name: (device, index)
        for name, device in requested.items()
        if (index := cuda_ordinal(device)) is not None
    }
    if not wanted:
        return None
    if not report.get("cuda_available"):
        detail = f"; {report['detail']}" if report.get("detail") else ""
        return (
            f"invocations {', '.join(sorted(wanted))} ask for CUDA but the pinned "
            f"environment's torch ({_torch_label(report)}, cuda={report.get('cuda')}) reports "
            f"it unavailable{detail}; set torch_spec/torch_index_url to a CUDA build"
        )
    count = int(report.get("device_count", 0))
    names = list(report.get("devices") or [])
    capabilities = list(report.get("capabilities") or [])
    arch_list = [str(a) for a in report.get("arch_list") or []]
    smoke = list(report.get("smoke") or [])
    for name, (device, index) in sorted(wanted.items()):
        if index >= count:
            return (
                f"invocation {name} asks for {device} but only {count} CUDA device(s) are "
                f"visible ({ENV_CUDA_VISIBLE_DEVICES}={visible!r})"
            )
        if index < len(capabilities) and not architecture_supported(capabilities[index], arch_list):
            major, minor = capabilities[index][0], capabilities[index][1]
            gpu = names[index] if index < len(names) else device
            return (
                f"invocation {name} asks for {device} ({gpu}, sm_{major}{minor}) but torch "
                f"{_torch_label(report)} has kernels only for {arch_list}; pin a torch build "
                "whose index supports this GPU"
            )
        if index < len(smoke) and smoke[index] is not None:
            gpu = names[index] if index < len(names) else device
            return (
                f"invocation {name} asks for {device} ({gpu}) but a CUDA kernel failed there "
                f"under torch {_torch_label(report)}: {smoke[index]}"
            )
    for ordinal, expected in sorted((expected_gpus or {}).items()):
        found = names[ordinal] if ordinal < len(names) else None
        if found != expected:
            return (
                f"cuda:{ordinal} is {found!r} but the card expects {expected!r} "
                f"({ENV_CUDA_VISIBLE_DEVICES}={visible!r}, visible GPUs {names}); "
                "the GPUs are not in the order the card was written for"
            )
    return None
