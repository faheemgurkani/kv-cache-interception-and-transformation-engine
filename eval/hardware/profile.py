"""Hardware profile collection for CUDA reference sweeps (Phase 10).

Scope: **single NVIDIA GPU per job** (Modal ``gpu=`` on ``eval_worker``). Multi-GPU
tiers (A100/H100/4090 matrix) are explicitly **not** implemented — see
``configs/modal.yaml`` and ``docs/RESEARCH_REDESIGN_PLAN.md`` Phase 10.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class HardwareProfile:
    """Runtime + configured hardware metadata for one eval job."""

    device_type: str
    device_index: int | None
    device_name: str | None
    total_memory_bytes: int | None
    compute_capability: str | None
    driver_version: str | None
    platform_system: str
    platform_machine: str
    execution_platform: str
    configured_gpu: str | None
    gpu_fallbacks: tuple[str, ...] = field(default_factory=tuple)
    single_gpu_policy: bool = True
    multi_gpu_matrix: bool = False
    kv_eval_device_env: str | None = None
    nvml_available: bool = False
    cuda_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_type": self.device_type,
            "device_index": self.device_index,
            "device_name": self.device_name,
            "total_memory_bytes": self.total_memory_bytes,
            "compute_capability": self.compute_capability,
            "driver_version": self.driver_version,
            "platform_system": self.platform_system,
            "platform_machine": self.platform_machine,
            "execution_platform": self.execution_platform,
            "configured_gpu": self.configured_gpu,
            "gpu_fallbacks": list(self.gpu_fallbacks),
            "single_gpu_policy": self.single_gpu_policy,
            "multi_gpu_matrix": self.multi_gpu_matrix,
            "kv_eval_device_env": self.kv_eval_device_env,
            "nvml_available": self.nvml_available,
            "cuda_available": self.cuda_available,
        }


def _query_nvidia_smi() -> dict[str, str | None]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return {"device_name": None, "total_memory_bytes": None, "driver_version": None}

    line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    if not line:
        return {"device_name": None, "total_memory_bytes": None, "driver_version": None}

    parts = [part.strip() for part in line.split(",")]
    name = parts[0] if parts else None
    mem_mb = None
    driver = None
    if len(parts) >= 2:
        try:
            mem_mb = float(parts[1])
        except ValueError:
            mem_mb = None
    if len(parts) >= 3:
        driver = parts[2]
    total_bytes = None if mem_mb is None else int(mem_mb * 1024 * 1024)
    return {"device_name": name, "total_memory_bytes": total_bytes, "driver_version": driver}


def _nvml_available() -> bool:
    try:
        import pynvml  # noqa: F401
    except ImportError:
        return False
    return True


def collect_hardware_profile(
    device: torch.device | None = None,
    *,
    configured_gpu: str | None = None,
    gpu_fallbacks: Sequence[str] | None = None,
    execution_platform: str | None = None,
) -> HardwareProfile:
    """Collect best-effort hardware metadata for export in job JSON."""
    dev = device or torch.device("cpu")
    smi = _query_nvidia_smi()

    device_name = smi["device_name"]
    total_memory_bytes = smi["total_memory_bytes"]
    driver_version = smi["driver_version"]
    compute_capability: str | None = None
    cuda_available = torch.cuda.is_available()

    if dev.type == "cuda" and cuda_available:
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        if device_name is None:
            device_name = torch.cuda.get_device_name(idx)
        if total_memory_bytes is None:
            props = torch.cuda.get_device_properties(idx)
            total_memory_bytes = props.total_memory
            compute_capability = f"{props.major}.{props.minor}"
        elif compute_capability is None:
            props = torch.cuda.get_device_properties(idx)
            compute_capability = f"{props.major}.{props.minor}"
    elif dev.type == "mps":
        device_name = device_name or "Apple MPS"
    else:
        device_name = device_name or "CPU"

    platform_env = execution_platform or os.environ.get("KV_EXECUTION_PLATFORM") or "local"
    configured = configured_gpu or os.environ.get("KV_HARDWARE_PROFILE") or os.environ.get("MODAL_GPU_REQUEST")

    fallbacks: tuple[str, ...] = tuple(gpu_fallbacks or ())
    if not fallbacks and configured:
        fallbacks = (configured,)

    return HardwareProfile(
        device_type=dev.type,
        device_index=dev.index,
        device_name=device_name,
        total_memory_bytes=total_memory_bytes,
        compute_capability=compute_capability,
        driver_version=driver_version,
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        execution_platform=platform_env,
        configured_gpu=configured,
        gpu_fallbacks=fallbacks,
        single_gpu_policy=True,
        multi_gpu_matrix=False,
        kv_eval_device_env=os.environ.get("KV_EVAL_DEVICE") or None,
        nvml_available=_nvml_available(),
        cuda_available=cuda_available,
    )


def hardware_metrics_enabled() -> bool:
    """True when peak VRAM + GPU util should run (Modal CUDA reference path)."""
    flag = os.environ.get("KV_COLLECT_HARDWARE_METRICS", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    if flag in {"0", "false", "no", "off"}:
        return False
    return os.environ.get("KV_EXECUTION_PLATFORM", "").lower() == "modal"
