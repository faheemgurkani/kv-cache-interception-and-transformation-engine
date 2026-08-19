"""Tests for Phase 10 hardware profile collection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from eval.hardware.profile import (
    HardwareProfile,
    collect_hardware_profile,
    hardware_metrics_enabled,
)


def test_hardware_profile_to_dict():
    profile = HardwareProfile(
        device_type="cuda",
        device_index=0,
        device_name="NVIDIA A10G",
        total_memory_bytes=24_000_000_000,
        compute_capability="8.6",
        driver_version="550.54",
        platform_system="Linux",
        platform_machine="x86_64",
        execution_platform="modal",
        configured_gpu="NVIDIA A10G",
        gpu_fallbacks=("a10g", "l4", "any"),
    )
    payload = profile.to_dict()
    assert payload["single_gpu_policy"] is True
    assert payload["multi_gpu_matrix"] is False
    assert payload["gpu_fallbacks"] == ["a10g", "l4", "any"]


def test_collect_hardware_profile_cpu(monkeypatch):
    monkeypatch.delenv("KV_EXECUTION_PLATFORM", raising=False)
    monkeypatch.delenv("KV_HARDWARE_PROFILE", raising=False)
    profile = collect_hardware_profile(torch.device("cpu"))
    assert profile.device_type == "cpu"
    assert profile.device_name == "CPU"
    assert profile.execution_platform == "local"
    assert profile.single_gpu_policy is True
    assert profile.multi_gpu_matrix is False


def test_collect_hardware_profile_modal_env(monkeypatch):
    monkeypatch.setenv("KV_EXECUTION_PLATFORM", "modal")
    monkeypatch.setenv("KV_HARDWARE_PROFILE", "NVIDIA A10G")
    monkeypatch.setenv("KV_EVAL_DEVICE", "cuda")
    monkeypatch.setenv("MODAL_GPU_REQUEST", "a10g")

    with patch("eval.hardware.profile._query_nvidia_smi") as mock_smi:
        mock_smi.return_value = {
            "device_name": "NVIDIA A10G",
            "total_memory_bytes": 24_000_000_000,
            "driver_version": "550.54",
        }
        profile = collect_hardware_profile(torch.device("cuda", 0))

    assert profile.execution_platform == "modal"
    assert profile.configured_gpu == "NVIDIA A10G"
    assert profile.device_name == "NVIDIA A10G"
    assert profile.driver_version == "550.54"


def test_hardware_metrics_enabled_modal_default(monkeypatch):
    monkeypatch.delenv("KV_COLLECT_HARDWARE_METRICS", raising=False)
    monkeypatch.setenv("KV_EXECUTION_PLATFORM", "modal")
    assert hardware_metrics_enabled() is True


def test_hardware_metrics_enabled_explicit_off(monkeypatch):
    monkeypatch.setenv("KV_COLLECT_HARDWARE_METRICS", "0")
    monkeypatch.setenv("KV_EXECUTION_PLATFORM", "modal")
    assert hardware_metrics_enabled() is False


def test_query_nvidia_smi_parses_csv():
    mock_proc = MagicMock()
    mock_proc.stdout = "NVIDIA A10G, 23028, 550.54.15\n"
    with patch("eval.hardware.profile.subprocess.run", return_value=mock_proc) as mock_run:
        from eval.hardware.profile import _query_nvidia_smi

        result = _query_nvidia_smi()
    mock_run.assert_called_once()
    assert mock_run.call_args.args[0][0] == "nvidia-smi"
    assert result["device_name"] == "NVIDIA A10G"
    assert result["driver_version"] == "550.54.15"
    assert result["total_memory_bytes"] == int(23028 * 1024 * 1024)


def test_query_nvidia_smi_missing_binary():
    from eval.hardware.profile import _query_nvidia_smi

    with patch("eval.hardware.profile.subprocess.run", side_effect=FileNotFoundError):
        result = _query_nvidia_smi()
    assert result["device_name"] is None
