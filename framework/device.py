"""Device selection for Apple Silicon (MPS), NVIDIA CUDA, and CPU fallback."""

from __future__ import annotations

import os

import torch


def get_device(prefer_mps: bool = True, prefer_cuda: bool = False) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_eval_device() -> torch.device:
    """
    Select eval runtime device.

    Local default: MPS on Apple Silicon, else CPU.
    Modal / CUDA hosts: set KV_EVAL_DEVICE=cuda.
    """
    forced = os.environ.get("KV_EVAL_DEVICE", "").strip().lower()
    if forced == "cuda":
        return get_device(prefer_cuda=True, prefer_mps=False)
    if forced == "cpu":
        return torch.device("cpu")
    if forced == "mps":
        return get_device(prefer_mps=True, prefer_cuda=False)
    return get_device(prefer_mps=True, prefer_cuda=False)
