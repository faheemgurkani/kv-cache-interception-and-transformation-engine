"""Cross-device peak memory and utilization sampling (CUDA, MPS, CPU)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import psutil
import torch


@dataclass
class PeakMemorySnapshot:
    peak_allocated_mb: float | None
    peak_reserved_mb: float | None
    peak_process_rss_mb: float
    memory_backend: str


class PeakMemoryTracker:
    """Sample peak device/process memory during a workload."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.peak_allocated_mb: float | None = None
        self.peak_reserved_mb: float | None = None
        self.peak_process_rss_mb = 0.0
        self.memory_backend = "process_rss"
        self._process = psutil.Process()

    def _sample_once(self) -> None:
        rss_mb = self._process.memory_info().rss / (1024 * 1024)
        self.peak_process_rss_mb = max(self.peak_process_rss_mb, rss_mb)

        if self.device.type == "cuda" and torch.cuda.is_available():
            self.memory_backend = "cuda"
            allocated = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            reserved = torch.cuda.max_memory_reserved(self.device) / (1024 * 1024)
            self.peak_allocated_mb = max(self.peak_allocated_mb or 0.0, allocated)
            self.peak_reserved_mb = max(self.peak_reserved_mb or 0.0, reserved)
            return

        if self.device.type == "mps" and torch.backends.mps.is_available():
            self.memory_backend = "mps"
            current_fn = getattr(torch.mps, "current_allocated_memory", None)
            driver_fn = getattr(torch.mps, "driver_allocated_memory", None)
            if current_fn is not None:
                current_mb = float(current_fn()) / (1024 * 1024)
                self.peak_allocated_mb = max(self.peak_allocated_mb or 0.0, current_mb)
            if driver_fn is not None:
                driver_mb = float(driver_fn()) / (1024 * 1024)
                self.peak_reserved_mb = max(self.peak_reserved_mb or 0.0, driver_mb)
            return

        self.memory_backend = "process_rss"
        self.peak_allocated_mb = max(self.peak_allocated_mb or 0.0, rss_mb)
        self.peak_reserved_mb = None

    def reset(self) -> None:
        self.peak_allocated_mb = None
        self.peak_reserved_mb = None
        self.peak_process_rss_mb = 0.0
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def sample(self) -> None:
        self._sample_once()

    def snapshot(self) -> PeakMemorySnapshot:
        self.sample()
        return PeakMemorySnapshot(
            peak_allocated_mb=self.peak_allocated_mb,
            peak_reserved_mb=self.peak_reserved_mb,
            peak_process_rss_mb=self.peak_process_rss_mb,
            memory_backend=self.memory_backend,
        )


class UtilizationSampler:
    """Background sampler for NVML GPU util (CUDA) or process CPU util (fallback)."""

    def __init__(self, device: torch.device, sample_interval_s: float = 0.05) -> None:
        self.device = device
        self.sample_interval_s = sample_interval_s
        self.utilization_backend = "unavailable"
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process()

    def _cuda_nvml_loop(self, handle) -> None:
        import pynvml

        while not self._stop.is_set():
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            self.samples.append(float(util.gpu))
            time.sleep(self.sample_interval_s)

    def _process_cpu_loop(self) -> None:
        while not self._stop.is_set():
            self.samples.append(float(self._process.cpu_percent(interval=None)))
            time.sleep(self.sample_interval_s)

    def start(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                import pynvml
            except ImportError:
                self.utilization_backend = "process_cpu"
                self._thread = threading.Thread(target=self._process_cpu_loop, daemon=True)
                self._thread.start()
                return

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index or 0)
            self.utilization_backend = "nvml"
            self._thread = threading.Thread(target=self._cuda_nvml_loop, args=(handle,), daemon=True)
            self._thread.start()
            self._nvml_handle = handle
            self._nvml = pynvml
            return

        self.utilization_backend = "process_cpu"
        self._thread = threading.Thread(target=self._process_cpu_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.utilization_backend == "nvml" and hasattr(self, "_nvml"):
            self._nvml.nvmlShutdown()

    @property
    def available(self) -> bool:
        return self.utilization_backend != "unavailable"

    def mean_pct(self) -> float | None:
        if not self.samples:
            return None
        return sum(self.samples) / len(self.samples)

    def max_pct(self) -> float | None:
        if not self.samples:
            return None
        return max(self.samples)
