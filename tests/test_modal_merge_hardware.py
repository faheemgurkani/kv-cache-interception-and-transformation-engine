"""Tests for Modal merge CSV flattening (Phase 10 hardware columns)."""

from modal_app.merge import CSV_FIELDNAMES, flatten_result_payload


def test_flatten_includes_hardware_and_system_metrics():
    payload = {
        "label": "turboquant_512",
        "compressor": "turboquant",
        "context_length": 512,
        "reference_gpu": "NVIDIA A10G",
        "hardware": {
            "device_name": "NVIDIA A10G",
            "configured_gpu": "NVIDIA A10G",
            "execution_platform": "modal",
            "single_gpu_policy": True,
        },
        "fidelity": {"representation": {}, "attention": {}, "memory": {}},
        "behavior": {"task_quality": {"perplexity": 12.3, "perplexity_baseline": 10.1}},
        "system": {
            "latency_throughput": {"tokens_per_second": 42.0, "ttft_ms": 100.0, "itl_ms_mean": 20.0},
            "peak_memory": {"peak_allocated_mb": 2048.0, "peak_reserved_mb": 2200.0},
            "gpu_utilization": {"mean_utilization_pct": 55.0, "max_utilization_pct": 88.0},
        },
    }
    row = flatten_result_payload(payload)
    assert row["peak_vram_allocated_mb"] == 2048.0
    assert row["gpu_util_max_pct"] == 88.0
    assert row["hardware_execution_platform"] == "modal"
    assert row["reference_gpu"] == "NVIDIA A10G"
    assert "hardware_device_name" in CSV_FIELDNAMES
