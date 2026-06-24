"""Throughput evaluation for KV-cache compression."""

from __future__ import annotations

import time
from collections.abc import Callable


def measure_tokens_per_second(generate_fn: Callable, num_tokens: int = 128) -> float:
    """Measure generation throughput."""
    start = time.perf_counter()
    generate_fn(num_tokens)
    elapsed = time.perf_counter() - start
    return num_tokens / elapsed if elapsed > 0 else 0.0
