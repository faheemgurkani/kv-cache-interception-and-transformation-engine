"""Compressor registry and factory."""

from __future__ import annotations

from compressors.base import KVCompressor
from compressors.identity import IdentityCompressor
from compressors.kivi import KIVICompressor
from compressors.qjl import QJLCompressor
from compressors.rocketkv import RocketKVCompressor
from compressors.turboquant import TurboQuantCompressor

COMPRESSORS: dict[str, type[KVCompressor]] = {
    "identity": IdentityCompressor,
    "turboquant": TurboQuantCompressor,
    "kivi": KIVICompressor,
    "qjl": QJLCompressor,
    "rocketkv": RocketKVCompressor,
}


def get_compressor(name: str, bitwidth: int | None = None, **kwargs) -> KVCompressor:
    if name not in COMPRESSORS:
        available = ", ".join(sorted(COMPRESSORS))
        raise ValueError(f"Unknown compressor '{name}'. Available: {available}")

    cls = COMPRESSORS[name]
    if bitwidth is not None:
        kwargs.setdefault("bitwidth", bitwidth)
    if not kwargs:
        return cls()
    return cls(**kwargs)
