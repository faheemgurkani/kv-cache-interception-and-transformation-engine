"""Palu online setup: bind weight factors before KVCacheEngine decode loops."""

from __future__ import annotations

from compressors.palu import PaluCompressor


def enable_palu_online(model, compressor: PaluCompressor) -> None:
    """Prepare Palu weight-factor decomposition; latent cache via kv_engine compress path."""
    if getattr(model, "_palu_online_enabled", False):
        return
    compressor.bind_model(model)
    model._palu_online_enabled = True
