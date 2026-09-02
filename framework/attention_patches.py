"""Restore vanilla attention after compressor-specific online patches.

Modal jobs each get a fresh model process. Local multi-job smokes reuse one
``ModelLayer``; without this, QJL/RocketKV/SnapKV ``attn.forward`` closures
leak into later compressors and invalidate every KPI after the first patched job.
"""

from __future__ import annotations

import torch

ONLINE_FLAGS: tuple[str, ...] = (
    "_qjl_online_enabled",
    "_rocketkv_online_enabled",
    "_snapkv_online_enabled",
    "_palu_online_enabled",
)


def decoder_layers(model):
    """Yield decoder layers across nested HF causal-LM wrappers (Gemma3, etc.)."""
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        inner = getattr(inner, "language_model", inner)
        layers = getattr(inner, "layers", None)
    if layers is None:
        raise AttributeError("Could not locate decoder layers for attention patches")
    return layers


def align_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    q_len: int,
    k_len: int,
) -> torch.Tensor | None:
    """Slice a 4D attention mask to the current query/key lengths.

    Gemma3 (and other sliding-window families) may pass a prefill-sized 4D mask
    into decode. Adding that tensor to [B, H, 1, k] scores silently broadcasts
    or writes -inf on the wrong positions — QJL's ProdQJL path was hitting this
    on Gemma3-270M (PPL ~90× identity).
    """
    if attention_mask is None or attention_mask.dim() != 4:
        return attention_mask
    if attention_mask.shape[-2] == q_len and attention_mask.shape[-1] == k_len:
        return attention_mask
    return attention_mask[..., -q_len:, -k_len:]


def _clear_online_flags(model) -> None:
    for name in ONLINE_FLAGS:
        if hasattr(model, name):
            delattr(model, name)


def ensure_vanilla_attention(model) -> None:
    """Reinstall the first-seen ``self_attn.forward`` on every layer.

    The first call snapshots the unpatched methods. Later calls restore them
    and drop online-enable flags so the next ``enable_*_online`` re-binds the
    *current* compressor instead of a stale closure.
    """
    layers = decoder_layers(model)
    stored = getattr(model, "_kvbench_vanilla_attn_forwards", None)
    if stored is None:
        model._kvbench_vanilla_attn_forwards = [layer.self_attn.forward for layer in layers]
    else:
        if len(stored) != len(layers):
            raise RuntimeError(
                f"Vanilla attention snapshot has {len(stored)} layers, model has {len(layers)}"
            )
        for layer, fwd in zip(layers, stored):
            layer.self_attn.forward = fwd
    _clear_online_flags(model)
