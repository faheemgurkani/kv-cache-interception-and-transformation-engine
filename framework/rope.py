"""Layer-aware RoPE context builder."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from framework.model_capabilities import resolve_model_capabilities


@dataclass
class RoPEContext:
    """Backwards-compatible RoPE lookup: global by default, per-layer-type when required."""

    _tables: dict[str | None, tuple[torch.Tensor, torch.Tensor]]
    _layer_types: list[str] | None

    def get_rope(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._layer_types and layer_idx < len(self._layer_types):
            layer_type = self._layer_types[layer_idx]
            if layer_type in self._tables:
                return self._tables[layer_type]
        if None in self._tables:
            return self._tables[None]
        if self._tables:
            return next(iter(self._tables.values()))
        raise RuntimeError("RoPEContext has no tables.")


def build_rope_context(
    model,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    config=None,
) -> RoPEContext:
    """Build a RoPE lookup table for all layers.

    OLMo2/Qwen3 reuse one global table for every layer. Gemma3 selects between
    precomputed sliding/full tables via ``get_rope(layer_idx)``.
    """
    config = config or model.config
    caps = resolve_model_capabilities(config)
    rotary_emb = model.model.rotary_emb
    layer_types = list(getattr(config, "layer_types", []) or [])

    if caps.rope_mode == "per_layer_type":
        unique_types = sorted(set(layer_types))
        tables: dict[str | None, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_type in unique_types:
            tables[layer_type] = rotary_emb(hidden_states, position_ids, layer_type=layer_type)
        return RoPEContext(_tables=tables, _layer_types=layer_types)

    global_table = rotary_emb(hidden_states, position_ids)
    return RoPEContext(_tables={None: global_table}, _layer_types=layer_types or None)
