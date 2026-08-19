"""State-aware compression dispatch (§28): attention K/V compress; recurrent passthrough."""

from __future__ import annotations

from dataclasses import dataclass

from compressors.base import CompressedKV, KVCompressor
from framework.model_capabilities import ModelCapabilities, StateType
from framework.state_interface import LayerState, RecurrentState, iter_layer_states


@dataclass(frozen=True)
class CompressedLayerState:
    """One layer after state-aware compression."""

    layer_idx: int
    state_type: StateType
    attention: CompressedKV | None = None
    recurrent: RecurrentState | None = None


def compress_layer_state(
    state: LayerState,
    compressor: KVCompressor,
) -> CompressedLayerState:
    """Compress attention K/V; preserve recurrent/Mamba state unchanged (§22, §28)."""
    attention_compressed: CompressedKV | None = None
    if state.attention is not None:
        attention_compressed = compressor.compress(
            state.attention.key,
            state.attention.value,
            layer=state.layer_idx,
        )
    return CompressedLayerState(
        layer_idx=state.layer_idx,
        state_type=state.state_type,
        attention=attention_compressed,
        recurrent=state.recurrent,
    )


def compress_state(
    past_key_values,
    compressor: KVCompressor,
    *,
    capabilities: ModelCapabilities | None = None,
) -> list[CompressedLayerState]:
    """State-aware entry point: ATTENTION_KV → compress_kv; RECURRENT → passthrough."""
    return [
        compress_layer_state(state, compressor)
        for state in iter_layer_states(past_key_values, capabilities=capabilities)
    ]


def compressed_attention_layers(states: list[CompressedLayerState]) -> list[CompressedKV]:
    """Extract attention payloads for legacy KV-only accounting paths."""
    layers: list[CompressedKV] = []
    for item in states:
        if item.attention is None:
            raise RuntimeError(
                f"Layer {item.layer_idx} has no compressible attention state; "
                "declared compression policy requires attention K/V."
            )
        layers.append(item.attention)
    return layers
