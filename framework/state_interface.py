"""Typed inference-state iteration for KVBench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch

from framework.model_capabilities import ModelCapabilities, StateType


@dataclass(frozen=True)
class AttentionKVState:
    key: torch.Tensor
    value: torch.Tensor
    attention_type: str | None = None
    window_size: int | None = None


@dataclass(frozen=True)
class RecurrentState:
    recurrent_states: torch.Tensor | None = None
    conv_states: torch.Tensor | None = None


@dataclass(frozen=True)
class LayerState:
    layer_idx: int
    state_type: StateType
    attention: AttentionKVState | None
    recurrent: RecurrentState | None = None
    latent_note: str | None = None


def _cache_layers(past_key_values):
    if hasattr(past_key_values, "layers"):
        return past_key_values.layers
    if hasattr(past_key_values, "key_cache"):
        return list(
            zip(past_key_values.key_cache, past_key_values.value_cache, strict=True)
        )
    return list(past_key_values)


def _extract_attention_from_layer(layer) -> AttentionKVState | None:
    if hasattr(layer, "keys") and hasattr(layer, "values"):
        keys = layer.keys
        values = layer.values
        if keys is None or values is None:
            return None
        attention_type = getattr(layer, "layer_type", None)
        window_size = getattr(layer, "sliding_window", None)
        return AttentionKVState(
            key=keys,
            value=values,
            attention_type=attention_type,
            window_size=int(window_size) if window_size is not None else None,
        )
    if isinstance(layer, tuple) and len(layer) >= 2:
        return AttentionKVState(key=layer[0], value=layer[1])
    return None


def _extract_recurrent_from_layer(layer) -> RecurrentState | None:
    recurrent_states = getattr(layer, "recurrent_states", None)
    conv_states = getattr(layer, "conv_states", None)
    if recurrent_states is None and conv_states is None:
        return None
    return RecurrentState(
        recurrent_states=recurrent_states,
        conv_states=conv_states,
    )


def _infer_state_type(attention: AttentionKVState | None, recurrent: RecurrentState | None) -> StateType:
    if recurrent is not None and attention is not None:
        return StateType.HYBRID
    if attention is not None:
        return StateType.CONVENTIONAL_KV
    if recurrent is not None:
        return StateType.HYBRID
    return StateType.CONVENTIONAL_KV


def iter_layer_states(
    past_key_values,
    *,
    capabilities: ModelCapabilities | None = None,
) -> Iterator[LayerState]:
    """Yield typed per-layer inference state.

    ``iter_layer_kv()`` in ``framework/kv_cache.py`` remains the compatibility view
    for conventional K/V only.
    """
    latent_note = capabilities.expanded_kv_disclosure if capabilities else None
    for layer_idx, layer in enumerate(_cache_layers(past_key_values)):
        attention = _extract_attention_from_layer(layer)
        recurrent = _extract_recurrent_from_layer(layer)
        state_type = (
            _infer_state_type(attention, recurrent)
            if capabilities is None
            else capabilities.state_type
        )
        yield LayerState(
            layer_idx=layer_idx,
            state_type=state_type,
            attention=attention,
            recurrent=recurrent,
            latent_note=latent_note,
        )


def state_semantics_issues(past_key_values, capabilities: ModelCapabilities) -> list[str]:
    """Return Gate-C issues when undeclared state components are present."""
    issues: list[str] = []
    for state in iter_layer_states(past_key_values, capabilities=capabilities):
        if state.recurrent is not None and not capabilities.has_recurrent_state:
            issues.append(
                f"Layer {state.layer_idx} exposes recurrent state but capabilities.has_recurrent_state=False."
            )
        if (
            state.recurrent is not None
            and capabilities.has_recurrent_state
            and not capabilities.state_semantics_complete
        ):
            issues.append(
                f"Layer {state.layer_idx} has recurrent state that is not yet included in "
                "reported memory accounting."
            )
    return issues


def total_state_bytes(past_key_values, capabilities: ModelCapabilities | None = None) -> int:
    """Count bytes across all accounted inference-state components."""
    total = 0
    include_recurrent = capabilities is not None and capabilities.state_semantics_complete
    for state in iter_layer_states(past_key_values, capabilities=capabilities):
        if state.attention is not None:
            total += state.attention.key.numel() * state.attention.key.element_size()
            total += state.attention.value.numel() * state.attention.value.element_size()
        if include_recurrent and state.recurrent is not None:
            for tensor in (state.recurrent.recurrent_states, state.recurrent.conv_states):
                if tensor is not None:
                    total += tensor.numel() * tensor.element_size()
    return total


def hybrid_layer_detected(past_key_values) -> bool:
    """True when any cache layer exposes recurrent/Mamba state."""
    for state in iter_layer_states(past_key_values):
        if state.recurrent is not None:
            return True
    return False
