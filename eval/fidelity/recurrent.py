"""FIDELITY / Recurrent: Mamba state preservation after K/V compression (§25)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from compressors.base import KVCompressor
from framework.kv_cache import decompress_to_legacy_cache
from framework.model_capabilities import resolve_model_capabilities
from framework.state_compression import compress_state, compressed_attention_layers
from framework.state_interface import iter_layer_states


@dataclass
class LayerRecurrentMetrics:
    layer: int
    recurrent_exact: bool
    conv_exact: bool
    max_abs_error: float


@dataclass
class RecurrentMetrics:
    """Whether recurrent/Mamba state is unchanged through a K/V compress round-trip."""

    applicable: bool
    layers_with_recurrent: int
    exact_preservation: bool
    max_abs_error: float
    per_layer: list[LayerRecurrentMetrics]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["per_layer"] = [asdict(item) for item in self.per_layer]
        return payload


def _tensor_max_abs(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    return (a.float() - b.float()).abs().max().item()


@torch.no_grad()
def evaluate_recurrent_fidelity(
    past_key_values,
    compressor: KVCompressor,
    model_config,
    *,
    device: torch.device | None = None,
) -> RecurrentMetrics:
    """Plan §25: assert R'_t = R_t after attention K/V compression (recurrent passthrough)."""
    caps = resolve_model_capabilities(model_config)
    if not caps.has_recurrent_state:
        return RecurrentMetrics(
            applicable=False,
            layers_with_recurrent=0,
            exact_preservation=True,
            max_abs_error=0.0,
            per_layer=[],
        )

    before_states = list(iter_layer_states(past_key_values, capabilities=caps))
    compressed_states = compress_state(past_key_values, compressor, capabilities=caps)
    attention_layers = compressed_attention_layers(compressed_states)
    merged = decompress_to_legacy_cache(
        attention_layers,
        compressor,
        model_config,
        device=device,
        template_cache=past_key_values,
    )
    after_states = list(iter_layer_states(merged, capabilities=caps))

    per_layer: list[LayerRecurrentMetrics] = []
    max_error = 0.0
    layers_with_recurrent = 0

    for before, after in zip(before_states, after_states, strict=True):
        if before.recurrent is None:
            continue
        layers_with_recurrent += 1
        rs_err = _tensor_max_abs(
            before.recurrent.recurrent_states,
            after.recurrent.recurrent_states if after.recurrent else None,
        )
        cs_err = _tensor_max_abs(
            before.recurrent.conv_states,
            after.recurrent.conv_states if after.recurrent else None,
        )
        layer_max = max(rs_err, cs_err)
        max_error = max(max_error, layer_max)
        rs_exact = rs_err == 0.0
        cs_exact = cs_err == 0.0
        if before.recurrent.recurrent_states is not None:
            rs_exact = torch.equal(before.recurrent.recurrent_states, after.recurrent.recurrent_states)
        if before.recurrent.conv_states is not None:
            cs_exact = torch.equal(before.recurrent.conv_states, after.recurrent.conv_states)
        per_layer.append(
            LayerRecurrentMetrics(
                layer=before.layer_idx,
                recurrent_exact=rs_exact,
                conv_exact=cs_exact,
                max_abs_error=layer_max,
            )
        )

    exact = all(item.recurrent_exact and item.conv_exact for item in per_layer)
    return RecurrentMetrics(
        applicable=True,
        layers_with_recurrent=layers_with_recurrent,
        exact_preservation=exact,
        max_abs_error=max_error,
        per_layer=per_layer,
    )
