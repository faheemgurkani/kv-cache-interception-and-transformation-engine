"""Unified compatibility probe consumed by FIDELITY / BEHAVIOR / SYSTEM (§29)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from framework.compatibility import GateCheckResult, evaluate_compatibility_gates
from framework.model import ModelLayer
from framework.model_capabilities import (
    ModelCapabilities,
    get_model_eval_metadata,
    load_compatibility_manifest,
    resolve_model_capabilities,
    validate_manifest,
)


@dataclass(frozen=True)
class CompatibilityProbe:
    """Single forward-pass probe: gates + manifest + metadata (§29)."""

    capabilities: ModelCapabilities
    gates: dict[str, GateCheckResult]
    manifest: dict[str, object]
    metadata: dict[str, object]
    forward_ok: bool

    @property
    def all_gates_passed(self) -> bool:
        return all(result.passed for result in self.gates.values())

    def gate_passed(self, gate_name: str) -> bool:
        return self.gates[gate_name].passed

    def gates_to_dict(self) -> dict[str, dict[str, object]]:
        return {
            name: {"gate": result.gate.value, "passed": result.passed, "detail": result.detail}
            for name, result in self.gates.items()
        }


@torch.no_grad()
def run_compatibility_probe(
    model_layer: ModelLayer,
    *,
    manifest: dict[str, object] | None = None,
    probe_text: str = "KVBench compatibility probe.",
) -> CompatibilityProbe:
    """Run Gate A/B/C once; FIDELITY/BEHAVIOR/SYSTEM consume this result (§29)."""
    config = model_layer.config
    caps = resolve_model_capabilities(config)
    manifest_payload = manifest or load_compatibility_manifest(config)
    validate_manifest(manifest_payload, caps, config)

    forward_ok = False
    past_key_values = None
    try:
        input_ids = model_layer.tokenize(probe_text)
        outputs = model_layer.model(input_ids, use_cache=True, return_dict=True)
        past_key_values = outputs.past_key_values
        forward_ok = past_key_values is not None
    except Exception:  # noqa: BLE001 - probe records failure via gates
        forward_ok = False

    gates = evaluate_compatibility_gates(
        config=config,
        model_loaded=True,
        forward_ok=forward_ok,
        past_key_values=past_key_values,
    )
    metadata = get_model_eval_metadata(config, local_path=str(model_layer.model_path))
    metadata["compatibility_gates"] = {
        name: {"gate": result.gate.value, "passed": result.passed, "detail": result.detail}
        for name, result in gates.items()
    }
    return CompatibilityProbe(
        capabilities=caps,
        gates=gates,
        manifest=manifest_payload,
        metadata=metadata,
        forward_ok=forward_ok,
    )
