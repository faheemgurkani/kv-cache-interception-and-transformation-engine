"""Explicit compatibility gate checks for KVBench model support."""

from __future__ import annotations

from dataclasses import dataclass

from framework.model_adapter import load_attention_ops, resolve_model_type
from framework.model_capabilities import CompatibilityGate, ModelCapabilities, resolve_model_capabilities
from framework.state_interface import iter_layer_states, state_semantics_issues


@dataclass(frozen=True)
class GateCheckResult:
    gate: CompatibilityGate
    passed: bool
    detail: str


def check_loader_state_gate(*, model_loaded: bool, forward_ok: bool, past_key_values) -> GateCheckResult:
    """Gate A — loader/state compatibility."""
    if not model_loaded:
        return GateCheckResult(CompatibilityGate.LOADER_STATE, False, "Model failed to load.")
    if not forward_ok:
        return GateCheckResult(CompatibilityGate.LOADER_STATE, False, "Forward pass failed.")
    if past_key_values is None:
        return GateCheckResult(CompatibilityGate.LOADER_STATE, False, "No past_key_values returned.")
    try:
        layers = list(iter_layer_states(past_key_values))
    except Exception as exc:  # noqa: BLE001 - gate diagnostics
        return GateCheckResult(
            CompatibilityGate.LOADER_STATE,
            False,
            f"State discovery failed: {exc}",
        )
    if not layers:
        return GateCheckResult(CompatibilityGate.LOADER_STATE, False, "No layer states discovered.")
    return GateCheckResult(CompatibilityGate.LOADER_STATE, True, f"Discovered {len(layers)} layer states.")


def check_attention_gate(config) -> GateCheckResult:
    """Gate B — attention-adapter compatibility."""
    caps = resolve_model_capabilities(config)
    if not caps.supports_gate(CompatibilityGate.ATTENTION):
        return GateCheckResult(
            CompatibilityGate.ATTENTION,
            False,
            f"No attention adapter registered for model_type={resolve_model_type(config)!r}.",
        )
    try:
        load_attention_ops(config)
    except NotImplementedError as exc:
        return GateCheckResult(CompatibilityGate.ATTENTION, False, str(exc))
    return GateCheckResult(
        CompatibilityGate.ATTENTION,
        True,
        f"Attention adapter available for model_type={caps.model_type!r}.",
    )


def check_state_semantics_gate(config, past_key_values) -> GateCheckResult:
    """Gate C — every inference-state component is accounted for."""
    caps = resolve_model_capabilities(config)
    issues = state_semantics_issues(past_key_values, caps)
    if issues:
        return GateCheckResult(CompatibilityGate.STATE_SEMANTICS, False, "; ".join(issues))
    if not caps.state_semantics_complete:
        return GateCheckResult(
            CompatibilityGate.STATE_SEMANTICS,
            False,
            f"Capability metadata marks state semantics incomplete for model_type={caps.model_type!r}.",
        )
    return GateCheckResult(
        CompatibilityGate.STATE_SEMANTICS,
        True,
        "All declared inference-state components are accounted for.",
    )


def evaluate_compatibility_gates(
    *,
    config,
    model_loaded: bool,
    forward_ok: bool,
    past_key_values,
) -> dict[str, GateCheckResult]:
    """Evaluate all three compatibility gates."""
    return {
        CompatibilityGate.LOADER_STATE.value: check_loader_state_gate(
            model_loaded=model_loaded,
            forward_ok=forward_ok,
            past_key_values=past_key_values,
        ),
        CompatibilityGate.ATTENTION.value: check_attention_gate(config),
        CompatibilityGate.STATE_SEMANTICS.value: check_state_semantics_gate(config, past_key_values),
    }


def assert_gate(caps: ModelCapabilities, gate: CompatibilityGate) -> None:
    """Raise when a model family does not support a compatibility gate."""
    if not caps.supports_gate(gate):
        raise RuntimeError(
            f"Model type {caps.model_type!r} does not support compatibility gate {gate.value!r}."
        )
