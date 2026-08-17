"""Core infrastructure for the KV-Cache Interception and Transformation Engine."""

from framework.compatibility import (
    CompatibilityGate,
    GateCheckResult,
    check_attention_gate,
    check_loader_state_gate,
    check_state_semantics_gate,
    evaluate_compatibility_gates,
)
from framework.config import load_eval_config, load_model_config
from framework.device import get_device
from framework.kv_cache import apply_compressor, extract_layer_kv, get_cache_size_bytes
from framework.kv_engine import CompressedCache, KVCacheEngine
from framework.model import ModelLayer
from framework.model_adapter import ATTENTION_ADAPTER_REGISTRY, load_attention_ops
from framework.model_capabilities import ModelCapabilities, resolve_model_capabilities
from framework.rope import RoPEContext, build_rope_context
from framework.state_interface import (
    LayerState,
    attention_kv_bytes,
    iter_layer_states,
    total_state_bytes,
    visible_state_bytes,
)

__all__ = [
    "ATTENTION_ADAPTER_REGISTRY",
    "CompressedCache",
    "CompatibilityGate",
    "GateCheckResult",
    "KVCacheEngine",
    "LayerState",
    "ModelCapabilities",
    "ModelLayer",
    "RoPEContext",
    "apply_compressor",
    "attention_kv_bytes",
    "build_rope_context",
    "check_attention_gate",
    "check_loader_state_gate",
    "check_state_semantics_gate",
    "evaluate_compatibility_gates",
    "extract_layer_kv",
    "get_cache_size_bytes",
    "get_device",
    "iter_layer_states",
    "load_attention_ops",
    "load_eval_config",
    "load_model_config",
    "resolve_model_capabilities",
    "total_state_bytes",
]
