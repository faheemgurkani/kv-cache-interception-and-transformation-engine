"""Controlled KV interception and experimental conditions (Phases 6–7).

Phase 6: same model, input construction, decode loop, and metric definitions;
only the compressor plug-in varies.

Phase 7: make every controlled axis explicit and exportable — model, tokenizer,
prompt/input construction, dataset, context length, generation length,
compression budget, hardware, batch size, decoding configuration, and metrics.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from compressors.base import KVCompressor

PHASE6_PRINCIPLE = (
    "Different KV transformations are executed through the same inference path "
    "under matched conditions; only the compressor plug-in varies."
)

PHASE7_PRINCIPLE = (
    "Same model + same input + same decode loop + same hardware + "
    "different KV transformation."
)

# Fixed axes that must appear in every exported contract (Phase 7 checklist).
REQUIRED_FIXED_AXES: tuple[str, ...] = (
    "model",
    "tokenizer",
    "dataset",
    "input_construction",
    "context_length",
    "generation_length",
    "batch_size",
    "decode_loop",
    "decoding_configuration",
    "hardware",
    "evaluation_metrics",
    "attention_implementation",
    "evaluation_orchestrator",
    "kv_interception_engine",
)

DEFAULT_DECODING_CONFIGURATION: dict[str, Any] = {
    "strategy": "greedy",
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "selection": "argmax",
    "engine": "framework/kv_engine.py::KVCacheEngine.generate",
}


def detect_hardware(device: torch.device | None = None) -> dict[str, Any]:
    """Best-effort hardware profile for the current eval runtime."""
    dev = device or torch.device("cpu")
    profile: dict[str, Any] = {
        "device_type": dev.type,
        "device_index": dev.index,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "kv_eval_device_env": os.environ.get("KV_EVAL_DEVICE") or None,
        "hardware_profile_env": os.environ.get("KV_HARDWARE_PROFILE") or None,
    }

    if dev.type == "cuda" and torch.cuda.is_available():
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        profile["device_name"] = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        profile["total_memory_bytes"] = props.total_memory
        profile["compute_capability"] = f"{props.major}.{props.minor}"
    elif dev.type == "mps" and torch.backends.mps.is_available():
        profile["device_name"] = "Apple MPS"
    else:
        profile["device_name"] = "CPU"

    # Reference sweeps document Modal A10G; allow explicit override without probing cloud.
    if profile["hardware_profile_env"]:
        profile["documented_reference_gpu"] = profile["hardware_profile_env"]
    else:
        profile["documented_reference_gpu"] = "NVIDIA A10G (Modal reference sweeps)"

    return profile


def build_tokenizer_metadata(
    tokenizer: PreTrainedTokenizerBase,
    *,
    model_path: Path | str | None = None,
) -> dict[str, Any]:
    """Tokenizer axis for controlled comparisons."""
    name_or_path = getattr(tokenizer, "name_or_path", None) or str(model_path or "")
    return {
        "name_or_path": name_or_path,
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }


def build_input_construction(
    *,
    dataset: str,
    dataset_split: str,
    context_length: int,
    generation_length: int,
    run_retrieval: bool,
    run_instruction_following: bool,
    run_reasoning: bool,
) -> dict[str, Any]:
    """How prompts/inputs are built for each evaluation branch."""
    wikitext_block = {
        "source": "wikitext2_concat",
        "dataset": dataset,
        "split": dataset_split,
        "builder": "data.loader.build_long_context_ids",
        "separator": "\n\n",
        "target_length": context_length,
        "add_special_tokens": False,
    }
    construction: dict[str, Any] = {
        "fidelity": dict(wikitext_block),
        "behavior_perplexity": dict(wikitext_block),
        "system_throughput_prefix": dict(wikitext_block),
    }
    if run_retrieval:
        construction["behavior_retrieval"] = {
            "source": "synthetic_needle_in_haystack",
            "module": "eval/behavior/retrieval.py",
            "context_length": context_length,
        }
    if run_instruction_following:
        construction["behavior_instruction_following"] = {
            "source": "synthetic_instruction_compliance",
            "module": "eval/behavior/instruction_following.py",
        }
    if run_reasoning:
        construction["behavior_reasoning"] = {
            "source": "synthetic_arithmetic_chain",
            "module": "eval/behavior/reasoning.py",
        }
    construction["system_throughput_prefix"]["generation_length"] = generation_length
    return construction


def build_decoding_configuration(
    *,
    generation_length: int,
    perplexity_stride: int,
) -> dict[str, Any]:
    """Decoding axis — KVCacheEngine uses deterministic greedy decode."""
    cfg = dict(DEFAULT_DECODING_CONFIGURATION)
    cfg["generation_length"] = generation_length
    cfg["perplexity_stride"] = perplexity_stride
    cfg["perplexity_scoring"] = "sliding_window_teacher_forcing"
    return cfg


def build_evaluation_metrics_profile(
    *,
    run_fidelity: bool,
    run_behavior: bool,
    run_perplexity: bool,
    run_retrieval: bool,
    run_reasoning: bool,
    run_instruction_following: bool,
    run_system: bool,
    run_throughput: bool,
    run_peak_memory: bool,
    run_memory_bandwidth: bool,
    run_kernel_cost: bool,
    run_gpu_utilization: bool,
    run_cost: bool,
    attention_fidelity_tokens: int,
) -> dict[str, Any]:
    """Which FIDELITY / BEHAVIOR / SYSTEM metrics are enabled for this run."""
    fidelity_metrics: list[str] = []
    if run_fidelity:
        fidelity_metrics = [
            "representation",
            "attention",
            "memory",
            "recurrent",
        ]

    behavior_metrics: list[str] = []
    if run_behavior:
        if run_perplexity:
            behavior_metrics.append("perplexity")
        if run_retrieval:
            behavior_metrics.append("retrieval")
        if run_instruction_following:
            behavior_metrics.append("instruction_following")
        if run_reasoning:
            behavior_metrics.append("reasoning")

    system_metrics: list[str] = []
    if run_system:
        if run_throughput:
            system_metrics.extend(["ttft", "itl", "throughput", "latency"])
        if run_peak_memory:
            system_metrics.append("peak_memory")
        if run_memory_bandwidth:
            system_metrics.append("memory_bandwidth")
        if run_kernel_cost:
            system_metrics.append("kernel_cost")
        if run_gpu_utilization:
            system_metrics.append("gpu_utilization")

    return {
        "branches": {
            "fidelity": run_fidelity,
            "behavior": run_behavior,
            "system": run_system,
            "cost": run_cost,
        },
        "fidelity": fidelity_metrics,
        "behavior": behavior_metrics,
        "system": system_metrics,
        "attention_fidelity_tokens": attention_fidelity_tokens,
        "definitions": "eval/{fidelity,behavior,system,cost}/",
    }


def extract_compression_budget(compressor: KVCompressor) -> dict[str, Any]:
    """Method-specific compression budget — the knob that defines aggressiveness."""
    budget: dict[str, Any] = {
        "compressor": compressor.name,
        "bitwidth": getattr(compressor, "bitwidth", None),
    }

    stage = getattr(compressor, "stage", None)
    if stage is not None:
        budget["stage"] = stage.value if hasattr(stage, "value") else str(stage)

    for attr in (
        "token_budget",
        "hsa_budget",
        "window_size",
        "max_capacity_prompt",
        "kernel_size",
        "compression_rate",
        "group_size",
        "rank",
    ):
        if hasattr(compressor, attr):
            budget[attr] = getattr(compressor, attr)

    return budget


@dataclass(frozen=True)
class ControlledInterceptionContract:
    """Matched conditions for fair KV-transformation comparison."""

    fixed: dict[str, Any] = field(default_factory=dict)
    variable: dict[str, Any] = field(default_factory=dict)
    evaluation_branches: tuple[str, ...] = ("fidelity", "behavior", "system")
    principle: str = PHASE7_PRINCIPLE
    phase: str = "7"

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "principle": self.principle,
            "fixed": self.fixed,
            "variable": self.variable,
            "evaluation_branches": list(self.evaluation_branches),
        }


def validate_controlled_contract(contract: ControlledInterceptionContract) -> None:
    """Raise ValueError if any Phase 7 fixed axis is missing."""
    missing = [key for key in REQUIRED_FIXED_AXES if key not in contract.fixed]
    if missing:
        raise ValueError(f"controlled contract missing fixed axes: {missing}")
    if "compressor" not in contract.variable:
        raise ValueError("controlled contract missing variable.compressor")
    if "compression_budget" not in contract.variable:
        raise ValueError("controlled contract missing variable.compression_budget")


def build_controlled_conditions(
    *,
    model_metadata: dict[str, Any] | None,
    eval_config: dict[str, Any],
    context_length: int,
    compressor: KVCompressor,
    tokenizer: PreTrainedTokenizerBase | None = None,
    model_path: Path | str | None = None,
    device: torch.device | None = None,
    dataset: str | None = None,
    dataset_split: str | None = None,
    perplexity_stride: int | None = None,
    generation_length: int | None = None,
    run_fidelity: bool = True,
    run_behavior: bool = True,
    run_perplexity: bool = True,
    run_retrieval: bool = True,
    run_reasoning: bool = False,
    run_instruction_following: bool = True,
    run_system: bool = True,
    run_throughput: bool = True,
    run_peak_memory: bool = False,
    run_memory_bandwidth: bool = False,
    run_kernel_cost: bool = False,
    run_gpu_utilization: bool = False,
    run_cost: bool = True,
) -> ControlledInterceptionContract:
    """Build the Phase 7 controlled-conditions contract for one evaluation run."""
    wikitext_cfg = eval_config.get("wikitext", {})
    ds_name = dataset or wikitext_cfg.get("config", "wikitext-2-raw-v1")
    ds_split = dataset_split or wikitext_cfg.get("split", "test")
    stride = perplexity_stride if perplexity_stride is not None else eval_config.get("perplexity_stride", 512)
    gen_len = generation_length if generation_length is not None else eval_config.get("generated_tokens", 64)
    attn_window = eval_config.get("attention_fidelity_tokens", 512)

    fixed: dict[str, Any] = {
        "model": model_metadata,
        "tokenizer": (
            None
            if tokenizer is None
            else build_tokenizer_metadata(tokenizer, model_path=model_path)
        ),
        "dataset": {
            "name": wikitext_cfg.get("name", "Salesforce/wikitext"),
            "config": ds_name,
            "split": ds_split,
        },
        "input_construction": build_input_construction(
            dataset=ds_name,
            dataset_split=ds_split,
            context_length=context_length,
            generation_length=gen_len,
            run_retrieval=run_retrieval,
            run_instruction_following=run_instruction_following,
            run_reasoning=run_reasoning,
        ),
        "context_length": context_length,
        "generation_length": gen_len,
        "batch_size": eval_config.get("batch_size", 1),
        "decode_loop": "incremental_kv_engine_no_recompression",
        "decoding_configuration": build_decoding_configuration(
            generation_length=gen_len,
            perplexity_stride=stride,
        ),
        "hardware": detect_hardware(device),
        "evaluation_metrics": build_evaluation_metrics_profile(
            run_fidelity=run_fidelity,
            run_behavior=run_behavior,
            run_perplexity=run_perplexity,
            run_retrieval=run_retrieval,
            run_reasoning=run_reasoning,
            run_instruction_following=run_instruction_following,
            run_system=run_system,
            run_throughput=run_throughput,
            run_peak_memory=run_peak_memory,
            run_memory_bandwidth=run_memory_bandwidth,
            run_kernel_cost=run_kernel_cost,
            run_gpu_utilization=run_gpu_utilization,
            run_cost=run_cost,
            attention_fidelity_tokens=attn_window,
        ),
        "attention_implementation": "eager",
        "attention_fidelity_tokens": attn_window,
        "perplexity_stride": stride,
        "evaluation_orchestrator": "eval/runner.py",
        "kv_interception_engine": "framework/kv_engine.py",
    }

    variable: dict[str, Any] = {
        "compressor": compressor.name,
        "compression_budget": extract_compression_budget(compressor),
    }

    contract = ControlledInterceptionContract(fixed=fixed, variable=variable)
    validate_controlled_contract(contract)
    return contract
