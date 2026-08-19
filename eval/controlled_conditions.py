"""Phase 6: controlled KV interception contract.

Documents which experimental axes are held fixed vs. which may vary across
method comparisons. Only the KV transformation (compressor plug-in) changes
between runs; model, input construction, decode loop, and metric definitions
stay shared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PHASE6_PRINCIPLE = (
    "Different KV transformations are executed through the same inference path "
    "under matched conditions; only the compressor plug-in varies."
)


@dataclass(frozen=True)
class ControlledInterceptionContract:
    """Matched conditions for fair KV-transformation comparison."""

    fixed: dict[str, Any] = field(default_factory=dict)
    variable: dict[str, Any] = field(default_factory=dict)
    evaluation_branches: tuple[str, ...] = ("fidelity", "behavior", "system")
    principle: str = PHASE6_PRINCIPLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "principle": self.principle,
            "fixed": self.fixed,
            "variable": self.variable,
            "evaluation_branches": list(self.evaluation_branches),
        }


def build_controlled_conditions(
    *,
    model_metadata: dict[str, Any] | None,
    eval_config: dict[str, Any],
    context_length: int,
    compressor_name: str,
    bitwidth: int | None = None,
    stage: str | None = None,
    dataset: str = "wikitext-2-raw-v1",
) -> ControlledInterceptionContract:
    """Build the Phase 6 contract for one evaluation run."""
    fixed: dict[str, Any] = {
        "model": model_metadata,
        "dataset": dataset,
        "context_length": context_length,
        "batch_size": eval_config.get("batch_size", 1),
        "perplexity_stride": eval_config.get("perplexity_stride", 512),
        "generated_tokens": eval_config.get("generated_tokens", 64),
        "attention_fidelity_tokens": eval_config.get("attention_fidelity_tokens", 512),
        "decode_loop": "incremental_kv_engine_no_recompression",
        "attention_implementation": "eager",
        "evaluation_orchestrator": "eval/runner.py",
        "metric_definitions": "eval/{fidelity,behavior,system}/",
        "kv_interception_engine": "framework/kv_engine.py",
    }
    variable: dict[str, Any] = {
        "compressor": compressor_name,
        "bitwidth": bitwidth,
        "stage": stage,
    }
    return ControlledInterceptionContract(fixed=fixed, variable=variable)
