"""Evaluation job specifications for Modal parallel sweeps."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from framework.config import load_model_config


@dataclass(frozen=True)
class EvalJobSpec:
    compressor: str
    context_length: int
    bitwidth: int | None = None
    stage: str | None = None
    label: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


SWEEP_CONFIGS: list[tuple[str, dict]] = [
    ("identity_baseline", {"name": "identity"}),
    ("tq_full_b2", {"name": "turboquant", "stage": "full", "bitwidth": 2}),
    ("tq_full_b3", {"name": "turboquant", "stage": "full", "bitwidth": 3}),
    ("tq_full_b4", {"name": "turboquant", "stage": "full", "bitwidth": 4}),
    ("tq_mse_b4", {"name": "turboquant", "stage": "wht_quant", "bitwidth": 4}),
]


def build_sweep_jobs(
    context_lengths: list[int] | None = None,
) -> list[EvalJobSpec]:
    model_config = load_model_config()
    lengths = context_lengths or model_config["context_lengths"]
    jobs: list[EvalJobSpec] = []
    for ctx in lengths:
        for label, cfg in SWEEP_CONFIGS:
            params = dict(cfg)
            name = params.pop("name")
            jobs.append(
                EvalJobSpec(
                    compressor=name,
                    context_length=ctx,
                    bitwidth=params.get("bitwidth"),
                    stage=params.get("stage"),
                    label=label,
                )
            )
    return jobs
