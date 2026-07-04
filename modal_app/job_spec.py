"""Evaluation job specifications for Modal parallel sweeps."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EvalJobSpec:
    compressor: str
    context_length: int
    bitwidth: int | None = None
    stage: str | None = None
    label: str = ""
    skip_perplexity: bool = False
    skip_throughput: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def result_stem(self) -> str:
        bit = self.bitwidth if self.bitwidth is not None else "na"
        stage = self.stage or "na"
        return f"{self.label}_ctx{self.context_length}_b{bit}_{stage}"


SWEEP_CONFIGS: list[tuple[str, dict]] = [
    ("identity_baseline", {"name": "identity"}),
    ("tq_full_b2", {"name": "turboquant", "stage": "full", "bitwidth": 2}),
    ("tq_full_b3", {"name": "turboquant", "stage": "full", "bitwidth": 3}),
    ("tq_full_b4", {"name": "turboquant", "stage": "full", "bitwidth": 4}),
    ("tq_mse_b4", {"name": "turboquant", "stage": "wht_quant", "bitwidth": 4}),
]


def default_context_lengths() -> list[int]:
    from framework.config import load_model_config

    return list(load_model_config().get("context_lengths", [128, 256, 512]))


def build_sweep_jobs(
    context_lengths: list[int] | None = None,
    labels: list[str] | None = None,
    skip_perplexity: bool = False,
    skip_throughput: bool = False,
) -> list[EvalJobSpec]:
    lengths = context_lengths or default_context_lengths()
    jobs: list[EvalJobSpec] = []
    for ctx in lengths:
        for label, cfg in SWEEP_CONFIGS:
            if labels and label not in labels:
                continue
            params = dict(cfg)
            name = params.pop("name")
            jobs.append(
                EvalJobSpec(
                    compressor=name,
                    context_length=ctx,
                    bitwidth=params.get("bitwidth"),
                    stage=params.get("stage"),
                    label=label,
                    skip_perplexity=skip_perplexity,
                    skip_throughput=skip_throughput,
                )
            )
    return jobs


def filter_existing_jobs(
    jobs: list[EvalJobSpec],
    completed_stems: set[str],
) -> list[EvalJobSpec]:
    return [job for job in jobs if job.result_stem not in completed_stems]
