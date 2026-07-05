"""Evaluation job specifications for Modal parallel sweeps."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from modal_app.settings import project_root

SWEEPS_CONFIG_PATH = project_root() / "configs" / "modal_sweeps.yaml"
PRESET_ORDER = ("baseline", "turboquant", "qjl", "rocketkv")


@dataclass
class EvalJobSpec:
    compressor: str
    context_length: int
    bitwidth: int | None = None
    stage: str | None = None
    label: str = ""
    skip_perplexity: bool = False
    skip_throughput: bool = False
    compressor_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> EvalJobSpec:
        payload = dict(data)
        payload.setdefault("compressor_kwargs", {})
        return cls(**payload)

    def get_compressor_kwargs(self) -> dict[str, Any]:
        kwargs = dict(self.compressor_kwargs)
        if self.bitwidth is not None:
            kwargs["bitwidth"] = self.bitwidth
        if self.stage is not None:
            kwargs["stage"] = self.stage
        return kwargs

    @property
    def result_stem(self) -> str:
        if "keep_ratio" in self.compressor_kwargs:
            keep_ratio = float(self.compressor_kwargs["keep_ratio"])
            window_size = int(self.compressor_kwargs.get("window_size", 32))
            dynamic_top_k = int(self.compressor_kwargs.get("dynamic_top_k", 64))
            r = int(round(keep_ratio * 100))
            return (
                f"{self.label}_ctx{self.context_length}_r{r}"
                f"_ws{window_size}_k{dynamic_top_k}"
            )

        bit = self.bitwidth if self.bitwidth is not None else "na"
        stage = self.stage or "na"
        return f"{self.label}_ctx{self.context_length}_b{bit}_{stage}"


def load_sweeps_config(path: Path | str | None = None) -> dict:
    config_path = Path(path) if path else SWEEPS_CONFIG_PATH
    with config_path.open() as handle:
        return yaml.safe_load(handle)


def _parse_preset_entries(entries: list[dict]) -> list[tuple[str, dict]]:
    parsed: list[tuple[str, dict]] = []
    for entry in entries:
        item = dict(entry)
        label = item.pop("label")
        parsed.append((label, item))
    return parsed


def get_sweep_configs(preset: str = "turboquant") -> list[tuple[str, dict]]:
    presets = load_sweeps_config().get("presets", {})
    if preset not in presets and preset != "all":
        available = ", ".join(sorted(set(presets) | {"all"}))
        raise ValueError(f"Unknown sweep preset '{preset}'. Available: {available}")

    if preset == "all":
        combined: list[tuple[str, dict]] = []
        seen_labels: set[str] = set()
        for preset_name in PRESET_ORDER:
            if preset_name == "baseline":
                continue
            for label, cfg in _parse_preset_entries(presets.get(preset_name, [])):
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                combined.append((label, cfg))
        return combined

    return _parse_preset_entries(presets[preset])


def turboquant_sweep_configs() -> list[tuple[str, dict]]:
    """Backward-compatible alias for the original TurboQuant sweep grid."""
    return get_sweep_configs("turboquant")


def default_context_lengths() -> list[int]:
    from framework.config import load_model_config

    return list(load_model_config().get("context_lengths", [128, 256, 512]))


def build_sweep_jobs(
    context_lengths: list[int] | None = None,
    labels: list[str] | None = None,
    preset: str = "turboquant",
    skip_perplexity: bool = False,
    skip_throughput: bool = False,
) -> list[EvalJobSpec]:
    lengths = context_lengths or default_context_lengths()
    sweep_configs = get_sweep_configs(preset)
    jobs: list[EvalJobSpec] = []
    for ctx in lengths:
        for label, cfg in sweep_configs:
            if labels and label not in labels:
                continue
            params = dict(cfg)
            name = params.pop("name")
            bitwidth = params.pop("bitwidth", None)
            stage = params.pop("stage", None)
            jobs.append(
                EvalJobSpec(
                    compressor=name,
                    context_length=ctx,
                    bitwidth=bitwidth,
                    stage=stage,
                    label=label,
                    skip_perplexity=skip_perplexity,
                    skip_throughput=skip_throughput,
                    compressor_kwargs=params,
                )
            )
    return jobs


def filter_existing_jobs(
    jobs: list[EvalJobSpec],
    completed_stems: set[str],
) -> list[EvalJobSpec]:
    return [job for job in jobs if job.result_stem not in completed_stems]
