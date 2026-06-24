"""Perplexity evaluation for compressed KV caches."""

from __future__ import annotations

from pathlib import Path

import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_eval_config(config_path: Path | str = "configs/eval.yaml") -> dict:
    with Path(config_path).open() as f:
        return yaml.safe_load(f)


def load_wikitext2(config: dict | None = None):
    config = config or load_eval_config()
    wikitext_cfg = config["wikitext"]
    return load_dataset(wikitext_cfg["name"], wikitext_cfg["config"], split=wikitext_cfg["split"])


def evaluate_perplexity(model_path: str, dataset=None) -> float:
    """Compute perplexity. Full implementation pending."""
    _ = AutoTokenizer.from_pretrained(model_path)
    _ = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    _ = dataset or load_wikitext2()
    raise NotImplementedError("Perplexity evaluation not yet implemented.")
