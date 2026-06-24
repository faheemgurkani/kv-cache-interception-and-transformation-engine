"""Verify KV-cache access on downloaded Qwen3-1.7B model."""

from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen3_1.7b"


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Model not downloaded")
def test_kv_cache_shapes():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto")
    model = model.to(device)
    model.eval()

    input_ids = tokenizer("Hello world", return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)

    pkv = outputs.past_key_values
    if hasattr(pkv, "layers"):
        key, value = pkv.layers[0].keys, pkv.layers[0].values
    elif hasattr(pkv, "key_cache"):
        key, value = pkv.key_cache[0], pkv.value_cache[0]
    else:
        key, value = pkv[0]

    assert key.ndim == 4
    assert value.ndim == 4
    assert key.shape == value.shape
