"""Verify direct access to past_key_values for KV-cache compression research."""

from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen3_1.7b"


def _first_layer_kv(past_key_values):
    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[0]
        return layer.keys, layer.values
    if hasattr(past_key_values, "key_cache"):
        return past_key_values.key_cache[0], past_key_values.value_cache[0]
    key, value = past_key_values[0]
    return key, value


def main() -> None:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_DIR}. Run scripts/download_model.py first."
        )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto")
    model = model.to(device)
    model.eval()

    input_ids = tokenizer("Hello world", return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)

    key, value = _first_layer_kv(outputs.past_key_values)
    print(f"Key shape:   {tuple(key.shape)}")
    print(f"Value shape: {tuple(value.shape)}")
    print("KV cache access verified.")


if __name__ == "__main__":
    main()
