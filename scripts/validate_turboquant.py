"""Step-by-step TurboQuant validation (Phases 2–4 of the implementation plan)."""

from __future__ import annotations

import argparse

import setup_path  # noqa: F401
import torch

from compressors.turboquant import TurboQuantCompressor
from framework.model import ModelLayer
from quantizers.turboquant_pipeline import TurboQuantStage


def validate_stages() -> None:
    key = torch.randn(1, 8, 4, 128)
    for stage in TurboQuantStage:
        compressor = TurboQuantCompressor(bitwidth=4, stage=stage)
        payload = compressor.compress(key, key, layer=0)
        restored_k, _ = compressor.decompress(payload)
        rmse = (key.float() - restored_k.float()).pow(2).mean().sqrt().item()
        print(f"[{stage.value}] nbytes={payload.nbytes:,}  key_rmse={rmse:.6f}")


def validate_kv_intercept(context_length: int = 64) -> None:
    model_layer = ModelLayer()
    compressor = TurboQuantCompressor(bitwidth=4, stage=TurboQuantStage.FULL)
    engine = model_layer.make_kv_engine(compressor)

    text = "TurboQuant KV interception test sequence."
    input_ids = model_layer.tokenize(text)[:, :context_length]

    outputs = model_layer.forward_with_cache(input_ids)
    cache = engine.compress_existing_cache(outputs.past_key_values)
    print(f"Compressed cache: {len(cache.layers)} layers, {cache.nbytes:,} bytes")

    logits, new_cache = engine.step(input_ids[:, :1], compressed_cache=cache)
    print(f"Step logits shape: {tuple(logits.shape)}")
    print(f"Post-step cache: {new_cache.nbytes:,} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TurboQuant pipeline stages.")
    parser.add_argument(
        "--phase",
        choices=["stages", "intercept", "all"],
        default="all",
    )
    parser.add_argument("--context-length", type=int, default=64)
    args = parser.parse_args()

    if args.phase in ("stages", "all"):
        print("=== TurboQuant stage ablation ===")
        validate_stages()

    if args.phase in ("intercept", "all"):
        print("\n=== KV cache interception ===")
        validate_kv_intercept(args.context_length)


if __name__ == "__main__":
    main()
