"""BEHAVIOR / Task quality: perplexity with compressed KV driving decode.

Protocol (identical for every compressor, and for the uncompressed baseline):

1. Prefill the first ``P = seq_len - decode_tokens`` tokens through ``KVCacheEngine``
   (the cache is compressed after the forward pass).
2. Teacher-force the remaining ``decode_tokens`` tokens one at a time, each step reading
   the *compressed* cache, and score the next-token NLL of every decode position
   (plus the prefill's last logit, which predicts token ``P``).

Why not score the prefill logits: prefill logits are computed before any compression
touches the KV cache, so a compressor that only transforms the stored cache (TurboQuant,
Palu, ...) would have perplexity identical to the uncompressed model regardless of how
lossy it is (audit B1, docs/audits/GEMMA3_POST_OVERHAUL_AUDIT.md). Scoring only
positions whose logits are produced *from* the compressed cache fixes that, and keeps
causality: no compressor ever sees future tokens.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from compressors.base import KVCompressor
from framework.model import ModelLayer

# Matches the generation length used by the SYSTEM throughput pass.
DEFAULT_PPL_DECODE_TOKENS = 64


def default_ppl_decode_tokens() -> int:
    """64 unless ``KV_PPL_DECODE_TOKENS`` is set (the test suite lowers it for CPU smoke runs)."""
    raw = os.environ.get("KV_PPL_DECODE_TOKENS", "").strip()
    return int(raw) if raw else DEFAULT_PPL_DECODE_TOKENS


@dataclass
class PerplexityResult:
    perplexity: float
    n_tokens: int
    nll_sum: float
    prefill_tokens: int = 0

    def __float__(self) -> float:
        return self.perplexity


def ppl_prefill_length(seq_len: int, decode_tokens: int = DEFAULT_PPL_DECODE_TOKENS) -> int:
    """Number of prefilled tokens; the remaining positions are scored under compression.

    Short inputs score at most half the sequence so there is always a non-trivial prefix.
    """
    if seq_len < 2:
        raise ValueError("Perplexity needs at least 2 tokens.")
    decode = max(1, min(int(decode_tokens), seq_len // 2))
    return seq_len - decode


def _window_ids(input_ids: torch.Tensor, max_length: int | None) -> torch.Tensor:
    """Keep the trailing ``max_length`` tokens so the cache never exceeds the window."""
    if max_length is not None and input_ids.size(1) > max_length:
        return input_ids[:, -max_length:]
    return input_ids


@torch.no_grad()
def evaluate_perplexity_baseline(
    model_layer: ModelLayer,
    input_ids: torch.Tensor,
    max_length: int | None = None,
    stride: int = 512,
    decode_tokens: int | None = None,
) -> float:
    """Uncompressed reference perplexity over the same scored positions."""
    return evaluate_perplexity_baseline_result(
        model_layer, input_ids, max_length=max_length, stride=stride, decode_tokens=decode_tokens
    ).perplexity


@torch.no_grad()
def evaluate_perplexity_baseline_result(
    model_layer: ModelLayer,
    input_ids: torch.Tensor,
    max_length: int | None = None,
    stride: int = 512,  # retained for API compatibility; the protocol no longer strides
    decode_tokens: int | None = None,
) -> PerplexityResult:
    """One uncompressed forward over the sequence, scoring the same positions as the
    compressed protocol (targets ``P .. seq_len-1``)."""
    del stride
    decode_tokens = default_ppl_decode_tokens() if decode_tokens is None else decode_tokens
    ids = _window_ids(input_ids, max_length).to(model_layer.device)
    seq_len = ids.size(1)
    prefill_len = ppl_prefill_length(seq_len, decode_tokens)

    logits = model_layer.model(ids).logits  # [1, T, V]
    pred = logits[0, prefill_len - 1 : seq_len - 1, :].float()
    target = ids[0, prefill_len:seq_len]
    nll_sum = F.cross_entropy(pred, target, reduction="sum").item()
    n_tokens = int(target.numel())
    return PerplexityResult(
        perplexity=math.exp(nll_sum / n_tokens),
        n_tokens=n_tokens,
        nll_sum=nll_sum,
        prefill_tokens=prefill_len,
    )


@torch.no_grad()
def evaluate_perplexity(
    model_layer: ModelLayer,
    input_ids: torch.Tensor,
    compressor: KVCompressor,
    max_length: int | None = None,
    stride: int = 512,
    decode_tokens: int | None = None,
) -> float:
    """Perplexity with compressed KV driving decode (see module docstring)."""
    return evaluate_perplexity_result(
        model_layer,
        input_ids,
        compressor,
        max_length=max_length,
        stride=stride,
        decode_tokens=decode_tokens,
    ).perplexity


@torch.no_grad()
def evaluate_perplexity_result(
    model_layer: ModelLayer,
    input_ids: torch.Tensor,
    compressor: KVCompressor,
    max_length: int | None = None,
    stride: int = 512,  # retained for API compatibility; the protocol no longer strides
    decode_tokens: int | None = None,
) -> PerplexityResult:
    """Prefill, compress, then teacher-force ``decode_tokens`` tokens through the
    compressed cache, scoring each next-token prediction."""
    del stride
    decode_tokens = default_ppl_decode_tokens() if decode_tokens is None else decode_tokens
    device = model_layer.device
    if hasattr(compressor, "reset_state"):
        compressor.reset_state()
    engine = model_layer.make_kv_engine(compressor)

    ids = _window_ids(input_ids, max_length).to(device)
    seq_len = ids.size(1)
    prefill_len = ppl_prefill_length(seq_len, decode_tokens)

    prefill_ids = ids[:, :prefill_len]
    logits, cache = engine.step(
        prefill_ids,
        attention_mask=torch.ones(1, prefill_len, device=device, dtype=torch.long),
        compressed_cache=None,
        position_ids=torch.arange(prefill_len, device=device, dtype=torch.long).unsqueeze(0),
    )
    # The last prefill logit predicts token `prefill_len` (first scored position).
    nll_sum = F.cross_entropy(logits[:, -1, :].float(), ids[:, prefill_len], reduction="sum").item()
    n_tokens = 1

    for pos in range(prefill_len, seq_len - 1):
        logits, cache = engine.step(
            ids[:, pos : pos + 1],
            attention_mask=torch.ones(1, cache.seq_length + 1, device=device, dtype=torch.long),
            compressed_cache=cache,
            position_ids=torch.tensor([[pos]], device=device, dtype=torch.long),
        )
        nll_sum += F.cross_entropy(logits[:, -1, :].float(), ids[:, pos + 1], reduction="sum").item()
        n_tokens += 1

    return PerplexityResult(
        perplexity=math.exp(nll_sum / n_tokens),
        n_tokens=n_tokens,
        nll_sum=nll_sum,
        prefill_tokens=prefill_len,
    )
