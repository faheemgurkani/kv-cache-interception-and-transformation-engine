"""Perplexity evaluation with compressed KV in the autoregressive loop."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from tqdm import tqdm

from compressors.base import KVCompressor
from framework.kv_cache import trim_compressed_cache
from framework.kv_engine import CompressedCache
from framework.model import ModelLayer


@torch.no_grad()
def evaluate_perplexity_baseline(
    model_layer: ModelLayer,
    input_ids: torch.Tensor,
    max_length: int | None = None,
    stride: int = 512,
) -> float:
    """Uncompressed sliding-window perplexity (baseline reference only)."""
    model = model_layer.model
    device = model_layer.device
    max_length = max_length or getattr(model.config, "max_position_embeddings", input_ids.size(1))
    seq_len = input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, stride), desc="ppl-baseline", leave=False):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        window = input_ids[:, begin_loc:end_loc].to(device)
        target_ids = window.clone()
        target_ids[:, :-trg_len] = -100

        outputs = model(window, labels=target_ids)
        neg_log_likelihood = outputs.loss

        num_valid_tokens = (target_ids != -100).sum().item()
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size
        nll_sum += neg_log_likelihood.item() * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    if n_tokens == 0:
        raise ValueError("No tokens available for perplexity evaluation.")

    return math.exp(nll_sum / n_tokens)


def _maybe_trim_cache(cache: CompressedCache, max_length: int, compressor: KVCompressor) -> CompressedCache:
    if cache.seq_length <= max_length:
        return cache
    return trim_compressed_cache(cache, cache.seq_length - max_length, compressor)


@torch.no_grad()
def evaluate_perplexity(
    model_layer: ModelLayer,
    input_ids: torch.Tensor,
    compressor: KVCompressor,
    max_length: int | None = None,
    stride: int = 512,
) -> float:
    """
    Sliding-window perplexity with compressed KV stored between autoregressive steps.

    Uses a single incremental compressed cache across stride windows (no prefix replay).
    Passes an explicit attention mask so past KV positions are visible to the model.
    """
    device = model_layer.device
    engine = model_layer.make_kv_engine(compressor)
    max_length = max_length or getattr(
        model_layer.config, "max_position_embeddings", input_ids.size(1)
    )
    seq_len = input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    cache: CompressedCache | None = None

    for begin_loc in tqdm(range(0, seq_len, stride), desc="ppl-compressed", leave=False):
        end_loc = min(begin_loc + max_length, seq_len)
        score_from = max(1, prev_end_loc + 1)
        score_to = end_loc - 1

        for t in range(prev_end_loc, end_loc):
            token = input_ids[:, t : t + 1].to(device)
            attn_len = t + 1
            attention_mask = torch.ones(
                token.shape[0],
                attn_len,
                device=device,
                dtype=torch.long,
            )
            logits, cache = engine.step(token, attention_mask=attention_mask, compressed_cache=cache)

            target_pos = t + 1
            if score_from <= target_pos <= score_to:
                target = input_ids[:, target_pos].to(device)
                nll = F.cross_entropy(logits[:, -1, :], target, reduction="sum")
                nll_sum += nll.item()
                n_tokens += 1

        cache = _maybe_trim_cache(cache, max_length, compressor) if cache is not None else None
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    if n_tokens == 0:
        raise ValueError("No tokens available for compressed perplexity evaluation.")

    return math.exp(nll_sum / n_tokens)
