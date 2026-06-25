"""Perplexity evaluation with compressed KV in the autoregressive loop."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from tqdm import tqdm

from compressors.base import KVCompressor
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

    Each token step runs through KVCacheEngine: compress after forward, decompress before next step.
    """
    device = model_layer.device
    engine = model_layer.make_kv_engine(compressor)
    max_length = max_length or getattr(model_layer.config, "max_position_embeddings", input_ids.size(1))
    seq_len = input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, stride), desc="ppl-compressed", leave=False):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        window = input_ids[:, begin_loc:end_loc].to(device)
        window_len = window.size(1)
        eval_start = window_len - trg_len

        cache = None
        for t in range(window_len):
            logits, cache = engine.step(window[:, t : t + 1], compressed_cache=cache)
            target_idx = t + 1
            if target_idx >= eval_start and target_idx < window_len:
                target = window[:, target_idx]
                nll = F.cross_entropy(logits[:, -1, :], target, reduction="sum")
                nll_sum += nll.item()
                n_tokens += 1

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    if n_tokens == 0:
        raise ValueError("No tokens available for compressed perplexity evaluation.")

    return math.exp(nll_sum / n_tokens)
