"""Audit B5: Palu FIDELITY must reflect its real (lossy) reconstruction."""

from pathlib import Path

import pytest
import torch

GEMMA3 = Path(__file__).resolve().parent.parent / "models" / "gemma3_270m"

_TEXT = (
    "The history of the printing press is closely tied to the spread of literacy in Europe. "
    "Before movable type, books were copied by hand, which made them rare and expensive. "
    "Johannes Gutenberg introduced a system that let printers assemble pages from reusable "
    "metal letters, and within a few decades presses had appeared in hundreds of towns. "
) * 2


def test_palu_attention_fidelity_is_not_a_pass_through():
    pytest.importorskip("transformers")
    if not (GEMMA3 / "config.json").exists():
        pytest.skip("Gemma3 checkpoint not present")
    from compressors.registry import get_compressor
    from eval.fidelity import evaluate_fidelity
    from framework.model import ModelLayer

    layer = ModelLayer(model_path=GEMMA3, device=torch.device("cpu"), torch_dtype=torch.float32)
    ids = layer.tokenize(_TEXT)[:, :160]
    comp = get_compressor("palu", compression_rate=0.25, group_size=1)
    comp.bind_model(layer.model)
    fid = evaluate_fidelity(layer, ids, comp)

    assert fid.representation.key_rmse > 0
    # Used to be exactly cos=1 / rmse=0 / KL=0 because compress_kv is a clone().
    assert fid.attention.rmse > 0
    assert fid.attention.cosine_similarity < 1.0
    assert fid.attention.distribution_kl_divergence is not None
    assert fid.attention.distribution_kl_divergence > 0
    # Shared factors are excluded from the headline ratio; the inclusive one is reported.
    assert fid.memory.compression_ratio_including_shared is not None
    assert fid.memory.compression_ratio_including_shared < fid.memory.compression_ratio


def test_palu_reports_no_calibration_data():
    from compressors.palu import PaluCompressor

    meta = PaluCompressor().offline_cost_metadata()
    assert meta.calibration_required is False
    assert meta.calibration_dataset is None and meta.calibration_tokens is None
