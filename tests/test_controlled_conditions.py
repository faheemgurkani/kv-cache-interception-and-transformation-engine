"""Unit tests for Phase 6 controlled interception contract."""

from eval.controlled_conditions import (
    PHASE6_PRINCIPLE,
    build_controlled_conditions,
)


def test_build_controlled_conditions_fixed_vs_variable():
    contract = build_controlled_conditions(
        model_metadata={"model_id": "Qwen/Qwen3-1.7B"},
        eval_config={
            "batch_size": 1,
            "perplexity_stride": 512,
            "generated_tokens": 64,
            "attention_fidelity_tokens": 512,
        },
        context_length=256,
        compressor_name="turboquant",
        bitwidth=4,
        stage="full",
    )

    assert contract.principle == PHASE6_PRINCIPLE
    assert contract.fixed["context_length"] == 256
    assert contract.fixed["decode_loop"] == "incremental_kv_engine_no_recompression"
    assert contract.fixed["model"]["model_id"] == "Qwen/Qwen3-1.7B"
    assert contract.variable["compressor"] == "turboquant"
    assert contract.variable["bitwidth"] == 4
    assert contract.variable["stage"] == "full"
    assert contract.evaluation_branches == ("fidelity", "behavior", "system")

    payload = contract.to_dict()
    assert payload["principle"] == PHASE6_PRINCIPLE
    assert "fidelity" in payload["evaluation_branches"]
    assert payload["variable"]["compressor"] == "turboquant"
