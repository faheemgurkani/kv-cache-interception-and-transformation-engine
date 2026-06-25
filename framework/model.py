"""Model layer: load Qwen3 with eager attention for KV interception."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from framework.config import PROJECT_ROOT, load_model_config
from framework.device import get_device


class ModelLayer:
    """Wrapper around a causal LM with consistent KV-cache access."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        device: torch.device | None = None,
        torch_dtype: torch.dtype | str = torch.float16,
        attn_implementation: str = "eager",
    ) -> None:
        config = load_model_config()
        self.model_path = Path(model_path or PROJECT_ROOT / config["local_path"])
        self.device = device or get_device()
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch_dtype,
            attn_implementation=attn_implementation,
        ).to(self.device)
        self.model.config.use_cache = True
        self.model.eval()

    @property
    def config(self):
        return self.model.config

    def tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    @torch.no_grad()
    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool = True,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        return self.model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=attention_mask,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        past_key_values=None,
    ) -> torch.Tensor:
        return self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            past_key_values=past_key_values,
            use_cache=True,
            do_sample=False,
        )

    def make_kv_engine(self, compressor):
        from framework.kv_engine import KVCacheEngine

        return KVCacheEngine(self.model, compressor)
