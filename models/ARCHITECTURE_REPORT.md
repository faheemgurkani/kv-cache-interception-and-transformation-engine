# Architecture Matrix — Downloaded Model Report

## allenai/OLMo-1B-hf → `olmo1b` (MHA)

- `model_type`: olmo
- `hidden_size`: 2048
- `num_hidden_layers`: 16
- `num_attention_heads`: 16
- `num_key_value_heads`: 16
- weight files: ['model.safetensors']
- other artifacts: ['generation_config.json', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json']

## Qwen/Qwen3-0.6B → `qwen3_0.6b` (GQA)

- `model_type`: qwen3
- `hidden_size`: 1024
- `num_hidden_layers`: 28
- `num_attention_heads`: 16
- `num_key_value_heads`: 8
- weight files: ['model.safetensors']
- other artifacts: ['generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json']

## FreedomIntelligence/TinyDeepSeek-0.5B-base → `tinydeepseek_0.5b` (MLA)

- `model_type`: deepseek_v3
- `hidden_size`: 1024
- `num_hidden_layers`: 26
- `num_attention_heads`: 4
- `num_key_value_heads`: 4
- `kv_lora_rank`: 256
- `qk_nope_head_dim`: 32
- `qk_rope_head_dim`: 32
- weight files: ['model.safetensors']
- other artifacts: ['configuration_tinydeepseek.py', 'generation_config.json', 'modeling_tinydeepseek.py', 'tokenizer.json', 'tokenizer_config.json']

## tiiuae/Falcon-H1-0.5B-Base → `falcon_h1_0.5b` (Hybrid Attention + Mamba2)

- `model_type`: falcon_h1
- `hidden_size`: 1024
- `num_hidden_layers`: 36
- `num_attention_heads`: 8
- `num_key_value_heads`: 2
- `mamba_n_heads`: 24
- weight files: ['model.safetensors']
- other artifacts: ['generation_config.json', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json']

## google/gemma-3-270m → `gemma3_270m` (MQA + local/global attention)

- `model_type`: gemma3_text
- `hidden_size`: 640
- `num_hidden_layers`: 18
- `num_attention_heads`: 4
- `num_key_value_heads`: 1
- `layer_types`: ['sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention']...['sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention'] (len=18)
- weight files: ['model.safetensors']
- other artifacts: ['added_tokens.json', 'generation_config.json', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json']
