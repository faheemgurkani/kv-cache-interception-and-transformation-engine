# KV-Cache Compression Benchmark

Reproduction and benchmarking framework for KV-cache compression methods:

**TurboQuant → KIVI → QJL → RocketKV**

Built on **Qwen3-1.7B** ([Hugging Face](https://huggingface.co/Qwen/Qwen3-1.7B)) with PyTorch and HuggingFace Transformers, optimized for Apple Silicon (MPS).

## Model

| Property | Value |
|---|---|
| Model | `Qwen/Qwen3-1.7B` |
| Parameters | 1.7B |
| Context | 32k |
| Architecture | GQA (Group Query Attention) |
| Backend | HuggingFace Transformers (not GGUF/Ollama) |

## Requirements

- Python 3.11
- Apple Silicon Mac (MPS) or CPU

> **Note:** `fast-hadamard-transform` requires CUDA/nvcc and is not available on Apple Silicon. TurboQuant Hadamard steps will use a fallback implementation until run on a CUDA machine.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your HuggingFace token:

```bash
HF_TOKEN=your_token_here
```

The `HF_TOKEN` environment variable is used automatically by `huggingface_hub` and Transformers for authentication.

## Download Model

```bash
python scripts/download_model.py
```

This saves the model to `models/qwen3_1.7b/`.

## Verify KV Cache Access

```bash
python scripts/verify_kv_cache.py
```

Confirms direct access to `past_key_values` — the foundation for all compression methods.

## Project Structure

```text
kv-cache-compression-benchmark/
├── configs/          # model.yaml, eval.yaml
├── models/           # downloaded model weights (gitignored)
├── quantizers/       # TurboQuant, QJL quantizer modules
├── baselines/        # KIVI, QJL, RocketKV baselines
├── eval/             # perplexity, memory, throughput
├── datasets/         # wikitext2, c4
├── results/          # experiment outputs
├── plots/            # figures
├── notebooks/        # exploratory analysis
├── scripts/          # download, verify, run baselines
└── tests/            # pytest suite
```

## Configuration

`configs/model.yaml` — model name, context lengths (4096–32768), bitwidths (2–4).

`configs/eval.yaml` — datasets (WikiText-2, C4), metrics, batch size.

## Datasets

```python
from datasets import load_dataset

load_dataset("wikitext", "wikitext-2-raw-v1")
```

Local cache directories: `datasets/wikitext2/`, `datasets/c4/`.

## Running Baselines

```bash
python scripts/run_baseline.py --baseline kivi
```

## Testing

```bash
pytest tests/
```

## Roadmap

1. **Phase 0** — Repository setup, model download, KV cache verification ✅
2. **Phase 1** — TurboQuant implementation
3. **Phase 2** — KIVI baseline
4. **Phase 3** — QJL baseline
5. **Phase 4** — RocketKV baseline

## License

See individual paper implementations for respective licenses.
