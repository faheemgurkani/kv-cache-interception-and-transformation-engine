# Contributing to the KV Cache Interception and Transformation Engine

Thank you for your interest in this project. The repository implements a unified KV interception and transformation engine with plug-in compressors and dual Section A/B evaluation. The reproducible benchmarking study is published under the name **KVBench** (research manuscript in preparation). Community contributions that strengthen the engine or evaluations are welcome.

## Ways to contribute

We especially welcome:

1. **Additional SLMs** — new model adapters (attention / RoPE / QK-norm), `configs/model_*.yaml`, Modal volume wiring, and smoke evals  
2. **Additional KV-compression algorithms** — new plug-ins under `compressors/` implementing the shared interface (encode/decode, optional fidelity hooks, online attention patches when required)  
3. **Additional evaluation datasets** — loaders and configs beyond WikiText-2 (e.g. LongBench, RULER)  
4. **Additional inference backends / providers** — local CUDA, other cloud runners, or serving stacks that preserve interceptable KV  
5. **Reproducibility improvements** — tests, deterministic seeds, clearer runbooks, CI smoke jobs  
6. **Documentation improvements** — clarity, diagrams, troubleshooting, result write-ups  

Bug reports and small fixes are also appreciated.

## Authorship

Significant research contributions may be considered for co-authorship in accordance with academic authorship standards.

Opening issues or pull requests does **not** by itself confer authorship. Decisions about manuscript credit are made by the maintainers based on substantive intellectual contribution (for example: new methods or models with complete evaluation, major experimental design, or analysis that changes the paper’s claims).

## Development setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch torchvision torchaudio
pip install -r requirements.txt
cp .env.example .env   # add a Hugging Face read token
pytest tests/ -q
```

Do not commit `.env`, model weights, or Modal secrets. Use `.env.example` as the template.

## Adding a compressor plug-in

1. Implement `compressors/<name>.py` subclassing `KVCompressor` (`compressors/base.py`).  
2. Register it in `compressors/registry.py`.  
3. Add unit tests under `tests/`.  
4. If the method needs a custom attention path, add an adapter in `framework/` (see `model_adapter.py`, `qjl_online.py`, `rocketkv_online.py`).  
5. Document the method in `docs/` and add a Modal sweep preset if applicable (`configs/modal_sweeps.yaml`).  
6. Prefer a smoke eval at `context_length=128` before a full grid.

## Adding an SLM

1. Confirm the model exposes interceptable `past_key_values` under `attn_implementation="eager"`.  
2. Extend `framework/model_adapter.py` for that `model_type` (Q/K-norm layout, RoPE import, pre/post-norm).  
3. Add `configs/model_<name>.yaml` and Modal volume settings.  
4. Run identity + TurboQuant smoke tests, then online methods (QJL / RocketKV) if adapters exist.  
5. Record architecture notes under `docs/` (see `OLMO2_ARCHITECTURE_PROBE.md`).

## Pull requests

- Keep PRs focused; separate refactors from new algorithms when possible.  
- Include tests or a short reproduction snippet for behavioral changes.  
- Update docs when you change public APIs, configs, or evaluation semantics.  
- Do not commit large binaries (`*.safetensors`, Modal volume dumps). Prefer summary CSV/JSON under `results/` as documented in `results/README.md`.  
- Follow existing code style; avoid drive-by reformatting.

## Code of conduct (short)

Be respectful and constructive. Assume good faith. Harassment or bad-faith spam will not be tolerated.

## Questions

Open a GitHub issue for design discussion before large features (new families of compressors, multi-provider backends, or paper-facing experimental redesigns).
