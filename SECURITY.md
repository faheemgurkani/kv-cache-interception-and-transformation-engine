# Security policy

## Reporting a vulnerability

If you discover a security issue in this repository (the KV Cache Interception and Transformation Engine / KVBench codebase — for example, accidental secret handling, unsafe deserialization of result payloads, or supply-chain concerns in scripts), please **do not** open a public GitHub issue with exploit details.

Email the maintainer at **faheemgurkani@gmail.com** with:

- a description of the issue  
- steps to reproduce  
- impact assessment if known  

## Secrets

Never commit Hugging Face tokens, Modal tokens, or `.env` files. Use `.env.example` and Modal secrets (`huggingface-secret`) as documented in `docs/REPRODUCIBILITY.md`.
