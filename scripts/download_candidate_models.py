"""Download candidate eval models into models/ for architecture probing."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# (hf_repo_id, local_dirname)
CANDIDATES = [
    ("allenai/OLMo-2-0425-1B", "olmo2_1b"),
    ("openbmb/MiniCPM4-0.5B", "minicpm4_0.5b"),
    ("ibm-granite/granite-4.0-350m", "granite_4.0_350m"),
    ("google/gemma-3-270m", "gemma3_270m"),
    ("Qwen/Qwen3.5-0.8B", "qwen3.5_0.8b"),
]


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN missing — set it in .env")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for repo_id, dirname in CANDIDATES:
        out = MODELS_DIR / dirname
        marker = out / "config.json"
        if marker.exists():
            print(f"SKIP {repo_id} → {out} (already present)")
            continue
        print(f"DOWNLOAD {repo_id} → {out}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(out),
            token=token,
        )
        print(f"DONE {repo_id}")


if __name__ == "__main__":
    main()
