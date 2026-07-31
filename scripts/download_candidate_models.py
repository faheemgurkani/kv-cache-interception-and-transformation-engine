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


def _is_complete(out: Path) -> bool:
    if not (out / "config.json").exists():
        return False
    weights = list(out.glob("*.safetensors"))
    if not weights:
        return False
    index = out / "model.safetensors.index.json"
    if index.exists():
        import json

        need = set(json.loads(index.read_text())["weight_map"].values())
        have = {p.name for p in weights}
        return need.issubset(have)
    return True


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN missing — set it in .env")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for repo_id, dirname in CANDIDATES:
        out = MODELS_DIR / dirname
        if _is_complete(out):
            print(f"SKIP {repo_id} → {out} (complete)")
            continue
        print(f"DOWNLOAD {repo_id} → {out}")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(out),
                token=token,
            )
        except Exception as exc:  # noqa: BLE001 — report and continue other models
            failures.append(f"{repo_id}: {exc}")
            print(f"FAIL {repo_id}: {exc}")
            continue
        if _is_complete(out):
            print(f"DONE {repo_id}")
        else:
            failures.append(f"{repo_id}: incomplete after download")
            print(f"INCOMPLETE {repo_id}")

    if failures:
        raise SystemExit("Some downloads failed:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    main()
