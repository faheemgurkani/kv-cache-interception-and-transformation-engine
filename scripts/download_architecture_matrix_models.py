"""Download the 5-model architecture-matrix shortlist (MHA/MQA/GQA/MLA/Hybrid) into models/.

Skips any repo that already exists complete under models/, and skips (without
failing) any repo that is gated behind a license click-through on Hugging Face —
those are logged to models/PENDING_LICENSE.txt for manual approval later.

After each successful download, probes the saved config for basic architecture
facts (model_type, head counts, layer count, hidden size) and appends them to
models/ARCHITECTURE_REPORT.md so downloaded artifacts are self-documenting.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
PENDING_FILE = MODELS_DIR / "PENDING_LICENSE.txt"
REPORT_FILE = MODELS_DIR / "ARCHITECTURE_REPORT.md"

# (hf_repo_id, local_dirname, architecture_family)
CANDIDATES = [
    ("allenai/OLMo-1B-hf", "olmo1b", "MHA"),
    ("Qwen/Qwen3-0.6B", "qwen3_0.6b", "GQA"),
    ("FreedomIntelligence/TinyDeepSeek-0.5B-base", "tinydeepseek_0.5b", "MLA"),
    ("tiiuae/Falcon-H1-0.5B-Base", "falcon_h1_0.5b", "Hybrid Attention + Mamba2"),
    ("google/gemma-3-270m", "gemma3_270m", "MQA + local/global attention"),
]


def _is_complete(out: Path) -> bool:
    if not (out / "config.json").exists():
        return False
    weights = list(out.glob("*.safetensors"))
    if not weights:
        return False
    index = out / "model.safetensors.index.json"
    if index.exists():
        need = set(json.loads(index.read_text())["weight_map"].values())
        have = {p.name for p in weights}
        return need.issubset(have)
    return True


def _probe_architecture(repo_id: str, dirname: str, family: str, out: Path) -> str:
    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text())
    fields = [
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "mamba_n_heads",
        "layer_types",
    ]
    lines = [f"## {repo_id} → `{dirname}` ({family})", ""]
    for f in fields:
        if f in cfg:
            v = cfg[f]
            if isinstance(v, list) and len(v) > 8:
                v = f"{v[:4]}...{v[-4:]} (len={len(v)})"
            lines.append(f"- `{f}`: {v}")
    weights = sorted(p.name for p in out.glob("*.safetensors"))
    lines.append(f"- weight files: {weights}")
    other = sorted(p.name for p in out.iterdir() if p.suffix in {".py", ".json"} and p.name != "config.json")
    lines.append(f"- other artifacts: {other}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN missing — set it in .env")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pending: list[str] = []
    failures: list[str] = []
    report_sections: list[str] = []

    for repo_id, dirname, family in CANDIDATES:
        out = MODELS_DIR / dirname
        if _is_complete(out):
            print(f"SKIP {repo_id} → {out} (already complete)")
            report_sections.append(_probe_architecture(repo_id, dirname, family, out))
            continue

        print(f"DOWNLOAD {repo_id} → {out} [{family}]")
        try:
            snapshot_download(repo_id=repo_id, local_dir=str(out), token=token)
        except GatedRepoError as exc:
            msg = f"{repo_id}: gated — visit https://huggingface.co/{repo_id} and accept the license, then re-run this script"
            print(f"GATED {msg}")
            pending.append(msg)
            continue
        except HfHubHTTPError as exc:
            if getattr(exc.response, "status_code", None) in (401, 403):
                msg = f"{repo_id}: access restricted (HTTP {exc.response.status_code}) — likely needs license acceptance"
                print(f"GATED {msg}")
                pending.append(msg)
                continue
            failures.append(f"{repo_id}: {exc}")
            print(f"FAIL {repo_id}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 — report and continue other models
            failures.append(f"{repo_id}: {exc}\n{traceback.format_exc()}")
            print(f"FAIL {repo_id}: {exc}")
            continue

        if _is_complete(out):
            print(f"DONE {repo_id}")
            report_sections.append(_probe_architecture(repo_id, dirname, family, out))
        else:
            failures.append(f"{repo_id}: incomplete after download")
            print(f"INCOMPLETE {repo_id}")

    if pending:
        PENDING_FILE.write_text("\n".join(pending) + "\n")
        print(f"\nWrote {len(pending)} pending-license repo(s) to {PENDING_FILE}")

    if report_sections:
        header = "# Architecture Matrix — Downloaded Model Report\n\n"
        REPORT_FILE.write_text(header + "\n".join(report_sections))
        print(f"Wrote architecture report to {REPORT_FILE}")

    if failures:
        raise SystemExit("Some downloads failed:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    main()
