"""Download Qwen3-1.7B from HuggingFace and save locally."""

from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

MODEL_NAME = "Qwen/Qwen3-1.7B"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen3_1.7b"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto")

    tokenizer.save_pretrained(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    print(f"Saved model and tokenizer to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
