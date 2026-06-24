"""Run a baseline KV-cache compression method."""

import argparse
from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict:
    with config_path.open() as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a KV-cache compression baseline.")
    parser.add_argument(
        "--baseline",
        choices=["kivi", "qjl", "rocketkv"],
        required=True,
        help="Baseline method to run.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval.yaml"),
        help="Path to evaluation config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Baseline: {args.baseline}")
    print(f"Config: {config}")
    print("Baseline runner placeholder — implement in Phase 1+.")


if __name__ == "__main__":
    main()
