#!/usr/bin/env python3
"""Compute Pareto frontier from evaluation bundles (Phase 9)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import setup_path  # noqa: F401
from eval.pareto.analysis import analyze_pareto, load_pareto_points_from_json
from eval.pareto.plot import save_pareto_figure
from framework.config import PROJECT_ROOT


def _collect_points(bundle_paths: list[Path]) -> list:
    points = []
    seen: set[str] = set()
    for path in bundle_paths:
        for pt in load_pareto_points_from_json(path):
            if pt.point_id in seen:
                continue
            seen.add(pt.point_id)
            points.append(pt)
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description="Pareto analysis for KVBench job bundles")
    parser.add_argument(
        "bundles",
        nargs="+",
        type=Path,
        help="Merged JSON bundle paths (e.g. results/phase5_modal_*/phase5_modal_*.json)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Filter to one context length (default: 512, matching paper figure)",
    )
    parser.add_argument(
        "--exclude-identity",
        action="store_true",
        help="Exclude identity baseline from frontier (paper offline-vs-online style)",
    )
    parser.add_argument(
        "--max-ppl-ratio",
        type=float,
        default=5.0,
        help=(
            "Exclude points with perplexity/baseline above this from the frontier "
            "(default 5.0). Pass 0 to disable."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "pareto",
        help="Directory for pareto.json and plot_pareto.pdf",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib figure export")
    args = parser.parse_args()

    points = _collect_points(args.bundles)
    if not points:
        raise SystemExit("No valid Pareto points extracted from bundles.")

    exclude = ["identity"] if args.exclude_identity else None
    max_ratio = None if args.max_ppl_ratio == 0 else args.max_ppl_ratio
    analysis = analyze_pareto(
        points,
        context_length=args.context_length,
        exclude_compressors=exclude,
        max_perplexity_ratio=max_ratio,
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"pareto_ctx{args.context_length}.json"
    json_path.write_text(json.dumps(analysis.to_dict(), indent=2))

    if not args.no_plot:
        plot_path = out_dir / f"plot_pareto_ctx{args.context_length}.pdf"
        save_pareto_figure(analysis, plot_path)
        print(f"Wrote {plot_path}")

    print(
        f"Pareto ctx={args.context_length}: {len(analysis.points)} points, "
        f"{len(analysis.pareto_optimal_ids)} optimal (3D), "
        f"{len(analysis.frontier_2d)} on 2D front"
    )
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
