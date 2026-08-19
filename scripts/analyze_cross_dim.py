#!/usr/bin/env python3
"""Cross-dimensional correlation + trade-off figures (Phases 24–25)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import setup_path  # noqa: F401
from eval.cross_dim.correlation import analyze_correlations
from eval.cross_dim.plot import (
    save_correlation_heatmap,
    save_tradeoff_3d_figure,
    save_tradeoff_figure,
)
from eval.cross_dim.points import load_cross_dim_points_from_json
from framework.config import PROJECT_ROOT


def _collect_points(bundle_paths: list[Path]) -> list:
    points = []
    seen: set[str] = set()
    for path in bundle_paths:
        for pt in load_cross_dim_points_from_json(path):
            if pt.point_id in seen:
                continue
            seen.add(pt.point_id)
            points.append(pt)
    return points


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-dimensional correlation and trade-off analysis for KVBench bundles"
    )
    parser.add_argument(
        "bundles",
        nargs="+",
        type=Path,
        help="Merged JSON bundle paths (e.g. results/phase5_modal_*/*.json)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Filter to one context length (default: 512)",
    )
    parser.add_argument(
        "--exclude-identity",
        action="store_true",
        help="Exclude identity baseline from analysis",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "cross_dim",
        help="Directory for JSON exports and figures",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib figure export")
    parser.add_argument("--3d", dest="plot_3d", action="store_true", help="Also write 3D trade-off figure")
    args = parser.parse_args()

    points = _collect_points(args.bundles)
    if not points:
        raise SystemExit("No valid cross-dimensional points extracted from bundles.")

    exclude = ["identity"] if args.exclude_identity else None
    analysis = analyze_correlations(
        points,
        context_length=args.context_length,
        exclude_compressors=exclude,
    )
    filtered = [
        p
        for p in points
        if p.context_length == args.context_length
        and (not exclude or p.compressor.lower() not in {c.lower() for c in exclude})
    ]

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"correlations_ctx{args.context_length}.json"
    json_path.write_text(json.dumps(analysis.to_dict(), indent=2))

    if not args.no_plot:
        tradeoff_path = out_dir / f"plot_tradeoff_ctx{args.context_length}.pdf"
        save_tradeoff_figure(filtered, tradeoff_path, context_length=args.context_length)
        print(f"Wrote {tradeoff_path}")

        heatmap_path = out_dir / f"plot_correlation_ctx{args.context_length}.pdf"
        try:
            save_correlation_heatmap(analysis, heatmap_path, context_length=args.context_length)
            print(f"Wrote {heatmap_path}")
        except ValueError as exc:
            print(f"Skipped correlation heatmap: {exc}")

        if args.plot_3d:
            path_3d = out_dir / f"plot_tradeoff_3d_ctx{args.context_length}.pdf"
            save_tradeoff_3d_figure(filtered, path_3d, context_length=args.context_length)
            print(f"Wrote {path_3d}")

    weak = [p.label for p in analysis.pairs if p.interpretable and p.pearson_r is not None and abs(p.pearson_r) < 0.5]
    print(
        f"Cross-dim ctx={args.context_length}: {analysis.point_count} points, "
        f"{sum(1 for p in analysis.pairs if p.interpretable)} interpretable pairs"
    )
    if weak:
        print("Weak predictors (|r|<0.5):", "; ".join(weak[:3]), "..." if len(weak) > 3 else "")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
