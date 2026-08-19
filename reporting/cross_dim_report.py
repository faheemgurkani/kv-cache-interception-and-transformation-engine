"""Persist cross-dimensional correlation analysis (Phases 24–25)."""

from __future__ import annotations

from pathlib import Path

from eval.cross_dim.correlation import analyze_correlations
from eval.cross_dim.plot import save_correlation_heatmap, save_tradeoff_figure
from eval.cross_dim.points import load_cross_dim_points_from_results
from eval.runner import EvaluationResult


def save_cross_dim_analysis(
    results: list[EvaluationResult],
    output_dir: Path | str,
    *,
    context_length: int | None = 512,
    exclude_identity: bool = False,
    write_plot: bool = True,
    name_prefix: str = "cross_dim",
) -> tuple[dict, Path]:
    """Analyze correlations and optionally write trade-off / heatmap PDFs."""
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points = load_cross_dim_points_from_results(results)
    exclude = ["identity"] if exclude_identity else None
    analysis = analyze_correlations(
        points,
        context_length=context_length,
        exclude_compressors=exclude,
    )
    payload = analysis.to_dict()
    ctx_suffix = f"ctx{context_length}" if context_length is not None else "all"
    json_path = output_dir / f"{name_prefix}_{ctx_suffix}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    filtered = [
        p
        for p in points
        if context_length is None or p.context_length == context_length
    ]
    if exclude:
        blocked = {c.lower() for c in exclude}
        filtered = [p for p in filtered if p.compressor.lower() not in blocked]

    if write_plot and filtered:
        save_tradeoff_figure(
            filtered,
            output_dir / f"plot_tradeoff_{ctx_suffix}.pdf",
            context_length=context_length,
        )
        try:
            save_correlation_heatmap(
                analysis,
                output_dir / f"plot_correlation_{ctx_suffix}.pdf",
            )
        except ValueError:
            pass

    return payload, json_path
