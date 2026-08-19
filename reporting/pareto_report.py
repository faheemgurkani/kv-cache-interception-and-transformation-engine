"""Persist Pareto analysis artifacts alongside sweep results."""

from __future__ import annotations

import json
from pathlib import Path

from eval.pareto.analysis import ParetoAnalysis, analyze_pareto, load_pareto_points_from_results
from eval.pareto.plot import save_pareto_figure
from eval.runner import EvaluationResult


def save_pareto_analysis(
    results: list[EvaluationResult],
    output_dir: Path | str,
    *,
    context_length: int | None = 512,
    exclude_identity: bool = False,
    write_plot: bool = True,
    name_prefix: str = "pareto",
) -> tuple[ParetoAnalysis, Path]:
    """Analyze results and write ``pareto.json`` (+ optional PDF figure)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points = load_pareto_points_from_results(results)
    exclude = ["identity"] if exclude_identity else None
    analysis = analyze_pareto(points, context_length=context_length, exclude_compressors=exclude)

    suffix = f"_ctx{context_length}" if context_length is not None else "_all"
    json_path = output_dir / f"{name_prefix}{suffix}.json"
    json_path.write_text(json.dumps(analysis.to_dict(), indent=2))

    if write_plot:
        save_pareto_figure(analysis, output_dir / f"plot_pareto{suffix}.pdf")

    return analysis, json_path
