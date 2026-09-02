"""Matplotlib Pareto figure export (Phase 9)."""

from __future__ import annotations

from pathlib import Path

from eval.pareto.analysis import ParetoAnalysis, ParetoPoint


def save_pareto_figure(
    analysis: ParetoAnalysis,
    path: Path | str,
    *,
    title: str | None = None,
    highlight_ids: set[str] | None = None,
) -> Path:
    """Save a paper-style Pareto plot (memory ratio vs log10 PPL ratio)."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    optimal = set(analysis.pareto_optimal_ids)
    excluded = set(analysis.excluded_from_frontier_ids)
    highlight = highlight_ids or optimal

    fig, ax = plt.subplots(figsize=(8, 5))
    xs: list[float] = []
    ys: list[float] = []
    for pt in analysis.points:
        xs.append(pt.compression_ratio)
        ys.append(pt.log10_perplexity_ratio)
        size = 40.0
        if pt.tokens_per_second is not None:
            size = max(20.0, min(400.0, pt.tokens_per_second * 20.0))
        if pt.point_id in excluded:
            ax.scatter(
                pt.compression_ratio,
                pt.log10_perplexity_ratio,
                s=size,
                alpha=0.45,
                marker="x",
                label=f"{pt.compressor} (excluded)",
            )
            continue
        alpha = 0.35 if pt.point_id not in optimal else 0.95
        marker = "o" if pt.point_id in highlight else "."
        ax.scatter(
            pt.compression_ratio,
            pt.log10_perplexity_ratio,
            s=size,
            alpha=alpha,
            marker=marker,
            label=pt.compressor if pt.point_id in highlight else None,
        )

    if analysis.frontier_2d:
        fx = [p.compression_ratio for p in analysis.frontier_2d]
        fy = [p.log10_perplexity_ratio for p in analysis.frontier_2d]
        ax.plot(fx, fy, linestyle="--", color="0.3", linewidth=1.5, label="empirical Pareto front")

    ctx = analysis.context_length
    ax.set_xlabel("Memory compression ratio (higher is better)")
    ax.set_ylabel(r"$\log_{10}$ perplexity ratio (lower is better)")
    ax.set_title(title or f"Pareto trade-off (ctx={ctx})")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
