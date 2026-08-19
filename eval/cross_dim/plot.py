"""Trade-off and correlation figures (Phases 24–25)."""

from __future__ import annotations

from pathlib import Path

from eval.cross_dim.correlation import CrossDimCorrelationAnalysis
from eval.cross_dim.points import CrossDimPoint


def save_tradeoff_figure(
    points: list[CrossDimPoint],
    path: Path | str,
    *,
    title: str | None = None,
    context_length: int | None = None,
) -> Path:
    """Phase 25: Quality ↔ Memory / Speed paired scatter."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    usable = [p for p in points if p.quality_score is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for pt in usable:
        size = 40.0
        if pt.tokens_per_second is not None:
            size = max(20.0, min(400.0, pt.tokens_per_second * 20.0))
        if pt.compression_ratio is not None:
            axes[0].scatter(
                pt.compression_ratio,
                pt.quality_score,
                s=size,
                alpha=0.75,
                label=pt.compressor,
            )
        if pt.tokens_per_second is not None:
            axes[1].scatter(
                pt.tokens_per_second,
                pt.quality_score,
                s=80,
                alpha=0.75,
                label=pt.compressor,
            )

    axes[0].set_xlabel("Memory compression ratio (higher → more savings)")
    axes[0].set_ylabel("Quality score (higher → better BEHAVIOR)")
    axes[0].set_title("Quality ↔ Memory")
    axes[0].grid(True, alpha=0.25)

    axes[1].set_xlabel("Throughput (tok/s)")
    axes[1].set_ylabel("Quality score (higher → better BEHAVIOR)")
    axes[1].set_title("Quality ↔ Speed")
    axes[1].grid(True, alpha=0.25)

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            by_label = dict(zip(labels, handles, strict=True))
            ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="best")

    ctx = context_length
    fig.suptitle(title or f"Compression trade-offs (ctx={ctx})")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_tradeoff_3d_figure(
    points: list[CrossDimPoint],
    path: Path | str,
    *,
    title: str | None = None,
    context_length: int | None = None,
) -> Path:
    """Optional 3D Quality × Memory × Speed scatter."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for pt in points:
        if (
            pt.compression_ratio is None
            or pt.tokens_per_second is None
            or pt.quality_score is None
        ):
            continue
        ax.scatter(
            pt.compression_ratio,
            pt.tokens_per_second,
            pt.quality_score,
            s=50,
            alpha=0.8,
            label=pt.compressor,
        )
    ax.set_xlabel("Memory compression ratio")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_zlabel("Quality score")
    ax.set_title(title or f"Quality–Memory–Speed (ctx={context_length})")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        by_label = dict(zip(labels, handles, strict=True))
        ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_correlation_heatmap(
    analysis: CrossDimCorrelationAnalysis,
    path: Path | str,
    *,
    title: str | None = None,
    context_length: int | None = None,
) -> Path:
    """Bar chart of Pearson r for each predefined metric pair."""
    import matplotlib.pyplot as plt
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    interpretable = [p for p in analysis.pairs if p.interpretable and p.pearson_r is not None]
    if not interpretable:
        raise ValueError("No interpretable correlation pairs to plot")

    labels = [f"{p.metric_x}\n↔\n{p.metric_y}" for p in interpretable]
    values = [p.pearson_r for p in interpretable]
    colors = ["#2ca02c" if abs(v) >= 0.5 else "#d62728" for v in values if v is not None]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4.5))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0.0, color="0.3", linewidth=0.8)
    ax.axvline(0.5, color="0.6", linestyle="--", linewidth=0.8, label="|r|=0.5")
    ax.axvline(-0.5, color="0.6", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Pearson r")
    ax.set_xlim(-1.0, 1.0)
    suffix = f" (ctx={context_length})" if context_length is not None else ""
    ax.set_title(title or f"Cross-dimensional correlations{suffix}")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
