"""Plot helpers for experiment metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


SeriesInput = Mapping[str, Sequence[float]] | Sequence[tuple[str, Sequence[float]]]


def save_line_plot(
    path: str | Path,
    series: SeriesInput,
    *,
    title: str | None = None,
    xlabel: str = "step",
    ylabel: str = "value",
    legend: bool = True,
    grid: bool = True,
    figsize: tuple[float, float] = (8.0, 4.5),
) -> Path:
    """Save a multi-series line plot using a non-interactive backend."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    try:
        items = series.items() if isinstance(series, Mapping) else series
        for label, values in items:
            y = np.asarray(values, dtype=float)
            x = np.arange(y.shape[0])
            ax.plot(x, y, label=label)
        if title:
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if grid:
            ax.grid(True, alpha=0.3)
        if legend and len(ax.lines) > 1:
            ax.legend()
        fig.savefig(output, dpi=150)
    finally:
        plt.close(fig)
    return output
