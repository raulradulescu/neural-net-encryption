"""Plot helpers for experiment metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import contextlib
import csv
import io
from pathlib import Path


SeriesInput = Mapping[str, Sequence[float]] | Sequence[tuple[str, Sequence[float]]]


def _series_items(series: SeriesInput) -> list[tuple[str, Sequence[float]]]:
    return list(series.items() if isinstance(series, Mapping) else series)


def _write_plot_fallback(path: Path, series: SeriesInput) -> Path:
    """Write plot data as CSV when Matplotlib is unavailable or broken."""

    fallback = path.with_suffix(".csv")
    items = _series_items(series)
    max_len = max((len(values) for _, values in items), default=0)
    with fallback.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", *[label for label, _ in items]])
        for index in range(max_len):
            row: list[float | int | str] = [index]
            for _, values in items:
                row.append(float(values[index]) if index < len(values) else "")
            writer.writerow(row)
    return fallback


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
    """Save a multi-series line plot, falling back to CSV on minimal Linux hosts."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
    except Exception:
        return _write_plot_fallback(output, series)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    try:
        for label, values in _series_items(series):
            y = [float(value) for value in values]
            x = list(range(len(y)))
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
    except Exception:
        plt.close(fig)
        return _write_plot_fallback(output, series)
    finally:
        plt.close(fig)
    return output
