from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


PLOT_FIGSIZE = (10, 6)
PLOT_DPI = 100


def _point_columns(rows: list[dict[str, float]]) -> tuple[str, str]:
    if rows and "x_value" in rows[0] and "y_value" in rows[0]:
        return "x_value", "y_value"
    return "x_norm", "y_norm"


def _finite_points(
    rows: list[dict[str, float]],
    x_key: str,
    y_key: str,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        x = row.get(x_key)
        y = row.get(y_key)
        if x is None or y is None:
            continue
        if math.isfinite(x) and math.isfinite(y):
            points.append((float(x), float(y)))
    return points


def _expanded_bounds(values: list[float]) -> tuple[float, float]:
    lower = min(values)
    upper = max(values)
    if lower != upper:
        padding = (upper - lower) * 0.03
        return lower - padding, upper + padding

    padding = abs(lower) * 0.03 if lower else 1.0
    return lower - padding, upper + padding


def write_line_plot(plot_path: Path, rows: list[dict[str, float]]) -> None:
    x_key, y_key = _point_columns(rows)
    points = _finite_points(rows, x_key, y_key)
    if not points:
        raise RuntimeError("No numeric CSV data points are available for plotting")

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = _expanded_bounds(xs)
    y_min, y_max = _expanded_bounds(ys)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    try:
        figure.patch.set_facecolor("white")
        axis.set_facecolor("#fbfcfe")

        if len(points) == 1:
            axis.scatter(xs, ys, s=32, color="#1667be", zorder=3)
        else:
            axis.plot(xs, ys, color="#1667be", linewidth=1.8)

        axis.set_title("Extracted Curve", fontsize=14, pad=14)
        axis.set_xlabel(x_key)
        axis.set_ylabel(y_key)
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
        axis.grid(True, color="#dfe4ea", linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#38404a")
        axis.spines["bottom"].set_color("#38404a")
        axis.tick_params(colors="#2e343c", labelsize=9)

        figure.tight_layout()
        figure.savefig(plot_path, format="png")
    finally:
        plt.close(figure)
