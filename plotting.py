from __future__ import annotations

import math
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


PLOT_FIGSIZE = (10, 6)
PLOT_DPI = 100


def _point_columns(rows: list[dict[str, float]]) -> tuple[str, str]:
    if rows and "x_value" in rows[0] and "y_value" in rows[0]:
        return "x_value", "y_value"
    if rows and "x_norm" in rows[0] and "y_norm" in rows[0]:
        return "x_norm", "y_norm"
    return "x_px", "y_px"


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


def write_line_plot(plot_path: Path | None, rows: list[dict[str, float]]) -> None:
    x_key, y_key = _point_columns(rows)
    points = _finite_points(rows, x_key, y_key)
    if not points:
        raise RuntimeError("No numeric CSV data points are available for plotting")

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = _expanded_bounds(xs)
    y_min, y_max = _expanded_bounds(ys)

    figure, axis = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    try:
        if figure.canvas.manager is not None:
            figure.canvas.manager.set_window_title("Extracted Curve")
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
        if y_key == "y_px":
            axis.invert_yaxis()
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
        axis.grid(True, color="#dfe4ea", linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#38404a")
        axis.spines["bottom"].set_color("#38404a")
        axis.tick_params(colors="#2e343c", labelsize=9)

        figure.tight_layout()
        if plot_path is not None:
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(plot_path, format="png")
        plt.show()
    finally:
        plt.close(figure)


def _channel_points(
    rows: list[dict[str, float | str | None]],
    channel: str,
    y_key: str,
    time_start_s: float | None,
    time_end_s: float | None,
) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        if row.get("channel") != channel:
            continue
        x = row.get("time_s")
        y = row.get(y_key)
        if not isinstance(x, float):
            continue
        if not math.isfinite(x):
            continue
        if time_start_s is not None and x < time_start_s:
            continue
        if time_end_s is not None and x > time_end_s:
            continue

        xs.append(x)
        ys.append(y if isinstance(y, float) and math.isfinite(y) else math.nan)
    return xs, ys


def write_wav_envelope_plot(
    plot_path: Path | None,
    rows: list[dict[str, float | str | None]],
    time_start_s: float | None = 0.0,
    time_end_s: float | None = 1.1,
    y_key: str = "envelope_normalized",
) -> None:
    channels = sorted(
        {
            str(row["channel"])
            for row in rows
            if isinstance(row.get("channel"), str)
        }
    )
    if not channels:
        raise RuntimeError("No channel data is available for plotting")

    series = [
        (channel, *_channel_points(rows, channel, y_key, time_start_s, time_end_s))
        for channel in channels
    ]
    series = [
        (channel, xs, ys)
        for channel, xs, ys in series
        if xs and any(math.isfinite(value) for value in ys)
    ]
    if not series:
        raise RuntimeError("No finite WAV envelope points are available for plotting")

    all_times = [value for _, xs, _ in series for value in xs]
    all_values = [
        value
        for _, _, ys in series
        for value in ys
        if math.isfinite(value)
    ]
    x_min, x_max = _expanded_bounds(all_times)
    if y_key == "envelope_normalized":
        y_min, y_max = 0.0, 1.0
    else:
        y_min, y_max = _expanded_bounds(all_values)

    figure, axis = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    try:
        if figure.canvas.manager is not None:
            figure.canvas.manager.set_window_title("Doppler STFT Rolloff Envelope")
        figure.patch.set_facecolor("white")
        axis.set_facecolor("#fbfcfe")

        for channel, xs, ys in series:
            axis.plot(xs, ys, linewidth=1.4, label=channel)

        axis.set_title("Doppler STFT Rolloff Envelope", fontsize=14, pad=14)
        axis.set_xlabel("time_s")
        axis.set_ylabel(y_key)
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=8))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
        axis.grid(True, color="#dfe4ea", linewidth=0.8)
        axis.legend(loc="best", frameon=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#38404a")
        axis.spines["bottom"].set_color("#38404a")
        axis.tick_params(colors="#2e343c", labelsize=9)

        figure.tight_layout()
        if plot_path is not None:
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(plot_path, format="png")
        plt.show()
    finally:
        plt.close(figure)
