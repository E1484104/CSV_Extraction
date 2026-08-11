from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


PLOT_FIGSIZE = (10, 6)
PLOT_DPI = 100
MATCH_DETAIL_PANE_LIMIT = 3
LONG_OVERLAY_ALPHA = 0.88
SHORT_OVERLAY_ALPHA = 0.55


@dataclass(frozen=True)
class MatchOverlaySeries:
    name: str
    aligned_times: Sequence[float]
    values: Sequence[float]
    y_label: str
    start_time: float
    end_time: float
    rank: int
    score: float


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


def _finite_xy(
    xs: Sequence[float],
    ys: Sequence[float],
) -> tuple[list[float], list[float]]:
    finite_xs: list[float] = []
    finite_ys: list[float] = []
    for x, y in zip(xs, ys):
        x_value = float(x)
        y_value = float(y)
        if math.isfinite(x_value) and math.isfinite(y_value):
            finite_xs.append(x_value)
            finite_ys.append(y_value)
    return finite_xs, finite_ys


def _windowed_y_bounds(
    xs: Sequence[float],
    ys: Sequence[float],
    x_min: float,
    x_max: float,
    fallback: tuple[float, float],
) -> tuple[float, float]:
    window_values = [
        float(y)
        for x, y in zip(xs, ys)
        if x_min <= float(x) <= x_max and math.isfinite(float(y))
    ]
    if not window_values:
        return fallback
    return _expanded_bounds(window_values)


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


def write_dual_axis_match_overlay_plot(
    plot_path: Path | None,
    *,
    long_times: Sequence[float],
    long_values: Sequence[float],
    short_matches: Sequence[MatchOverlaySeries],
    long_name: str,
    long_x_label: str,
    long_y_label: str,
    metric: str,
    show: bool = False,
) -> None:
    long_xs, long_ys = _finite_xy(long_times, long_values)
    if not long_xs:
        raise RuntimeError("No finite long CSV points are available for plotting")

    prepared_matches: list[tuple[MatchOverlaySeries, list[float], list[float]]] = []
    for match in short_matches:
        short_xs, short_ys = _finite_xy(match.aligned_times, match.values)
        if short_xs:
            prepared_matches.append((match, short_xs, short_ys))
    if not prepared_matches:
        raise RuntimeError("No finite matched short CSV points are available for plotting")

    x_min, x_max = _expanded_bounds(long_xs)
    long_y_min, long_y_max = _expanded_bounds(long_ys)
    short_values_all = [
        value
        for _, _, short_ys in prepared_matches
        for value in short_ys
    ]
    short_y_min, short_y_max = _expanded_bounds(short_values_all)

    long_color = "#1f5f9e"
    short_colors = [
        "#c2571a",
        "#178a63",
        "#7b4ab8",
        "#b33c60",
        "#8a6f12",
        "#156f7a",
    ]
    short_y_labels = sorted({match.y_label for match, _, _ in prepared_matches})
    if len(short_y_labels) == 1:
        short_y_label = f"Normalized Ultrasound Data"
    else:
        short_y_label = "short signal"

    detail_matches = prepared_matches[:MATCH_DETAIL_PANE_LIMIT]
    detail_count = max(1, len(detail_matches))

    figure = plt.figure(figsize=(12, 8), dpi=PLOT_DPI)
    grid = figure.add_gridspec(
        2,
        detail_count,
        height_ratios=[2.25, 1.0],
        hspace=0.38,
        wspace=0.34,
    )
    axis = figure.add_subplot(grid[0, :])
    try:
        if figure.canvas.manager is not None:
            figure.canvas.manager.set_window_title("Matched CSV Overlay")
        figure.patch.set_facecolor("white")
        axis.set_facecolor("#fbfcfe")

        short_axis = axis.twinx()
        short_axis.patch.set_alpha(0.0)
        long_line = axis.plot(
            long_xs,
            long_ys,
            color=long_color,
            linewidth=1.5,
            alpha=LONG_OVERLAY_ALPHA,
            label=f"Wearable Data",
            zorder=2,
        )

        short_lines = []
        for index, (match, short_xs, short_ys) in enumerate(prepared_matches):
            color = short_colors[index % len(short_colors)]
            axis.axvspan(match.start_time, match.end_time, color=color, alpha=0.07, linewidth=0)
            lines = short_axis.plot(
                short_xs,
                short_ys,
                color=color,
                linewidth=1.8,
                alpha=SHORT_OVERLAY_ALPHA,
                label=(
                    f"Ultrasound Data | phase {match.rank} | "
                    f"{match.start_time:.6g}-{match.end_time:.6g} | {match.score:.4g}"
                ),
                zorder=3,
            )
            short_lines.extend(lines)

        axis.set_title(
            f"Matched Sample Interval | metric={metric}",
            fontsize=14,
            pad=14,
        )
        axis.set_xlabel(f"Sample Time (s)")
        axis.set_ylabel(f"Normalized Wearable Data", color=long_color)
        short_axis.set_ylabel(short_y_label)
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(long_y_min, long_y_max)
        short_axis.set_ylim(short_y_min, short_y_max)

        lines = [*long_line, *short_lines]
        labels = [line.get_label() for line in lines]
        axis.legend(lines, labels, loc="best", frameon=False, fontsize=8)

        axis.xaxis.set_major_locator(MaxNLocator(nbins=10))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
        short_axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
        axis.grid(True, color="#dfe4ea", linewidth=0.8)

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(long_color)
        axis.spines["bottom"].set_color("#38404a")
        short_axis.spines["top"].set_visible(False)
        short_axis.spines["right"].set_color("#38404a")
        axis.tick_params(axis="x", colors="#2e343c", labelsize=9)
        axis.tick_params(axis="y", colors=long_color, labelsize=9)
        short_axis.tick_params(axis="y", colors="#2e343c", labelsize=9)

        for index, (match, short_xs, short_ys) in enumerate(detail_matches):
            detail_axis = figure.add_subplot(grid[1, index])
            detail_short_axis = detail_axis.twinx()
            detail_short_axis.patch.set_alpha(0.0)
            color = short_colors[index % len(short_colors)]

            duration = match.end_time - match.start_time
            padding = duration * 0.03 if duration > 0 else 1.0
            detail_x_min = match.start_time - padding
            detail_x_max = match.end_time + padding
            detail_long_y_min, detail_long_y_max = _windowed_y_bounds(
                long_xs,
                long_ys,
                detail_x_min,
                detail_x_max,
                fallback=(long_y_min, long_y_max),
            )
            detail_short_y_min, detail_short_y_max = _windowed_y_bounds(
                short_xs,
                short_ys,
                detail_x_min,
                detail_x_max,
                fallback=(short_y_min, short_y_max),
            )

            detail_axis.set_facecolor("#fbfcfe")
            detail_axis.plot(
                long_xs,
                long_ys,
                color=long_color,
                linewidth=1.25,
                alpha=LONG_OVERLAY_ALPHA,
                label="long",
                zorder=2,
            )
            detail_short_axis.plot(
                short_xs,
                short_ys,
                color=color,
                linewidth=1.7,
                alpha=SHORT_OVERLAY_ALPHA,
                label="short",
                zorder=3,
            )
            detail_axis.set_title(
                f"phase {match.rank} | {match.start_time:.6g}-{match.end_time:.6g}",
                fontsize=9,
                pad=8,
            )
            detail_axis.set_xlim(detail_x_min, detail_x_max)
            detail_axis.set_ylim(detail_long_y_min, detail_long_y_max)
            detail_short_axis.set_ylim(detail_short_y_min, detail_short_y_max)
            detail_axis.set_xlabel(f"Sample Time (s)", fontsize=8)
            if index == 0:
                detail_axis.set_ylabel(f"Normalized Wearable Data", color=long_color, fontsize=8)
            else:
                detail_axis.tick_params(axis="y", labelleft=False)
            if index == len(detail_matches) - 1:
                detail_short_axis.set_ylabel(short_y_label, fontsize=8)
            else:
                detail_short_axis.tick_params(axis="y", labelright=False)

            detail_axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
            detail_axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
            detail_short_axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
            detail_axis.grid(True, color="#dfe4ea", linewidth=0.8)

            detail_axis.spines["top"].set_visible(False)
            detail_axis.spines["right"].set_visible(False)
            detail_axis.spines["left"].set_color(long_color)
            detail_axis.spines["bottom"].set_color("#38404a")
            detail_short_axis.spines["top"].set_visible(False)
            detail_short_axis.spines["right"].set_color("#38404a")
            detail_axis.tick_params(axis="x", colors="#2e343c", labelsize=8)
            detail_axis.tick_params(axis="y", colors=long_color, labelsize=8)
            detail_short_axis.tick_params(axis="y", colors="#2e343c", labelsize=8)

        figure.subplots_adjust(
            left=0.07,
            right=0.93,
            top=0.93,
            bottom=0.08,
        )
        if plot_path is not None:
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(plot_path, format="png")
        if show:
            plt.show()
    finally:
        plt.close(figure)


def write_dual_axis_match_plot(
    plot_path: Path | None,
    *,
    aligned_times: Sequence[float],
    long_values: Sequence[float],
    short_values: Sequence[float],
    long_name: str,
    short_name: str,
    long_x_label: str,
    short_x_label: str,
    long_y_label: str,
    short_y_label: str,
    match_start_time: float,
    short_time_origin: float,
    metric: str,
    score: float,
    show: bool = False,
) -> None:
    long_xs, long_ys = _finite_xy(aligned_times, long_values)
    short_xs, short_ys = _finite_xy(aligned_times, short_values)
    if not long_xs or not short_xs:
        raise RuntimeError("No finite matched points are available for plotting")

    x_min, x_max = _expanded_bounds([*long_xs, *short_xs])
    long_y_min, long_y_max = _expanded_bounds(long_ys)
    short_y_min, short_y_max = _expanded_bounds(short_ys)

    long_color = "#1667be"
    short_color = "#c2571a"

    figure, axis = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    try:
        if figure.canvas.manager is not None:
            figure.canvas.manager.set_window_title("Matched CSV Overlay")
        figure.patch.set_facecolor("white")
        axis.set_facecolor("#fbfcfe")

        short_axis = axis.twinx()
        short_axis.patch.set_alpha(0.0)
        long_line = axis.plot(
            long_xs,
            long_ys,
            color=long_color,
            linewidth=1.8,
            alpha=LONG_OVERLAY_ALPHA,
            label=f"Wearable Data",
        )
        short_line = short_axis.plot(
            short_xs,
            short_ys,
            color=short_color,
            linewidth=1.8,
            alpha=SHORT_OVERLAY_ALPHA,
            label=f"Ultrasound Data",
        )

        axis.set_title(
            f"Matched CSV Overlay | {metric} score={score:.6g}",
            fontsize=14,
            pad=14,
        )
        axis.set_xlabel(f"long {long_x_label}")
        axis.set_ylabel(f"long {long_y_label}", color=long_color)
        short_axis.set_ylabel(f"short {short_y_label}", color=short_color)
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(long_y_min, long_y_max)
        short_axis.set_ylim(short_y_min, short_y_max)

        def to_short_time(long_time: float) -> float:
            return long_time - match_start_time + short_time_origin

        def to_long_time(short_time: float) -> float:
            return short_time - short_time_origin + match_start_time

        top_axis = axis.secondary_xaxis(
            "top",
            functions=(to_short_time, to_long_time),
        )
        top_axis.set_xlabel(f"aligned short {short_x_label}")

        lines = [*long_line, *short_line]
        labels = [line.get_label() for line in lines]
        axis.legend(lines, labels, loc="best", frameon=False)

        axis.xaxis.set_major_locator(MaxNLocator(nbins=8))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
        short_axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
        top_axis.xaxis.set_major_locator(MaxNLocator(nbins=8))
        axis.grid(True, color="#dfe4ea", linewidth=0.8)

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(long_color)
        axis.spines["bottom"].set_color("#38404a")
        short_axis.spines["top"].set_visible(False)
        short_axis.spines["right"].set_color(short_color)
        top_axis.spines["top"].set_color("#38404a")
        axis.tick_params(axis="x", colors="#2e343c", labelsize=9)
        axis.tick_params(axis="y", colors=long_color, labelsize=9)
        short_axis.tick_params(axis="y", colors=short_color, labelsize=9)
        top_axis.tick_params(axis="x", colors="#2e343c", labelsize=9)

        figure.tight_layout()
        if plot_path is not None:
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(plot_path, format="png")
        if show:
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
