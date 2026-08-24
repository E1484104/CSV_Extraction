from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import MaxNLocator


PLOT_FIGSIZE = (10, 6)
PLOT_DPI = 100
MATCH_DETAIL_PANE_LIMIT = 4
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
    matched_long_times: Sequence[float] | None = None
    matched_long_values: Sequence[float] | None = None


@dataclass(frozen=True)
class PairedWindowPlotSeries:
    name: str
    times: Sequence[float]
    short_values: Sequence[float]
    long_values: Sequence[float]
    short_label: str
    long_label: str
    rank: int
    pearson: float
    pearson_p: float
    start_time: float
    end_time: float
    duration_s: float
    point_count: int


@dataclass(frozen=True)
class PairedWindowZoomSeries:
    name: str
    overlay_times: Sequence[float]
    overlay_short_values: Sequence[float]
    window_times: Sequence[float]
    window_short_values: Sequence[float]
    window_long_values: Sequence[float]
    short_label: str
    long_label: str
    rank: int
    pearson: float
    pearson_p: float
    start_time: float
    end_time: float
    duration_s: float
    point_count: int


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


def _finite_paired_window_points(
    xs: Sequence[float],
    short_values: Sequence[float],
    long_values: Sequence[float],
) -> tuple[list[float], list[float], list[float]]:
    finite_xs: list[float] = []
    finite_short_values: list[float] = []
    finite_long_values: list[float] = []
    for x, short_value, long_value in zip(xs, short_values, long_values):
        x_float = float(x)
        short_float = float(short_value)
        long_float = float(long_value)
        if (
            math.isfinite(x_float)
            and math.isfinite(short_float)
            and math.isfinite(long_float)
        ):
            finite_xs.append(x_float)
            finite_short_values.append(short_float)
            finite_long_values.append(long_float)
    return finite_xs, finite_short_values, finite_long_values


def _format_plot_p_value(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.3e}"


def _format_plot_number(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.6g}"


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
    plot_mode: str = "raw",
    show: bool = False,
) -> None:
    long_xs, long_ys = _finite_xy(long_times, long_values)
    if not long_xs:
        raise RuntimeError("No finite long CSV points are available for plotting")

    prepared_matches: list[
        tuple[MatchOverlaySeries, list[float], list[float], list[float], list[float]]
    ] = []
    for match in short_matches:
        short_xs, short_ys = _finite_xy(match.aligned_times, match.values)
        if short_xs:
            matched_long_xs: list[float] = []
            matched_long_ys: list[float] = []
            if match.matched_long_times is not None and match.matched_long_values is not None:
                matched_long_xs, matched_long_ys = _finite_xy(
                    match.matched_long_times,
                    match.matched_long_values,
                )
            prepared_matches.append(
                (match, short_xs, short_ys, matched_long_xs, matched_long_ys)
            )
    if not prepared_matches:
        raise RuntimeError("No finite matched short CSV points are available for plotting")

    x_min, x_max = _expanded_bounds(long_xs)
    long_y_min, long_y_max = _expanded_bounds(long_ys)
    short_values_all = [
        value
        for _, _, short_ys, _, _ in prepared_matches
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
    short_y_labels = sorted({match.y_label for match, _, _, _, _ in prepared_matches})
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
        for index, (
            match,
            short_xs,
            short_ys,
            matched_long_xs,
            matched_long_ys,
        ) in enumerate(prepared_matches):
            color = short_colors[index % len(short_colors)]
            axis.axvspan(match.start_time, match.end_time, color=color, alpha=0.07, linewidth=0)
            if matched_long_xs:
                axis.plot(
                    matched_long_xs,
                    matched_long_ys,
                    color=long_color,
                    linewidth=1.9,
                    alpha=0.48,
                    zorder=3,
                )
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

        title = f"Matched Sample Interval | metric={metric}"
        if plot_mode != "raw":
            title += f" | plot={plot_mode}"
        axis.set_title(title, fontsize=14, pad=14)
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

        for index, (
            match,
            short_xs,
            short_ys,
            matched_long_xs,
            matched_long_ys,
        ) in enumerate(detail_matches):
            detail_axis = figure.add_subplot(grid[1, index])
            detail_short_axis = detail_axis.twinx()
            detail_short_axis.patch.set_alpha(0.0)
            color = short_colors[index % len(short_colors)]
            detail_long_xs = matched_long_xs or long_xs
            detail_long_ys = matched_long_ys or long_ys

            duration = match.end_time - match.start_time
            padding = duration * 0.03 if duration > 0 else 1.0
            detail_x_min = match.start_time - padding
            detail_x_max = match.end_time + padding
            detail_long_y_min, detail_long_y_max = _windowed_y_bounds(
                detail_long_xs,
                detail_long_ys,
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
                detail_long_xs,
                detail_long_ys,
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


def write_paired_match_points_plot(
    plot_path: Path | None,
    *,
    matches: Sequence[MatchOverlaySeries],
    metric: str,
    sample_method: str,
    show: bool = False,
) -> None:
    prepared: list[tuple[MatchOverlaySeries, list[float], list[float], list[float]]] = []
    for match in matches:
        if match.matched_long_values is None:
            continue
        xs: list[float] = []
        short_ys: list[float] = []
        long_ys: list[float] = []
        for x, short_y, long_y in zip(
            match.aligned_times,
            match.values,
            match.matched_long_values,
        ):
            x_value = float(x)
            short_value = float(short_y)
            long_value = float(long_y)
            if (
                math.isfinite(x_value)
                and math.isfinite(short_value)
                and math.isfinite(long_value)
            ):
                xs.append(x_value)
                short_ys.append(short_value)
                long_ys.append(long_value)
        if xs:
            prepared.append((match, xs, short_ys, long_ys))

    if not prepared:
        raise RuntimeError("No finite paired match points are available for plotting")

    long_color = "#1f5f9e"
    short_colors = [
        "#c2571a",
        "#178a63",
        "#7b4ab8",
        "#b33c60",
        "#8a6f12",
        "#156f7a",
    ]
    pane_count = len(prepared)
    figure_width = max(10.0, 3.8 * pane_count)
    figure, axes = plt.subplots(
        1,
        pane_count,
        figsize=(figure_width, 4.2),
        dpi=PLOT_DPI,
        squeeze=False,
    )

    try:
        if figure.canvas.manager is not None:
            figure.canvas.manager.set_window_title("Pearson Paired Points")
        figure.patch.set_facecolor("white")
        figure.suptitle(
            f"Pearson Paired Points | metric={metric} | sample={sample_method}",
            fontsize=14,
            y=0.98,
        )

        for index, (match, xs, short_ys, long_ys) in enumerate(prepared):
            axis = axes[0][index]
            short_axis = axis.twinx()
            short_axis.patch.set_alpha(0.0)
            color = short_colors[index % len(short_colors)]
            marker = "." if len(xs) <= 700 else None
            marker_size = 2.4 if marker else 0.0

            long_line = axis.plot(
                xs,
                long_ys,
                color=long_color,
                linewidth=1.35,
                marker=marker,
                markersize=marker_size,
                alpha=LONG_OVERLAY_ALPHA,
                label="Wearable paired points",
            )
            short_line = short_axis.plot(
                xs,
                short_ys,
                color=color,
                linewidth=1.35,
                marker=marker,
                markersize=marker_size,
                alpha=SHORT_OVERLAY_ALPHA,
                label="Ultrasound paired points",
            )

            x_min, x_max = _expanded_bounds(xs)
            long_y_min, long_y_max = _expanded_bounds(long_ys)
            short_y_min, short_y_max = _expanded_bounds(short_ys)

            axis.set_title(
                f"phase {match.rank} | {match.start_time:.6g}-{match.end_time:.6g}",
                fontsize=10,
                pad=8,
            )
            axis.set_xlim(x_min, x_max)
            axis.set_ylim(long_y_min, long_y_max)
            short_axis.set_ylim(short_y_min, short_y_max)
            axis.set_xlabel("Sample Time (s)", fontsize=9)
            if index == 0:
                axis.set_ylabel("Wearable paired value", color=long_color, fontsize=9)
            else:
                axis.tick_params(axis="y", labelleft=False)
            if index == pane_count - 1:
                short_axis.set_ylabel("Ultrasound paired value", fontsize=9)
            else:
                short_axis.tick_params(axis="y", labelright=False)

            axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
            short_axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.grid(True, color="#dfe4ea", linewidth=0.8)

            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_color(long_color)
            axis.spines["bottom"].set_color("#38404a")
            short_axis.spines["top"].set_visible(False)
            short_axis.spines["right"].set_color("#38404a")
            axis.tick_params(axis="x", colors="#2e343c", labelsize=8)
            axis.tick_params(axis="y", colors=long_color, labelsize=8)
            short_axis.tick_params(axis="y", colors="#2e343c", labelsize=8)

            if index == 0:
                lines = [*long_line, *short_line]
                labels = [line.get_label() for line in lines]
                axis.legend(lines, labels, loc="best", frameon=False, fontsize=8)

        figure.subplots_adjust(
            left=0.07,
            right=0.93,
            top=0.84,
            bottom=0.14,
            wspace=0.32,
        )
        if plot_path is not None:
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(plot_path, format="png")
        if show:
            plt.show()
    finally:
        plt.close(figure)


def _common_axis_label(labels: Sequence[str], fallback: str) -> str:
    unique_labels = sorted({label for label in labels if label})
    if len(unique_labels) == 1:
        return unique_labels[0]
    return fallback


def write_paired_window_overlay_zoom_plot(
    plot_path: Path | None,
    *,
    overlay_long_times: Sequence[float],
    overlay_long_values: Sequence[float],
    overlay_long_label: str,
    windows: Sequence[PairedWindowZoomSeries],
    metric: str | None = None,
    show: bool = False,
) -> None:
    prepared: list[
        tuple[
            PairedWindowZoomSeries,
            list[float],
            list[float],
            list[float],
            list[float],
            list[float],
        ]
    ] = []
    long_xs, long_ys = _finite_xy(overlay_long_times, overlay_long_values)
    if not long_xs:
        raise RuntimeError("No finite long CSV points are available for plotting")

    for window in windows:
        overlay_xs, overlay_short_ys = _finite_xy(
            window.overlay_times,
            window.overlay_short_values,
        )
        window_xs, window_short_ys, window_long_ys = _finite_paired_window_points(
            window.window_times,
            window.window_short_values,
            window.window_long_values,
        )
        if overlay_xs and window_xs:
            prepared.append(
                (
                    window,
                    overlay_xs,
                    overlay_short_ys,
                    window_xs,
                    window_short_ys,
                    window_long_ys,
                )
            )

    if not prepared:
        raise RuntimeError("No finite paired window points are available for plotting")

    long_color = "#1f5f9e"
    short_colors = [
        "#c2571a",
        "#178a63",
        "#7b4ab8",
        "#b33c60",
        "#8a6f12",
        "#156f7a",
    ]
    pane_count = len(prepared)
    figure_width = max(11.0, 3.9 * pane_count)
    figure = plt.figure(figsize=(figure_width, 8.2), dpi=PLOT_DPI)
    grid = figure.add_gridspec(
        2,
        pane_count,
        height_ratios=[2.25, 1.12],
        hspace=0.55,
        wspace=0.34,
    )
    overlay_axis = figure.add_subplot(grid[0, :])

    try:
        if figure.canvas.manager is not None:
            figure.canvas.manager.set_window_title("Rank-1 Window Overlay")
        figure.patch.set_facecolor("white")
        overlay_axis.set_facecolor("#fbfcfe")

        overlay_short_axis = overlay_axis.twinx()
        overlay_short_axis.patch.set_alpha(0.0)

        all_overlay_short_ys = [
            y
            for _, _, overlay_short_ys, _, _, _ in prepared
            for y in overlay_short_ys
        ]
        x_min, x_max = _expanded_bounds(long_xs)
        long_y_min, long_y_max = _expanded_bounds(long_ys)
        short_y_min, short_y_max = _expanded_bounds(all_overlay_short_ys)

        long_line = overlay_axis.plot(
            long_xs,
            long_ys,
            color=long_color,
            linewidth=1.5,
            alpha=LONG_OVERLAY_ALPHA,
            label="Wearable",
            zorder=2,
        )

        for index, (
            window,
            overlay_xs,
            overlay_short_ys,
            _,
            _,
            _,
        ) in enumerate(prepared):
            color = short_colors[index % len(short_colors)]
            overlay_short_axis.plot(
                overlay_xs,
                overlay_short_ys,
                color=color,
                linewidth=1.8,
                alpha=SHORT_OVERLAY_ALPHA,
                label=f"phase{index + 1}",
                zorder=3,
            )
            overlay_axis.axvspan(
                window.start_time,
                window.end_time,
                color=color,
                alpha=0.065,
                linewidth=0,
                zorder=1,
            )

        duration_s = prepared[0][0].duration_s
        point_counts = sorted({window.point_count for window, _, _, _, _, _ in prepared})
        point_text = (
            f"{point_counts[0]} pts"
            if len(point_counts) == 1
            else f"{point_counts[0]}-{point_counts[-1]} pts"
        )
        title = (
            "Rank-1 Selected Window on Matched Overlay | "
            f"size={_format_plot_number(duration_s)} s ({point_text})"
        )
        if metric:
            title += f" | metric={metric}"
        overlay_axis.set_title(
            title,
            fontsize=14,
            pad=14,
        )
        overlay_axis.set_ylabel(
            overlay_long_label,
            color=long_color,
        )
        overlay_short_axis.set_ylabel(
            _common_axis_label(
                [window.short_label for window, _, _, _, _, _ in prepared],
                "Ultrasound paired value",
            )
        )
        overlay_axis.set_xlim(x_min, x_max)
        overlay_axis.set_ylim(long_y_min, long_y_max)
        overlay_short_axis.set_ylim(short_y_min, short_y_max)

        overlay_axis.xaxis.set_major_locator(MaxNLocator(nbins=10))
        overlay_axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
        overlay_short_axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
        overlay_axis.grid(True, color="#dfe4ea", linewidth=0.8)

        overlay_axis.spines["top"].set_visible(False)
        overlay_axis.spines["right"].set_visible(False)
        overlay_axis.spines["left"].set_color(long_color)
        overlay_axis.spines["bottom"].set_color("#38404a")
        overlay_short_axis.spines["top"].set_visible(False)
        overlay_short_axis.spines["right"].set_color("#38404a")
        overlay_axis.tick_params(axis="x", colors="#2e343c", labelsize=9)
        overlay_axis.tick_params(axis="y", colors=long_color, labelsize=9)
        overlay_short_axis.tick_params(axis="y", colors="#2e343c", labelsize=9)

        legend_lines = list(long_line)
        legend_labels = [line.get_label() for line in legend_lines]
        short_lines, short_labels = overlay_short_axis.get_legend_handles_labels()
        overlay_axis.legend(
            [*legend_lines, *short_lines],
            [*legend_labels, *short_labels],
            loc="best",
            frameon=False,
            fontsize=8,
            ncol=min(4, pane_count + 1),
        )

        detail_axes = []
        for index, (
            window,
            _,
            _,
            window_xs,
            window_short_ys,
            window_long_ys,
        ) in enumerate(prepared):
            detail_axis = figure.add_subplot(grid[1, index])
            detail_short_axis = detail_axis.twinx()
            detail_short_axis.patch.set_alpha(0.0)
            detail_axes.append(detail_axis)

            color = short_colors[index % len(short_colors)]
            detail_axis.set_facecolor("#fbfcfe")
            detail_axis.plot(
                window_xs,
                window_long_ys,
                color=long_color,
                linewidth=1.35,
                alpha=LONG_OVERLAY_ALPHA,
                label="Wearable",
                zorder=2,
            )
            detail_short_axis.plot(
                window_xs,
                window_short_ys,
                color=color,
                linewidth=1.55,
                alpha=SHORT_OVERLAY_ALPHA,
                label="Ultrasound",
                zorder=3,
            )

            duration = max(0.0, window.end_time - window.start_time)
            padding = duration * 0.03 if duration > 0 else 1.0
            detail_x_min = window.start_time - padding
            detail_x_max = window.end_time + padding
            detail_axis.set_xlim(detail_x_min, detail_x_max)
            detail_axis.set_ylim(*_expanded_bounds(window_long_ys))
            detail_short_axis.set_ylim(*_expanded_bounds(window_short_ys))

            detail_axis.set_title(
                "\n".join(
                    [
                        f"phase{index + 1}",
                        (
                            f"Pearson r = {_format_plot_number(window.pearson)} | "
                            f"p-value = {_format_plot_p_value(window.pearson_p)}"
                        ),
                        (
                            f"start-end = {_format_plot_number(window.start_time)}"
                            f"-{_format_plot_number(window.end_time)} s"
                        ),
                    ]
                ),
                fontsize=9,
                pad=10,
            )
            detail_axis.set_xlabel("Sample Time (s)", fontsize=9)
            if index == 0:
                detail_axis.set_ylabel(window.long_label, color=long_color, fontsize=9)
            else:
                detail_axis.tick_params(axis="y", labelleft=False)
            if index == pane_count - 1:
                detail_short_axis.set_ylabel(window.short_label, fontsize=9)
            else:
                detail_short_axis.tick_params(axis="y", labelright=False)

            detail_axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
            detail_axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
            detail_short_axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
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

            overlay_axis.add_patch(
                Rectangle(
                    (window.start_time, 0.0),
                    window.end_time - window.start_time,
                    1.0,
                    transform=overlay_axis.get_xaxis_transform(),
                    fill=False,
                    edgecolor=color,
                    linewidth=1.15,
                    linestyle="--",
                    zorder=5,
                )
            )

        overlay_bottom_y = overlay_axis.get_ylim()[0]
        for (window, _, _, _, _, _), detail_axis in zip(prepared, detail_axes):
            for overlay_x, detail_x in (
                (window.start_time, 0.0),
                (window.end_time, 1.0),
            ):
                figure.add_artist(
                    ConnectionPatch(
                        xyA=(overlay_x, overlay_bottom_y),
                        xyB=(detail_x, 1.0),
                        coordsA="data",
                        coordsB="axes fraction",
                        axesA=overlay_axis,
                        axesB=detail_axis,
                        color="black",
                        linewidth=1.1,
                        alpha=0.86,
                        clip_on=False,
                        zorder=4,
                    )
                )

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


def write_paired_window_rank_plot(
    plot_path: Path | None,
    *,
    windows: Sequence[PairedWindowPlotSeries],
    show: bool = False,
) -> None:
    prepared: list[tuple[PairedWindowPlotSeries, list[float], list[float], list[float]]] = []
    for window in windows:
        xs, short_ys, long_ys = _finite_paired_window_points(
            window.times,
            window.short_values,
            window.long_values,
        )
        if xs:
            prepared.append((window, xs, short_ys, long_ys))

    if not prepared:
        raise RuntimeError("No finite paired window points are available for plotting")

    long_color = "#1f5f9e"
    short_colors = [
        "#c2571a",
        "#178a63",
        "#7b4ab8",
        "#b33c60",
        "#8a6f12",
        "#156f7a",
    ]
    pane_count = len(prepared)
    figure_width = max(10.0, 3.8 * pane_count)
    figure, axes = plt.subplots(
        1,
        pane_count,
        figsize=(figure_width, 5.0),
        dpi=PLOT_DPI,
        squeeze=False,
    )

    try:
        if figure.canvas.manager is not None:
            figure.canvas.manager.set_window_title("Characteristic Paired Value")
        figure.patch.set_facecolor("white")
        duration_s = prepared[0][0].duration_s
        point_counts = sorted({window.point_count for window, _, _, _ in prepared})
        point_text = (
            f"{point_counts[0]} pts"
            if len(point_counts) == 1
            else f"{point_counts[0]}-{point_counts[-1]} pts"
        )
        figure.suptitle(
            (
                "Characteristic Paired Value | "
                f"size={_format_plot_number(duration_s)} s ({point_text})"
            ),
            fontsize=14,
            y=0.98,
        )

        for index, (window, xs, short_ys, long_ys) in enumerate(prepared):
            axis = axes[0][index]
            short_axis = axis.twinx()
            short_axis.patch.set_alpha(0.0)
            color = short_colors[index % len(short_colors)]

            long_line = axis.plot(
                xs,
                long_ys,
                color=long_color,
                linewidth=1.35,
                alpha=LONG_OVERLAY_ALPHA,
                label="Wearable",
            )
            short_line = short_axis.plot(
                xs,
                short_ys,
                color=color,
                linewidth=1.35,
                alpha=SHORT_OVERLAY_ALPHA,
                label="Ultrasound",
            )

            x_min, x_max = _expanded_bounds(xs)
            long_y_min, long_y_max = _expanded_bounds(long_ys)
            short_y_min, short_y_max = _expanded_bounds(short_ys)

            axis.set_title(
                "\n".join(
                    [
                        f"phase{index + 1}",
                        (
                            f"Pearson r = {_format_plot_number(window.pearson)} | "
                            f"p-value = {_format_plot_p_value(window.pearson_p)}"
                        ),
                        (
                            f"start-end = {_format_plot_number(window.start_time)}"
                            f"-{_format_plot_number(window.end_time)} s"
                        ),
                    ]
                ),
                fontsize=9,
                pad=38,
            )
            axis.set_xlim(x_min, x_max)
            axis.set_ylim(long_y_min, long_y_max)
            short_axis.set_ylim(short_y_min, short_y_max)
            axis.set_xlabel("Sample Time (s)", fontsize=9)
            if index == 0:
                axis.set_ylabel(window.long_label, color=long_color, fontsize=9)
            else:
                axis.tick_params(axis="y", labelleft=False)
            if index == pane_count - 1:
                short_axis.set_ylabel(window.short_label, fontsize=9)
            else:
                short_axis.tick_params(axis="y", labelright=False)

            axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
            short_axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.grid(True, color="#dfe4ea", linewidth=0.8)

            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_color(long_color)
            axis.spines["bottom"].set_color("#38404a")
            short_axis.spines["top"].set_visible(False)
            short_axis.spines["right"].set_color("#38404a")
            axis.tick_params(axis="x", colors="#2e343c", labelsize=8)
            axis.tick_params(axis="y", colors=long_color, labelsize=8)
            short_axis.tick_params(axis="y", colors="#2e343c", labelsize=8)

            lines = [*long_line, *short_line]
            labels = [line.get_label() for line in lines]
            axis.legend(
                lines,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.01),
                ncol=2,
                frameon=False,
                fontsize=8,
                borderaxespad=0.0,
                columnspacing=1.0,
                handlelength=1.7,
            )

        figure.subplots_adjust(
            left=0.07,
            right=0.93,
            top=0.66,
            bottom=0.14,
            wspace=0.32,
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
