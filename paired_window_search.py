from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from config import BPI_PROCESSED_DIRNAME


DEFAULT_PATTERN = "matched_paired_points*.csv"
DEFAULT_RANK1_PLOT_FILENAME = "paired_window_rank1_char.png"
DEFAULT_TIME_CANDIDATES = [
    "short_offset_s",
    "matched_time_s",
    "time_s",
    "x_norm",
    "x_value",
    "x_px",
]
DEFAULT_ABSOLUTE_TIME_CANDIDATES = ["matched_time_s"]
DEFAULT_LEFT_CANDIDATES = ["short_value", "ultrasound_value", "y_norm", "y_value"]
DEFAULT_RIGHT_CANDIDATES = ["long_value", "wearable_value", "bpi_normalized", "BPI"]
EPSILON = 1e-12


@dataclass(frozen=True)
class PairedSeries:
    path: Path
    time_column: str
    absolute_time_column: str | None
    left_column: str
    right_column: str
    times: np.ndarray
    absolute_times: np.ndarray | None
    left_values: np.ndarray
    right_values: np.ndarray
    source_rows: int
    valid_rows: int
    duplicate_time_rows: int
    short_csv: str | None
    metric: str | None
    match_start_time: float | None
    match_end_time: float | None

    @property
    def duration_s(self) -> float:
        return float(self.times[-1] - self.times[0])


@dataclass(frozen=True)
class WindowResult:
    series: PairedSeries
    pearson: float
    pearson_p: float
    start_s: float
    end_s: float
    absolute_start_s: float | None
    absolute_end_s: float | None
    start_index: int
    end_index_exclusive: int

    @property
    def points(self) -> int:
        return self.end_index_exclusive - self.start_index


@dataclass(frozen=True)
class CommonDurationResult:
    duration_s: float
    average_pearson: float
    min_pearson: float
    max_pearson: float
    windows: list[WindowResult]


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def parse_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def column_key(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def choose_column(
    fieldnames: list[str],
    requested: str | None,
    candidates: list[str],
    *,
    required: bool = True,
) -> str | None:
    if requested:
        if requested in fieldnames:
            return requested
        requested_key = column_key(requested)
        for fieldname in fieldnames:
            if column_key(fieldname) == requested_key:
                return fieldname
        raise RuntimeError(f"Column {requested!r} is missing")

    lookup = {column_key(fieldname): fieldname for fieldname in fieldnames}
    for candidate in candidates:
        key = column_key(candidate)
        if key in lookup:
            return lookup[key]

    if not required:
        return None

    candidate_text = ", ".join(candidates)
    raise RuntimeError(f"Could not find a CSV column among: {candidate_text}")


def collapse_duplicate_times(
    times: np.ndarray,
    absolute_times: np.ndarray | None,
    left_values: np.ndarray,
    right_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, int]:
    if len(times) <= 1:
        return times, absolute_times, left_values, right_values, 0

    unique_times, inverse, counts = np.unique(times, return_inverse=True, return_counts=True)
    duplicate_rows = int(np.sum(counts - 1))
    if duplicate_rows == 0:
        return times, absolute_times, left_values, right_values, 0

    left_means = np.bincount(inverse, weights=left_values) / counts
    right_means = np.bincount(inverse, weights=right_values) / counts
    absolute_means = None
    if absolute_times is not None:
        absolute_means = np.bincount(inverse, weights=absolute_times) / counts

    return (
        unique_times.astype(float),
        None if absolute_means is None else absolute_means.astype(float),
        left_means.astype(float),
        right_means.astype(float),
        duplicate_rows,
    )


def read_paired_series(
    path: Path,
    *,
    time_column: str | None,
    absolute_time_column: str | None,
    left_column: str | None,
    right_column: str | None,
    expected_points: int | None,
) -> PairedSeries:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        fieldnames = list(reader.fieldnames)
        chosen_time = choose_column(fieldnames, time_column, DEFAULT_TIME_CANDIDATES)
        chosen_absolute_time = choose_column(
            fieldnames,
            absolute_time_column,
            DEFAULT_ABSOLUTE_TIME_CANDIDATES,
            required=False,
        )
        chosen_left = choose_column(fieldnames, left_column, DEFAULT_LEFT_CANDIDATES)
        chosen_right = choose_column(fieldnames, right_column, DEFAULT_RIGHT_CANDIDATES)
        if chosen_time is None or chosen_left is None or chosen_right is None:
            raise RuntimeError(f"Missing required columns in {path}")

        times: list[float] = []
        absolute_times: list[float] = []
        left_values: list[float] = []
        right_values: list[float] = []
        source_rows = 0
        valid_rows = 0
        short_csv = None
        metric = None
        match_start_time = None
        match_end_time = None

        for row in reader:
            source_rows += 1
            if short_csv is None:
                short_csv = parse_text(row.get("short_csv"))
            if metric is None:
                metric = parse_text(row.get("metric"))
            if match_start_time is None:
                match_start_time = parse_float(row.get("match_start_time"))
            if match_end_time is None:
                match_end_time = parse_float(row.get("match_end_time"))
            time_value = parse_float(row.get(chosen_time))
            absolute_time_value = (
                parse_float(row.get(chosen_absolute_time))
                if chosen_absolute_time is not None
                else None
            )
            left_value = parse_float(row.get(chosen_left))
            right_value = parse_float(row.get(chosen_right))
            if time_value is None or left_value is None or right_value is None:
                continue
            if chosen_absolute_time is not None and absolute_time_value is None:
                continue
            times.append(time_value)
            if absolute_time_value is not None:
                absolute_times.append(absolute_time_value)
            left_values.append(left_value)
            right_values.append(right_value)
            valid_rows += 1

    if expected_points is not None and valid_rows != expected_points:
        raise RuntimeError(
            f"{path.name}: expected {expected_points} valid paired points, got {valid_rows}"
        )
    if valid_rows < 3:
        raise RuntimeError(f"Need at least 3 valid paired points in {path}")

    time_array = np.asarray(times, dtype=float)
    absolute_time_array = (
        np.asarray(absolute_times, dtype=float)
        if len(absolute_times) == len(times)
        else None
    )
    left_array = np.asarray(left_values, dtype=float)
    right_array = np.asarray(right_values, dtype=float)
    order = np.argsort(time_array, kind="stable")
    time_array = time_array[order]
    if absolute_time_array is not None:
        absolute_time_array = absolute_time_array[order]
    left_array = left_array[order]
    right_array = right_array[order]
    time_array, absolute_time_array, left_array, right_array, duplicate_rows = (
        collapse_duplicate_times(time_array, absolute_time_array, left_array, right_array)
    )

    if len(time_array) < 3:
        raise RuntimeError(f"Need at least 3 unique time points in {path}")
    if time_array[-1] <= time_array[0]:
        raise RuntimeError(f"Time axis has no positive duration in {path}")

    return PairedSeries(
        path=path,
        time_column=chosen_time,
        absolute_time_column=chosen_absolute_time,
        left_column=chosen_left,
        right_column=chosen_right,
        times=time_array,
        absolute_times=absolute_time_array,
        left_values=left_array,
        right_values=right_array,
        source_rows=source_rows,
        valid_rows=valid_rows,
        duplicate_time_rows=duplicate_rows,
        short_csv=short_csv,
        metric=metric,
        match_start_time=match_start_time,
        match_end_time=match_end_time,
    )


def paired_csv_paths(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    if not input_dir.is_dir():
        raise RuntimeError(f"--input is not a directory: {input_dir}")
    iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
    paths = sorted(path for path in iterator if path.is_file())
    if not paths:
        raise RuntimeError(f"No paired CSV files found in {input_dir} with pattern {pattern!r}")
    return paths


def candidate_durations(
    series_items: list[PairedSeries],
    *,
    min_duration_s: float,
    max_duration_s: float | None,
    duration_step_s: float,
) -> np.ndarray:
    if min_duration_s <= 0:
        raise RuntimeError("--min-duration-s must be greater than 0")
    if duration_step_s <= 0:
        raise RuntimeError("--duration-step-s must be greater than 0")

    common_max = min(item.duration_s for item in series_items)
    if max_duration_s is not None:
        if max_duration_s <= 0:
            raise RuntimeError("--max-duration-s must be greater than 0")
        common_max = min(common_max, max_duration_s)
    if common_max + EPSILON < min_duration_s:
        raise RuntimeError(
            "No common duration can satisfy "
            f"min_duration_s={min_duration_s:g}; shortest CSV duration is {common_max:g}s"
        )

    count = int(math.floor((common_max - min_duration_s) / duration_step_s)) + 1
    durations = min_duration_s + np.arange(count, dtype=float) * duration_step_s
    return durations[durations <= common_max + EPSILON]


def candidate_starts(times: np.ndarray, duration_s: float, step_s: float) -> np.ndarray:
    if step_s <= 0:
        raise RuntimeError("--start-step-s must be greater than 0")
    latest_start = float(times[-1] - duration_s)
    first_start = float(times[0])
    if latest_start + EPSILON < first_start:
        return np.asarray([], dtype=float)
    count = int(math.floor((latest_start - first_start) / step_s)) + 1
    starts = first_start + np.arange(count, dtype=float) * step_s
    return starts[starts <= latest_start + EPSILON]


def prefix_sums(values: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(values, dtype=float)))


def pearsons_for_ranges(
    left_values: np.ndarray,
    right_values: np.ndarray,
    start_indices: np.ndarray,
    end_indices: np.ndarray,
    *,
    min_points: int,
) -> np.ndarray:
    lengths = (end_indices - start_indices).astype(float)
    left_prefix = prefix_sums(left_values)
    right_prefix = prefix_sums(right_values)
    left_sq_prefix = prefix_sums(left_values * left_values)
    right_sq_prefix = prefix_sums(right_values * right_values)
    cross_prefix = prefix_sums(left_values * right_values)

    left_sum = left_prefix[end_indices] - left_prefix[start_indices]
    right_sum = right_prefix[end_indices] - right_prefix[start_indices]
    left_sq_sum = left_sq_prefix[end_indices] - left_sq_prefix[start_indices]
    right_sq_sum = right_sq_prefix[end_indices] - right_sq_prefix[start_indices]
    cross_sum = cross_prefix[end_indices] - cross_prefix[start_indices]

    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = cross_sum - (left_sum * right_sum / lengths)
        left_var = left_sq_sum - (left_sum * left_sum / lengths)
        right_var = right_sq_sum - (right_sum * right_sum / lengths)
        denominator = np.sqrt(left_var * right_var)

    scores = np.full(len(start_indices), np.nan, dtype=float)
    valid = (
        (lengths >= min_points)
        & (left_var > EPSILON)
        & (right_var > EPSILON)
        & (denominator > EPSILON)
    )
    scores[valid] = numerator[valid] / denominator[valid]
    return scores


def absolute_time_at(series: PairedSeries, time_s: float) -> float | None:
    if series.absolute_times is None:
        return None
    return float(np.interp(time_s, series.times, series.absolute_times))


def pearson_p_value(pearson: float, points: int) -> float:
    if points < 2 or not math.isfinite(pearson):
        return math.nan
    if points == 2:
        return 1.0
    clipped = max(-1.0, min(1.0, pearson))
    if abs(clipped) >= 1.0:
        return 0.0
    degrees_of_freedom = points - 2
    statistic = abs(clipped) * math.sqrt(degrees_of_freedom / (1.0 - clipped * clipped))
    return float(2.0 * stats.t.sf(statistic, degrees_of_freedom))


def make_window_result(
    series: PairedSeries,
    *,
    pearson: float,
    start_s: float,
    end_s: float,
    start_index: int,
    end_index_exclusive: int,
) -> WindowResult:
    points = end_index_exclusive - start_index
    return WindowResult(
        series=series,
        pearson=pearson,
        pearson_p=pearson_p_value(pearson, points),
        start_s=start_s,
        end_s=end_s,
        absolute_start_s=absolute_time_at(series, start_s),
        absolute_end_s=absolute_time_at(series, end_s),
        start_index=start_index,
        end_index_exclusive=end_index_exclusive,
    )


def windows_for_duration(
    series: PairedSeries,
    *,
    duration_s: float,
    start_step_s: float,
    min_points: int,
) -> list[WindowResult]:
    starts = candidate_starts(series.times, duration_s, start_step_s)
    if len(starts) == 0:
        return []

    ends = starts + duration_s
    start_indices = np.searchsorted(series.times, starts, side="left")
    end_indices = np.searchsorted(series.times, ends, side="right")
    scores = pearsons_for_ranges(
        series.left_values,
        series.right_values,
        start_indices,
        end_indices,
        min_points=min_points,
    )
    finite = np.isfinite(scores)
    if not np.any(finite):
        return []

    finite_indices = np.flatnonzero(finite)
    ordered_indices = finite_indices[np.argsort(scores[finite_indices])[::-1]]
    return [
        make_window_result(
            series,
            pearson=float(scores[index]),
            start_s=float(starts[index]),
            end_s=float(ends[index]),
            start_index=int(start_indices[index]),
            end_index_exclusive=int(end_indices[index]),
        )
        for index in ordered_indices
    ]


def common_duration_result(
    duration_s: float,
    windows: list[WindowResult],
) -> CommonDurationResult:
    scores = np.asarray([window.pearson for window in windows], dtype=float)
    return CommonDurationResult(
        duration_s=duration_s,
        average_pearson=float(np.mean(scores)),
        min_pearson=float(np.min(scores)),
        max_pearson=float(np.max(scores)),
        windows=windows,
    )


def score_key(result: CommonDurationResult) -> tuple[float, float, float, float]:
    return (
        result.average_pearson,
        result.min_pearson,
        result.max_pearson,
        -result.duration_s,
    )


def score_common_windows_by_duration(
    series_items: list[PairedSeries],
    *,
    durations: np.ndarray,
    start_step_s: float,
    min_points: int,
) -> list[CommonDurationResult]:
    results: list[CommonDurationResult] = []
    for duration_s in durations:
        candidate_lists = [
            windows_for_duration(
                series,
                duration_s=float(duration_s),
                start_step_s=start_step_s,
                min_points=min_points,
            )
            for series in series_items
        ]
        if all(candidate_lists):
            results.append(
                common_duration_result(
                    float(duration_s),
                    [candidates[0] for candidates in candidate_lists],
                )
            )

    if not results:
        raise RuntimeError("No valid windows were found")
    results.sort(key=score_key, reverse=True)
    return results


def select_top_common_windows(
    series_items: list[PairedSeries],
    *,
    durations: np.ndarray,
    start_step_s: float,
    min_points: int,
    top: int,
) -> list[CommonDurationResult]:
    results = score_common_windows_by_duration(
        series_items,
        durations=durations,
        start_step_s=start_step_s,
        min_points=min_points,
    )
    return results[:top]


def format_optional_time(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def format_p_value(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.3e}"


def print_results(results: list[CommonDurationResult]) -> None:
    csv_width = max(
        len("csv"),
        *(len(window.series.path.name) for result in results for window in result.windows),
    )
    header = (
        f"{'rank':>4}  {'duration_s':>10}  {'avg_pearson':>12}  "
        f"{'csv':<{csv_width}}  {'pearson':>10}  {'p_value':>12}  "
        f"{'matched_start_s':>15}  {'matched_end_s':>13}"
    )
    print(header)
    print("-" * len(header))
    for rank, result in enumerate(results, start=1):
        if rank > 1:
            print("-" * len(header))
        for window in result.windows:
            print(
                f"{rank:>4}  {result.duration_s:>10.6f}  "
                f"{result.average_pearson:>12.10f}  "
                f"{window.series.path.name:<{csv_width}}  "
                f"{window.pearson:>10.10f}  "
                f"{format_p_value(window.pearson_p):>12}  "
                f"{format_optional_time(window.absolute_start_s):>15}  "
                f"{format_optional_time(window.absolute_end_s):>13}"
            )


def print_duration_sweep(results: list[CommonDurationResult]) -> None:
    print()
    print("duration_sweep")
    print("duration_s,rank_by_score,avg_pearson,min_pearson,max_pearson,csv_count")
    ranks = {id(result): rank for rank, result in enumerate(results, start=1)}
    for result in sorted(results, key=lambda item: item.duration_s):
        print(
            f"{result.duration_s:.6f},{ranks[id(result)]},"
            f"{result.average_pearson:.10f},{result.min_pearson:.10f},"
            f"{result.max_pearson:.10f},{len(result.windows)}"
        )


def paired_plot_output_path(input_dir: Path, plot_output: Path | None) -> Path:
    if plot_output is None:
        return input_dir / DEFAULT_RANK1_PLOT_FILENAME
    if plot_output.suffix:
        return plot_output
    return plot_output / DEFAULT_RANK1_PLOT_FILENAME


def paired_axis_label(column: str, default_label: str) -> str:
    key = column_key(column)
    if key in {"short_value", "ultrasound_value"}:
        return "Ultrasound paired value"
    if key in {"long_value", "wearable_value", "bpi_normalized", "bpi"}:
        return "Wearable paired value"
    return default_label


def window_plot_times(window: WindowResult) -> tuple[np.ndarray, float, float]:
    window_slice = slice(window.start_index, window.end_index_exclusive)
    if window.series.absolute_times is not None:
        times = window.series.absolute_times[window_slice]
        start_time = (
            window.absolute_start_s
            if window.absolute_start_s is not None
            else float(times[0])
        )
        end_time = (
            window.absolute_end_s
            if window.absolute_end_s is not None
            else float(times[-1])
        )
        return times, start_time, end_time

    times = window.series.times[window_slice]
    if window.series.match_start_time is not None:
        absolute_times = times + window.series.match_start_time
        return (
            absolute_times,
            window.series.match_start_time + window.start_s,
            window.series.match_start_time + window.end_s,
        )
    return times, window.start_s, window.end_s


def series_plot_times(series: PairedSeries) -> np.ndarray:
    if series.absolute_times is not None:
        return series.absolute_times
    if series.match_start_time is not None:
        return series.times + series.match_start_time
    return series.times


def unique_non_none(values: list[str | None]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None or value in seen:
            continue
        output.append(value)
        seen.add(value)
    return output


def resolve_overlay_long_path(
    input_dir: Path,
    *,
    overlay_long: Path | None,
    overlay_long_dir: Path | None,
    overlay_long_pattern: str,
) -> Path:
    if overlay_long is not None:
        if not overlay_long.is_file():
            raise RuntimeError(f"--overlay-long is not a file: {overlay_long}")
        return overlay_long

    candidate_dirs = (
        [overlay_long_dir]
        if overlay_long_dir is not None
        else [
            input_dir / BPI_PROCESSED_DIRNAME,
            input_dir.parent / BPI_PROCESSED_DIRNAME,
        ]
    )
    seen_dirs: set[Path] = set()
    for directory in candidate_dirs:
        if directory is None:
            continue
        resolved = directory.resolve()
        if resolved in seen_dirs or not directory.is_dir():
            continue
        seen_dirs.add(resolved)
        paths = sorted(path for path in directory.glob(overlay_long_pattern) if path.is_file())
        if len(paths) == 1:
            return paths[0]
        if len(paths) > 1:
            candidates = ", ".join(path.name for path in paths)
            raise RuntimeError(
                "Multiple overlay long CSV files were found in "
                f"{directory}: {candidates}. Use --overlay-long to choose one."
            )

    raise RuntimeError(
        "Could not find the interval matcher long CSV for the overlay. "
        "Use --overlay-long or --overlay-long-dir."
    )


def read_overlay_long_series(args: argparse.Namespace):
    from csv_interval_matcher import read_series

    long_path = resolve_overlay_long_path(
        args.input,
        overlay_long=args.overlay_long,
        overlay_long_dir=args.overlay_long_dir,
        overlay_long_pattern=args.overlay_long_pattern,
    )
    return read_series(
        long_path,
        x_column=args.overlay_long_x_column,
        y_column=args.overlay_long_y_column,
        sample_rate_hz=args.overlay_long_sample_rate,
    )


def write_rank1_paired_window_plot(
    result: CommonDurationResult,
    *,
    plot_path: Path,
    overlay_long,
    show: bool,
) -> None:
    from plotting import PairedWindowZoomSeries, write_paired_window_overlay_zoom_plot

    plot_windows: list[PairedWindowZoomSeries] = []
    for window in result.windows:
        window_slice = slice(window.start_index, window.end_index_exclusive)
        times, start_time, end_time = window_plot_times(window)
        plot_windows.append(
            PairedWindowZoomSeries(
                name=window.series.path.name,
                overlay_times=series_plot_times(window.series),
                overlay_short_values=window.series.left_values,
                window_times=times,
                window_short_values=window.series.left_values[window_slice],
                window_long_values=window.series.right_values[window_slice],
                short_label=paired_axis_label(
                    window.series.left_column,
                    window.series.left_column,
                ),
                long_label=paired_axis_label(
                    window.series.right_column,
                    window.series.right_column,
                ),
                rank=1,
                pearson=window.pearson,
                pearson_p=window.pearson_p,
                start_time=start_time,
                end_time=end_time,
                duration_s=result.duration_s,
                point_count=window.points,
            )
        )

    metrics = unique_non_none([window.series.metric for window in result.windows])
    write_paired_window_overlay_zoom_plot(
        plot_path,
        overlay_long_times=overlay_long.times,
        overlay_long_values=overlay_long.values,
        overlay_long_label=paired_axis_label(overlay_long.y_column, overlay_long.y_column),
        windows=plot_windows,
        metric=metrics[0] if len(metrics) == 1 else None,
        show=show,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search paired-point CSV files for same-duration windows with high Pearson "
            "correlation."
        ),
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Folder containing paired-point CSV files produced by csv_interval_matcher.py.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Filename pattern inside input folder. Default: {DEFAULT_PATTERN}.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input folder recursively.",
    )
    parser.add_argument(
        "--time-column",
        help=(
            "Window time column. Defaults to short_offset_s when available, then "
            "matched_time_s/time_s/x_norm/x_value/x_px."
        ),
    )
    parser.add_argument(
        "--absolute-time-column",
        help="Optional absolute time column to display. Defaults to matched_time_s when present.",
    )
    parser.add_argument("--left-column", help="First paired value column. Default: short_value.")
    parser.add_argument("--right-column", help="Second paired value column. Default: long_value.")
    parser.add_argument(
        "--expected-points",
        type=int,
        default=3000,
        help="Require this many valid paired rows per CSV. Use 0 to disable. Default: 3000.",
    )
    parser.add_argument(
        "--min-duration-s",
        type=float,
        default=5.0,
        help="Shortest tested window duration in seconds. Default: 5.",
    )
    parser.add_argument(
        "--max-duration-s",
        type=float,
        help="Longest tested window duration in seconds. Default: shortest CSV duration.",
    )
    parser.add_argument(
        "--duration-step-s",
        type=float,
        default=1.0,
        help="Step between tested window durations in seconds. Default: 1.",
    )
    parser.add_argument(
        "--start-step-s",
        type=float,
        default=0.1,
        help="Step between tested window starts in seconds. Default: 0.1.",
    )
    parser.add_argument(
        "--min-points-per-window",
        type=int,
        default=3,
        help="Minimum valid paired points required inside a window. Default: 3.",
    )
    parser.add_argument(
        "--show-duration-sweep",
        action="store_true",
        help="Print one summary row for every tested window duration.",
    )
    parser.add_argument("--top", type=int, default=5, help="Number of ranks to print. Default: 5.")
    parser.add_argument(
        "--plot-output",
        type=Path,
        help=(
            "PNG path, or folder, for the rank-1 overlay/char zoom plot. "
            f"Default: INPUT/{DEFAULT_RANK1_PLOT_FILENAME}."
        ),
    )
    parser.add_argument(
        "--overlay-long",
        type=Path,
        help=(
            "Original long CSV used by csv_interval_matcher.py for the top overlay. "
            "Defaults to INPUT/BPI_Processed/*.csv, then INPUT_PARENT/BPI_Processed/*.csv."
        ),
    )
    parser.add_argument(
        "--overlay-long-dir",
        type=Path,
        help="Folder containing the original long CSV for the top overlay.",
    )
    parser.add_argument(
        "--overlay-long-pattern",
        default="*.csv",
        help="Filename pattern for --overlay-long-dir. Default: *.csv.",
    )
    parser.add_argument(
        "--overlay-long-x-column",
        help="Long CSV time/x column for the top overlay. Use row_index to synthesize time from rows.",
    )
    parser.add_argument("--overlay-long-y-column", help="Long CSV signal/y column for the top overlay.")
    parser.add_argument(
        "--overlay-long-sample-rate",
        type=float,
        help="Hz used when --overlay-long-x-column row_index is selected.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Show the rank-1 overlay/char Matplotlib window after saving it.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving and showing the rank-1 overlay/char plot.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.expected_points < 0:
        print("error: --expected-points must be at least 0", file=sys.stderr)
        return 1
    expected_points = None if args.expected_points == 0 else args.expected_points

    try:
        paths = paired_csv_paths(args.input, args.pattern, args.recursive)
        series_items = [
            read_paired_series(
                path,
                time_column=args.time_column,
                absolute_time_column=args.absolute_time_column,
                left_column=args.left_column,
                right_column=args.right_column,
                expected_points=expected_points,
            )
            for path in paths
        ]
        durations = candidate_durations(
            series_items,
            min_duration_s=args.min_duration_s,
            max_duration_s=args.max_duration_s,
            duration_step_s=args.duration_step_s,
        )
        if args.top <= 0:
            raise RuntimeError("--top must be greater than 0")
        if args.min_points_per_window < 2:
            raise RuntimeError("--min-points-per-window must be at least 2")
        scored_durations = score_common_windows_by_duration(
            series_items,
            durations=durations,
            start_step_s=args.start_step_s,
            min_points=args.min_points_per_window,
        )
        results = scored_durations[:args.top]
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print_results(results)
    if args.show_duration_sweep:
        print_duration_sweep(scored_durations)
    if not args.no_plot:
        try:
            plot_path = paired_plot_output_path(args.input, args.plot_output)
            overlay_long = read_overlay_long_series(args)
            write_rank1_paired_window_plot(
                results[0],
                plot_path=plot_path,
                overlay_long=overlay_long,
                show=args.show_plot,
            )
        except Exception as exc:
            print(f"error: failed to write rank1 plot: {exc}", file=sys.stderr)
            return 1
        print(f"\nwrote rank1 overlay/char plot {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
