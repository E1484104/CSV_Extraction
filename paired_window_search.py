from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_PATTERN = "matched_paired_points*.csv"
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

    @property
    def duration_s(self) -> float:
        return float(self.times[-1] - self.times[0])


@dataclass(frozen=True)
class WindowResult:
    series: PairedSeries
    pearson: float
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

        for row in reader:
            source_rows += 1
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


def make_window_result(
    series: PairedSeries,
    *,
    pearson: float,
    start_s: float,
    end_s: float,
    start_index: int,
    end_index_exclusive: int,
) -> WindowResult:
    return WindowResult(
        series=series,
        pearson=pearson,
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


def overlap_fraction(left: WindowResult, right: WindowResult) -> float:
    if left.series.path != right.series.path:
        return 0.0
    overlap = max(0.0, min(left.end_s, right.end_s) - max(left.start_s, right.start_s))
    shorter = min(left.end_s - left.start_s, right.end_s - right.start_s)
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def first_independent_window(
    candidates: list[WindowResult],
    selected: list[WindowResult],
    *,
    max_overlap_fraction: float,
) -> WindowResult | None:
    for candidate in candidates:
        if all(
            overlap_fraction(candidate, existing) <= max_overlap_fraction + EPSILON
            for existing in selected
        ):
            return candidate
    return None


def select_top_common_windows(
    series_items: list[PairedSeries],
    *,
    durations: np.ndarray,
    start_step_s: float,
    min_points: int,
    top: int,
    max_overlap_fraction: float,
) -> list[CommonDurationResult]:
    candidates_by_duration: list[tuple[float, list[list[WindowResult]]]] = []
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
            candidates_by_duration.append((float(duration_s), candidate_lists))

    if not candidates_by_duration:
        raise RuntimeError("No valid windows were found")

    selected_by_series: list[list[WindowResult]] = [[] for _ in series_items]
    results: list[CommonDurationResult] = []
    while len(results) < top:
        best_result: CommonDurationResult | None = None
        for duration_s, candidate_lists in candidates_by_duration:
            windows: list[WindowResult] = []
            for series_index, candidates in enumerate(candidate_lists):
                window = first_independent_window(
                    candidates,
                    selected_by_series[series_index],
                    max_overlap_fraction=max_overlap_fraction,
                )
                if window is None:
                    break
                windows.append(window)
            if len(windows) != len(series_items):
                continue

            result = common_duration_result(duration_s, windows)
            if best_result is None or score_key(result) > score_key(best_result):
                best_result = result

        if best_result is None:
            break
        results.append(best_result)
        for series_index, window in enumerate(best_result.windows):
            selected_by_series[series_index].append(window)

    if not results:
        raise RuntimeError("No independent window sets were found")
    return results


def format_optional_time(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def print_series_summary(series_items: list[PairedSeries]) -> None:
    print("paired_csv_summary")
    print(
        "csv,points,duration_s,time_column,absolute_time_column,left_column,right_column,"
        "duplicate_time_rows"
    )
    for series in series_items:
        print(
            f"{series.path.name},{len(series.times)},{series.duration_s:.6f},"
            f"{series.time_column},{series.absolute_time_column or ''},"
            f"{series.left_column},{series.right_column},{series.duplicate_time_rows}"
        )


def print_results(results: list[CommonDurationResult]) -> None:
    print()
    print("top_independent_common_windows")
    print("rank,duration_s,avg_pearson,min_pearson,max_pearson,csv_count")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank},{result.duration_s:.6f},{result.average_pearson:.10f},"
            f"{result.min_pearson:.10f},{result.max_pearson:.10f},{len(result.windows)}"
        )
        print(
            "csv,pearson,relative_start_s,relative_end_s,matched_start_s,"
            "matched_end_s,start_idx,end_idx_exclusive,points"
        )
        for window in result.windows:
            print(
                f"{window.series.path.name},{window.pearson:.10f},"
                f"{window.start_s:.6f},{window.end_s:.6f},"
                f"{format_optional_time(window.absolute_start_s)},"
                f"{format_optional_time(window.absolute_end_s)},"
                f"{window.start_index},{window.end_index_exclusive},{window.points}"
            )
        print()


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
        default=1.0,
        help="Step between tested window starts in seconds. Default: 1.",
    )
    parser.add_argument(
        "--min-points-per-window",
        type=int,
        default=3,
        help="Minimum valid paired points required inside a window. Default: 3.",
    )
    parser.add_argument(
        "--max-overlap-fraction",
        type=float,
        default=0.2,
        help=(
            "Maximum allowed overlap with earlier printed windows in the same CSV. "
            "Default: 0.2. Use 0 for strict non-overlap."
        ),
    )
    parser.add_argument("--top", type=int, default=5, help="Number of ranks to print. Default: 5.")
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
        if args.max_overlap_fraction < 0:
            raise RuntimeError("--max-overlap-fraction must be at least 0")
        results = select_top_common_windows(
            series_items,
            durations=durations,
            start_step_s=args.start_step_s,
            min_points=args.min_points_per_window,
            top=args.top,
            max_overlap_fraction=args.max_overlap_fraction,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print_series_summary(series_items)
    print(
        "\nscan_settings="
        f"min_duration_s:{args.min_duration_s:g},"
        f"max_duration_s:{float(durations[-1]):g},"
        f"duration_step_s:{args.duration_step_s:g},"
        f"start_step_s:{args.start_step_s:g},"
        f"max_overlap_fraction:{args.max_overlap_fraction:g},"
        f"durations_tested:{len(durations)}"
    )
    print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
