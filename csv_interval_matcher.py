from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from config import ROOT_PATH, bpi_processed_path, norm_data_path


ROW_INDEX_COLUMN_NAMES = {"row_index", "index", "__index__"}
DEFAULT_X_CANDIDATES = [
    "time_s",
    "x_norm",
    "x_value",
    "x_px",
    "Timestamp",
    "timestamp",
    "time",
]
DEFAULT_Y_CANDIDATES = [
    "bpi_normalized",
    "y_norm",
    "y_value",
    "y_px",
    "BPI",
    "bpi",
    "PPG",
    "ppg",
]
METRICS = [
    "fusion",
    "fusion_derivative",
    "pearson",
    "spearman",
    "rmse_z",
    "mae_z",
    "rmse_abs",
    "mae_abs",
    "bland_altman",
    "bland_altman_abs",
    "bias_abs",
    "derivative",
    "smooth_pearson",
    "smooth_derivative",
    "sign_agreement",
    "feature_corr",
]
EPSILON = 1e-12


@dataclass(frozen=True)
class CsvSeries:
    path: Path
    x_column: str
    y_column: str
    times: np.ndarray
    values: np.ndarray
    source_rows: int
    valid_rows: int
    duplicate_time_rows: int

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0])


@dataclass(frozen=True)
class WindowDiagnostics:
    pearson: float
    pearson_p: float
    spearman: float
    spearman_p: float
    rmse_z: float
    mae_z: float
    rmse_abs: float
    mae_abs: float
    bland_altman: float
    bland_altman_abs: float
    bias_abs: float
    derivative: float
    smooth_pearson: float
    smooth_pearson_p: float
    smooth_derivative: float
    sign_agreement: float
    feature_corr: float


@dataclass(frozen=True)
class MatchResult:
    rank: int
    short_name: str
    metric: str
    start_time: float
    end_time: float
    score: float
    diagnostics: WindowDiagnostics
    mean_time_error: float
    max_time_error: float
    points: int


@dataclass(frozen=True)
class ScoredSeries:
    short: CsvSeries
    grid_offsets: np.ndarray
    starts: np.ndarray
    scores_by_metric: dict[str, np.ndarray]
    diagnostics_by_start: list[WindowDiagnostics]
    mean_gaps: np.ndarray
    max_gaps: np.ndarray


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


def choose_column(fieldnames: list[str], requested: str | None, candidates: list[str]) -> str:
    if requested:
        if column_key(requested) in ROW_INDEX_COLUMN_NAMES:
            return "row_index"
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

    candidate_text = ", ".join(candidates)
    raise RuntimeError(f"Could not find a CSV column among: {candidate_text}")


def collapse_duplicate_times(
    times: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    if len(times) <= 1:
        return times, values, 0

    unique_times, inverse, counts = np.unique(times, return_inverse=True, return_counts=True)
    duplicate_rows = int(np.sum(counts - 1))
    if duplicate_rows == 0:
        return times, values, 0

    sums = np.bincount(inverse, weights=values)
    means = sums / counts
    return unique_times.astype(float), means.astype(float), duplicate_rows


def read_series(
    path: Path,
    *,
    x_column: str | None,
    y_column: str | None,
    sample_rate_hz: float | None,
) -> CsvSeries:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        fieldnames = list(reader.fieldnames)
        chosen_x = choose_column(fieldnames, x_column, DEFAULT_X_CANDIDATES)
        chosen_y = choose_column(fieldnames, y_column, DEFAULT_Y_CANDIDATES)

        times: list[float] = []
        values: list[float] = []
        source_rows = 0
        valid_rows = 0
        for row_index, row in enumerate(reader):
            source_rows += 1
            y = parse_float(row.get(chosen_y))
            if chosen_x == "row_index":
                if sample_rate_hz is None:
                    x = float(row_index)
                else:
                    if sample_rate_hz <= 0:
                        raise RuntimeError("Sample rate must be greater than 0")
                    x = float(row_index) / sample_rate_hz
            else:
                x = parse_float(row.get(chosen_x))

            if x is None or y is None:
                continue
            times.append(x)
            values.append(y)
            valid_rows += 1

    if valid_rows < 3:
        raise RuntimeError(f"Need at least 3 valid data points in {path}")

    time_array = np.asarray(times, dtype=float)
    value_array = np.asarray(values, dtype=float)
    order = np.argsort(time_array, kind="stable")
    time_array = time_array[order]
    value_array = value_array[order]
    time_array, value_array, duplicate_rows = collapse_duplicate_times(time_array, value_array)

    if len(time_array) < 3:
        raise RuntimeError(f"Need at least 3 unique time points in {path}")
    if time_array[-1] <= time_array[0]:
        raise RuntimeError(f"Time axis has no positive duration in {path}")

    return CsvSeries(
        path=path,
        x_column=chosen_x,
        y_column=chosen_y,
        times=time_array,
        values=value_array,
        source_rows=source_rows,
        valid_rows=valid_rows,
        duplicate_time_rows=duplicate_rows,
    )


def short_paths_from_args(args: argparse.Namespace) -> list[Path]:
    if args.short is not None:
        return [args.short]
    if args.short_dir is None:
        args.short_dir = norm_data_path(args.root)
    if not args.short_dir.is_dir():
        raise RuntimeError(f"--short-dir is not a directory: {args.short_dir}")
    paths = sorted(path for path in args.short_dir.glob(args.short_pattern) if path.is_file())
    if not paths:
        raise RuntimeError(f"No short CSV files found in {args.short_dir}")
    return paths


def long_path_from_args(args: argparse.Namespace) -> Path:
    if args.long is not None:
        return args.long

    long_dir = args.long_dir or bpi_processed_path(args.root)
    if not long_dir.is_dir():
        raise RuntimeError(f"--long-dir is not a directory: {long_dir}")

    paths = sorted(path for path in long_dir.glob(args.long_pattern) if path.is_file())
    if not paths:
        raise RuntimeError(f"No long CSV files found in {long_dir}")
    if len(paths) > 1:
        candidates = ", ".join(path.name for path in paths)
        raise RuntimeError(
            "Multiple long CSV files were found in "
            f"{long_dir}: {candidates}. Use --long to choose one explicitly."
        )
    return paths[0]


def zscore(values: np.ndarray) -> np.ndarray | None:
    std = float(np.std(values))
    if std <= EPSILON:
        return None
    return (values - float(np.mean(values))) / std


def finite_pairs(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(left) & np.isfinite(right)
    return left[mask].astype(float), right[mask].astype(float)


def pearson_test(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left_values, right_values = finite_pairs(left, right)
    if len(left_values) < 2:
        return math.nan, math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
        result = stats.pearsonr(left_values, right_values)
    return float(result.statistic), float(result.pvalue)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    return pearson_test(left, right)[0]


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0
        index = end
    return ranks


def spearman_test(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left_values, right_values = finite_pairs(left, right)
    if len(left_values) < 2:
        return math.nan, math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
        result = stats.spearmanr(left_values, right_values)
    return float(result.statistic), float(result.pvalue)


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    return spearman_test(left, right)[0]


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    window = min(window, len(values) // 2 * 2 + 1)
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def z_rmse(left: np.ndarray, right: np.ndarray) -> float:
    left_z = zscore(left)
    right_z = zscore(right)
    if left_z is None or right_z is None:
        return math.nan
    return float(np.sqrt(np.mean((left_z - right_z) ** 2)))


def z_mae(left: np.ndarray, right: np.ndarray) -> float:
    left_z = zscore(left)
    right_z = zscore(right)
    if left_z is None or right_z is None:
        return math.nan
    return float(np.mean(np.abs(left_z - right_z)))


def raw_rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def raw_mae(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left - right)))


def bland_altman_width(left: np.ndarray, right: np.ndarray) -> float:
    left_z = zscore(left)
    right_z = zscore(right)
    if left_z is None or right_z is None:
        return math.nan
    differences = left_z - right_z
    return float(abs(np.mean(differences)) + 1.96 * np.std(differences))


def bland_altman_raw_width(left: np.ndarray, right: np.ndarray) -> float:
    differences = left - right
    return float(abs(np.mean(differences)) + 1.96 * np.std(differences))


def absolute_bias(left: np.ndarray, right: np.ndarray) -> float:
    return float(abs(np.mean(left - right)))


def sign_agreement(left: np.ndarray, right: np.ndarray) -> float:
    left_sign = np.sign(np.diff(left))
    right_sign = np.sign(np.diff(right))
    valid = (left_sign != 0) & (right_sign != 0)
    if not np.any(valid):
        return math.nan
    return float(np.mean(left_sign[valid] == right_sign[valid]) * 2.0 - 1.0)


def morphology_features(values: np.ndarray) -> np.ndarray:
    values_z = zscore(values)
    if values_z is None:
        values_z = np.zeros_like(values)
    thirds = np.array_split(values_z, 3)
    return np.concatenate(
        [
            np.quantile(values_z, [0.05, 0.25, 0.5, 0.75, 0.95]),
            np.asarray([part.mean() for part in thirds]),
            np.asarray([part.std() for part in thirds]),
        ]
    )


def nearest_sample(
    long_times: np.ndarray,
    long_values: np.ndarray,
    target_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(long_times, target_times, side="left")
    right = np.clip(right, 0, len(long_times) - 1)
    left = np.clip(right - 1, 0, len(long_times) - 1)

    left_gap = np.abs(target_times - long_times[left])
    right_gap = np.abs(long_times[right] - target_times)
    use_right = right_gap < left_gap
    indices = np.where(use_right, right, left)
    gaps = np.abs(long_times[indices] - target_times)
    return long_values[indices], gaps


def linear_sample(
    long_times: np.ndarray,
    long_values: np.ndarray,
    target_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sampled_values = np.interp(target_times, long_times, long_values)
    _, gaps = nearest_sample(long_times, long_values, target_times)
    return sampled_values, gaps


def candidate_start_times(
    long_times: np.ndarray,
    short_duration: float,
    *,
    start_step_rows: int,
    start_step_s: float | None,
) -> np.ndarray:
    latest_start = long_times[-1] - short_duration
    if latest_start < long_times[0]:
        raise RuntimeError("Short CSV duration is longer than the long CSV duration")

    if start_step_s is not None:
        if start_step_s <= 0:
            raise RuntimeError("--start-step-s must be greater than 0")
        count = int(math.floor((latest_start - long_times[0]) / start_step_s)) + 1
        starts = long_times[0] + np.arange(count, dtype=float) * start_step_s
        return starts[starts <= latest_start + EPSILON]

    if start_step_rows <= 0:
        raise RuntimeError("--start-step-rows must be greater than 0")
    valid = long_times[long_times <= latest_start + EPSILON]
    return valid[::start_step_rows]


def diagnostics_for_window(
    short_values: np.ndarray,
    long_values: np.ndarray,
    grid_offsets: np.ndarray,
    smooth_window: int,
) -> WindowDiagnostics:
    short_smooth = smooth(short_values, smooth_window)
    long_smooth = smooth(long_values, smooth_window)
    short_derivative = np.gradient(short_values, grid_offsets)
    long_derivative = np.gradient(long_values, grid_offsets)
    short_smooth_derivative = np.gradient(short_smooth, grid_offsets)
    long_smooth_derivative = np.gradient(long_smooth, grid_offsets)
    pearson_r, pearson_p = pearson_test(short_values, long_values)
    spearman_r, spearman_p = spearman_test(short_values, long_values)
    smooth_pearson_r, smooth_pearson_p = pearson_test(short_smooth, long_smooth)

    return WindowDiagnostics(
        pearson=pearson_r,
        pearson_p=pearson_p,
        spearman=spearman_r,
        spearman_p=spearman_p,
        rmse_z=z_rmse(short_values, long_values),
        mae_z=z_mae(short_values, long_values),
        rmse_abs=raw_rmse(short_values, long_values),
        mae_abs=raw_mae(short_values, long_values),
        bland_altman=bland_altman_width(short_values, long_values),
        bland_altman_abs=bland_altman_raw_width(short_values, long_values),
        bias_abs=absolute_bias(short_values, long_values),
        derivative=correlation(short_derivative, long_derivative),
        smooth_pearson=smooth_pearson_r,
        smooth_pearson_p=smooth_pearson_p,
        smooth_derivative=correlation(short_smooth_derivative, long_smooth_derivative),
        sign_agreement=sign_agreement(short_smooth, long_smooth),
        feature_corr=correlation(morphology_features(short_smooth), morphology_features(long_smooth)),
    )


def safe(value: float, fallback: float = 0.0) -> float:
    return value if math.isfinite(value) else fallback


def agreement_from_error(error: float) -> float:
    if not math.isfinite(error):
        return 0.0
    return max(0.0, 1.0 - error)


def score_from_diagnostics(metric: str, diagnostics: WindowDiagnostics) -> float:
    if metric == "pearson":
        return diagnostics.pearson
    if metric == "spearman":
        return diagnostics.spearman
    if metric == "rmse_z":
        return -diagnostics.rmse_z
    if metric == "mae_z":
        return -diagnostics.mae_z
    if metric == "rmse_abs":
        return -diagnostics.rmse_abs
    if metric == "mae_abs":
        return -diagnostics.mae_abs
    if metric == "bland_altman":
        return -diagnostics.bland_altman
    if metric == "bland_altman_abs":
        return -diagnostics.bland_altman_abs
    if metric == "bias_abs":
        return -diagnostics.bias_abs
    if metric == "derivative":
        return diagnostics.derivative
    if metric == "smooth_pearson":
        return diagnostics.smooth_pearson
    if metric == "smooth_derivative":
        return diagnostics.smooth_derivative
    if metric == "sign_agreement":
        return diagnostics.sign_agreement
    if metric == "feature_corr":
        return diagnostics.feature_corr
    if metric == "fusion_derivative":
        return (
            0.388889 * safe(diagnostics.smooth_pearson)
            + 0.277778 * safe(diagnostics.smooth_derivative)
            + 0.222222 * safe(diagnostics.spearman)
            + 0.111111 * safe(diagnostics.sign_agreement)
        )
    if metric == "fusion":
        return (
            0.222222 * safe(diagnostics.smooth_pearson)
            + 0.333333 * safe(diagnostics.pearson)
            + 0.333333 * safe(diagnostics.smooth_derivative)
            + 0.111112 * safe(diagnostics.feature_corr)
        )
    raise RuntimeError(f"Unsupported metric: {metric}")


def scored_series_for_short(
    short: CsvSeries,
    long: CsvSeries,
    args: argparse.Namespace,
) -> ScoredSeries:
    if args.resample_points < 3:
        raise RuntimeError("--resample-points must be at least 3")

    short_offsets = short.times - short.times[0]
    grid_offsets = np.linspace(0.0, short_offsets[-1], args.resample_points)
    short_values = np.interp(grid_offsets, short_offsets, short.values)

    starts = candidate_start_times(
        long.times,
        grid_offsets[-1],
        start_step_rows=args.start_step_rows,
        start_step_s=args.start_step_s,
    )
    if len(starts) == 0:
        raise RuntimeError("No valid candidate start times were generated")

    diagnostics_by_start: list[WindowDiagnostics] = []
    mean_gaps: list[float] = []
    max_gaps: list[float] = []
    scores_by_metric = {metric: [] for metric in METRICS}

    for start_time in starts:
        target_times = float(start_time) + grid_offsets
        if args.sample_method == "linear":
            sampled_values, gaps = linear_sample(long.times, long.values, target_times)
        else:
            sampled_values, gaps = nearest_sample(long.times, long.values, target_times)

        if args.max_nearest_gap_s is not None and np.any(gaps > args.max_nearest_gap_s):
            diagnostics = WindowDiagnostics(*(math.nan for _ in range(17)))
        else:
            diagnostics = diagnostics_for_window(
                short_values,
                sampled_values,
                grid_offsets,
                args.smooth_window_points,
            )

        diagnostics_by_start.append(diagnostics)
        mean_gaps.append(float(np.mean(gaps)))
        max_gaps.append(float(np.max(gaps)))
        for metric in METRICS:
            scores_by_metric[metric].append(score_from_diagnostics(metric, diagnostics))

    return ScoredSeries(
        short=short,
        grid_offsets=grid_offsets,
        starts=starts,
        scores_by_metric={
            metric: np.asarray(scores, dtype=float)
            for metric, scores in scores_by_metric.items()
        },
        diagnostics_by_start=diagnostics_by_start,
        mean_gaps=np.asarray(mean_gaps, dtype=float),
        max_gaps=np.asarray(max_gaps, dtype=float),
    )


def result_from_index(
    scored: ScoredSeries,
    index: int,
    *,
    rank: int,
    metric: str,
) -> MatchResult:
    return MatchResult(
        rank=rank,
        short_name=scored.short.path.name,
        metric=metric,
        start_time=float(scored.starts[index]),
        end_time=float(scored.starts[index] + scored.grid_offsets[-1]),
        score=float(scored.scores_by_metric[metric][index]),
        diagnostics=scored.diagnostics_by_start[index],
        mean_time_error=float(scored.mean_gaps[index]),
        max_time_error=float(scored.max_gaps[index]),
        points=len(scored.grid_offsets),
    )


def select_top_results(
    scored: ScoredSeries,
    *,
    metric: str,
    top_count: int,
    min_start_separation_s: float,
) -> list[MatchResult]:
    if top_count <= 0:
        raise RuntimeError("--top must be greater than 0")
    if min_start_separation_s < 0:
        raise RuntimeError("--min-start-separation-s cannot be negative")

    scores = scored.scores_by_metric[metric]
    ranked_indices = np.argsort(scores)[::-1]
    selected: list[int] = []
    for index in ranked_indices:
        score = float(scores[index])
        if not math.isfinite(score):
            continue
        start = float(scored.starts[index])
        if all(abs(start - float(scored.starts[existing])) >= min_start_separation_s for existing in selected):
            selected.append(int(index))
            if len(selected) >= top_count:
                break

    return [
        result_from_index(scored, index, rank=rank, metric=metric)
        for rank, index in enumerate(selected, start=1)
    ]


def prefix_best_indices(starts: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    best_scores = np.empty(len(starts), dtype=float)
    best_indices = np.empty(len(starts), dtype=int)
    best_score = -math.inf
    best_index = -1
    for index, score in enumerate(scores):
        score = float(score)
        if math.isfinite(score) and score > best_score:
            best_score = score
            best_index = index
        best_scores[index] = best_score
        best_indices[index] = best_index
    return best_scores, best_indices


def select_ordered_non_overlapping(
    scored_items: list[ScoredSeries],
    *,
    metric: str,
    gap_s: float,
    overlap_s: float,
    three_short_skip_after_first_s: float,
) -> list[MatchResult]:
    if len(scored_items) == 1:
        return select_top_results(
            scored_items[0],
            metric=metric,
            top_count=1,
            min_start_separation_s=0.0,
        )
    if gap_s < 0:
        raise RuntimeError("--ordered-gap-s cannot be negative")
    if overlap_s < 0:
        raise RuntimeError("--ordered-overlap-s cannot be negative")
    if three_short_skip_after_first_s < 0:
        raise RuntimeError("--three-short-skip-after-first-s cannot be negative")

    previous = scored_items[0]
    previous_scores = previous.scores_by_metric[metric]
    previous_paths = [[index] for index in range(len(previous.starts))]

    for item_index in range(1, len(scored_items)):
        current = scored_items[item_index]
        previous_duration = scored_items[item_index - 1].grid_offsets[-1]
        prefix_scores, prefix_indices = prefix_best_indices(previous.starts, previous_scores)

        current_scores = np.full(len(current.starts), -math.inf, dtype=float)
        current_paths: list[list[int]] = [[] for _ in range(len(current.starts))]
        for index, start_time in enumerate(current.starts):
            if len(scored_items) == 3 and item_index == 1:
                transition_gap_s = max(gap_s, three_short_skip_after_first_s)
                previous_limit = float(start_time) - previous_duration - transition_gap_s
            else:
                previous_limit = float(start_time) - previous_duration - gap_s + overlap_s
            previous_index = int(np.searchsorted(previous.starts, previous_limit, side="right") - 1)
            own_score = float(current.scores_by_metric[metric][index])
            if previous_index < 0 or not math.isfinite(own_score):
                continue
            best_previous_index = int(prefix_indices[previous_index])
            if best_previous_index < 0:
                continue
            current_scores[index] = prefix_scores[previous_index] + own_score
            current_paths[index] = previous_paths[best_previous_index] + [index]

        previous = current
        previous_scores = current_scores
        previous_paths = current_paths

    final_index = int(np.argmax(previous_scores))
    if final_index < 0 or not math.isfinite(float(previous_scores[final_index])):
        raise RuntimeError("No ordered non-overlapping match could be selected")

    indices = previous_paths[final_index]
    return [
        result_from_index(item, index, rank=rank, metric=metric)
        for rank, (item, index) in enumerate(zip(scored_items, indices), start=1)
    ]


def metrics_from_args(args: argparse.Namespace) -> list[str]:
    if args.metric == "all":
        return list(METRICS)
    return [args.metric]


def read_inputs(args: argparse.Namespace) -> tuple[list[CsvSeries], CsvSeries]:
    short_paths = short_paths_from_args(args)
    long_path = long_path_from_args(args)
    short_items = [
        read_series(
            path,
            x_column=args.short_x_column,
            y_column=args.short_y_column,
            sample_rate_hz=args.short_sample_rate,
        )
        for path in short_paths
    ]
    long = read_series(
        long_path,
        x_column=args.long_x_column,
        y_column=args.long_y_column,
        sample_rate_hz=args.long_sample_rate,
    )
    return short_items, long


def format_series_summary(series: CsvSeries, label: str) -> str:
    text = (
        f"{label}: {series.path} | x={series.x_column} y={series.y_column} | "
        f"{len(series.times)}/{series.source_rows} points | duration={series.duration:.8g}"
    )
    if series.duplicate_time_rows:
        text += f" | collapsed duplicate-time rows={series.duplicate_time_rows}"
    return text


def format_results(results: list[MatchResult], *, compact: bool = False) -> str:
    if compact:
        headers = ["rank", "short_csv", "metric", "start_time", "end_time", "score"]
        rows = [
            [
                str(result.rank),
                result.short_name,
                result.metric,
                f"{result.start_time:.10g}",
                f"{result.end_time:.10g}",
                f"{result.score:.6f}",
            ]
            for result in results
        ]
    else:
        headers = [
            "rank",
            "short_csv",
            "metric",
            "start_time",
            "end_time",
            "score",
            "pearson",
            "pearson_p",
            "spearman",
            "spearman_p",
            "rmse_z",
            "mae_abs",
            "rmse_abs",
            "ba_width",
            "ba_abs",
            "smooth_r",
            "smooth_r_p",
            "smooth_dr",
            "mean_gap",
            "max_gap",
        ]
        rows = [
            [
                str(result.rank),
                result.short_name,
                result.metric,
                f"{result.start_time:.10g}",
                f"{result.end_time:.10g}",
                f"{result.score:.6f}",
                f"{result.diagnostics.pearson:.6f}",
                f"{result.diagnostics.pearson_p:.3g}",
                f"{result.diagnostics.spearman:.6f}",
                f"{result.diagnostics.spearman_p:.3g}",
                f"{result.diagnostics.rmse_z:.6f}",
                f"{result.diagnostics.mae_abs:.6f}",
                f"{result.diagnostics.rmse_abs:.6f}",
                f"{result.diagnostics.bland_altman:.6f}",
                f"{result.diagnostics.bland_altman_abs:.6f}",
                f"{result.diagnostics.smooth_pearson:.6f}",
                f"{result.diagnostics.smooth_pearson_p:.3g}",
                f"{result.diagnostics.smooth_derivative:.6f}",
                f"{result.mean_time_error:.6g}",
                f"{result.max_time_error:.6g}",
            ]
            for result in results
        ]

    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]
    output = ["  ".join(headers[index].ljust(widths[index]) for index in range(len(headers)))]
    output.append("  ".join("-" * width for width in widths))
    for row in rows:
        output.append("  ".join(row[index].ljust(widths[index]) for index in range(len(row))))
    return "\n".join(output)


def write_results(path: Path, results: list[MatchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "rank",
            "short_csv",
            "metric",
            "start_time",
            "end_time",
            "score",
            "pearson",
            "pearson_p",
            "spearman",
            "spearman_p",
            "rmse_z",
            "mae_z",
            "rmse_abs",
            "mae_abs",
            "bland_altman",
            "bland_altman_abs",
            "bias_abs",
            "derivative",
            "smooth_pearson",
            "smooth_pearson_p",
            "smooth_derivative",
            "sign_agreement",
            "feature_corr",
            "mean_time_error",
            "max_time_error",
            "points",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            diagnostics = result.diagnostics
            writer.writerow(
                {
                    "rank": result.rank,
                    "short_csv": result.short_name,
                    "metric": result.metric,
                    "start_time": f"{result.start_time:.10g}",
                    "end_time": f"{result.end_time:.10g}",
                    "score": f"{result.score:.10g}",
                    "pearson": f"{diagnostics.pearson:.10g}",
                    "pearson_p": f"{diagnostics.pearson_p:.10g}",
                    "spearman": f"{diagnostics.spearman:.10g}",
                    "spearman_p": f"{diagnostics.spearman_p:.10g}",
                    "rmse_z": f"{diagnostics.rmse_z:.10g}",
                    "mae_z": f"{diagnostics.mae_z:.10g}",
                    "rmse_abs": f"{diagnostics.rmse_abs:.10g}",
                    "mae_abs": f"{diagnostics.mae_abs:.10g}",
                    "bland_altman": f"{diagnostics.bland_altman:.10g}",
                    "bland_altman_abs": f"{diagnostics.bland_altman_abs:.10g}",
                    "bias_abs": f"{diagnostics.bias_abs:.10g}",
                    "derivative": f"{diagnostics.derivative:.10g}",
                    "smooth_pearson": f"{diagnostics.smooth_pearson:.10g}",
                    "smooth_pearson_p": f"{diagnostics.smooth_pearson_p:.10g}",
                    "smooth_derivative": f"{diagnostics.smooth_derivative:.10g}",
                    "sign_agreement": f"{diagnostics.sign_agreement:.10g}",
                    "feature_corr": f"{diagnostics.feature_corr:.10g}",
                    "mean_time_error": f"{result.mean_time_error:.10g}",
                    "max_time_error": f"{result.max_time_error:.10g}",
                    "points": result.points,
                }
            )


def paired_values_for_match(
    scored: ScoredSeries,
    result: MatchResult,
    long: CsvSeries,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    short_offsets = scored.short.times - scored.short.times[0]
    aligned_times = result.start_time + scored.grid_offsets
    short_values = np.interp(scored.grid_offsets, short_offsets, scored.short.values)
    if args.sample_method == "linear":
        long_values, _ = linear_sample(long.times, long.values, aligned_times)
    else:
        long_values, _ = nearest_sample(long.times, long.values, aligned_times)
    return aligned_times, short_values, long_values


def paired_points_csv_filename(result: MatchResult) -> str:
    short_stem = safe_filename_component(Path(result.short_name).stem)
    metric = safe_filename_component(result.metric)
    return f"matched_paired_points_phase{result.rank}_{short_stem}_{metric}.csv"


def write_paired_match_csvs(
    plot_requests: list[tuple[ScoredSeries, MatchResult]],
    long: CsvSeries,
    args: argparse.Namespace,
) -> list[Path]:
    if not plot_requests:
        return []

    output_dir = args.root
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    fieldnames = [
        "phase",
        "short_csv",
        "metric",
        "match_start_time",
        "match_end_time",
        "point_index",
        "matched_time_s",
        "short_offset_s",
        "short_value",
        "long_value",
    ]

    for scored, result in plot_requests:
        aligned_times, short_values, long_values = paired_values_for_match(
            scored,
            result,
            long,
            args,
        )
        path = output_dir / paired_points_csv_filename(result)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for index, (time_s, short_value, long_value) in enumerate(
                zip(aligned_times, short_values, long_values)
            ):
                writer.writerow(
                    {
                        "phase": result.rank,
                        "short_csv": result.short_name,
                        "metric": result.metric,
                        "match_start_time": f"{result.start_time:.10g}",
                        "match_end_time": f"{result.end_time:.10g}",
                        "point_index": index,
                        "matched_time_s": f"{float(time_s):.10g}",
                        "short_offset_s": f"{float(time_s - result.start_time):.10g}",
                        "short_value": f"{float(short_value):.10g}",
                        "long_value": f"{float(long_value):.10g}",
                    }
                )
        written_paths.append(path)
    return written_paths


def safe_filename_component(text: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    cleaned = "".join(character if character in allowed else "_" for character in text)
    return cleaned.strip("._") or "match"


def overlay_plot_filename(metric: str) -> str:
    return f"matched_overlay_{safe_filename_component(metric)}.png"


def paired_points_plot_filename(metric: str) -> str:
    return f"matched_paired_points_{safe_filename_component(metric)}.png"


def match_overlay_plot_path(
    plot_output: Path | None,
    *,
    metric: str,
    total_groups: int,
) -> Path | None:
    if plot_output is None:
        return None
    if plot_output.suffix:
        if total_groups > 1:
            raise RuntimeError(
                "--plot-output must be a directory when plots are generated for multiple metrics"
            )
        return plot_output
    return plot_output / overlay_plot_filename(metric)


def paired_points_plot_path(
    plot_output: Path | None,
    *,
    metric: str,
    total_groups: int,
) -> Path | None:
    if plot_output is None:
        return None
    if plot_output.suffix:
        if total_groups > 1:
            raise RuntimeError(
                "--plot-output must be a directory when plots are generated for multiple metrics"
            )
        return plot_output.with_name(
            f"{plot_output.stem}_paired_points{plot_output.suffix}"
        )
    return plot_output / paired_points_plot_filename(metric)


def equivalent_smooth_window_points(
    series_times: np.ndarray,
    grid_offsets: np.ndarray,
    smooth_window_points: int,
) -> int:
    if smooth_window_points <= 1 or len(series_times) < 2 or len(grid_offsets) < 2:
        return 1

    grid_deltas = np.diff(grid_offsets)
    grid_deltas = grid_deltas[grid_deltas > EPSILON]
    series_deltas = np.diff(series_times)
    series_deltas = series_deltas[series_deltas > EPSILON]
    if len(grid_deltas) == 0 or len(series_deltas) == 0:
        return 1

    smooth_window_s = float(np.median(grid_deltas)) * smooth_window_points
    series_step_s = float(np.median(series_deltas))
    return max(1, int(round(smooth_window_s / series_step_s)))


def write_match_plots(
    plot_requests: list[tuple[ScoredSeries, MatchResult]],
    long: CsvSeries,
    args: argparse.Namespace,
) -> list[Path]:
    if not plot_requests:
        return []
    if args.plot_output is None and args.no_plot:
        return []

    from plotting import (
        MatchOverlaySeries,
        write_dual_axis_match_overlay_plot,
        write_paired_match_points_plot,
    )

    grouped_requests: dict[str, list[tuple[ScoredSeries, MatchResult]]] = {}
    for scored, result in plot_requests:
        grouped_requests.setdefault(result.metric, []).append((scored, result))

    written_paths: list[Path] = []
    total_groups = len(grouped_requests)
    show_plot = not args.no_plot or args.show_plot
    for metric, requests in grouped_requests.items():
        plot_path = match_overlay_plot_path(
            args.plot_output,
            metric=metric,
            total_groups=total_groups,
        )
        paired_plot_path = paired_points_plot_path(
            args.plot_output,
            metric=metric,
            total_groups=total_groups,
        )
        plot_mode = "smooth_r" if getattr(args, "plot_smooth_r", False) else "raw"
        plot_long_values = long.values
        if getattr(args, "plot_smooth_r", False):
            long_smooth_window = equivalent_smooth_window_points(
                long.times,
                requests[0][0].grid_offsets,
                args.smooth_window_points,
            )
            plot_long_values = smooth(long.values, long_smooth_window)

        short_matches: list[MatchOverlaySeries] = []
        paired_matches: list[MatchOverlaySeries] = []
        for scored, result in requests:
            aligned_times, raw_short_values, raw_matched_long_values = (
                paired_values_for_match(scored, result, long, args)
            )
            short_values = raw_short_values
            overlay_matched_long_times = None
            overlay_matched_long_values = None
            if getattr(args, "plot_smooth_r", False):
                short_values = smooth(short_values, args.smooth_window_points)
                overlay_matched_long_times = aligned_times
                overlay_matched_long_values = smooth(
                    raw_matched_long_values,
                    args.smooth_window_points,
                )
            short_matches.append(
                MatchOverlaySeries(
                    name=scored.short.path.name,
                    aligned_times=aligned_times,
                    values=short_values,
                    y_label=scored.short.y_column,
                    start_time=result.start_time,
                    end_time=result.end_time,
                    rank=result.rank,
                    score=result.score,
                    matched_long_times=overlay_matched_long_times,
                    matched_long_values=overlay_matched_long_values,
                )
            )
            paired_matches.append(
                MatchOverlaySeries(
                    name=scored.short.path.name,
                    aligned_times=aligned_times,
                    values=raw_short_values,
                    y_label=scored.short.y_column,
                    start_time=result.start_time,
                    end_time=result.end_time,
                    rank=result.rank,
                    score=result.score,
                    matched_long_times=aligned_times,
                    matched_long_values=raw_matched_long_values,
                )
            )

        write_dual_axis_match_overlay_plot(
            plot_path,
            long_times=long.times,
            long_values=plot_long_values,
            short_matches=short_matches,
            long_name=long.path.name,
            long_x_label=long.x_column,
            long_y_label=long.y_column,
            metric=metric,
            plot_mode=plot_mode,
            show=show_plot,
        )
        if plot_path is not None:
            written_paths.append(plot_path)
        write_paired_match_points_plot(
            paired_plot_path,
            matches=paired_matches,
            metric=metric,
            sample_method=args.sample_method,
            show=show_plot,
        )
        if paired_plot_path is not None:
            written_paths.append(paired_plot_path)
    return written_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find the long-CSV interval whose waveform best matches shorter CSV data.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT_PATH,
        help=(
            "Experiment root folder. Used for omitted --short/--short-dir and "
            f"--long/--long-dir. Default: {ROOT_PATH}"
        ),
    )
    short_input = parser.add_mutually_exclusive_group()
    short_input.add_argument("--short", type=Path, help="Short CSV path.")
    short_input.add_argument(
        "--short-dir",
        type=Path,
        help="Folder of short CSV files, processed in sorted order. Default: ROOT/Norm_Data.",
    )
    parser.add_argument(
        "--short-pattern",
        default="*.csv",
        help="Filename pattern for --short-dir. Default: *.csv.",
    )
    long_input = parser.add_mutually_exclusive_group()
    long_input.add_argument("--long", type=Path, help="Long CSV path.")
    long_input.add_argument(
        "--long-dir",
        type=Path,
        help="Folder containing exactly one long CSV. Default: ROOT/BPI_Processed.",
    )
    parser.add_argument(
        "--long-pattern",
        default="*.csv",
        help="Filename pattern for --long-dir. Default: *.csv.",
    )
    parser.add_argument("--short-x-column", help="Short CSV time/x column. Use row_index to synthesize time from rows.")
    parser.add_argument("--short-y-column", help="Short CSV signal/y column.")
    parser.add_argument("--long-x-column", help="Long CSV time/x column. Use row_index to synthesize time from rows.")
    parser.add_argument("--long-y-column", help="Long CSV signal/y column.")
    parser.add_argument("--short-sample-rate", type=float, help="Hz used when --short-x-column row_index is selected.")
    parser.add_argument("--long-sample-rate", type=float, help="Hz used when --long-x-column row_index is selected.")
    parser.add_argument(
        "--metric",
        choices=["all", *METRICS],
        default="fusion",
        help="Scoring metric. Use all to compare every supported metric. Default: fusion.",
    )
    parser.add_argument(
        "--sample-method",
        choices=["nearest", "linear"],
        default="nearest",
        help="How to sample long CSV at shifted short time points. Default: nearest.",
    )
    parser.add_argument(
        "--resample-points",
        type=int,
        default=300,
        help="Resample each candidate window to this many points before scoring. Default: 300.",
    )
    parser.add_argument(
        "--smooth-window-points",
        type=int,
        default=15,
        help="Moving-average window, in resampled points, for smooth metrics. Default: 15.",
    )
    parser.add_argument("--top", type=int, default=5, help="Top starts to print for single-short mode. Default: 5.")
    parser.add_argument("--start-step-rows", type=int, default=1, help="Slide by this many long-CSV rows. Default: 1.")
    parser.add_argument("--start-step-s", type=float, help="Fixed start-time step; overrides --start-step-rows.")
    parser.add_argument(
        "--min-start-separation-s",
        type=float,
        default=0.0,
        help="Minimum separation between reported starts in single-short mode. Default: 0.",
    )
    parser.add_argument(
        "--independent",
        action="store_true",
        help="For --short-dir, print independent top matches instead of one ordered assignment.",
    )
    parser.add_argument(
        "--ordered-gap-s",
        type=float,
        default=0.0,
        help="Extra required gap between adjacent ordered windows. Default: 0.",
    )
    parser.add_argument(
        "--ordered-overlap-s",
        type=float,
        default=5.0,
        help=(
            "Allowed overlap between adjacent ordered windows, in seconds. "
            "Default: 5."
        ),
    )
    parser.add_argument(
        "--three-short-skip-after-first-s",
        type=float,
        default=30.0,
        help=(
            "When ordered matching has exactly three short CSVs, require the second "
            "match to start at least this many seconds after the first match ends. "
            "Use 0 to disable. Default: 30."
        ),
    )
    parser.add_argument("--max-nearest-gap-s", type=float, help="Reject windows with a larger nearest-time gap.")
    parser.add_argument("--output", type=Path, help="Optional CSV path for the match table.")
    parser.add_argument(
        "--plot-output",
        type=Path,
        help=(
            "PNG path for one matched dual-axis overlay, or directory when plotting multiple metrics. "
            "Uses the full long time axis and aligns every selected short curve onto it."
        ),
    )
    parser.add_argument(
        "--plot-smooth-r",
        action="store_true",
        help=(
            "Plot the smoothed short and matched-long window values used by "
            "smooth_r/smooth_pearson instead of raw resampled values."
        ),
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Show matched dual-axis Matplotlib windows. This is the default unless --no-plot is used.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip showing matched dual-axis Matplotlib windows. --plot-output can still save PNG files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        short_items, long = read_inputs(args)
        scored_items = [
            scored_series_for_short(short, long, args)
            for short in short_items
        ]
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_series_summary(long, "long"))
    for short in short_items:
        print(format_series_summary(short, "short"))
    print(
        f"sample={args.sample_method} | resample_points={args.resample_points} | "
        f"smooth_window_points={args.smooth_window_points}"
    )

    metrics = metrics_from_args(args)
    all_results: list[MatchResult] = []
    plot_requests: list[tuple[ScoredSeries, MatchResult]] = []
    if len(scored_items) == 1 or args.independent:
        for scored in scored_items:
            for metric in metrics:
                results = select_top_results(
                    scored,
                    metric=metric,
                    top_count=args.top,
                    min_start_separation_s=args.min_start_separation_s,
                )
                all_results.extend(results)
                plot_requests.extend((scored, result) for result in results)
                print()
                print(format_results(results, compact=args.metric == "all"))
    else:
        for metric in metrics:
            results = select_ordered_non_overlapping(
                scored_items,
                metric=metric,
                gap_s=args.ordered_gap_s,
                overlap_s=args.ordered_overlap_s,
                three_short_skip_after_first_s=args.three_short_skip_after_first_s,
            )
            all_results.extend(results)
            plot_requests.extend(zip(scored_items, results))
            print()
            constraint_text = (
                f"ordered selection | metric={metric} | "
                f"overlap_s={args.ordered_overlap_s:g}"
            )
            if len(scored_items) == 3 and args.three_short_skip_after_first_s > 0:
                constraint_text += (
                    " | "
                    f"three_short_skip_after_first_s={args.three_short_skip_after_first_s:g}"
                )
            print(constraint_text)
            print(format_results(results, compact=args.metric == "all"))

    if args.output is not None:
        write_results(args.output, all_results)
        print(f"\nwrote {args.output}")

    paired_csv_paths = write_paired_match_csvs(plot_requests, long, args)
    if paired_csv_paths:
        print(f"\nwrote {len(paired_csv_paths)} paired-point CSV files to {args.root}")

    plot_paths = write_match_plots(plot_requests, long, args)
    if plot_paths:
        if len(plot_paths) == 1:
            print(f"\nwrote match plot {plot_paths[0]}")
        else:
            print(f"\nwrote {len(plot_paths)} match plots")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
