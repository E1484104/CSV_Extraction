#!/usr/bin/env python3
"""Compare time-matched relative peak heights, FWHM, and beat timing.

Each CSV file, or each worksheet in an Excel workbook, is treated as one
dataset when it contains both ``short_value`` and ``long_value`` columns.
Peaks are detected independently in the short and long signals, then paired
one-to-one by time. Relative peak height is the SciPy peak prominence, i.e.
the peak height above its local baseline.
The script is standalone: it does not import anything from this project.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.signal import find_peaks, peak_widths, savgol_filter
except ImportError as exc:  # pragma: no cover - only used for a friendly startup error
    missing = getattr(exc, "name", "a required package")
    raise SystemExit(
        f"Missing dependency: {missing}\n"
        "Install the required packages with:\n"
        "  pip install numpy pandas scipy matplotlib openpyxl"
    ) from exc


SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls", ".xlsm"}
SHORT_ALIASES = ("short_value", "shortvalue", "short")
LONG_ALIASES = ("long_value", "longvalue", "long")
TIME_ALIASES = (
    "short_offset_s",
    "matched_time_s",
    "time_s",
    "time",
    "timestamp",
    "point_index",
)


@dataclass
class Dataset:
    label: str
    source: str
    time: np.ndarray
    short: np.ndarray
    long: np.ndarray


@dataclass
class PairedFeatures:
    relative_peak_height_long: np.ndarray
    relative_peak_height_short: np.ndarray
    fwhm_long: np.ndarray
    fwhm_short: np.ndarray
    peak_delay_s: np.ndarray
    ibi_long_s: np.ndarray
    ibi_short_s: np.ndarray
    matched_count: int
    short_peak_count: int
    long_peak_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create relative-peak-height, FWHM, and beat-timing comparisons "
            "for independently detected, time-matched short_value/long_value peaks."
        )
    )
    parser.add_argument(
        "input_folder",
        nargs="?",
        type=Path,
        help="Folder containing three CSV files and/or Excel worksheets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory used when saving (default: input folder).",
    )
    parser.add_argument(
        "--expected-datasets",
        type=int,
        default=3,
        help="Required number of valid CSV files/worksheets; use 0 for any number (default: 3).",
    )
    parser.add_argument(
        "--smooth-window-s",
        type=float,
        default=0.15,
        help="Savitzky-Golay smoothing window in seconds (default: 0.15).",
    )
    parser.add_argument(
        "--min-peak-distance-s",
        type=float,
        default=0.35,
        help="Minimum distance between detected peaks in seconds (default: 0.35).",
    )
    parser.add_argument(
        "--prominence-ratio",
        type=float,
        default=0.08,
        help="Minimum peak prominence as a fraction of the 5th-95th percentile range (default: 0.08).",
    )
    parser.add_argument(
        "--max-pair-delay-s",
        type=float,
        default=0.30,
        help="Maximum time separation when pairing short and long peaks (default: 0.30).",
    )
    parser.add_argument(
        "--global-fit",
        action="store_true",
        help="Also draw one dashed fit through the combined data from all datasets.",
    )
    parser.add_argument(
        "--save-dpi",
        type=int,
        default=300,
        help="PNG resolution when saving (default: 300 dpi).",
    )
    parser.add_argument(
        "--save-without-show",
        action="store_true",
        help="Save immediately without opening Matplotlib windows.",
    )
    return parser.parse_args()


def choose_input_folder() -> Path:
    """Open a native folder chooser when no command-line folder is supplied."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askdirectory(title="Select folder containing paired data")
        root.destroy()
    except Exception as exc:
        raise SystemExit(
            "No input folder was supplied and the folder chooser could not be opened.\n"
            "Run: python compare_peak_fwhm.py <input_folder>"
        ) from exc
    if not selected:
        raise SystemExit("No input folder selected.")
    return Path(selected)


def normalized_name(value: object) -> str:
    return "_".join(str(value).strip().lower().replace("-", " ").split())


def find_column(columns: Iterable[object], aliases: Iterable[str]) -> object | None:
    normalized_columns = {normalized_name(original): original for original in columns}
    for alias in aliases:
        if alias in normalized_columns:
            return normalized_columns[alias]
    return None


def frame_to_dataset(frame: pd.DataFrame, label: str, source: str) -> Dataset | None:
    short_col = find_column(frame.columns, SHORT_ALIASES)
    long_col = find_column(frame.columns, LONG_ALIASES)
    if short_col is None or long_col is None:
        return None

    time_col = find_column(frame.columns, TIME_ALIASES)
    short = pd.to_numeric(frame[short_col], errors="coerce")
    long = pd.to_numeric(frame[long_col], errors="coerce")
    if time_col is None:
        time = pd.Series(np.arange(len(frame), dtype=float), index=frame.index)
    else:
        time = pd.to_numeric(frame[time_col], errors="coerce")

    valid = short.notna() & long.notna() & time.notna()
    cleaned = pd.DataFrame(
        {"time": time[valid], "short": short[valid], "long": long[valid]}
    ).sort_values("time")
    cleaned = cleaned.drop_duplicates(subset="time", keep="first")
    if len(cleaned) < 5:
        raise ValueError(f"{source}: fewer than five valid paired rows")

    return Dataset(
        label=label,
        source=source,
        time=cleaned["time"].to_numpy(dtype=float),
        short=cleaned["short"].to_numpy(dtype=float),
        long=cleaned["long"].to_numpy(dtype=float),
    )


def load_datasets(folder: Path) -> list[Dataset]:
    datasets: list[Dataset] = []
    files = sorted(
        (path for path in folder.iterdir() if path.is_file()),
        key=lambda path: path.name.lower(),
    )
    for path in files:
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES or path.name.startswith("~$"):
            continue
        if suffix == ".csv":
            frame = pd.read_csv(path)
            dataset = frame_to_dataset(frame, path.stem, str(path))
            if dataset is not None:
                datasets.append(dataset)
            continue

        workbook = pd.ExcelFile(path)
        for sheet_name in workbook.sheet_names:
            frame = pd.read_excel(workbook, sheet_name=sheet_name)
            source = f"{path} :: {sheet_name}"
            dataset = frame_to_dataset(frame, f"{path.stem} - {sheet_name}", source)
            if dataset is not None:
                datasets.append(dataset)
    return datasets


def median_sample_step(time: np.ndarray) -> float:
    differences = np.diff(time)
    positive = differences[np.isfinite(differences) & (differences > 0)]
    if positive.size == 0:
        raise ValueError("Time/index values must increase.")
    return float(np.median(positive))


def smooth_signal(values: np.ndarray, requested_window_s: float, step_s: float) -> np.ndarray:
    if requested_window_s <= 0:
        return values.copy()
    window = max(5, int(round(requested_window_s / step_s)))
    if window % 2 == 0:
        window += 1
    largest_odd = len(values) if len(values) % 2 == 1 else len(values) - 1
    window = min(window, largest_odd)
    if window < 5:
        return values.copy()
    return savgol_filter(values, window_length=window, polyorder=min(3, window - 2))


def detect_peaks(
    values: np.ndarray,
    step_s: float,
    smooth_window_s: float,
    min_distance_s: float,
    prominence_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    smoothed = smooth_signal(values, smooth_window_s, step_s)
    q05, q95 = np.nanpercentile(smoothed, [5.0, 95.0])
    robust_range = float(q95 - q05)
    if not np.isfinite(robust_range) or robust_range <= 0:
        raise ValueError("Signal has no usable variation for peak detection.")
    prominence = prominence_ratio * robust_range
    distance = max(1, int(round(min_distance_s / step_s)))
    peaks, properties = find_peaks(
        smoothed, prominence=prominence, distance=distance
    )
    if peaks.size == 0:
        empty = np.array([], dtype=float)
        return peaks, empty, empty, smoothed

    _, _, left_ips, right_ips = peak_widths(smoothed, peaks, rel_height=0.5)
    sample_positions = np.arange(len(values), dtype=float)
    left_times = np.interp(left_ips, sample_positions, sample_positions * step_s)
    right_times = np.interp(right_ips, sample_positions, sample_positions * step_s)
    widths_s = right_times - left_times
    relative_heights = np.asarray(properties["prominences"], dtype=float)
    return peaks, relative_heights, widths_s, smoothed


def pair_peaks(
    short_peaks: np.ndarray,
    long_peaks: np.ndarray,
    time: np.ndarray,
    max_delay_s: float,
) -> list[tuple[int, int]]:
    """Greedily make unique nearest-time pairs, starting with the closest."""
    candidates: list[tuple[float, int, int]] = []
    for short_position, short_index in enumerate(short_peaks):
        delays = np.abs(time[long_peaks] - time[short_index])
        for long_position in np.flatnonzero(delays <= max_delay_s):
            candidates.append(
                (float(delays[long_position]), short_position, int(long_position))
            )

    used_short: set[int] = set()
    used_long: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, short_position, long_position in sorted(candidates):
        if short_position in used_short or long_position in used_long:
            continue
        used_short.add(short_position)
        used_long.add(long_position)
        pairs.append((short_position, long_position))
    return sorted(pairs, key=lambda pair: time[short_peaks[pair[0]]])


def extract_features(dataset: Dataset, args: argparse.Namespace) -> PairedFeatures:
    step_s = median_sample_step(dataset.time)
    short_peaks, short_relative_heights, short_widths, _ = detect_peaks(
        dataset.short,
        step_s,
        args.smooth_window_s,
        args.min_peak_distance_s,
        args.prominence_ratio,
    )
    long_peaks, long_relative_heights, long_widths, _ = detect_peaks(
        dataset.long,
        step_s,
        args.smooth_window_s,
        args.min_peak_distance_s,
        args.prominence_ratio,
    )
    pairs = pair_peaks(short_peaks, long_peaks, dataset.time, args.max_pair_delay_s)
    if len(pairs) < 2:
        raise ValueError(
            f"{dataset.source}: only {len(pairs)} matched peak(s). "
            "Try reducing --prominence-ratio or increasing --max-pair-delay-s."
        )

    short_positions = np.array([pair[0] for pair in pairs], dtype=int)
    long_positions = np.array([pair[1] for pair in pairs], dtype=int)
    short_peak_times = dataset.time[short_peaks[short_positions]]
    long_peak_times = dataset.time[long_peaks[long_positions]]

    # An IBI is retained only when both matched peak sequences advance by
    # exactly one independently detected peak. This prevents a missed or
    # extra peak from turning one interval into an apparent double interval.
    consecutive = (
        (np.diff(short_positions) == 1) & (np.diff(long_positions) == 1)
    )
    ibi_short_s = np.diff(short_peak_times)[consecutive]
    ibi_long_s = np.diff(long_peak_times)[consecutive]
    return PairedFeatures(
        relative_peak_height_long=long_relative_heights[long_positions],
        relative_peak_height_short=short_relative_heights[short_positions],
        fwhm_long=long_widths[long_positions],
        fwhm_short=short_widths[short_positions],
        peak_delay_s=short_peak_times - long_peak_times,
        ibi_long_s=ibi_long_s,
        ibi_short_s=ibi_short_s,
        matched_count=len(pairs),
        short_peak_count=len(short_peaks),
        long_peak_count=len(long_peaks),
    )


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float] | None:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2 or np.ptp(x) == 0:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual_sum = float(np.sum((y - predicted) ** 2))
    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = np.nan if total_sum == 0 else 1.0 - residual_sum / total_sum
    return float(slope), float(intercept), float(r_squared)


def add_series(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    label: str,
    color: object,
) -> None:
    axis.scatter(x, y, s=35, alpha=0.72, color=color, edgecolors="white", linewidths=0.5)
    fit = fit_line(x, y)
    if fit is None:
        axis.plot([], [], color=color, label=f"{label}: fit unavailable (n={len(x)})")
        return
    slope, intercept, r_squared = fit
    fit_x = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    axis.plot(
        fit_x,
        slope * fit_x + intercept,
        color=color,
        linewidth=2,
        label=f"{label}: y={slope:.4g}x{intercept:+.4g}, R²={r_squared:.3f}, n={len(x)}",
    )


def create_comparison_figure(
    datasets: list[Dataset],
    features: list[PairedFeatures],
    kind: str,
    include_global_fit: bool,
) -> plt.Figure:
    if kind == "peak":
        title = "Time-matched relative peak height comparison"
        x_label = "Long relative peak height"
        y_label = "Short relative peak height"
        x_arrays = [feature.relative_peak_height_long for feature in features]
        y_arrays = [feature.relative_peak_height_short for feature in features]
    else:
        title = "Paired FWHM comparison"
        x_label = "Long FWHM (s)"
        y_label = "Short FWHM (s)"
        x_arrays = [feature.fwhm_long for feature in features]
        y_arrays = [feature.fwhm_short for feature in features]

    figure, axis = plt.subplots(figsize=(10.5, 7.0), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    for index, (dataset, x, y) in enumerate(zip(datasets, x_arrays, y_arrays)):
        add_series(axis, x, y, dataset.label, colors[index % len(colors)])

    if include_global_fit:
        combined_x = np.concatenate(x_arrays)
        combined_y = np.concatenate(y_arrays)
        fit = fit_line(combined_x, combined_y)
        if fit is not None:
            slope, intercept, r_squared = fit
            fit_x = np.linspace(float(np.min(combined_x)), float(np.max(combined_x)), 200)
            axis.plot(
                fit_x,
                slope * fit_x + intercept,
                color="black",
                linestyle="--",
                linewidth=2.2,
                label=(
                    f"Combined: y={slope:.4g}x{intercept:+.4g}, "
                    f"R²={r_squared:.3f}, n={len(combined_x)}"
                ),
            )

    axis.set_title(title, fontsize=15, pad=12)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.75)
    axis.legend(fontsize=8.5, frameon=True)
    return figure


def create_beat_timing_figure(
    datasets: list[Dataset], features: list[PairedFeatures]
) -> plt.Figure:
    """Create IBI agreement and peak-delay diagnostics for one input folder."""
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)
    trace_axis, scatter_axis, bland_altman_axis, delay_axis = axes.flat
    colors = plt.get_cmap("tab10").colors
    all_ibi_long: list[np.ndarray] = []
    all_ibi_short: list[np.ndarray] = []

    for index, (dataset, feature) in enumerate(zip(datasets, features)):
        color = colors[index % len(colors)]
        phase_match = re.search(r"phase\s*([0-9]+)", dataset.label, re.IGNORECASE)
        display_label = (
            f"Phase {phase_match.group(1)}" if phase_match else f"Dataset {index + 1}"
        )
        beat_number = np.arange(1, len(feature.ibi_long_s) + 1)
        trace_axis.plot(
            beat_number,
            feature.ibi_long_s,
            color=color,
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            label=f"{display_label} - long",
        )
        trace_axis.plot(
            beat_number,
            feature.ibi_short_s,
            color=color,
            marker="x",
            markersize=4.0,
            linewidth=1.2,
            linestyle="--",
            label=f"{display_label} - short",
        )

        if len(feature.ibi_long_s) >= 2:
            add_series(
                scatter_axis,
                feature.ibi_long_s,
                feature.ibi_short_s,
                display_label,
                color,
            )
        elif len(feature.ibi_long_s) == 1:
            scatter_axis.scatter(
                feature.ibi_long_s,
                feature.ibi_short_s,
                color=color,
                label=f"{display_label}: n=1",
            )

        ibi_mean_s = (feature.ibi_long_s + feature.ibi_short_s) / 2.0
        ibi_difference_ms = (feature.ibi_short_s - feature.ibi_long_s) * 1000.0
        bland_altman_axis.scatter(
            ibi_mean_s,
            ibi_difference_ms,
            s=32,
            alpha=0.72,
            color=color,
            edgecolors="white",
            linewidths=0.5,
            label=display_label,
        )
        delay_axis.plot(
            np.arange(1, len(feature.peak_delay_s) + 1),
            feature.peak_delay_s * 1000.0,
            color=color,
            marker="o",
            markersize=3.5,
            linewidth=1.3,
            label=display_label,
        )
        all_ibi_long.append(feature.ibi_long_s)
        all_ibi_short.append(feature.ibi_short_s)

    nonempty_long = [values for values in all_ibi_long if len(values)]
    nonempty_short = [values for values in all_ibi_short if len(values)]
    if nonempty_long and nonempty_short:
        combined_long = np.concatenate(nonempty_long)
        combined_short = np.concatenate(nonempty_short)
        lower = float(min(np.min(combined_long), np.min(combined_short)))
        upper = float(max(np.max(combined_long), np.max(combined_short)))
        padding = max((upper - lower) * 0.05, 0.001)
        identity = np.array([lower - padding, upper + padding])
        scatter_axis.plot(
            identity,
            identity,
            color="black",
            linestyle=":",
            linewidth=1.6,
            label="Identity: y=x",
        )

        differences_ms = (combined_short - combined_long) * 1000.0
        bias_ms = float(np.mean(differences_ms))
        difference_sd_ms = (
            float(np.std(differences_ms, ddof=1)) if len(differences_ms) > 1 else 0.0
        )
        lower_limit_ms = bias_ms - 1.96 * difference_sd_ms
        upper_limit_ms = bias_ms + 1.96 * difference_sd_ms
        bland_altman_axis.axhline(
            bias_ms,
            color="black",
            linewidth=1.7,
            label=f"Bias={bias_ms:.2f} ms",
        )
        bland_altman_axis.axhline(
            lower_limit_ms,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"95% limits: {lower_limit_ms:.2f}, {upper_limit_ms:.2f} ms",
        )
        bland_altman_axis.axhline(
            upper_limit_ms, color="black", linestyle="--", linewidth=1.2
        )

    trace_axis.set_title("Beat-to-beat inter-beat intervals")
    trace_axis.set_xlabel("Valid consecutive beat interval number")
    trace_axis.set_ylabel("IBI (s)")
    trace_axis.legend(fontsize=6.8, ncol=2)

    scatter_axis.set_title("IBI comparison")
    scatter_axis.set_xlabel("Long IBI (s)")
    scatter_axis.set_ylabel("Short IBI (s)")
    scatter_axis.legend(fontsize=6.8)

    bland_altman_axis.set_title("IBI Bland-Altman agreement")
    bland_altman_axis.set_xlabel("Mean IBI (s)")
    bland_altman_axis.set_ylabel("Short - long IBI (ms)")
    bland_altman_axis.legend(fontsize=6.8)

    delay_axis.axhline(0.0, color="black", linestyle=":", linewidth=1.3)
    delay_axis.set_title("Matched peak timing difference")
    delay_axis.set_xlabel("Matched beat number")
    delay_axis.set_ylabel("Short - long peak time (ms)")
    delay_axis.legend(fontsize=6.8)

    for axis in axes.flat:
        axis.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.75)
    figure.suptitle("Beat-to-beat timing comparison", fontsize=16)
    return figure


def save_figures(figures: dict[str, plt.Figure], output_dir: Path, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "peak": "relative_peak_height_comparison.png",
        "fwhm": "fwhm_comparison.png",
        "timing": "beat_timing_comparison.png",
    }
    for kind, figure in figures.items():
        path = output_dir / names[kind]
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {path}")


def main() -> int:
    args = parse_args()
    if args.prominence_ratio <= 0:
        raise SystemExit("--prominence-ratio must be greater than 0.")
    if args.min_peak_distance_s <= 0 or args.max_pair_delay_s < 0:
        raise SystemExit("Peak distance must be positive and pair delay cannot be negative.")

    folder = (args.input_folder or choose_input_folder()).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"Input folder does not exist: {folder}")

    datasets = load_datasets(folder)
    if not datasets:
        raise SystemExit(
            "No valid dataset found. A CSV file or Excel worksheet must contain "
            "short_value and long_value columns."
        )
    if args.expected_datasets > 0 and len(datasets) != args.expected_datasets:
        found = "\n".join(f"  - {dataset.source}" for dataset in datasets)
        raise SystemExit(
            f"Expected {args.expected_datasets} valid datasets, found {len(datasets)}:\n{found}\n"
            "Use --expected-datasets 0 to accept any number."
        )

    feature_sets: list[PairedFeatures] = []
    print(f"Loaded {len(datasets)} datasets from {folder}")
    for dataset in datasets:
        try:
            feature = extract_features(dataset, args)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        feature_sets.append(feature)
        print(
            f"  {dataset.label}: short peaks={feature.short_peak_count}, "
            f"long peaks={feature.long_peak_count}, paired={feature.matched_count}, "
            f"valid consecutive IBIs={len(feature.ibi_long_s)}"
        )

    figures = {
        "peak": create_comparison_figure(
            datasets, feature_sets, "peak", args.global_fit
        ),
        "fwhm": create_comparison_figure(
            datasets, feature_sets, "fwhm", args.global_fit
        ),
        "timing": create_beat_timing_figure(datasets, feature_sets),
    }
    output_dir = (args.output_dir or folder).expanduser().resolve()

    if args.save_without_show:
        save_figures(figures, output_dir, args.save_dpi)
        plt.close("all")
        return 0

    print("\nThree Matplotlib windows are open. Inspect them, then close all windows.")
    plt.show()
    answer = input("Save all three figures as PNG files? [y/N]: ").strip().lower()
    if answer in {"y", "yes", "s", "save", "是"}:
        save_figures(figures, output_dir, args.save_dpi)
    else:
        print("Figures were not saved.")
    plt.close("all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
