from __future__ import annotations

import argparse
import csv
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from config import ROOT_PATH
from plotting import PLOT_DPI, PLOT_FIGSIZE


TIME_COLUMN = "time_s"
BPI_NORMALIZED_COLUMN = "bpi_normalized"
BPI_SMOOTHED_COLUMN = "bpi_smoothed"
DEFAULT_OUTPUT_DIRNAME = "BPI_Processed"
DEFAULT_OUTPUT_SUFFIX = "_bpi_processed"
DEFAULT_RAW_PLOT_SUFFIX = "_bpi"
DEFAULT_SMOOTHED_PLOT_SUFFIX = "_bpi_smoothed"
DEFAULT_NORMALIZED_PLOT_SUFFIX = "_bpi_normalized"
DEFAULT_NOISE_SAMPLE_COUNT = 2000
DEFAULT_NOISE_SEED = 0
DEFAULT_BPI_SMOOTH_WINDOW_POINTS = 1


@dataclass(frozen=True)
class WearableCsvData:
    path: Path
    fieldnames: list[str]
    rows: list[dict[str, str]]


@dataclass(frozen=True)
class WearableBpiStats:
    timestamp_column: str
    bpi_column: str
    first_timestamp: float
    last_timestamp: float
    bpi_min: float
    bpi_max: float
    normalize_bpi: bool
    noise_floor: float | None
    removed_noise_points: int
    smooth_window_points: int
    bottom_envelop: bool
    bottom_envelope_min: float | None
    bottom_envelope_max: float | None
    row_count: int
    point_count: int


@dataclass(frozen=True)
class ProcessedWearableBpi:
    input_path: Path
    csv_path: Path
    plot_path: Path | None
    stats: WearableBpiStats


def wearable_csv_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            return []
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.iterdir() if path.suffix.lower() == ".csv")
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def read_wearable_csv(path: Path) -> WearableCsvData:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        fieldnames = list(reader.fieldnames)
        rows = [dict(row) for row in reader]
    return WearableCsvData(path=path, fieldnames=fieldnames, rows=rows)


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


def _column_key(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def choose_column(fieldnames: list[str], requested: str | None, candidates: list[str]) -> str:
    if requested:
        if requested in fieldnames:
            return requested
        requested_key = _column_key(requested)
        for fieldname in fieldnames:
            if _column_key(fieldname) == requested_key:
                return fieldname
        raise RuntimeError(f"Column {requested!r} is missing from the wearable CSV")

    lookup = {_column_key(fieldname): fieldname for fieldname in fieldnames}
    for candidate in candidates:
        if _column_key(candidate) in lookup:
            return lookup[_column_key(candidate)]

    candidate_text = ", ".join(candidates)
    raise RuntimeError(f"Could not find a wearable CSV column among: {candidate_text}")


def normalize_value(value: float, lower: float, upper: float) -> float:
    if upper == lower:
        return 0.0
    return (value - lower) / (upper - lower)


def processed_fieldnames(
    fieldnames: list[str],
    *,
    include_normalized_bpi: bool,
    include_smoothed_bpi: bool,
) -> list[str]:
    result = list(fieldnames)
    extra_fieldnames = [TIME_COLUMN]
    if include_normalized_bpi:
        extra_fieldnames.append(BPI_NORMALIZED_COLUMN)
    if include_smoothed_bpi:
        extra_fieldnames.append(BPI_SMOOTHED_COLUMN)
    for fieldname in extra_fieldnames:
        if fieldname not in result:
            result.append(fieldname)
    return result


def selected_noise_indices(
    normalized_points: list[tuple[int, float, float]],
    sample_count: int,
    seed: int,
    path: Path,
) -> tuple[set[int], float | None]:
    if sample_count < 0:
        raise RuntimeError("--noise-sample-count cannot be negative")
    if sample_count == 0:
        return set(), None

    if len(normalized_points) <= sample_count:
        raise RuntimeError(
            f"Need more than {sample_count} valid BPI points to estimate noise floor "
            f"and keep data in {path}"
        )

    rng = random.Random(seed)
    sampled_positions = set(rng.sample(range(len(normalized_points)), sample_count))
    sampled = [normalized_points[position] for position in sampled_positions]
    noise_floor = sum(normalized for _, _, normalized in sampled) / sample_count
    if abs(noise_floor) <= 1e-12:
        raise RuntimeError("Estimated noise floor is zero; cannot divide BPI values")

    return {index for index, _, _ in sampled}, noise_floor


def smooth_values(values: list[float], window_points: int) -> list[float]:
    if window_points < 1:
        raise RuntimeError("--bpi-smooth-window-points must be greater than 0")
    if window_points % 2 == 0:
        raise RuntimeError("--bpi-smooth-window-points must be an odd number")
    if window_points == 1 or len(values) <= 1:
        return list(values)

    radius = window_points // 2
    padded = [values[0]] * radius + list(values) + [values[-1]] * radius
    prefix = [0.0]
    for value in padded:
        prefix.append(prefix[-1] + value)

    smoothed: list[float] = []
    for index in range(len(values)):
        start = index
        end = index + window_points
        smoothed.append((prefix[end] - prefix[start]) / window_points)
    return smoothed


def lower_envelope_values(values: list[float]) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return list(values)

    minima_indices: list[int] = []
    if values[0] <= values[1]:
        minima_indices.append(0)

    for index in range(1, len(values) - 1):
        current = values[index]
        previous_value = values[index - 1]
        next_value = values[index + 1]
        if (
            current <= previous_value
            and current <= next_value
            and (current < previous_value or current < next_value)
        ):
            minima_indices.append(index)

    last_index = len(values) - 1
    if values[last_index] <= values[last_index - 1]:
        minima_indices.append(last_index)

    if not minima_indices:
        floor = min(values)
        return [floor] * len(values)

    envelope = [values[minima_indices[0]]] * len(values)
    for left_index, right_index in zip(minima_indices, minima_indices[1:]):
        left_value = values[left_index]
        right_value = values[right_index]
        width = right_index - left_index
        for index in range(left_index, right_index + 1):
            ratio = (index - left_index) / width
            envelope[index] = left_value + (right_value - left_value) * ratio

    last_minimum_index = minima_indices[-1]
    for index in range(last_minimum_index + 1, len(values)):
        envelope[index] = values[last_minimum_index]

    return [
        min(envelope_value, value)
        for envelope_value, value in zip(envelope, values)
    ]


def bottom_zero_values(values: list[float]) -> tuple[list[float], list[float]]:
    envelope = lower_envelope_values(values)
    corrected = [
        0.0 if abs(value - envelope_value) <= 1e-12 else value - envelope_value
        for value, envelope_value in zip(values, envelope)
    ]
    return corrected, envelope


def processed_rows(
    data: WearableCsvData,
    timestamp_column: str,
    bpi_column: str,
    normalize_bpi: bool,
    noise_sample_count: int,
    noise_seed: int,
    bpi_smooth_window_points: int,
    bottom_envelop: bool,
) -> tuple[list[dict[str, str]], WearableBpiStats]:
    valid_points: list[tuple[int, float, float]] = []
    for index, row in enumerate(data.rows):
        timestamp = parse_float(row.get(timestamp_column))
        bpi = parse_float(row.get(bpi_column))
        if timestamp is not None and bpi is not None:
            valid_points.append((index, timestamp, bpi))

    if not valid_points:
        raise RuntimeError(f"No finite wearable BPI points were found in {data.path}")

    raw_bpi_values = [bpi for _, _, bpi in valid_points]
    bpi_min = min(raw_bpi_values)
    bpi_max = max(raw_bpi_values)
    if bpi_smooth_window_points < 1:
        raise RuntimeError("--bpi-smooth-window-points must be greater than 0")
    if bpi_smooth_window_points % 2 == 0:
        raise RuntimeError("--bpi-smooth-window-points must be an odd number")

    if not normalize_bpi:
        if noise_sample_count:
            raise RuntimeError("--denoise requires --normalize-bpi")
        if bottom_envelop:
            raise RuntimeError("--bottom-envelop requires --normalize-bpi")

        first_timestamp = valid_points[0][1]
        last_timestamp = valid_points[-1][1]
        smoothed_by_index: dict[int, float] = {}
        if bpi_smooth_window_points > 1:
            smoothed_values = smooth_values(raw_bpi_values, bpi_smooth_window_points)
            smoothed_by_index = {
                index: smoothed
                for (index, _, _), smoothed in zip(valid_points, smoothed_values)
            }

        output: list[dict[str, str]] = []
        point_count = 0

        for index, row in enumerate(data.rows):
            output_row = dict(row)
            timestamp = parse_float(row.get(timestamp_column))
            bpi = parse_float(row.get(bpi_column))

            if timestamp is None:
                output_row[TIME_COLUMN] = ""
            else:
                output_row[TIME_COLUMN] = f"{timestamp - first_timestamp:.8g}"

            if bpi_smooth_window_points > 1:
                smoothed = smoothed_by_index.get(index)
                if smoothed is None:
                    output_row[BPI_SMOOTHED_COLUMN] = ""
                else:
                    output_row[BPI_SMOOTHED_COLUMN] = f"{smoothed:.8g}"

            if timestamp is not None and bpi is not None:
                point_count += 1
            output.append(output_row)

        return output, WearableBpiStats(
            timestamp_column=timestamp_column,
            bpi_column=bpi_column,
            first_timestamp=first_timestamp,
            last_timestamp=last_timestamp,
            bpi_min=bpi_min,
            bpi_max=bpi_max,
            normalize_bpi=False,
            noise_floor=None,
            removed_noise_points=0,
            smooth_window_points=bpi_smooth_window_points,
            bottom_envelop=False,
            bottom_envelope_min=None,
            bottom_envelope_max=None,
            row_count=len(data.rows),
            point_count=point_count,
        )

    normalized_points = [
        (index, timestamp, normalize_value(bpi, bpi_min, bpi_max))
        for index, timestamp, bpi in valid_points
    ]
    noise_indices, noise_floor = selected_noise_indices(
        normalized_points,
        noise_sample_count,
        noise_seed,
        data.path,
    )

    valid_remaining: list[tuple[int, float, float]] = []
    for index, timestamp, normalized in normalized_points:
        if index in noise_indices:
            continue
        adjusted = normalized / noise_floor if noise_floor is not None else normalized
        valid_remaining.append((index, timestamp, adjusted))

    if not valid_remaining:
        raise RuntimeError(f"No finite wearable BPI points remain after denoising {data.path}")

    first_timestamp = valid_remaining[0][1]
    last_timestamp = valid_remaining[-1][1]
    smoothed_values = smooth_values(
        [adjusted for _, _, adjusted in valid_remaining],
        bpi_smooth_window_points,
    )
    if bottom_envelop:
        output_values, envelope_values = bottom_zero_values(smoothed_values)
    else:
        output_values = smoothed_values
        envelope_values = []

    adjusted_by_index = {
        index: adjusted
        for (index, _, _), adjusted in zip(valid_remaining, output_values)
    }
    output: list[dict[str, str]] = []
    point_count = 0

    for index, row in enumerate(data.rows):
        if index in noise_indices:
            continue

        output_row = dict(row)
        timestamp = parse_float(row.get(timestamp_column))
        adjusted = adjusted_by_index.get(index)

        if timestamp is None:
            output_row[TIME_COLUMN] = ""
        else:
            output_row[TIME_COLUMN] = f"{timestamp - first_timestamp:.8g}"

        if adjusted is None:
            output_row[BPI_NORMALIZED_COLUMN] = ""
        else:
            output_row[BPI_NORMALIZED_COLUMN] = f"{adjusted:.8g}"

        if timestamp is not None and adjusted is not None:
            point_count += 1
        output.append(output_row)

    return output, WearableBpiStats(
        timestamp_column=timestamp_column,
        bpi_column=bpi_column,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        bpi_min=bpi_min,
        bpi_max=bpi_max,
        normalize_bpi=True,
        noise_floor=noise_floor,
        removed_noise_points=len(noise_indices),
        smooth_window_points=bpi_smooth_window_points,
        bottom_envelop=bottom_envelop,
        bottom_envelope_min=min(envelope_values) if envelope_values else None,
        bottom_envelope_max=max(envelope_values) if envelope_values else None,
        row_count=len(data.rows),
        point_count=point_count,
    )


def default_processed_output_for(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.with_name(f"{input_path.stem}{DEFAULT_OUTPUT_SUFFIX}.csv")
    return input_path / DEFAULT_OUTPUT_DIRNAME


def processed_output_path_for(input_csv: Path, input_root: Path, output: Path | None) -> Path:
    destination = output or default_processed_output_for(input_root)
    default_name = f"{input_csv.stem}{DEFAULT_OUTPUT_SUFFIX}.csv"

    if input_root.is_file():
        if destination.suffix.lower() == ".csv":
            return destination
        return destination / default_name

    return destination / default_name


def plot_output_path_for(
    input_csv: Path,
    input_root: Path,
    processed_csv_path: Path,
    plot_output: Path | None,
    *,
    y_column: str,
) -> Path:
    if plot_output is None:
        return processed_csv_path.with_suffix(".png")
    if input_root.is_file() and plot_output.suffix:
        return plot_output
    if y_column == BPI_NORMALIZED_COLUMN:
        suffix = DEFAULT_NORMALIZED_PLOT_SUFFIX
    elif y_column == BPI_SMOOTHED_COLUMN:
        suffix = DEFAULT_SMOOTHED_PLOT_SUFFIX
    else:
        suffix = DEFAULT_RAW_PLOT_SUFFIX
    return plot_output / f"{input_csv.stem}{suffix}.png"


def write_processed_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _expanded_bounds(values: list[float]) -> tuple[float, float]:
    lower = min(values)
    upper = max(values)
    if lower != upper:
        padding = (upper - lower) * 0.03
        return lower - padding, upper + padding

    padding = abs(lower) * 0.03 if lower else 1.0
    return lower - padding, upper + padding


def _plot_points(rows: list[dict[str, str]], y_column: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        x = parse_float(row.get(TIME_COLUMN))
        y = parse_float(row.get(y_column))
        if x is not None and y is not None:
            points.append((x, y))
    return points


def show_wearable_bpi_plot(
    plot_path: Path | None,
    rows: list[dict[str, str]],
    *,
    y_column: str,
    save_plot: bool,
) -> Path | None:
    points = _plot_points(rows, y_column)
    if not points:
        raise RuntimeError("No finite wearable BPI points are available for plotting")

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = _expanded_bounds(xs)
    y_min, y_max = _expanded_bounds(ys)

    figure, axis = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    try:
        if y_column == BPI_NORMALIZED_COLUMN:
            title = "Wearable BFI Normalized"
        elif y_column == BPI_SMOOTHED_COLUMN:
            title = "Wearable BFI Smoothed"
        else:
            title = "Wearable BFI"
        if figure.canvas.manager is not None:
            figure.canvas.manager.set_window_title(title)
        figure.patch.set_facecolor("white")
        axis.set_facecolor("#fbfcfe")

        if len(points) == 1:
            axis.scatter(xs, ys, s=32, color="#1667be", zorder=3)
        else:
            axis.plot(xs, ys, color="#1667be", linewidth=1.4)

        axis.set_title(title, fontsize=14, pad=14)
        axis.set_xlabel(TIME_COLUMN)
        axis.set_ylabel(re.sub("bpi", "BFI", y_column, flags=re.IGNORECASE))
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(min(0.0, y_min), y_max)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=8))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
        axis.grid(True, color="#dfe4ea", linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#38404a")
        axis.spines["bottom"].set_color("#38404a")
        axis.tick_params(colors="#2e343c", labelsize=9)

        figure.tight_layout()
        saved_plot_path = None
        if save_plot and plot_path is not None:
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(plot_path, format="png")
            saved_plot_path = plot_path

        plt.show()
        return saved_plot_path
    finally:
        plt.close(figure)


def output_signal_column(stats: WearableBpiStats) -> str:
    if stats.normalize_bpi:
        return BPI_NORMALIZED_COLUMN
    if stats.smooth_window_points > 1:
        return BPI_SMOOTHED_COLUMN
    return stats.bpi_column


def process_wearable_bpi_csv(
    csv_path: Path,
    input_root: Path,
    output: Path | None,
    args: argparse.Namespace,
) -> ProcessedWearableBpi:
    output_csv_path = processed_output_path_for(csv_path, input_root, output)
    if output_csv_path.resolve() == csv_path.resolve():
        raise RuntimeError("Output CSV must be different from the input CSV")

    data = read_wearable_csv(csv_path)
    timestamp_column = choose_column(
        data.fieldnames,
        getattr(args, "timestamp_column", None),
        ["Timestamp", "timestamp", "time", "time_s", "unix_timestamp"],
    )
    bpi_column = choose_column(
        data.fieldnames,
        getattr(args, "bpi_column", None),
        ["BPI", "bpi", "BPI Raw", "BPI_raw", "bpi_raw", "bpi_raw_data"],
    )
    rows, stats = processed_rows(
        data,
        timestamp_column=timestamp_column,
        bpi_column=bpi_column,
        normalize_bpi=getattr(args, "normalize_bpi", False),
        noise_sample_count=(
            args.noise_sample_count
            if getattr(args, "denoise", False) and not getattr(args, "no_denoise", False)
            else 0
        ),
        noise_seed=args.noise_seed,
        bpi_smooth_window_points=args.bpi_smooth_window_points,
        bottom_envelop=getattr(args, "bottom_envelop", False),
    )
    write_processed_csv(
        output_csv_path,
        processed_fieldnames(
            data.fieldnames,
            include_normalized_bpi=getattr(args, "normalize_bpi", False),
            include_smoothed_bpi=(
                not getattr(args, "normalize_bpi", False)
                and args.bpi_smooth_window_points > 1
            ),
        ),
        rows,
    )

    saved_plot_path = None
    if not getattr(args, "no_plot", False):
        plot_y_column = output_signal_column(stats)
        candidate_plot_path = plot_output_path_for(
            input_csv=csv_path,
            input_root=input_root,
            processed_csv_path=output_csv_path,
            plot_output=getattr(args, "plot_output", None),
            y_column=plot_y_column,
        )
        auto_save = bool(
            getattr(args, "save_plot", False) or getattr(args, "plot_output", None) is not None
        )
        saved_plot_path = show_wearable_bpi_plot(
            candidate_plot_path,
            rows,
            y_column=plot_y_column,
            save_plot=auto_save,
        )

    return ProcessedWearableBpi(
        input_path=csv_path,
        csv_path=output_csv_path,
        plot_path=saved_plot_path,
        stats=stats,
    )


def process_wearable_bpi_csvs(args: argparse.Namespace) -> list[ProcessedWearableBpi]:
    output = args.output or default_processed_output_for(args.input)
    if args.input.is_dir() and output.suffix.lower() == ".csv":
        raise RuntimeError("Output must be a directory when wearable CSV input is a directory")
    if (
        args.input.is_dir()
        and getattr(args, "plot_output", None) is not None
        and args.plot_output.suffix
    ):
        raise RuntimeError("Plot output must be a directory when wearable CSV input is a directory")

    inputs = wearable_csv_inputs(args.input)
    if not inputs:
        raise RuntimeError(f"No wearable CSV files found in {args.input}")

    return [
        process_wearable_bpi_csv(
            csv_path=csv_path,
            input_root=args.input,
            output=output,
            args=args,
        )
        for csv_path in inputs
    ]


def format_wearable_bpi_result(processed: ProcessedWearableBpi) -> str:
    stats = processed.stats
    duration = stats.last_timestamp - stats.first_timestamp
    plot_info = f" | plot {processed.plot_path}" if processed.plot_path else ""
    if stats.normalize_bpi:
        if stats.noise_floor is None:
            noise_info = " | denoise off"
        else:
            noise_info = (
                f" | noise_floor={stats.noise_floor:.8g} "
                f"removed={stats.removed_noise_points}"
            )
        smooth_info = f" | smooth_window={stats.smooth_window_points}"
        if stats.bottom_envelop and stats.bottom_envelope_min is not None:
            bottom_info = (
                f" | bottom_envelop=[{stats.bottom_envelope_min:.8g}, "
                f"{stats.bottom_envelope_max:.8g}]"
            )
        else:
            bottom_info = ""
        signal_info = (
            f"BPI {stats.bpi_column} [{stats.bpi_min:.8g}, {stats.bpi_max:.8g}] "
            f"-> {BPI_NORMALIZED_COLUMN}{noise_info}"
            f"{smooth_info}"
            f"{bottom_info}"
        )
    else:
        if stats.smooth_window_points > 1:
            signal_info = (
                f"BPI {stats.bpi_column} [{stats.bpi_min:.8g}, {stats.bpi_max:.8g}] "
                f"-> {BPI_SMOOTHED_COLUMN} | normalization off | "
                f"smooth_window={stats.smooth_window_points}"
            )
        else:
            signal_info = (
                f"BPI {stats.bpi_column} [{stats.bpi_min:.8g}, {stats.bpi_max:.8g}] "
                "preserved | normalization off | smoothing off"
            )
    return (
        f"{processed.input_path} -> {processed.csv_path} | "
        f"{stats.point_count}/{stats.row_count} points | "
        f"time {stats.timestamp_column} -> [0, {duration:.8g}] | "
        f"{signal_info}"
        f"{plot_info}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Add a zero-based time_s axis to wearable BPI CSV data. BPI "
            "normalization is optional."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT_PATH,
        help=(
            "Experiment root folder. Used only when --input is omitted. "
            f"Default: {ROOT_PATH}"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Wearable CSV file or folder. Default: ROOT.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Processed CSV path for one input CSV, or output folder for an input "
            "folder. Default: next to the CSV, or INPUT/BPI_Processed for a folder."
        ),
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        help=(
            "PNG path for one input CSV, or output folder for an input folder. "
            "Also enables automatic plot saving."
        ),
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="Automatically save plot PNG files in addition to showing Matplotlib windows.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip showing and saving plots; only write processed CSV files.",
    )
    parser.add_argument(
        "--timestamp-column",
        help="Timestamp column. Default: auto-detect Timestamp.",
    )
    parser.add_argument(
        "--bpi-column",
        help="Raw BPI column. Default: auto-detect BPI.",
    )
    normalize_group = parser.add_mutually_exclusive_group()
    normalize_group.add_argument(
        "--normalize-bpi",
        action="store_true",
        dest="normalize_bpi",
        help=(
            "Also write bpi_normalized using min-max BPI normalization. "
            "Default: off."
        ),
    )
    normalize_group.add_argument(
        "--no-normalize-bpi",
        action="store_false",
        dest="normalize_bpi",
        help="Only write time_s and preserve raw BPI. This is the default.",
    )
    parser.set_defaults(normalize_bpi=False)
    parser.add_argument(
        "--noise-sample-count",
        type=int,
        default=DEFAULT_NOISE_SAMPLE_COUNT,
        help=(
            "Used only with --denoise. After min-max normalization, randomly sample "
            "this many valid points, average them as the noise floor, remove those "
            "rows, and divide the remaining normalized BPI values by that floor. "
            "Requires --normalize-bpi. "
            f"Default: {DEFAULT_NOISE_SAMPLE_COUNT}."
        ),
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=DEFAULT_NOISE_SEED,
        help=f"Random seed for noise-floor sampling. Default: {DEFAULT_NOISE_SEED}.",
    )
    denoise_group = parser.add_mutually_exclusive_group()
    denoise_group.add_argument(
        "--denoise",
        action="store_true",
        help="Enable post-normalization noise-floor removal and division. Requires --normalize-bpi. Default: off.",
    )
    denoise_group.add_argument(
        "--no-denoise",
        action="store_false",
        dest="denoise",
        help="Keep post-normalization noise-floor removal disabled. This is the default.",
    )
    parser.set_defaults(denoise=False)
    parser.add_argument(
        "--bpi-smooth-window-points",
        type=int,
        default=DEFAULT_BPI_SMOOTH_WINDOW_POINTS,
        help=(
            "Centered moving-average window. With --normalize-bpi it is applied to "
            "bpi_normalized; otherwise it writes bpi_smoothed from raw BPI. "
            "Use 1 to disable. Must be odd. "
            f"Default: {DEFAULT_BPI_SMOOTH_WINDOW_POINTS}."
        ),
    )
    parser.add_argument(
        "--bottom-envelop",
        "--bottom-envelope",
        action="store_true",
        dest="bottom_envelop",
        help=(
            "After wearable BPI normalization and smoothing, estimate the lower "
            "envelope from local minima and subtract it from each bpi_normalized "
            "point. Requires --normalize-bpi. Default: off."
        ),
    )
    return parser


def apply_root_defaults(args: argparse.Namespace) -> None:
    if args.input is None:
        args.input = args.root


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_root_defaults(args)

    try:
        if not args.input.exists():
            raise FileNotFoundError(f"Input path does not exist: {args.input}")
        results = process_wearable_bpi_csvs(args)
        for result in results:
            print(format_wearable_bpi_result(result))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
