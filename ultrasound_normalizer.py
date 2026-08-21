from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from config import ROOT_PATH, norm_data_path, raw_data_path


DEFAULT_OUTPUT_DIRNAME = "normalized"
DEFAULT_SINGLE_SUFFIX = "_normalized"


@dataclass(frozen=True)
class CsvData:
    path: Path
    fieldnames: list[str]
    rows: list[dict[str, str]]


@dataclass(frozen=True)
class NormalizationStats:
    x_column: str
    y_column: str
    y_min: float
    y_max: float
    duration_s: float
    point_count: int


def csv_paths(input_path: Path, pattern: str, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            raise RuntimeError(f"Input file is not a CSV: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    iterator = input_path.rglob(pattern) if recursive else input_path.glob(pattern)
    paths = sorted(path for path in iterator if path.is_file())
    if not paths:
        raise RuntimeError(f"No CSV files found in {input_path}")
    return paths


def read_csv(path: Path) -> CsvData:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        fieldnames = list(reader.fieldnames)
        rows = [dict(row) for row in reader]
    return CsvData(path=path, fieldnames=fieldnames, rows=rows)


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


def choose_column(data: list[CsvData], requested: str | None, candidates: list[str]) -> str:
    if requested:
        missing = [item.path for item in data if requested not in item.fieldnames]
        if missing:
            raise RuntimeError(f"Column {requested!r} is missing from {missing[0]}")
        return requested

    for candidate in candidates:
        if all(candidate in item.fieldnames for item in data):
            return candidate

    candidate_text = ", ".join(candidates)
    raise RuntimeError(f"Could not find a common column among: {candidate_text}")


def collect_stats(
    data: list[CsvData],
    x_column: str,
    y_column: str,
    duration_s: float,
) -> NormalizationStats:
    ys: list[float] = []
    point_count = 0

    for item in data:
        for row in item.rows:
            x = parse_float(row.get(x_column))
            y = parse_float(row.get(y_column))
            if y is not None:
                ys.append(y)
            if x is not None and y is not None:
                point_count += 1

    if not ys:
        raise RuntimeError("No finite numeric y values were found for normalization")

    return NormalizationStats(
        x_column=x_column,
        y_column=y_column,
        y_min=min(ys),
        y_max=max(ys),
        duration_s=duration_s,
        point_count=point_count,
    )


def file_x_range(item: CsvData, x_column: str) -> tuple[float, float]:
    xs = [
        x
        for row in item.rows
        if (x := parse_float(row.get(x_column))) is not None
    ]
    if not xs:
        raise RuntimeError(f"No finite numeric x values were found in {item.path}")
    return min(xs), max(xs)


def normalize_value(value: float, lower: float, upper: float) -> float:
    if upper == lower:
        return 0.0
    return (value - lower) / (upper - lower)


def scale_value(
    value: float,
    input_lower: float,
    input_upper: float,
    output_lower: float,
    output_upper: float,
) -> float:
    fraction = normalize_value(value, input_lower, input_upper)
    return output_lower + fraction * (output_upper - output_lower)


def normalized_rows(
    item: CsvData,
    stats: NormalizationStats,
    invert_y: bool,
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    x_min, x_max = file_x_range(item, stats.x_column)
    for row in item.rows:
        normalized = dict(row)
        x = parse_float(row.get(stats.x_column))
        y = parse_float(row.get(stats.y_column))
        if x is None:
            normalized["x_norm"] = ""
        else:
            x_norm = scale_value(x, x_min, x_max, 0.0, stats.duration_s)
            normalized["x_norm"] = f"{x_norm:.8g}"

        if y is None:
            normalized["y_norm"] = ""
        else:
            if invert_y:
                y_norm = normalize_value(stats.y_max - y, 0.0, stats.y_max - stats.y_min)
            else:
                y_norm = normalize_value(y, stats.y_min, stats.y_max)
            normalized["y_norm"] = f"{y_norm:.8g}"
        output.append(normalized)
    return output


def output_fieldnames(fieldnames: list[str]) -> list[str]:
    result = list(fieldnames)
    for field in ("x_norm", "y_norm"):
        if field not in result:
            result.append(field)
    return result


def default_output_for(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.with_name(f"{input_path.stem}{DEFAULT_SINGLE_SUFFIX}.csv")
    return input_path / DEFAULT_OUTPUT_DIRNAME


def output_path_for(
    input_path: Path,
    input_root: Path,
    output: Path,
    recursive: bool,
) -> Path:
    if input_root.is_file():
        if output.suffix.lower() == ".csv":
            return output
        return output / input_path.name

    if recursive:
        relative = input_path.relative_to(input_root)
        return output / relative
    return output / input_path.name


def write_normalized_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_folder(args: argparse.Namespace) -> tuple[NormalizationStats, int]:
    input_path = args.input
    output = args.output or default_output_for(input_path)
    if output.resolve() == input_path.resolve():
        raise RuntimeError("Output must be different from the input path")
    if input_path.is_dir() and output.suffix.lower() == ".csv":
        raise RuntimeError("Output must be a directory when the input is a directory")
    if args.duration_s <= 0:
        raise RuntimeError("--duration-s must be greater than 0")

    paths = csv_paths(input_path, pattern=args.pattern, recursive=args.recursive)
    data = [read_csv(path) for path in paths]
    x_column = choose_column(data, args.x_column, ["x_px", "x_value"])
    y_column = choose_column(data, args.y_column, ["y_px", "y_value"])
    stats = collect_stats(
        data,
        x_column=x_column,
        y_column=y_column,
        duration_s=args.duration_s,
    )

    for item in data:
        destination = output_path_for(
            input_path=item.path,
            input_root=input_path,
            output=output,
            recursive=args.recursive,
        )
        rows = normalized_rows(item, stats=stats, invert_y=not args.no_invert_y)
        write_normalized_csv(
            destination,
            fieldnames=output_fieldnames(item.fieldnames),
            rows=rows,
        )

    return stats, len(data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize one CSV file or a CSV folder with per-file x mapped to a fixed "
            "duration and y normalized by one shared range."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT_PATH,
        help=(
            "Experiment root folder. Used only for omitted --input/--output. "
            f"Default: {ROOT_PATH}"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        help=f"CSV file or folder containing CSV files to normalize. Default: ROOT/{raw_data_path().name}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output CSV path for one input CSV, or output folder for an input folder. "
            f"Default: ROOT/{norm_data_path().name}."
        ),
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="CSV filename pattern. Default: *.csv.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search CSV files recursively and preserve relative paths in the output folder.",
    )
    parser.add_argument(
        "--x-column",
        help="Column to normalize as x. Default: x_px, falling back to x_value.",
    )
    parser.add_argument(
        "--y-column",
        help="Column to normalize as y. Default: y_px, falling back to y_value.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=29.8,
        help="Map each CSV's x range to 0..duration seconds. Default: 29.8.",
    )
    parser.add_argument(
        "--no-invert-y",
        action="store_true",
        help="Do not invert y during normalization. By default y_px is normalized bottom-to-top.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.input is None:
        args.input = raw_data_path(args.root)
    if args.output is None:
        args.output = norm_data_path(args.root)
    try:
        stats, file_count = normalize_folder(args)
    except Exception as exc:
        print(f"error: {exc}")
        return 1

    output = args.output or default_output_for(args.input)
    print(
        f"{file_count} CSV files -> {output} | "
        f"x {stats.x_column} per file -> [0, {stats.duration_s:.8g}] | "
        f"y {stats.y_column} [{stats.y_min:.8g}, {stats.y_max:.8g}] | "
        f"{stats.point_count} points"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
