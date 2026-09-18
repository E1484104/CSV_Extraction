#!/usr/bin/env python3
"""Plot IBI agreement and matched-peak timing from Phase 1/2/3 CSV files.

The script is standalone and only expects a folder containing one CSV file
for each phase. File names must contain ``phase1``, ``phase2``, and ``phase3``.
The left panel reproduces the IBI comparison from ``compare_peak_fwhm.py``.
The right panel pools all phases into a histogram of matched-peak timing
differences. The two panels are arranged side by side in one figure.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import linregress
except ImportError as exc:  # pragma: no cover - startup guidance only
    missing = getattr(exc, "name", "a required package")
    raise SystemExit(
        f"Missing dependency: {missing}\n"
        "Install the required packages with:\n"
        "  pip install numpy pandas scipy matplotlib"
    ) from exc


REQUIRED_COLUMNS = (
    "long_interval_start_peak_time_s",
    "long_interval_end_peak_time_s",
    "short_interval_start_peak_time_s",
    "short_interval_end_peak_time_s",
    "long_ibi_s",
    "short_ibi_s",
)
PHASE_PATTERN = re.compile(r"phase[\s_-]*([123])", re.IGNORECASE)
CONDITION_LABELS = {
    1: "Baseline",
    2: "Reperfusion",
    3: "Post-recovery",
}


@dataclass
class PhaseData:
    phase: int
    source: Path
    frame: pd.DataFrame
    peak_delays_ms: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a two-panel IBI comparison from Phase 1, 2, and 3 CSV files."
        )
    )
    parser.add_argument(
        "input_folder",
        nargs="?",
        type=Path,
        help="Folder containing the Phase 1, Phase 2, and Phase 3 IBI CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG output path (default: <input folder>/ibi_comparison.png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Saved PNG resolution (default: 300 dpi).",
    )
    parser.add_argument(
        "--delay-limit-ms",
        type=float,
        default=60.0,
        help=(
            "Merge timing differences at or beyond +/- this value into the "
            "two edge bars (default: 60 ms)."
        ),
    )
    parser.add_argument(
        "--save-without-show",
        action="store_true",
        help="Save the PNG immediately without opening a Matplotlib window.",
    )
    return parser.parse_args()


def choose_input_folder() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askdirectory(
            title="Select folder containing Phase 1, 2, and 3 IBI CSV files"
        )
        root.destroy()
    except Exception as exc:
        raise SystemExit(
            "No input folder was supplied and the folder chooser could not be opened.\n"
            "Run: python plot_ibi_comparison.py <input_folder>"
        ) from exc
    if not selected:
        raise SystemExit("No input folder selected.")
    return Path(selected)


def identify_phase(path: Path) -> int | None:
    match = PHASE_PATTERN.search(path.stem)
    return int(match.group(1)) if match else None


def reconstruct_peak_delays_ms(frame: pd.DataFrame) -> np.ndarray:
    """Return unique retained peak delays from the interval endpoints.

    Consecutive IBI rows share an endpoint. It is included only once, while
    endpoints separated by an invalid interval remain separate observations.
    """
    long_start = frame["long_interval_start_peak_time_s"].to_numpy(dtype=float)
    long_end = frame["long_interval_end_peak_time_s"].to_numpy(dtype=float)
    short_start = frame["short_interval_start_peak_time_s"].to_numpy(dtype=float)
    short_end = frame["short_interval_end_peak_time_s"].to_numpy(dtype=float)

    peak_pairs: list[tuple[float, float]] = []
    for ls, le, ss, se in zip(long_start, long_end, short_start, short_end):
        start = (ls, ss)
        end = (le, se)
        if not peak_pairs or not (
            np.isclose(peak_pairs[-1][0], start[0])
            and np.isclose(peak_pairs[-1][1], start[1])
        ):
            peak_pairs.append(start)
        if not (
            np.isclose(peak_pairs[-1][0], end[0])
            and np.isclose(peak_pairs[-1][1], end[1])
        ):
            peak_pairs.append(end)

    return np.asarray([(short - long) * 1000.0 for long, short in peak_pairs])


def load_phase_data(folder: Path) -> list[PhaseData]:
    phase_files: dict[int, Path] = {}
    for path in sorted(folder.glob("*.csv"), key=lambda item: item.name.lower()):
        phase = identify_phase(path)
        if phase is None:
            continue
        if phase in phase_files:
            raise ValueError(
                f"More than one CSV file was identified as Phase {phase}:\n"
                f"  {phase_files[phase]}\n  {path}"
            )
        phase_files[phase] = path

    missing = [phase for phase in (1, 2, 3) if phase not in phase_files]
    if missing:
        missing_text = ", ".join(f"Phase {phase}" for phase in missing)
        raise ValueError(f"Missing CSV file(s): {missing_text}")

    datasets: list[PhaseData] = []
    for phase in (1, 2, 3):
        path = phase_files[phase]
        frame = pd.read_csv(path, encoding="utf-8-sig")
        absent = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if absent:
            raise ValueError(f"{path.name} is missing columns: {', '.join(absent)}")

        numeric = frame.loc[:, REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
        invalid_rows = numeric.index[numeric.isna().any(axis=1)].to_numpy() + 2
        if invalid_rows.size:
            examples = ", ".join(str(row) for row in invalid_rows[:10])
            raise ValueError(f"{path.name} contains non-numeric/blank data at row(s): {examples}")
        if len(numeric) == 0:
            raise ValueError(f"{path.name} contains no data rows.")
        if (numeric[["long_ibi_s", "short_ibi_s"]] <= 0).any().any():
            raise ValueError(f"{path.name} contains a non-positive IBI value.")

        datasets.append(
            PhaseData(
                phase=phase,
                source=path,
                frame=numeric,
                peak_delays_ms=reconstruct_peak_delays_ms(numeric),
            )
        )
    return datasets


def fit_line(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float, float] | None:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2 or np.ptp(x) == 0:
        return None
    result = linregress(x, y)
    return (
        float(result.slope),
        float(result.intercept),
        float(result.rvalue**2),
        float(result.pvalue),
    )


def add_ibi_series(
    axis: plt.Axes, x: np.ndarray, y: np.ndarray, label: str, color: object
) -> None:
    axis.scatter(
        x,
        y,
        s=35,
        alpha=0.72,
        color=color,
        edgecolors="white",
        linewidths=0.5,
    )
    fit = fit_line(x, y)
    if fit is None:
        axis.plot([], [], color=color, label=f"{label}: fit unavailable (n={len(x)})")
        return
    slope, intercept, r_squared, p_value = fit
    fit_x = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    axis.plot(
        fit_x,
        slope * fit_x + intercept,
        color=color,
        linewidth=2,
        label=(
            f"{label}: y={slope:.4g}x{intercept:+.4g}, "
            f"R²={r_squared:.3f}, p={p_value:.2e}, n={len(x)}"
        ),
    )


def create_figure(datasets: list[PhaseData], delay_limit_ms: float) -> plt.Figure:
    figure, (scatter_axis, delay_axis) = plt.subplots(
        1, 2, figsize=(14.0, 5.4), constrained_layout=True
    )
    colors = plt.get_cmap("tab10").colors
    all_long: list[np.ndarray] = []
    all_short: list[np.ndarray] = []

    all_delays: list[np.ndarray] = []
    for index, dataset in enumerate(datasets):
        label = CONDITION_LABELS[dataset.phase]
        color = colors[index]
        long_ibi = dataset.frame["long_ibi_s"].to_numpy(dtype=float)
        short_ibi = dataset.frame["short_ibi_s"].to_numpy(dtype=float)
        add_ibi_series(scatter_axis, long_ibi, short_ibi, label, color)
        all_long.append(long_ibi)
        all_short.append(short_ibi)

        all_delays.append(dataset.peak_delays_ms)

    combined_long = np.concatenate(all_long)
    combined_short = np.concatenate(all_short)
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

    scatter_axis.set_title("Paired Inter-Beat Interval Comparison")
    scatter_axis.set_xlabel("Wearable IBI (s)")
    scatter_axis.set_ylabel("Ultrasound IBI (s)")
    scatter_axis.legend(fontsize=7.2)

    combined_delays = np.concatenate(all_delays)
    delay_step_ms = 10.0
    limit_ms = delay_limit_ms
    rounded_delays = np.rint(combined_delays / delay_step_ms) * delay_step_ms
    clipped_delays = np.clip(rounded_delays, -limit_ms, limit_ms)
    delay_values = np.arange(-limit_ms, limit_ms + delay_step_ms / 2, delay_step_ms)
    counts = np.asarray(
        [np.count_nonzero(np.isclose(clipped_delays, value)) for value in delay_values]
    )
    bars = delay_axis.bar(
        delay_values,
        counts,
        width=delay_step_ms * 0.82,
        color="#4c78a8",
        edgecolor="white",
        linewidth=0.8,
    )
    delay_axis.bar_label(
        bars,
        labels=[str(count) if count else "" for count in counts],
        padding=2,
        fontsize=8,
    )
    tick_labels = [f"{value:.0f}" for value in delay_values]
    tick_labels[0] = f"≤{delay_values[0]:.0f}"
    tick_labels[-1] = f"≥{delay_values[-1]:.0f}"
    delay_axis.set_xticks(delay_values, tick_labels)
    delay_axis.set_title("Distribution of Peak Timing Differences")
    delay_axis.set_xlabel("Peak timing difference (ms)")
    delay_axis.set_ylabel("Peak count")

    for axis in (scatter_axis, delay_axis):
        axis.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.75)
    scatter_axis.text(
        0.5,
        -0.17,
        "a",
        transform=scatter_axis.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="top",
    )
    delay_axis.text(
        0.5,
        -0.17,
        "b",
        transform=delay_axis.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="top",
    )
    figure.suptitle(
        "Comparison of Wearable and Ultrasound Beat Timing Measurements",
        fontsize=15,
    )
    return figure


def save_figure(figure: plt.Figure, output: Path, dpi: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {output}")


def main() -> int:
    args = parse_args()
    if args.dpi <= 0:
        raise SystemExit("--dpi must be greater than zero.")
    if args.delay_limit_ms <= 0 or args.delay_limit_ms % 10 != 0:
        raise SystemExit("--delay-limit-ms must be a positive multiple of 10.")

    folder = (args.input_folder or choose_input_folder()).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"Input folder does not exist: {folder}")
    output = (
        args.output.expanduser().resolve()
        if args.output
        else folder / "ibi_comparison.png"
    )
    if output.suffix.lower() != ".png":
        raise SystemExit("--output must use the .png extension.")

    try:
        datasets = load_phase_data(folder)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Loaded Phase 1, 2, and 3 data from: {folder}")
    for dataset in datasets:
        print(
            f"  Phase {dataset.phase}: {len(dataset.frame)} IBIs, "
            f"{len(dataset.peak_delays_ms)} retained matched peaks"
        )

    figure = create_figure(datasets, args.delay_limit_ms)
    if args.save_without_show:
        save_figure(figure, output, args.dpi)
        plt.close(figure)
        return 0

    print("\nA Matplotlib window is open. Inspect it, then close the window.")
    plt.show()
    answer = input("Save the figure as a PNG file? [y/N]: ").strip().lower()
    if answer in {"y", "yes", "s", "save", "是"}:
        save_figure(figure, output, args.dpi)
    else:
        print("Figure was not saved.")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
