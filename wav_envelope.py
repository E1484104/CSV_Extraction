from __future__ import annotations

import argparse
import csv
import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plotting import write_wav_envelope_plot


CsvValue = float | str | None


@dataclass(frozen=True)
class WavEnvelopeResult:
    wav_path: Path
    csv_path: Path
    plot_path: Path | None
    sample_rate: int
    channels: int
    frames: int
    duration_s: float
    rows: list[dict[str, CsvValue]]


def wav_csv_path_for(wav_path: Path, output: Path) -> Path:
    if output.suffix.lower() == ".csv":
        return output
    return output / "continuous_envelope.csv"


def wav_plot_path_for(wav_path: Path, csv_path: Path, plot_output: Path | None) -> Path:
    if plot_output is None:
        return csv_path.with_suffix(".png")
    if plot_output.suffix:
        return plot_output
    return plot_output / "continuous_envelope.png"


def read_pcm16_wav(wav_path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(wav_path), "rb") as wav_file:
        if wav_file.getcomptype() != "NONE":
            raise RuntimeError(f"Compressed WAV is not supported: {wav_file.getcomptype()}")
        if wav_file.getsampwidth() != 2:
            raise RuntimeError(
                f"Only signed 16-bit PCM WAV is supported, got sample width "
                f"{wav_file.getsampwidth()} bytes"
            )

        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        raw = wav_file.readframes(frames)

    samples = np.frombuffer(raw, dtype="<i2")
    if samples.size != frames * channels:
        raise RuntimeError("WAV data size does not match the header frame/channel count")

    audio = samples.reshape(frames, channels).astype(np.float64) / 32768.0
    return sample_rate, audio


def validate_wav_args(args: argparse.Namespace) -> None:
    if args.wav_stft_window <= 0:
        raise ValueError("--wav-stft-window must be positive")
    if args.wav_stft_hop <= 0:
        raise ValueError("--wav-stft-hop must be positive")
    if not 0 <= args.wav_noise_percentile <= 100:
        raise ValueError("--wav-noise-percentile must be between 0 and 100")
    if args.wav_min_frequency < 0:
        raise ValueError("--wav-min-frequency must be non-negative")
    if args.wav_max_frequency <= args.wav_min_frequency:
        raise ValueError("--wav-max-frequency must be greater than --wav-min-frequency")
    if not 0 < args.wav_rolloff_percentile < 100:
        raise ValueError("--wav-rolloff-percentile must be between 0 and 100")
    if args.wav_min_consecutive_bins < 1:
        raise ValueError("--wav-min-consecutive-bins must be at least 1")
    if not 0 <= args.wav_min_run_energy_fraction <= 1:
        raise ValueError("--wav-min-run-energy-fraction must be between 0 and 1")
    if args.wav_median_window < 1:
        raise ValueError("--wav-median-window must be at least 1")
    if args.wav_smooth_window < 1:
        raise ValueError("--wav-smooth-window must be at least 1")
    if args.wav_fill_gaps < 0:
        raise ValueError("--wav-fill-gaps must be non-negative")
    if args.wav_max_jump_hz is not None and args.wav_max_jump_hz < 0:
        raise ValueError("--wav-max-jump-hz must be non-negative")
    if not 0 <= args.wav_min_confidence <= 1:
        raise ValueError("--wav-min-confidence must be between 0 and 1")
    if args.wav_plot_duration is not None and args.wav_plot_duration <= 0:
        raise ValueError("--wav-plot-duration must be positive")


def frame_starts(sample_count: int, window_size: int, hop_size: int) -> np.ndarray:
    if sample_count <= 0:
        raise RuntimeError("WAV file contains no samples")

    if sample_count <= window_size:
        return np.array([0], dtype=np.int64)

    starts = np.arange(0, sample_count - window_size + 1, hop_size, dtype=np.int64)
    last_start = sample_count - window_size
    if starts[-1] != last_start:
        starts = np.append(starts, last_start)
    return starts


def stft_power_db(
    samples: np.ndarray,
    sample_rate: int,
    window_size: int,
    hop_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = frame_starts(samples.size, window_size, hop_size)
    padded = samples
    if samples.size < window_size:
        padded = np.pad(samples, (0, window_size - samples.size), mode="constant")

    frame_indexes = starts[:, None] + np.arange(window_size)
    frames = padded[frame_indexes]
    frames = frames - frames.mean(axis=1, keepdims=True)
    window = np.hanning(window_size)
    frames = frames * window

    spectra = np.fft.rfft(frames, axis=1)
    power = (np.abs(spectra) ** 2) / max(float(np.sum(window**2)), 1.0)
    power_db = 10.0 * np.log10(power + np.finfo(np.float64).eps)
    times = np.minimum(
        (starts.astype(np.float64) + window_size / 2.0) / sample_rate,
        samples.size / sample_rate,
    )
    frequencies = np.fft.rfftfreq(window_size, d=1.0 / sample_rate)
    return times, frequencies, power_db


def frequency_mask(
    frequencies: np.ndarray,
    min_frequency: float,
    max_frequency: float,
) -> np.ndarray:
    mask = (frequencies >= min_frequency) & (frequencies <= max_frequency)
    if not np.any(mask):
        raise ValueError("No STFT frequency bins remain after applying frequency limits")
    return mask


def run_slices(mask: np.ndarray) -> list[slice]:
    runs: list[slice] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if (not value or index == mask.size - 1) and start is not None:
            end = index + 1 if value and index == mask.size - 1 else index
            runs.append(slice(start, end))
            start = None
    return runs


def keep_runs(mask: np.ndarray, min_length: int) -> np.ndarray:
    if min_length <= 1:
        return mask

    kept = np.zeros(mask.shape, dtype=bool)
    for run in run_slices(mask):
        if run.stop - run.start >= min_length:
            kept[run] = True
    return kept


def significant_run_mask(
    active: np.ndarray,
    excess_power: np.ndarray,
    min_length: int,
    min_energy_fraction: float,
) -> np.ndarray:
    active = keep_runs(active, min_length)
    runs = run_slices(active)
    if not runs:
        return active

    energies = np.array([float(np.sum(excess_power[run])) for run in runs])
    max_energy = float(np.max(energies))
    if max_energy <= 0:
        return np.zeros(active.shape, dtype=bool)

    kept = np.zeros(active.shape, dtype=bool)
    for run, energy in zip(runs, energies):
        if energy >= max_energy * min_energy_fraction:
            kept[run] = True
    return kept


def rolloff_frequency(
    frequencies: np.ndarray,
    signal_power: np.ndarray,
    rolloff_fraction: float,
) -> float:
    total_power = float(np.sum(signal_power))
    if total_power <= 0:
        return math.nan

    target = total_power * rolloff_fraction
    cumulative = np.cumsum(signal_power)
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(index, frequencies.size - 1)
    if index == 0 or signal_power[index] <= 0:
        return float(frequencies[index])

    previous_cumulative = cumulative[index - 1]
    fraction = (target - previous_cumulative) / signal_power[index]
    return float(
        frequencies[index - 1]
        + fraction * (frequencies[index] - frequencies[index - 1])
    )


def confidence_from_detection(
    snr_db: float,
    threshold_db: float,
    active_bins: int,
    min_bins: int,
) -> float:
    if not math.isfinite(snr_db):
        return 0.0

    snr_confidence = max(0.0, min(1.0, (snr_db - threshold_db) / 12.0))
    width_confidence = max(0.0, min(1.0, active_bins / max(1.0, min_bins * 3.0)))
    return 0.75 * snr_confidence + 0.25 * width_confidence


def extract_rolloff_frequencies(
    frequencies: np.ndarray,
    power_db: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    allowed = frequency_mask(
        frequencies,
        min_frequency=args.wav_min_frequency,
        max_frequency=args.wav_max_frequency,
    )
    band_frequencies = frequencies[allowed]
    band_power_db = power_db[:, allowed]
    noise_floor = np.percentile(
        band_power_db,
        args.wav_noise_percentile,
        axis=0,
    )
    whitened_db = band_power_db - noise_floor

    raw_peaks = np.full(power_db.shape[0], np.nan, dtype=np.float64)
    snr_db = np.full(power_db.shape[0], np.nan, dtype=np.float64)
    confidence = np.zeros(power_db.shape[0], dtype=np.float64)
    rolloff_fraction = args.wav_rolloff_percentile / 100.0

    for frame_index, frame_whitened_db in enumerate(whitened_db):
        max_whitened_db = float(np.max(frame_whitened_db))
        snr_db[frame_index] = max_whitened_db

        active = frame_whitened_db >= args.wav_threshold_db
        excess_power = np.maximum(np.power(10.0, frame_whitened_db / 10.0) - 1.0, 0.0)
        signal_mask = significant_run_mask(
            active=active,
            excess_power=excess_power,
            min_length=args.wav_min_consecutive_bins,
            min_energy_fraction=args.wav_min_run_energy_fraction,
        )
        if not np.any(signal_mask):
            continue

        frame_confidence = confidence_from_detection(
            snr_db=max_whitened_db,
            threshold_db=args.wav_threshold_db,
            active_bins=int(np.count_nonzero(signal_mask)),
            min_bins=args.wav_min_consecutive_bins,
        )
        confidence[frame_index] = frame_confidence
        if frame_confidence < args.wav_min_confidence:
            continue

        signal_power = np.where(signal_mask, excess_power, 0.0)
        raw_peaks[frame_index] = rolloff_frequency(
            frequencies=band_frequencies,
            signal_power=signal_power,
            rolloff_fraction=rolloff_fraction,
        )

    return raw_peaks, snr_db, confidence


def rolling_nanmedian(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return values.copy()
    if window_size % 2 == 0:
        window_size += 1

    radius = window_size // 2
    filtered = np.full(values.shape, np.nan, dtype=np.float64)
    for index in range(values.size):
        if not math.isfinite(float(values[index])):
            continue
        start = max(0, index - radius)
        end = min(values.size, index + radius + 1)
        window = values[start:end]
        if np.any(np.isfinite(window)):
            filtered[index] = float(np.nanmedian(window))
    return filtered


def rolling_nanmean(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return values.copy()
    if window_size % 2 == 0:
        window_size += 1

    radius = window_size // 2
    smoothed = np.full(values.shape, np.nan, dtype=np.float64)
    for index in range(values.size):
        if not math.isfinite(float(values[index])):
            continue
        start = max(0, index - radius)
        end = min(values.size, index + radius + 1)
        window = values[start:end]
        if np.any(np.isfinite(window)):
            smoothed[index] = float(np.nanmean(window))
    return smoothed


def automatic_max_jump_hz(values: np.ndarray, frequency_resolution: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return max(frequency_resolution * 6.0, 1.0)

    diffs = np.abs(np.diff(finite))
    if diffs.size == 0:
        return max(frequency_resolution * 6.0, 1.0)

    median = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - median)))
    return max(frequency_resolution * 6.0, median + 6.0 * mad)


def remove_isolated_spikes(values: np.ndarray, max_jump_hz: float) -> np.ndarray:
    cleaned = values.copy()
    if values.size < 3 or max_jump_hz <= 0:
        return cleaned

    for index in range(1, values.size - 1):
        previous_value = cleaned[index - 1]
        current_value = cleaned[index]
        next_value = cleaned[index + 1]
        if not (
            math.isfinite(previous_value)
            and math.isfinite(current_value)
            and math.isfinite(next_value)
        ):
            continue
        if (
            abs(current_value - previous_value) > max_jump_hz
            and abs(current_value - next_value) > max_jump_hz
            and abs(next_value - previous_value) <= max_jump_hz
        ):
            cleaned[index] = math.nan
    return cleaned


def fill_short_gaps(values: np.ndarray, max_gap_frames: int) -> tuple[np.ndarray, np.ndarray]:
    interpolated = np.zeros(values.shape, dtype=bool)
    if max_gap_frames <= 0:
        return values, interpolated

    filled = values.copy()
    finite = np.isfinite(filled)
    index = 0
    while index < filled.size:
        if finite[index]:
            index += 1
            continue

        start = index
        while index < filled.size and not finite[index]:
            index += 1
        end = index
        gap_length = end - start
        if (
            gap_length <= max_gap_frames
            and start > 0
            and end < filled.size
            and finite[start - 1]
            and finite[end]
        ):
            filled[start:end] = np.linspace(
                filled[start - 1],
                filled[end],
                gap_length + 2,
            )[1:-1]
            finite[start:end] = True
            interpolated[start:end] = True

    return filled, interpolated


def postprocess_peak_frequencies(
    raw_peaks: np.ndarray,
    frequency_resolution: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    processed = raw_peaks.copy()
    max_jump_hz = args.wav_max_jump_hz
    if max_jump_hz is None:
        max_jump_hz = automatic_max_jump_hz(processed, frequency_resolution)
    processed = remove_isolated_spikes(processed, max_jump_hz)
    processed = rolling_nanmedian(processed, args.wav_median_window)
    processed, interpolated = fill_short_gaps(processed, args.wav_fill_gaps)
    processed = rolling_nanmean(processed, args.wav_smooth_window)
    return processed, interpolated


def normalize_frequency(
    peak_frequency_hz: float,
    min_frequency: float,
    max_frequency: float,
) -> float:
    normalized = (peak_frequency_hz - min_frequency) / (max_frequency - min_frequency)
    return max(0.0, min(1.0, normalized))


def envelope_rows_for_channel(
    channel_name: str,
    times: np.ndarray,
    raw_peaks: np.ndarray,
    processed_peaks: np.ndarray,
    interpolated: np.ndarray,
    snr_db: np.ndarray,
    confidence: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, CsvValue]]:
    rows: list[dict[str, CsvValue]] = []
    for index, time_s in enumerate(times):
        raw_peak = raw_peaks[index]
        processed_peak = processed_peaks[index]
        if math.isfinite(float(processed_peak)):
            peak_value: float | None = float(processed_peak)
            normalized: float | None = normalize_frequency(
                peak_frequency_hz=peak_value,
                min_frequency=args.wav_min_frequency,
                max_frequency=args.wav_max_frequency,
            )
        else:
            peak_value = None
            normalized = None

        rows.append(
            {
                "time_s": float(time_s),
                "channel": channel_name,
                "peak_frequency_hz": peak_value,
                "envelope_normalized": normalized,
                "raw_peak_frequency_hz": (
                    float(raw_peak) if math.isfinite(float(raw_peak)) else None
                ),
                "confidence": float(confidence[index]),
                "snr_db": float(snr_db[index]) if math.isfinite(float(snr_db[index])) else None,
                "is_interpolated": 1.0 if interpolated[index] else 0.0,
            }
        )
    return rows


def channel_processing_order(channel_count: int) -> list[int]:
    if channel_count <= 1:
        return [0]
    return [1, 0, *range(2, channel_count)]


def extract_wav_envelope_rows(
    wav_path: Path,
    args: argparse.Namespace,
) -> tuple[int, int, int, float, list[dict[str, CsvValue]]]:
    sample_rate, audio = read_pcm16_wav(wav_path)
    frame_count, channel_count = audio.shape
    duration_s = frame_count / sample_rate
    rows: list[dict[str, CsvValue]] = []

    for channel_index in channel_processing_order(channel_count):
        times, frequencies, power_db = stft_power_db(
            audio[:, channel_index],
            sample_rate=sample_rate,
            window_size=args.wav_stft_window,
            hop_size=args.wav_stft_hop,
        )
        raw_peaks, snr_db, confidence = extract_rolloff_frequencies(
            frequencies=frequencies,
            power_db=power_db,
            args=args,
        )
        frequency_resolution = sample_rate / args.wav_stft_window
        processed_peaks, interpolated = postprocess_peak_frequencies(
            raw_peaks=raw_peaks,
            frequency_resolution=frequency_resolution,
            args=args,
        )
        rows.extend(
            envelope_rows_for_channel(
                channel_name=f"channel_{channel_index}",
                times=times,
                raw_peaks=raw_peaks,
                processed_peaks=processed_peaks,
                interpolated=interpolated,
                snr_db=snr_db,
                confidence=confidence,
                args=args,
            )
        )

    return sample_rate, channel_count, frame_count, duration_s, rows


def _csv_value(value: CsvValue) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return f"{value:.10g}"


def write_wav_envelope_csv(
    csv_path: Path,
    rows: list[dict[str, CsvValue]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "time_s",
        "channel",
        "peak_frequency_hz",
        "envelope_normalized",
        "raw_peak_frequency_hz",
        "confidence",
        "snr_db",
        "is_interpolated",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def process_wav(args: argparse.Namespace) -> WavEnvelopeResult:
    wav_path = Path(args.input)
    if not wav_path.is_file():
        raise FileNotFoundError(f"WAV input path does not exist: {wav_path}")

    validate_wav_args(args)
    csv_path = wav_csv_path_for(wav_path, args.output)
    sample_rate, channels, frames, duration_s, rows = extract_wav_envelope_rows(wav_path, args)
    write_wav_envelope_csv(csv_path, rows)

    plot_path = None
    if not getattr(args, "no_plot", False):
        if getattr(args, "save_plot", False) or getattr(args, "plot_output", None) is not None:
            plot_path = wav_plot_path_for(
                wav_path=wav_path,
                csv_path=csv_path,
                plot_output=getattr(args, "plot_output", None),
            )
        plot_end_s = (
            None
            if args.wav_plot_duration is None
            else args.wav_plot_start + args.wav_plot_duration
        )
        write_wav_envelope_plot(
            plot_path,
            rows,
            time_start_s=args.wav_plot_start,
            time_end_s=plot_end_s,
            y_key="envelope_normalized",
        )

    return WavEnvelopeResult(
        wav_path=wav_path,
        csv_path=csv_path,
        plot_path=plot_path,
        sample_rate=sample_rate,
        channels=channels,
        frames=frames,
        duration_s=duration_s,
        rows=rows,
    )


def format_wav_envelope_result(result: WavEnvelopeResult) -> str:
    finite_points = sum(
        1
        for row in result.rows
        if isinstance(row["peak_frequency_hz"], float)
        and math.isfinite(row["peak_frequency_hz"])
    )
    plot_info = f" | plot {result.plot_path}" if result.plot_path else ""
    return (
        f"{result.wav_path} -> {result.csv_path} | "
        f"{result.channels} channels | {result.sample_rate} Hz | "
        f"{result.frames} frames | {result.duration_s:.6f} s | "
        f"{finite_points} envelope points"
        f"{plot_info}"
    )
