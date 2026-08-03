from __future__ import annotations

import argparse
from pathlib import Path

from config import DEBUG_IMAGE_PATH, INPUT_PATH, OUTPUT_PATH
from parsing import parse_color


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract colored plot curves from local images, or Doppler STFT peak "
            "envelopes from PCM WAV files, into CSV files."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
        help=f"Image file or directory. Default: {INPUT_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=(
            "CSV path for one image, or output directory for an input directory. "
            f"Default: {OUTPUT_PATH}"
        ),
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        help=(
            "PNG path for one image, or output directory for an input directory. "
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
        help="Skip showing and saving plots after CSV extraction.",
    )
    parser.add_argument(
        "--target-color",
        type=parse_color,
        help="Curve color as #RRGGBB or R,G,B. If omitted, the script chooses automatically.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=55.0,
        help="RGB color distance tolerance. Default: 55.",
    )
    parser.add_argument(
        "--roi",
        help=(
            "Optional x,y,width,height crop. Use pixels, or normalized values from 0 to 1 "
            "relative to image size."
        ),
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=1,
        help="Horizontal pixel window size per output sample. Default: 1.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Rolling median smoothing window. Use 1 to disable. Default: 1.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=30,
        help="Minimum connected component pixel area. Default: 30.",
    )
    parser.add_argument(
        "--min-span-ratio",
        type=float,
        default=0.15,
        help="Preferred minimum horizontal span relative to ROI width. Default: 0.15.",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=1,
        help="Remove colored pixels with fewer than this many colored neighbors. Default: 1.",
    )
    parser.add_argument(
        "--allow-flat",
        action="store_true",
        help="Allow nearly horizontal components to win automatic selection.",
    )
    parser.add_argument(
        "--min-brightness",
        type=int,
        default=45,
        help="Auto-color minimum RGB channel max. Default: 45.",
    )
    parser.add_argument(
        "--min-saturation",
        type=int,
        default=35,
        help="Auto-color minimum RGB channel spread. Default: 35.",
    )
    parser.add_argument(
        "--auto-candidates",
        type=int,
        default=8,
        help="Number of dominant color candidates to try in auto mode. Default: 8.",
    )
    parser.add_argument("--x-min", type=float, help="Real x-axis minimum for calibrated output.")
    parser.add_argument("--x-max", type=float, help="Real x-axis maximum for calibrated output.")
    parser.add_argument("--y-min", type=float, help="Real y-axis minimum for calibrated output.")
    parser.add_argument("--y-max", type=float, help="Real y-axis maximum for calibrated output.")
    parser.add_argument(
        "--debug-image",
        type=Path,
        default=DEBUG_IMAGE_PATH,
        help="Optional path for an overlay image showing ROI, selected component, and points.",
    )
    parser.add_argument(
        "--wav-stft-window",
        type=int,
        default=2048,
        help="WAV STFT window size in samples. Default: 2048.",
    )
    parser.add_argument(
        "--wav-stft-hop",
        type=int,
        default=256,
        help="WAV STFT hop size in samples. Default: 256.",
    )
    parser.add_argument(
        "--wav-min-frequency",
        type=float,
        default=100.0,
        help="Minimum frequency considered for WAV peak extraction. Default: 100.",
    )
    parser.add_argument(
        "--wav-max-frequency",
        type=float,
        default=8000.0,
        help="Maximum frequency considered for WAV peak extraction. Default: 8000.",
    )
    parser.add_argument(
        "--wav-noise-percentile",
        type=float,
        default=20.0,
        help="Per-frequency noise floor percentile for WAV thresholding. Default: 20.",
    )
    parser.add_argument(
        "--wav-threshold-db",
        type=float,
        default=8.0,
        help="Required dB margin above the whitened WAV noise floor. Default: 8.",
    )
    parser.add_argument(
        "--wav-edge-drop-db",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--wav-rolloff-percentile",
        type=float,
        default=95.0,
        help="Spectral roll-off percentile for WAV envelope extraction. Default: 95.",
    )
    parser.add_argument(
        "--wav-min-consecutive-bins",
        type=int,
        default=3,
        help="Minimum contiguous STFT bins required for a valid WAV region. Default: 3.",
    )
    parser.add_argument(
        "--wav-min-run-energy-fraction",
        type=float,
        default=0.05,
        help="Keep active frequency runs with at least this fraction of dominant-run energy. Default: 0.05.",
    )
    parser.add_argument(
        "--wav-median-window",
        type=int,
        default=7,
        help="Median filter window over WAV peak-frequency frames. Default: 7.",
    )
    parser.add_argument(
        "--wav-smooth-window",
        type=int,
        default=3,
        help="Light moving-average smoothing window after median filtering. Default: 3.",
    )
    parser.add_argument(
        "--wav-fill-gaps",
        type=int,
        default=2,
        help="Linearly fill WAV detection gaps up to this many STFT frames. Default: 2.",
    )
    parser.add_argument(
        "--wav-max-jump-hz",
        type=float,
        help=(
            "Maximum one-frame WAV envelope jump treated as continuous. "
            "Default: auto; use 0 to disable spike removal."
        ),
    )
    parser.add_argument(
        "--wav-min-confidence",
        type=float,
        default=0.05,
        help="Minimum confidence required before accepting a WAV roll-off point. Default: 0.05.",
    )
    parser.add_argument(
        "--wav-plot-start",
        type=float,
        default=0.0,
        help="Start time for the WAV envelope plot. Default: 0.",
    )
    parser.add_argument(
        "--wav-plot-duration",
        type=float,
        default=1.1,
        help="Duration shown in the WAV envelope plot. Default: 1.1 seconds.",
    )
    return parser
