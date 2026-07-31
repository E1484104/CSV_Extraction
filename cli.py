from __future__ import annotations

import argparse
from pathlib import Path

from config import DEBUG_IMAGE_PATH, INPUT_PATH, OUTPUT_PATH
from parsing import parse_color


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract a colored plot curve from local images into CSV files.",
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
        default=3,
        help="Rolling median smoothing window. Use 1 to disable. Default: 3.",
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
    return parser
