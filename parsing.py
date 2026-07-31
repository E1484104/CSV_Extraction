from __future__ import annotations

import argparse

from models import Calibration, Roi


def parse_color(raw: str) -> tuple[int, int, int]:
    value = raw.strip()
    if value.startswith("#"):
        value = value[1:]
    if len(value) == 6 and all(ch in "0123456789abcdefABCDEF" for ch in value):
        return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))

    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Color must be #RRGGBB or R,G,B")
    try:
        rgb = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Color channels must be integers") from exc
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise argparse.ArgumentTypeError("Color channels must be between 0 and 255")
    return rgb


def parse_roi(raw: str, image_width: int, image_height: int) -> Roi:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x,y,width,height")

    values = [float(part) for part in parts]
    if all(0.0 <= value <= 1.0 for value in values):
        x = round(values[0] * image_width)
        y = round(values[1] * image_height)
        width = round(values[2] * image_width)
        height = round(values[3] * image_height)
    else:
        x, y, width, height = [round(value) for value in values]

    if width <= 0 or height <= 0:
        raise ValueError("ROI width and height must be positive")
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise ValueError(
            f"ROI {x},{y},{width},{height} is outside image bounds "
            f"{image_width}x{image_height}"
        )
    return Roi(x=x, y=y, width=width, height=height)


def parse_calibration(args: argparse.Namespace) -> Calibration | None:
    values = [args.x_min, args.x_max, args.y_min, args.y_max]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("Calibration requires all of --x-min, --x-max, --y-min, --y-max")
    if args.x_min == args.x_max or args.y_min == args.y_max:
        raise ValueError("Calibration min and max values must differ")
    return Calibration(
        x_min=float(args.x_min),
        x_max=float(args.x_max),
        y_min=float(args.y_min),
        y_max=float(args.y_max),
    )
