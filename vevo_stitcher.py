from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from config import ROOT_PATH, raw_images_path, stitched_images_path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_COLOR_TOLERANCE = 300.0
BLUE_TRACE_COLOR = (86, 94, 255)
GREEN_TRACE_COLOR = (86, 255, 94)
MIN_TRACE_SATURATION = 35.0
DEFAULT_ROIS_BY_IMAGE_SIZE = {
    # Full-screen Vevo 3100 exports used in the current data sets.
    # This covers the velocity waveform plot area but excludes the lower BPM/RR/C panel.
    (1412, 932): (267, 155, 1024, 616),
}


@dataclass(frozen=True)
class StitchPair:
    previous: Path
    current: Path
    shift_px: int
    score: float
    seam_connected: bool
    seam_horizontal_gap_px: int | None
    seam_vertical_gap_px: float | None
    seam_warning: str | None = None


@dataclass(frozen=True)
class SkippedPair:
    previous: Path
    current: Path
    shift_px: int
    score: float
    reason: str
    seam_horizontal_gap_px: int | None = None
    seam_vertical_gap_px: float | None = None


@dataclass(frozen=True)
class StitchResult:
    output_path: Path
    image_paths: list[Path]
    roi: tuple[int, int, int, int]
    pairs: list[StitchPair]
    skipped: list[SkippedPair]
    reversed_order: bool
    size: tuple[int, int]


def parse_roi(raw: str) -> tuple[int, int, int, int]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height")
    try:
        x, y, width, height = [int(round(float(part))) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI values must be numeric") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("ROI width and height must be positive")
    return x, y, width, height


def image_paths_from_inputs(inputs: list[Path]) -> list[Path]:
    if len(inputs) == 1 and inputs[0].is_dir():
        paths = sorted(
            path
            for path in inputs[0].iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    else:
        paths = inputs

    if len(paths) < 2:
        raise RuntimeError("At least two images are required for stitching")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input image does not exist: {missing[0]}")
    return paths


def load_images(paths: list[Path]) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    expected_height: int | None = None
    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if expected_height is None:
            expected_height = array.shape[0]
        elif array.shape[0] != expected_height:
            raise RuntimeError(
                f"All images must have the same height. "
                f"{path} has height {array.shape[0]}, expected {expected_height}. "
                "Different widths are allowed, but different heights usually mean "
                "the waveform y-scale is inconsistent."
            )
        images.append(array)
    return images


def default_roi_for_images(images: list[np.ndarray]) -> tuple[int, int, int, int] | None:
    if not images:
        return None

    image_height, image_width = images[0].shape[:2]
    roi = DEFAULT_ROIS_BY_IMAGE_SIZE.get((image_width, image_height))
    if roi is None:
        return None

    x, y, width, height = roi
    if x + width > image_width or y + height > image_height:
        return None
    return roi


def trace_color_distance_mask(
    image: np.ndarray,
    target_color: tuple[int, int, int],
    tolerance: float,
    dominant_channel: int,
) -> np.ndarray:
    pixels = image.astype(np.float64)
    target = np.asarray(target_color, dtype=np.float64)
    distance = np.linalg.norm(pixels - target, axis=2)
    channel_max = pixels.max(axis=2)
    channel_min = pixels.min(axis=2)
    saturation = channel_max - channel_min
    dominant_slack = max(15.0, float(tolerance) * 0.15)
    dominant = pixels[:, :, dominant_channel] + dominant_slack >= channel_max
    return (distance <= tolerance) & (saturation >= MIN_TRACE_SATURATION) & dominant


def blue_mask(
    image: np.ndarray,
    tolerance: float = DEFAULT_COLOR_TOLERANCE,
) -> np.ndarray:
    red = image[:, :, 0].astype(np.float64)
    green = image[:, :, 1].astype(np.float64)
    blue = image[:, :, 2].astype(np.float64)
    strict = (blue > 120) & (blue > red * 1.25) & (blue > green * 1.05)
    tolerant = trace_color_distance_mask(
        image,
        target_color=BLUE_TRACE_COLOR,
        tolerance=tolerance,
        dominant_channel=2,
    )
    return strict | tolerant


def green_mask(
    image: np.ndarray,
    tolerance: float = DEFAULT_COLOR_TOLERANCE,
) -> np.ndarray:
    red = image[:, :, 0].astype(np.float64)
    green = image[:, :, 1].astype(np.float64)
    blue = image[:, :, 2].astype(np.float64)
    strict = (green > 80) & (green > red * 1.35) & (green > blue * 1.35)
    tolerant = trace_color_distance_mask(
        image,
        target_color=GREEN_TRACE_COLOR,
        tolerance=tolerance,
        dominant_channel=1,
    )
    return strict | tolerant


def gray_signal_mask(image: np.ndarray, threshold: int) -> np.ndarray:
    gray = image.max(axis=2)
    mask = gray > threshold
    red = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    blue = image[:, :, 2].astype(np.int16)
    saturation = np.maximum.reduce([red, green, blue]) - np.minimum.reduce([red, green, blue])
    # Keep mostly gray Doppler speckle and reject colored UI legends.
    mask &= saturation < 35
    return mask


def remove_static_rows(mask: np.ndarray, occupancy_threshold: float) -> np.ndarray:
    cleaned = mask.copy()
    row_occupancy = cleaned.mean(axis=1)
    cleaned[row_occupancy > occupancy_threshold, :] = False
    return cleaned


def dilate_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask.copy()

    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    dilated = np.zeros(mask.shape, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            dilated |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return dilated


def feature_mask(
    image: np.ndarray,
    feature: str,
    gray_threshold: int,
    static_row_occupancy: float,
    color_tolerance: float,
) -> np.ndarray:
    if feature == "blue":
        mask = blue_mask(image, tolerance=color_tolerance)
        return remove_static_rows(mask, occupancy_threshold=static_row_occupancy)
    if feature == "green":
        mask = green_mask(image, tolerance=color_tolerance)
        return remove_static_rows(mask, occupancy_threshold=static_row_occupancy)

    if feature == "gray":
        return remove_static_rows(
            gray_signal_mask(image, threshold=gray_threshold),
            occupancy_threshold=static_row_occupancy,
        )

    blue = remove_static_rows(
        blue_mask(image, tolerance=color_tolerance),
        occupancy_threshold=static_row_occupancy,
    )
    green = remove_static_rows(
        green_mask(image, tolerance=color_tolerance),
        occupancy_threshold=static_row_occupancy,
    )
    if blue.sum() >= green.sum():
        return blue
    return green


def curve_mask(
    image: np.ndarray,
    feature: str,
    static_row_occupancy: float,
    color_tolerance: float,
) -> np.ndarray:
    if feature == "blue":
        return remove_static_rows(
            blue_mask(image, tolerance=color_tolerance),
            occupancy_threshold=static_row_occupancy,
        )
    if feature == "green":
        return remove_static_rows(
            green_mask(image, tolerance=color_tolerance),
            occupancy_threshold=static_row_occupancy,
        )
    blue = remove_static_rows(
        blue_mask(image, tolerance=color_tolerance),
        occupancy_threshold=static_row_occupancy,
    )
    green = remove_static_rows(
        green_mask(image, tolerance=color_tolerance),
        occupancy_threshold=static_row_occupancy,
    )
    if blue.sum() >= green.sum():
        return blue
    return green


def raw_curve_mask(
    image: np.ndarray,
    feature: str,
    color_tolerance: float,
) -> np.ndarray:
    if feature == "blue":
        return blue_mask(image, tolerance=color_tolerance)
    if feature == "green":
        return green_mask(image, tolerance=color_tolerance)

    blue = blue_mask(image, tolerance=color_tolerance)
    green = green_mask(image, tolerance=color_tolerance)
    if blue.sum() >= green.sum():
        return blue
    return green


def resolve_curve_feature(
    crops: list[np.ndarray],
    feature: str,
    static_row_occupancy: float,
    color_tolerance: float,
) -> str:
    if feature != "auto":
        return feature

    blue_total = 0
    green_total = 0
    for crop in crops:
        blue_total += int(
            remove_static_rows(
                blue_mask(crop, tolerance=color_tolerance),
                occupancy_threshold=static_row_occupancy,
            ).sum()
        )
        green_total += int(
            remove_static_rows(
                green_mask(crop, tolerance=color_tolerance),
                occupancy_threshold=static_row_occupancy,
            ).sum()
        )
    return "blue" if blue_total >= green_total else "green"


def output_trace_only(
    crops: list[np.ndarray],
    masks: list[np.ndarray],
    background: int,
) -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    for crop, mask in zip(crops, masks):
        output = np.full_like(crop, background)
        preserve = dilate_mask(mask, radius=1)
        output[preserve] = crop[preserve]
        outputs.append(output)
    return outputs


def white_axis_y(crops: list[np.ndarray]) -> int | None:
    if not crops:
        return None

    row_scores = np.zeros(crops[0].shape[0], dtype=np.float64)
    for crop in crops[: min(5, len(crops))]:
        pixels = crop.astype(np.int16)
        min_channel = pixels.min(axis=2)
        max_channel = pixels.max(axis=2)
        saturation = max_channel - min_channel
        white = (min_channel > 140) & (saturation < 30)
        row_scores += white.mean(axis=1)

    row_scores /= min(5, len(crops))
    y = int(np.argmax(row_scores))
    if row_scores[y] < 0.25:
        return None
    return y


def draw_horizontal_axis(image: np.ndarray, y: int | None, background: int) -> None:
    if y is None or y < 0 or y >= image.shape[0]:
        return
    empty = image[y].max(axis=1) <= background + 5
    image[y, empty] = np.array([190, 190, 190], dtype=np.uint8)


def cleanup_preserve_mask(
    image: np.ndarray,
    feature: str,
    gray_threshold: int,
    static_row_occupancy: float,
    color_tolerance: float,
) -> np.ndarray:
    if feature in {"auto", "blue", "green"}:
        mask = curve_mask(
            image,
            feature=feature,
            static_row_occupancy=static_row_occupancy,
            color_tolerance=color_tolerance,
        )
        return dilate_mask(mask, radius=1)
    if feature == "gray":
        return dilate_mask(gray_signal_mask(image, threshold=gray_threshold), radius=1)
    return np.zeros(image.shape[:2], dtype=bool)


def bbox_from_masks(masks: list[np.ndarray]) -> tuple[int, int, int, int]:
    xs_all: list[np.ndarray] = []
    ys_all: list[np.ndarray] = []
    for mask in masks:
        ys, xs = np.where(mask)
        if xs.size:
            xs_all.append(xs)
            ys_all.append(ys)

    if not xs_all:
        raise RuntimeError("Could not auto-detect waveform ROI from image features")

    xs = np.concatenate(xs_all)
    ys = np.concatenate(ys_all)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_from_main_waveform_band(masks: list[np.ndarray]) -> tuple[int, int, int, int] | None:
    if not masks:
        return None

    height = masks[0].shape[0]
    if any(mask.shape[0] != height for mask in masks):
        return None

    row_counts = np.zeros(height, dtype=np.int64)
    for mask in masks:
        row_counts += mask.sum(axis=1)

    active_rows = row_counts > 8
    best: tuple[int, int, int, int, int, int, int] | None = None
    start: int | None = None

    for y, is_active in enumerate(active_rows):
        if is_active and start is None:
            start = y
        last_row = y == len(active_rows) - 1
        if (not is_active or last_row) and start is not None:
            end = y if not is_active else y + 1
            xs_all: list[np.ndarray] = []
            total = 0
            for mask in masks:
                band = mask[start:end]
                band_ys, band_xs = np.where(band)
                if band_xs.size:
                    xs_all.append(band_xs)
                    total += int(band.sum())
            if xs_all:
                xs = np.concatenate(xs_all)
                min_x = int(xs.min())
                max_x = int(xs.max())
                span = max_x - min_x + 1
                if total >= 50 and span >= 50:
                    score = total * span
                    candidate = (score, min_x, start, max_x, end - 1, total, span)
                    if best is None or candidate[0] > best[0]:
                        best = candidate
            start = None

    if best is None:
        return None
    _, min_x, min_y, max_x, max_y, _, _ = best
    return min_x, min_y, max_x, max_y


def auto_roi(
    images: list[np.ndarray],
    feature: str,
    gray_threshold: int,
    static_row_occupancy: float,
    color_tolerance: float,
    x_padding: int,
    y_padding_top: int,
    y_padding_bottom: int,
) -> tuple[int, int, int, int]:
    masks = [
        feature_mask(
            image,
            feature=feature,
            gray_threshold=gray_threshold,
            static_row_occupancy=static_row_occupancy,
            color_tolerance=color_tolerance,
        )
        for image in images
    ]
    main_band = bbox_from_main_waveform_band(masks)
    if main_band is None:
        min_x, min_y, max_x, max_y = bbox_from_masks(masks)
    else:
        min_x, min_y, max_x, max_y = main_band
    image_height, image_width = images[0].shape[:2]

    x = max(0, min_x - x_padding)
    y = max(0, min_y - y_padding_top)
    right = min(image_width, max_x + x_padding + 1)
    bottom = min(image_height, max_y + y_padding_bottom + 1)
    return x, y, right - x, bottom - y


def crop_image(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = roi
    return image[y : y + height, x : x + width]


def overlap_slices_for_shift(
    previous_width: int,
    current_width: int,
    shift: int,
    min_overlap: int = 1,
) -> tuple[slice, slice, int] | None:
    if shift <= 0 or shift >= previous_width:
        return None

    overlap_width = min(previous_width - shift, current_width)
    if overlap_width < min_overlap:
        return None

    previous_slice = slice(shift, shift + overlap_width)
    current_slice = slice(0, overlap_width)
    return previous_slice, current_slice, overlap_width


def append_start_for_shift(previous_width: int, current_width: int, shift: int) -> int | None:
    slices = overlap_slices_for_shift(
        previous_width=previous_width,
        current_width=current_width,
        shift=shift,
    )
    if slices is None:
        return None

    _, _, overlap_width = slices
    return overlap_width


def automatic_max_shift(previous_width: int, current_width: int, min_overlap: int) -> int:
    return min(previous_width, current_width) - min_overlap


def f_score_for_shift(
    previous: np.ndarray,
    current: np.ndarray,
    shift: int,
    min_overlap: int,
) -> float:
    slices = overlap_slices_for_shift(
        previous_width=previous.shape[1],
        current_width=current.shape[1],
        shift=shift,
        min_overlap=min_overlap,
    )
    if slices is None:
        return -1.0

    previous_slice, current_slice, _ = slices
    previous_overlap = previous[:, previous_slice]
    current_overlap = current[:, current_slice]
    intersection = np.logical_and(previous_overlap, current_overlap).sum()
    total = previous_overlap.sum() + current_overlap.sum()
    if total == 0:
        return -1.0
    return float(2.0 * intersection / total)


def estimate_shift(
    previous: np.ndarray,
    current: np.ndarray,
    min_shift: int,
    max_shift: int | None,
    min_overlap: int,
) -> tuple[int, float]:
    resolved_max_shift = automatic_max_shift(
        previous_width=previous.shape[1],
        current_width=current.shape[1],
        min_overlap=min_overlap,
    )
    if max_shift is not None:
        resolved_max_shift = min(resolved_max_shift, max_shift)
    if min_shift > resolved_max_shift:
        raise ValueError(
            "Shift search range is outside the available overlap. "
            f"min_shift={min_shift}, max_shift={resolved_max_shift}, "
            f"min_overlap={min_overlap}."
        )

    scores = [
        (shift, f_score_for_shift(previous, current, shift, min_overlap=min_overlap))
        for shift in range(min_shift, resolved_max_shift + 1)
    ]
    return max(scores, key=lambda item: item[1])


def median_recent_shift_per_frame(values: list[float], limit: int = 8) -> float | None:
    if not values:
        return None
    recent = values[-limit:]
    return float(np.median(np.asarray(recent, dtype=np.float64)))


def stitch_warning(
    connected: bool,
    seam_reason: str | None,
    frame_gap: int,
) -> str | None:
    warnings: list[str] = []
    if not connected and seam_reason:
        warnings.append(seam_reason)
    if frame_gap > 1:
        warnings.append(f"skipped-frame recovery: gap {frame_gap}")
    if not warnings:
        return None
    return "; ".join(warnings)


def seam_check_enabled(mode: str, previous_mask: np.ndarray, current_mask: np.ndarray) -> bool:
    if mode == "off":
        return False
    if mode == "on":
        return True
    previous_count = previous_mask.sum()
    current_count = current_mask.sum()
    return previous_count >= 500 and current_count >= 500


def edge_column_points(
    mask: np.ndarray,
    start_x: int,
    end_x: int,
    prefer: str,
) -> tuple[int, np.ndarray] | None:
    start_x = max(0, start_x)
    end_x = min(mask.shape[1], end_x)
    if start_x >= end_x:
        return None

    columns = range(start_x, end_x)
    if prefer == "right":
        columns = range(end_x - 1, start_x - 1, -1)

    for x in columns:
        ys = np.flatnonzero(mask[:, x])
        if ys.size:
            return x, ys
    return None


def minimum_vertical_gap(previous_ys: np.ndarray, current_ys: np.ndarray) -> float:
    differences = np.abs(previous_ys[:, None] - current_ys[None, :])
    return float(np.min(differences))


def median_y(ys: np.ndarray) -> int:
    return int(round(float(np.median(ys))))


def feature_color(crop: np.ndarray, x: int, ys: np.ndarray) -> np.ndarray:
    pixels = crop[ys, x]
    if pixels.size == 0:
        return np.array([86, 94, 255], dtype=np.uint8)
    color = np.percentile(pixels, 80, axis=0)
    return np.clip(color, 0, 255).astype(np.uint8)


def seam_connection(
    previous_mask: np.ndarray,
    current_mask: np.ndarray,
    shift: int,
    seam_window: int,
    max_horizontal_gap: int,
    max_vertical_gap: float,
) -> tuple[bool, int | None, float | None]:
    slices = overlap_slices_for_shift(
        previous_width=previous_mask.shape[1],
        current_width=current_mask.shape[1],
        shift=shift,
    )
    if slices is None:
        return False, None, None

    _, _, overlap_width = slices
    previous_width = previous_mask.shape[1]
    strip_start = overlap_width

    previous_point = edge_column_points(
        previous_mask,
        start_x=previous_width - seam_window,
        end_x=previous_width,
        prefer="right",
    )
    current_point = edge_column_points(
        current_mask,
        start_x=strip_start,
        end_x=strip_start + seam_window,
        prefer="left",
    )
    if previous_point is None or current_point is None:
        return False, None, None

    previous_x, previous_ys = previous_point
    current_x, current_ys = current_point
    horizontal_gap = (previous_width - 1 - previous_x) + (current_x - strip_start)
    vertical_gap = minimum_vertical_gap(previous_ys, current_ys)
    connected = horizontal_gap <= max_horizontal_gap and vertical_gap <= max_vertical_gap
    return connected, int(horizontal_gap), vertical_gap


def max_blend_overlap(
    stitched: np.ndarray,
    current_crop: np.ndarray,
    append_start: int,
    blend_width: int,
) -> None:
    if blend_width <= 0:
        return

    width = min(blend_width, stitched.shape[1], append_start)
    if width <= 0:
        return

    stitched_tail = stitched[:, -width:]
    current_overlap = current_crop[:, append_start - width : append_start]
    stitched[:, -width:] = np.maximum(stitched_tail, current_overlap)


def draw_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: np.ndarray,
    width: int,
) -> None:
    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0), 1) + 1
    radius = max(0, width // 2)

    for x_float, y_float in zip(np.linspace(x0, x1, steps), np.linspace(y0, y1, steps)):
        x = int(round(float(x_float)))
        y = int(round(float(y_float)))
        x_min = max(0, x - radius)
        x_max = min(image.shape[1], x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(image.shape[0], y + radius + 1)
        image[y_min:y_max, x_min:x_max] = np.maximum(
            image[y_min:y_max, x_min:x_max],
            color,
        )


def bridge_seam(
    stitched: np.ndarray,
    previous_crop: np.ndarray,
    current_crop: np.ndarray,
    previous_mask: np.ndarray,
    current_mask: np.ndarray,
    shift: int,
    seam_window: int,
    max_vertical_gap: float,
    line_width: int,
) -> None:
    slices = overlap_slices_for_shift(
        previous_width=previous_crop.shape[1],
        current_width=current_crop.shape[1],
        shift=shift,
    )
    if slices is None:
        return

    _, _, overlap_width = slices
    previous_width = previous_crop.shape[1]
    strip_start = overlap_width
    append_start = append_start_for_shift(
        previous_width=previous_width,
        current_width=current_crop.shape[1],
        shift=shift,
    )
    if append_start is None:
        return
    appended_width = current_crop.shape[1] - append_start
    if appended_width <= 0:
        return
    stitched_width_before_append = stitched.shape[1] - appended_width

    previous_point = edge_column_points(
        previous_mask,
        start_x=previous_width - seam_window,
        end_x=previous_width,
        prefer="right",
    )
    current_point = edge_column_points(
        current_mask,
        start_x=strip_start,
        end_x=strip_start + seam_window,
        prefer="left",
    )
    if previous_point is None or current_point is None:
        return

    previous_x, previous_ys = previous_point
    current_x, current_ys = current_point
    previous_y = median_y(previous_ys)
    current_y = median_y(current_ys)
    if abs(previous_y - current_y) > max_vertical_gap:
        return

    start_x = stitched_width_before_append - previous_width + previous_x
    end_x = stitched_width_before_append + (current_x - strip_start)
    color = np.maximum(
        feature_color(previous_crop, previous_x, previous_ys),
        feature_color(current_crop, current_x, current_ys),
    )
    draw_line(
        stitched,
        start=(start_x, previous_y),
        end=(end_x, current_y),
        color=color,
        width=line_width,
    )


def mean_order_score(
    masks: list[np.ndarray],
    min_shift: int,
    max_shift: int | None,
    min_overlap: int,
) -> float:
    scores = [
        estimate_shift(
            masks[index],
            masks[index + 1],
            min_shift,
            max_shift,
            min_overlap,
        )[1]
        for index in range(len(masks) - 1)
    ]
    return float(np.mean(scores)) if scores else -1.0


def choose_order(
    paths: list[Path],
    crops: list[np.ndarray],
    masks: list[np.ndarray],
    order: str,
    min_shift: int,
    max_shift: int | None,
    min_overlap: int,
) -> tuple[list[Path], list[np.ndarray], list[np.ndarray], bool]:
    if order == "given":
        return paths, crops, masks, False
    if order == "reverse":
        return list(reversed(paths)), list(reversed(crops)), list(reversed(masks)), True

    given_score = mean_order_score(masks, min_shift, max_shift, min_overlap)
    reversed_masks = list(reversed(masks))
    reversed_score = mean_order_score(reversed_masks, min_shift, max_shift, min_overlap)
    if reversed_score > given_score:
        return (
            list(reversed(paths)),
            list(reversed(crops)),
            reversed_masks,
            True,
        )
    return paths, crops, masks, False


def static_overlay_mask(crops: list[np.ndarray], brightness_threshold: int = 35) -> np.ndarray:
    if len(crops) < 2:
        return np.zeros(crops[0].shape[:2], dtype=bool)

    stack = np.stack(crops).astype(np.float64)
    gray = stack.max(axis=3)
    mean_gray = gray.mean(axis=0)
    std_gray = gray.std(axis=0)
    return (mean_gray > brightness_threshold) & (std_gray < 3.0)


def apply_static_cleanup(
    crops: list[np.ndarray],
    preserve_masks: list[np.ndarray],
    background: int,
) -> list[np.ndarray]:
    mask = static_overlay_mask(crops)
    cleaned: list[np.ndarray] = []
    for crop, preserve_mask in zip(crops, preserve_masks):
        output = crop.copy()
        output[mask & ~preserve_mask] = background
        cleaned.append(output)
    return cleaned


def stitch_crops(
    paths: list[Path],
    crops: list[np.ndarray],
    masks: list[np.ndarray],
    min_shift: int,
    max_shift: int | None,
    min_overlap: int,
    min_score: float,
    seam_check: str,
    seam_window: int,
    max_seam_gap: int,
    max_seam_y_gap: float,
    seam_score_bypass: float,
    skip_recovery_mode: str,
    skip_recovery_tolerance: float,
    blend_width: int,
    bridge_seams: bool,
    bridge_width: int,
) -> tuple[np.ndarray, list[StitchPair], list[SkippedPair]]:
    stitched = crops[0]
    pairs: list[StitchPair] = []
    skipped: list[SkippedPair] = []
    accepted_index = 0
    shift_per_frame_values: list[float] = []

    for candidate_index in range(1, len(crops)):
        frame_gap = candidate_index - accepted_index
        typical_shift = median_recent_shift_per_frame(shift_per_frame_values)
        expected_shift = None
        effective_max_shift = max_shift
        if (
            frame_gap > 1
            and skip_recovery_mode == "expected-shift"
            and typical_shift is not None
            and max_shift is not None
        ):
            expected_shift = typical_shift * frame_gap
            effective_max_shift = max(
                max_shift,
                int(round(expected_shift + skip_recovery_tolerance)),
            )

        shift, score = estimate_shift(
            masks[accepted_index],
            masks[candidate_index],
            min_shift,
            effective_max_shift,
            min_overlap,
        )
        if score < min_score:
            skipped.append(
                SkippedPair(
                    previous=paths[accepted_index],
                    current=paths[candidate_index],
                    shift_px=shift,
                    score=score,
                    reason="low alignment score",
                )
            )
            continue

        skipped_frame_recovery = frame_gap > 1
        shift_is_plausible = True
        if skipped_frame_recovery:
            if skip_recovery_mode == "off":
                skipped.append(
                    SkippedPair(
                        previous=paths[accepted_index],
                        current=paths[candidate_index],
                        shift_px=shift,
                        score=score,
                        reason="skipped-frame recovery disabled",
                    )
                )
                continue

            if skip_recovery_mode == "expected-shift" and expected_shift is not None:
                shift_is_plausible = abs(shift - expected_shift) <= skip_recovery_tolerance
            if score < seam_score_bypass or not shift_is_plausible:
                reason = "low skipped-frame recovery score"
                if not shift_is_plausible:
                    reason = "implausible skipped-frame shift"
                skipped.append(
                    SkippedPair(
                        previous=paths[accepted_index],
                        current=paths[candidate_index],
                        shift_px=shift,
                        score=score,
                        reason=reason,
                    )
                )
                continue

        append_start = append_start_for_shift(
            previous_width=crops[accepted_index].shape[1],
            current_width=crops[candidate_index].shape[1],
            shift=shift,
        )
        if append_start is None or append_start >= crops[candidate_index].shape[1]:
            skipped.append(
                SkippedPair(
                    previous=paths[accepted_index],
                    current=paths[candidate_index],
                    shift_px=shift,
                    score=score,
                    reason="no new right-side content",
                )
            )
            continue

        connected = True
        seam_horizontal_gap: int | None = None
        seam_vertical_gap: float | None = None
        if seam_check_enabled(seam_check, masks[accepted_index], masks[candidate_index]):
            connected, seam_horizontal_gap, seam_vertical_gap = seam_connection(
                previous_mask=masks[accepted_index],
                current_mask=masks[candidate_index],
                shift=shift,
                seam_window=seam_window,
                max_horizontal_gap=max_seam_gap,
                max_vertical_gap=max_seam_y_gap,
            )
            if not connected:
                seam_reason = (
                    "seam disconnected"
                    if seam_horizontal_gap is not None
                    else "missing seam feature"
                )
                can_accept_high_score = (
                    seam_check == "auto"
                    and (
                        candidate_index == accepted_index + 1
                        or (
                            skipped_frame_recovery
                            and skip_recovery_mode != "off"
                            and shift_is_plausible
                        )
                    )
                    and score >= seam_score_bypass
                )
                if not can_accept_high_score:
                    skipped.append(
                        SkippedPair(
                            previous=paths[accepted_index],
                            current=paths[candidate_index],
                            shift_px=shift,
                            score=score,
                            reason=seam_reason,
                            seam_horizontal_gap_px=seam_horizontal_gap,
                            seam_vertical_gap_px=seam_vertical_gap,
                        )
                    )
                    continue

        max_blend_overlap(
            stitched=stitched,
            current_crop=crops[candidate_index],
            append_start=append_start,
            blend_width=blend_width,
        )
        new_strip = crops[candidate_index][:, append_start:]
        stitched = np.concatenate([stitched, new_strip], axis=1)
        if bridge_seams:
            bridge_seam(
                stitched=stitched,
                previous_crop=crops[accepted_index],
                current_crop=crops[candidate_index],
                previous_mask=masks[accepted_index],
                current_mask=masks[candidate_index],
                shift=shift,
                seam_window=seam_window,
                max_vertical_gap=max_seam_y_gap,
                line_width=bridge_width,
            )
        pairs.append(
            StitchPair(
                previous=paths[accepted_index],
                current=paths[candidate_index],
                shift_px=shift,
                score=score,
                seam_connected=connected,
                seam_horizontal_gap_px=seam_horizontal_gap,
                seam_vertical_gap_px=seam_vertical_gap,
                seam_warning=stitch_warning(
                    connected=connected,
                    seam_reason=seam_reason if not connected else None,
                    frame_gap=frame_gap,
                ),
            )
        )
        shift_per_frame_values.append(shift / frame_gap)
        accepted_index = candidate_index

    if not pairs:
        raise RuntimeError("No image pairs passed the stitch checks")

    return stitched, pairs, skipped


def directory_image_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def child_image_directories(directory: Path) -> list[Path]:
    return sorted(
        child
        for child in directory.iterdir()
        if child.is_dir() and len(directory_image_paths(child)) >= 2
    )


def series_number(path: Path) -> str | None:
    match = re.search(r"(?:series[_-]?)(\d+)$", path.name, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def experiment_output_stem_for_directory(directory: Path) -> str | None:
    number = series_number(directory)
    if number is None:
        return None
    if directory.parent.name.lower() != "raw_images":
        return None
    experiment_name = directory.parent.parent.name
    if not experiment_name:
        return None
    return f"{experiment_name}-{number}"


def output_name_for_inputs(inputs: list[Path]) -> str:
    if len(inputs) == 1 and inputs[0].is_dir():
        experiment_stem = experiment_output_stem_for_directory(inputs[0])
        if experiment_stem is not None:
            return f"{experiment_stem}.png"
        return f"{inputs[0].name}.png"
    if inputs:
        first_parent = inputs[0].parent
        if all(path.parent == first_parent for path in inputs):
            experiment_stem = experiment_output_stem_for_directory(first_parent)
            if experiment_stem is not None:
                return f"{experiment_stem}.png"
        return f"{inputs[0].parent.name or 'stitched_waveform'}.png"
    return "stitched_waveform.png"


def output_path_for_stitch(inputs: list[Path], output: Path, batch: bool) -> Path:
    if batch:
        if output.suffix:
            raise RuntimeError("Output must be a directory when stitching multiple groups")
        return output / output_name_for_inputs(inputs)

    if output.suffix:
        return output
    return output / output_name_for_inputs(inputs)


def stitch_jobs(args: argparse.Namespace) -> list[tuple[list[Path], Path]]:
    missing = [path for path in args.input if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Input path does not exist: {missing[0]}")

    files = [path for path in args.input if path.is_file()]
    directories = [path for path in args.input if path.is_dir()]
    if files and directories:
        raise RuntimeError("Do not mix image files and directories in one stitch command")

    if files:
        output_path = output_path_for_stitch(files, args.output, batch=False)
        return [(files, output_path)]

    if len(directories) == 1:
        directory = directories[0]
        if len(directory_image_paths(directory)) >= 2:
            output_path = output_path_for_stitch([directory], args.output, batch=False)
            return [([directory], output_path)]

        child_directories = child_image_directories(directory)
        if not child_directories:
            raise RuntimeError(f"No stitchable image groups found in {directory}")
        return [
            ([child], output_path_for_stitch([child], args.output, batch=True))
            for child in child_directories
        ]

    if directories:
        jobs: list[tuple[list[Path], Path]] = []
        for directory in directories:
            if len(directory_image_paths(directory)) < 2:
                raise RuntimeError(f"Directory has fewer than two supported images: {directory}")
            jobs.append(
                ([directory], output_path_for_stitch([directory], args.output, batch=True))
            )
        return jobs

    raise RuntimeError("No input images or directories were provided")


def stitch_image_group(
    inputs: list[Path],
    output_path: Path,
    args: argparse.Namespace,
) -> StitchResult:
    paths = image_paths_from_inputs(inputs)
    images = load_images(paths)
    roi = args.roi
    if roi is None:
        if not args.auto_roi:
            roi = default_roi_for_images(images)
        if roi is None:
            roi = auto_roi(
                images,
                feature=args.feature,
                gray_threshold=args.gray_threshold,
                static_row_occupancy=args.static_row_occupancy,
                color_tolerance=args.tolerance,
                x_padding=args.x_padding,
                y_padding_top=(
                    args.y_padding if args.y_padding_top is None else args.y_padding_top
                ),
                y_padding_bottom=(
                    args.y_padding if args.y_padding_bottom is None else args.y_padding_bottom
                ),
            )

    crops = [crop_image(image, roi) for image in images]
    curve_feature = resolve_curve_feature(
        crops,
        feature=args.feature,
        static_row_occupancy=args.static_row_occupancy,
        color_tolerance=args.tolerance,
    )
    masks = [
        feature_mask(
            crop,
            feature=curve_feature,
            gray_threshold=args.gray_threshold,
            static_row_occupancy=args.static_row_occupancy,
            color_tolerance=args.tolerance,
        )
        for crop in crops
    ]
    output_masks = [
        raw_curve_mask(
            crop,
            feature=curve_feature,
            color_tolerance=args.tolerance,
        )
        for crop in crops
    ]
    axis_y = white_axis_y(crops) if args.draw_axis else None
    paths, crops, masks, reversed_order = choose_order(
        paths=paths,
        crops=crops,
        masks=masks,
        order=args.order,
        min_shift=args.min_shift,
        max_shift=args.max_shift,
        min_overlap=args.min_overlap,
    )
    if reversed_order:
        output_masks = list(reversed(output_masks))

    if args.keep_static:
        output_crops = crops
    else:
        output_crops = output_trace_only(
            crops,
            masks=output_masks,
            background=args.background,
        )

    stitched, pairs, skipped = stitch_crops(
        paths=paths,
        crops=output_crops,
        masks=masks,
        min_shift=args.min_shift,
        max_shift=args.max_shift,
        min_overlap=args.min_overlap,
        min_score=args.min_score,
        seam_check=args.seam_check,
        seam_window=args.seam_window,
        max_seam_gap=args.max_seam_gap,
        max_seam_y_gap=args.max_seam_y_gap,
        seam_score_bypass=args.seam_score_bypass,
        skip_recovery_mode=args.skip_recovery_mode,
        skip_recovery_tolerance=args.skip_recovery_tolerance,
        blend_width=args.blend_width,
        bridge_seams=args.bridge_seams,
        bridge_width=args.bridge_width,
    )
    if args.draw_axis and not args.keep_static:
        draw_horizontal_axis(stitched, axis_y, background=args.background)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(stitched).save(output_path)
    return StitchResult(
        output_path=output_path,
        image_paths=paths,
        roi=roi,
        pairs=pairs,
        skipped=skipped,
        reversed_order=reversed_order,
        size=(stitched.shape[1], stitched.shape[0]),
    )


def stitch_images(args: argparse.Namespace) -> StitchResult:
    jobs = stitch_jobs(args)
    if len(jobs) != 1:
        raise RuntimeError("stitch_images expected exactly one stitch job")
    inputs, output_path = jobs[0]
    return stitch_image_group(inputs=inputs, output_path=output_path, args=args)


def stitch_all(args: argparse.Namespace) -> list[StitchResult]:
    return [
        stitch_image_group(inputs=inputs, output_path=output_path, args=args)
        for inputs, output_path in stitch_jobs(args)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stitch Vevo rolling-window waveform PNG exports into one long image.",
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
        nargs="+",
        type=Path,
        help=(
            "Input image files, one directory containing image files, multiple "
            "directories, or one parent directory containing stitchable subdirectories. "
            f"Default: ROOT/{raw_images_path().name}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output PNG path for one stitch group, or output directory for multiple "
            "stitch groups. For one group, a directory output writes <input-name>.png. "
            f"Default: ROOT/{stitched_images_path().name}"
        ),
    )
    parser.add_argument(
        "--roi",
        type=parse_roi,
        help="Optional x,y,width,height crop. Overrides built-in and auto ROI detection.",
    )
    parser.add_argument(
        "--auto-roi",
        action="store_true",
        help="Force curve-based auto ROI instead of the built-in full-screen Vevo ROI.",
    )
    parser.add_argument(
        "--feature",
        choices=["auto", "blue", "green"],
        default="auto",
        help="Curve color used for alignment. Default: auto chooses blue or green.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_COLOR_TOLERANCE,
        help="RGB color distance tolerance for blue/green trace detection. Default: 150.",
    )
    parser.add_argument(
        "--order",
        choices=["auto", "given", "reverse"],
        default="auto",
        help="Input order handling. Default: auto chooses given or reverse by match score.",
    )
    parser.add_argument("--min-shift", type=int, default=20, help="Minimum shift to search. Default: 20.")
    parser.add_argument(
        "--max-shift",
        type=int,
        help=(
            "Maximum shift to search. Default: auto, using the narrower image "
            "width minus --min-overlap for each pair."
        ),
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=50,
        help="Minimum same-width overlap region required when scoring a shift. Default: 50.",
    )
    parser.add_argument("--min-score", type=float, default=0.30, help="Minimum accepted pair score. Default: 0.30.")
    parser.add_argument(
        "--seam-check",
        choices=["auto", "on", "off"],
        default="auto",
        help="Check tail/head seam continuity. Auto allows high-score adjacent pairs. Default: auto.",
    )
    parser.add_argument(
        "--seam-window",
        type=int,
        default=16,
        help="Columns searched on each side of the stitch seam. Default: 16.",
    )
    parser.add_argument(
        "--max-seam-gap",
        type=int,
        default=8,
        help="Maximum horizontal feature gap allowed at the seam. Default: 8.",
    )
    parser.add_argument(
        "--max-seam-y-gap",
        type=float,
        default=20.0,
        help="Maximum vertical feature gap allowed at the seam. Default: 20.",
    )
    parser.add_argument(
        "--seam-score-bypass",
        type=float,
        default=0.95,
        help=(
            "In auto seam mode, accept adjacent pairs above this score even if "
            "the seam endpoint check is inconclusive. Default: 0.95."
        ),
    )
    parser.add_argument(
        "--skip-recovery-tolerance",
        type=float,
        default=25.0,
        help=(
            "Allowed shift error when recovering after skipped frames, in pixels. "
            "Used by --skip-recovery-mode expected-shift. Default: 25."
        ),
    )
    parser.add_argument(
        "--skip-recovery-mode",
        choices=["score", "expected-shift", "off"],
        default="score",
        help=(
            "How to recover after skipped frames. score uses only overlap score; "
            "expected-shift also checks recent shift trend; off disables recovery. "
            "Default: score."
        ),
    )
    parser.add_argument(
        "--blend-width",
        type=int,
        default=8,
        help="Columns of overlap to max-blend before appending each new strip. Default: 8.",
    )
    parser.add_argument(
        "--bridge-seams",
        dest="bridge_seams",
        action="store_true",
        default=False,
        help="Draw a short bridge between detected curve endpoints at each seam. Default: off.",
    )
    parser.add_argument(
        "--no-bridge-seams",
        dest="bridge_seams",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--bridge-width",
        type=int,
        default=3,
        help="Line width used for seam bridges. Default: 3.",
    )
    parser.add_argument("--x-padding", type=int, default=0, help="Auto ROI horizontal padding. Default: 0.")
    parser.add_argument("--y-padding", type=int, default=70, help="Auto ROI vertical padding. Default: 70.")
    parser.add_argument(
        "--y-padding-top",
        type=int,
        help="Auto ROI top padding. Overrides --y-padding for the top side.",
    )
    parser.add_argument(
        "--y-padding-bottom",
        type=int,
        help="Auto ROI bottom padding. Overrides --y-padding for the bottom side.",
    )
    parser.add_argument("--gray-threshold", type=int, default=18, help=argparse.SUPPRESS)
    parser.add_argument(
        "--static-row-occupancy",
        type=float,
        default=0.35,
        help="Rows above this feature occupancy are treated as static and ignored. Default: 0.35.",
    )
    parser.add_argument(
        "--keep-static",
        action="store_true",
        help="Keep original non-curve pixels and static overlays in stitched output.",
    )
    parser.add_argument(
        "--draw-axis",
        action="store_true",
        help="Draw one continuous horizontal white axis behind the stitched curve.",
    )
    parser.add_argument(
        "--background",
        type=int,
        default=0,
        help="Background value used when removing static overlays. Default: 0.",
    )
    return parser


def print_result(result: StitchResult) -> None:
    x, y, width, height = result.roi
    order = "reversed" if result.reversed_order else "given"
    print(
        f"{len(result.image_paths)} images -> {result.output_path} | "
        f"size {result.size[0]}x{result.size[1]} | "
        f"roi {x},{y},{width},{height} | order {order}"
    )
    for pair in result.pairs:
        seam_info = ""
        if pair.seam_horizontal_gap_px is not None and pair.seam_vertical_gap_px is not None:
            seam_info = (
                f" | seam dx {pair.seam_horizontal_gap_px}px"
                f" dy {pair.seam_vertical_gap_px:.1f}px"
            )
        if pair.seam_warning:
            seam_info += f" | seam warning: {pair.seam_warning}"
        print(
            f"{pair.previous.name} -> {pair.current.name} | "
            f"shift {pair.shift_px}px | score {pair.score:.4f}"
            f"{seam_info}"
        )
    for skipped in result.skipped:
        seam_info = ""
        if (
            skipped.seam_horizontal_gap_px is not None
            and skipped.seam_vertical_gap_px is not None
        ):
            seam_info = (
                f" | seam dx {skipped.seam_horizontal_gap_px}px"
                f" dy {skipped.seam_vertical_gap_px:.1f}px"
            )
        print(
            f"skipped {skipped.current.name} after {skipped.previous.name} | "
            f"shift {skipped.shift_px}px | score {skipped.score:.4f} | "
            f"{skipped.reason}"
            f"{seam_info}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.input is None:
        args.input = [raw_images_path(args.root)]
    if args.output is None:
        args.output = stitched_images_path(args.root)
    try:
        results = stitch_all(args)
    except Exception as exc:
        print(f"error: {exc}")
        return 1
    for index, result in enumerate(results):
        if index:
            print()
        print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
