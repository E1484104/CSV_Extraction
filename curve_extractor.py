from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from models import Calibration, Component, ExtractResult, Roi
from parsing import parse_calibration, parse_roi


def color_distance_mask(
    rgb: np.ndarray,
    target_color: tuple[int, int, int],
    tolerance: float,
) -> np.ndarray:
    target = np.array(target_color, dtype=np.float32)
    distance = np.linalg.norm(rgb.astype(np.float32) - target, axis=2)
    return distance <= tolerance


def neighbor_count(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    counts = np.zeros(mask.shape, dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            if dx == 1 and dy == 1:
                continue
            counts += padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return counts


def clean_mask(mask: np.ndarray, min_neighbors: int) -> np.ndarray:
    if min_neighbors <= 0:
        return mask
    return mask & (neighbor_count(mask) >= min_neighbors)


def connected_components(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[np.ndarray] = []
    starts = np.argwhere(mask)

    for start_y, start_x in starts:
        if visited[start_y, start_x]:
            continue

        queue: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []

        while queue:
            y, x = queue.popleft()
            pixels.append((x, y))

            for ny in (y - 1, y, y + 1):
                if ny < 0 or ny >= height:
                    continue
                for nx in (x - 1, x, x + 1):
                    if nx < 0 or nx >= width or (nx == x and ny == y):
                        continue
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

        if len(pixels) >= min_area:
            components.append(np.array(pixels, dtype=np.int32))

    return components


def score_component(pixels: np.ndarray, roi_width: int, roi_height: int) -> Component:
    xs = pixels[:, 0]
    ys = pixels[:, 1]
    min_x = int(xs.min())
    max_x = int(xs.max())
    min_y = int(ys.min())
    max_y = int(ys.max())
    unique_x_count = int(np.unique(xs).size)
    area = int(pixels.shape[0])

    x_span = max_x - min_x + 1
    y_span = max_y - min_y + 1
    span_ratio = x_span / max(1, roi_width)
    coverage_ratio = unique_x_count / max(1, x_span)
    height_ratio = y_span / max(1, roi_height)

    flat_penalty = 0.35 if height_ratio < 0.01 else 1.0
    score = (
        area
        * math.sqrt(max(1, x_span))
        * max(0.05, span_ratio)
        * max(0.05, coverage_ratio)
        * flat_penalty
    )

    return Component(
        pixels=pixels,
        area=area,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        unique_x_count=unique_x_count,
        score=score,
    )


def select_component(
    mask: np.ndarray,
    min_area: int,
    min_span_ratio: float,
    allow_flat: bool,
) -> Component:
    components = connected_components(mask, min_area=min_area)
    if not components:
        raise RuntimeError("No colored curve-like component was found")

    height, width = mask.shape
    scored = [score_component(component, width, height) for component in components]
    candidates = [
        component
        for component in scored
        if (component.max_x - component.min_x + 1) / max(1, width) >= min_span_ratio
    ]
    if not allow_flat:
        candidates = [
            component
            for component in candidates
            if component.max_y - component.min_y + 1 >= max(3, round(height * 0.005))
        ]
    if not candidates:
        candidates = scored

    return max(candidates, key=lambda component: component.score)


def auto_candidate_colors(
    rgb: np.ndarray,
    min_brightness: int,
    min_saturation: int,
    max_candidates: int,
) -> list[tuple[int, int, int]]:
    rgb_int = rgb.astype(np.int16)
    channel_max = rgb_int.max(axis=2)
    channel_min = rgb_int.min(axis=2)
    saturation = channel_max - channel_min
    candidate_mask = (channel_max >= min_brightness) & (saturation >= min_saturation)
    pixels = rgb[candidate_mask]

    if pixels.size == 0:
        raise RuntimeError(
            "Auto color detection found no bright saturated pixels. "
            "Try --target-color or lower --min-brightness/--min-saturation."
        )

    bins = (pixels // 32).astype(np.int32)
    keys = bins[:, 0] * 64 + bins[:, 1] * 8 + bins[:, 2]
    counts = np.bincount(keys, minlength=512)
    top_keys = np.argsort(counts)[::-1]

    colors: list[tuple[int, int, int]] = []
    for key in top_keys:
        if counts[key] == 0:
            break
        in_bin = keys == key
        color = tuple(int(round(value)) for value in pixels[in_bin].mean(axis=0))
        colors.append(color)
        if len(colors) >= max_candidates:
            break

    return colors


def build_mask_for_color(
    rgb: np.ndarray,
    color: tuple[int, int, int],
    tolerance: float,
    min_neighbors: int,
) -> np.ndarray:
    mask = color_distance_mask(rgb, color, tolerance)
    return clean_mask(mask, min_neighbors=min_neighbors)


def choose_mask_and_color(
    rgb: np.ndarray,
    target_color: tuple[int, int, int] | None,
    tolerance: float,
    min_neighbors: int,
    min_area: int,
    min_span_ratio: float,
    allow_flat: bool,
    min_brightness: int,
    min_saturation: int,
    auto_candidates: int,
) -> tuple[np.ndarray, tuple[int, int, int], Component]:
    if target_color is not None:
        mask = build_mask_for_color(rgb, target_color, tolerance, min_neighbors)
        component = select_component(mask, min_area, min_span_ratio, allow_flat)
        return mask, target_color, component

    best: tuple[np.ndarray, tuple[int, int, int], Component] | None = None
    for color in auto_candidate_colors(
        rgb,
        min_brightness=min_brightness,
        min_saturation=min_saturation,
        max_candidates=auto_candidates,
    ):
        mask = build_mask_for_color(rgb, color, tolerance, min_neighbors)
        try:
            component = select_component(mask, min_area, min_span_ratio, allow_flat)
        except RuntimeError:
            continue
        if best is None or component.score > best[2].score:
            best = (mask, color, component)

    if best is None:
        raise RuntimeError("Auto color detection did not produce a usable curve component")
    return best


def component_mask(shape: tuple[int, int], component: Component) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[component.pixels[:, 1], component.pixels[:, 0]] = True
    return mask


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    smoothed = np.empty_like(values, dtype=np.float64)
    for index in range(values.size):
        smoothed[index] = np.median(padded[index : index + window])
    return smoothed


def trace_component(
    component: Component,
    roi: Roi,
    sample_step: int,
    smooth_window: int,
    calibration: Calibration | None,
) -> list[dict[str, float]]:
    pixels = component.pixels
    min_x = component.min_x
    max_x = component.max_x
    min_y = component.min_y
    max_y = component.max_y

    rows: list[dict[str, float]] = []
    for start_x in range(min_x, max_x + 1, sample_step):
        end_x = min(start_x + sample_step - 1, max_x)
        in_window = (pixels[:, 0] >= start_x) & (pixels[:, 0] <= end_x)
        window_pixels = pixels[in_window]
        if window_pixels.size == 0:
            continue
        x_roi = float(np.median(window_pixels[:, 0]))
        y_roi = float(np.median(window_pixels[:, 1]))
        rows.append({"x_roi": x_roi, "y_roi": y_roi})

    if not rows:
        raise RuntimeError("Selected component contains no trace points")

    y_values = np.array([row["y_roi"] for row in rows], dtype=np.float64)
    y_values = rolling_median(y_values, smooth_window)
    for row, y_value in zip(rows, y_values):
        row["y_roi"] = float(y_value)

    x_range = max(1.0, float(max_x - min_x))
    y_range = max(1.0, float(max_y - min_y))
    output_rows: list[dict[str, float]] = []
    for row in rows:
        x_roi = row["x_roi"]
        y_roi = row["y_roi"]
        x_fraction = (x_roi - min_x) / x_range
        y_fraction = (max_y - y_roi) / y_range
        output_row = {
            "x_px": x_roi + roi.x,
            "y_px": y_roi + roi.y,
        }
        if calibration is not None:
            output_row["x_value"] = calibration.x_min + x_fraction * (
                calibration.x_max - calibration.x_min
            )
            output_row["y_value"] = calibration.y_min + y_fraction * (
                calibration.y_max - calibration.y_min
            )
        output_rows.append(output_row)

    return output_rows


def extract_curve(image_path: Path, args: argparse.Namespace) -> ExtractResult:
    image = Image.open(image_path).convert("RGB")
    rgb_full = np.array(image)
    image_height, image_width = rgb_full.shape[:2]
    roi = (
        parse_roi(args.roi, image_width, image_height)
        if args.roi
        else Roi(0, 0, image_width, image_height)
    )
    rgb = rgb_full[roi.y : roi.y2, roi.x : roi.x2]
    calibration = parse_calibration(args)
    mask, target_color, component = choose_mask_and_color(
        rgb,
        target_color=args.target_color,
        tolerance=args.tolerance,
        min_neighbors=args.min_neighbors,
        min_area=args.min_area,
        min_span_ratio=args.min_span_ratio,
        allow_flat=args.allow_flat,
        min_brightness=args.min_brightness,
        min_saturation=args.min_saturation,
        auto_candidates=args.auto_candidates,
    )
    trace_rows = trace_component(
        component,
        roi=roi,
        sample_step=max(1, args.sample_step),
        smooth_window=max(1, args.smooth_window),
        calibration=calibration,
    )
    if args.debug_image:
        write_debug_image(
            image=image,
            debug_path=Path(args.debug_image),
            roi=roi,
            mask=component_mask(mask.shape, component),
            component=component,
        )
    return ExtractResult(
        rows=trace_rows,
        target_color=target_color,
        roi=roi,
        component=component,
    )


def write_debug_image(
    image: Image.Image,
    debug_path: Path,
    roi: Roi,
    mask: np.ndarray,
    component: Component,
) -> None:
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    output = image.copy()
    overlay = Image.new("RGBA", output.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    ys, xs = np.where(mask)
    for x, y in zip(xs, ys):
        draw.point((int(x + roi.x), int(y + roi.y)), fill=(255, 40, 40, 210))

    draw.rectangle(
        [
            roi.x + component.min_x,
            roi.y + component.min_y,
            roi.x + component.max_x,
            roi.y + component.max_y,
        ],
        outline=(255, 220, 0, 255),
        width=2,
    )
    draw.rectangle([roi.x, roi.y, roi.x2 - 1, roi.y2 - 1], outline=(0, 200, 255, 255), width=2)
    Image.alpha_composite(output.convert("RGBA"), overlay).convert("RGB").save(debug_path)
