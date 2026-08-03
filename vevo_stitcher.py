from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class StitchPair:
    previous: Path
    current: Path
    shift_px: int
    score: float
    seam_connected: bool
    seam_horizontal_gap_px: int | None
    seam_vertical_gap_px: float | None


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
    expected_shape: tuple[int, int, int] | None = None
    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if expected_shape is None:
            expected_shape = array.shape
        elif array.shape != expected_shape:
            raise RuntimeError(
                f"All images must have the same size. "
                f"{path} has {array.shape[1]}x{array.shape[0]}, expected "
                f"{expected_shape[1]}x{expected_shape[0]}."
            )
        images.append(array)
    return images


def blue_mask(image: np.ndarray) -> np.ndarray:
    red = image[:, :, 0].astype(np.float64)
    green = image[:, :, 1].astype(np.float64)
    blue = image[:, :, 2].astype(np.float64)
    return (blue > 120) & (blue > red * 1.25) & (blue > green * 1.05)


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
) -> np.ndarray:
    if feature == "blue":
        mask = blue_mask(image)
    elif feature == "gray":
        mask = gray_signal_mask(image, threshold=gray_threshold)
    else:
        candidate = blue_mask(image)
        if candidate.sum() >= 500:
            mask = candidate
        else:
            mask = gray_signal_mask(image, threshold=gray_threshold)
    return remove_static_rows(mask, occupancy_threshold=static_row_occupancy)


def cleanup_preserve_mask(image: np.ndarray, feature: str, gray_threshold: int) -> np.ndarray:
    blue = blue_mask(image)
    if feature in {"auto", "blue"} and blue.sum() >= 500:
        return dilate_mask(blue, radius=1)
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


def auto_roi(
    images: list[np.ndarray],
    feature: str,
    gray_threshold: int,
    static_row_occupancy: float,
    x_padding: int,
    y_padding: int,
) -> tuple[int, int, int, int]:
    masks = [
        feature_mask(
            image,
            feature=feature,
            gray_threshold=gray_threshold,
            static_row_occupancy=static_row_occupancy,
        )
        for image in images
    ]
    min_x, min_y, max_x, max_y = bbox_from_masks(masks)
    image_height, image_width = images[0].shape[:2]

    x = max(0, min_x - x_padding)
    y = max(0, min_y - y_padding)
    right = min(image_width, max_x + x_padding + 1)
    bottom = min(image_height, max_y + y_padding + 1)
    return x, y, right - x, bottom - y


def crop_image(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = roi
    return image[y : y + height, x : x + width]


def f_score_for_shift(previous: np.ndarray, current: np.ndarray, shift: int) -> float:
    if shift <= 0 or shift >= previous.shape[1]:
        return -1.0
    previous_overlap = previous[:, shift:]
    current_overlap = current[:, :-shift]
    intersection = np.logical_and(previous_overlap, current_overlap).sum()
    total = previous_overlap.sum() + current_overlap.sum()
    if total == 0:
        return -1.0
    return float(2.0 * intersection / total)


def estimate_shift(
    previous: np.ndarray,
    current: np.ndarray,
    min_shift: int,
    max_shift: int,
) -> tuple[int, float]:
    max_shift = min(max_shift, previous.shape[1] - 1)
    if min_shift > max_shift:
        raise ValueError("Shift search range is outside the ROI width")

    scores = [
        (shift, f_score_for_shift(previous, current, shift))
        for shift in range(min_shift, max_shift + 1)
    ]
    return max(scores, key=lambda item: item[1])


def seam_check_enabled(mode: str, previous_crop: np.ndarray, current_crop: np.ndarray) -> bool:
    if mode == "off":
        return False
    if mode == "on":
        return True
    return blue_mask(previous_crop).sum() >= 500 and blue_mask(current_crop).sum() >= 500


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
    width = previous_mask.shape[1]
    strip_start = width - shift

    previous_point = edge_column_points(
        previous_mask,
        start_x=width - seam_window,
        end_x=width,
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
    horizontal_gap = (width - 1 - previous_x) + (current_x - strip_start)
    vertical_gap = minimum_vertical_gap(previous_ys, current_ys)
    connected = horizontal_gap <= max_horizontal_gap and vertical_gap <= max_vertical_gap
    return connected, int(horizontal_gap), vertical_gap


def max_blend_overlap(
    stitched: np.ndarray,
    current_crop: np.ndarray,
    shift: int,
    blend_width: int,
) -> None:
    if blend_width <= 0:
        return

    crop_width = current_crop.shape[1]
    strip_start = crop_width - shift
    width = min(blend_width, stitched.shape[1], strip_start)
    if width <= 0:
        return

    stitched_tail = stitched[:, -width:]
    current_overlap = current_crop[:, strip_start - width : strip_start]
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
    crop_width = previous_crop.shape[1]
    strip_start = crop_width - shift
    stitched_width_before_append = stitched.shape[1] - shift

    previous_point = edge_column_points(
        previous_mask,
        start_x=crop_width - seam_window,
        end_x=crop_width,
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

    start_x = stitched_width_before_append - crop_width + previous_x
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
    max_shift: int,
) -> float:
    scores = [
        estimate_shift(masks[index], masks[index + 1], min_shift, max_shift)[1]
        for index in range(len(masks) - 1)
    ]
    return float(np.mean(scores)) if scores else -1.0


def choose_order(
    paths: list[Path],
    crops: list[np.ndarray],
    masks: list[np.ndarray],
    order: str,
    min_shift: int,
    max_shift: int,
) -> tuple[list[Path], list[np.ndarray], list[np.ndarray], bool]:
    if order == "given":
        return paths, crops, masks, False
    if order == "reverse":
        return list(reversed(paths)), list(reversed(crops)), list(reversed(masks)), True

    given_score = mean_order_score(masks, min_shift, max_shift)
    reversed_masks = list(reversed(masks))
    reversed_score = mean_order_score(reversed_masks, min_shift, max_shift)
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
    max_shift: int,
    min_score: float,
    seam_check: str,
    seam_window: int,
    max_seam_gap: int,
    max_seam_y_gap: float,
    blend_width: int,
    bridge_seams: bool,
    bridge_width: int,
) -> tuple[np.ndarray, list[StitchPair], list[SkippedPair]]:
    stitched = crops[0]
    pairs: list[StitchPair] = []
    skipped: list[SkippedPair] = []
    accepted_index = 0

    for candidate_index in range(1, len(crops)):
        shift, score = estimate_shift(
            masks[accepted_index],
            masks[candidate_index],
            min_shift,
            max_shift,
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

        connected = True
        seam_horizontal_gap: int | None = None
        seam_vertical_gap: float | None = None
        if seam_check_enabled(seam_check, crops[accepted_index], crops[candidate_index]):
            connected, seam_horizontal_gap, seam_vertical_gap = seam_connection(
                previous_mask=masks[accepted_index],
                current_mask=masks[candidate_index],
                shift=shift,
                seam_window=seam_window,
                max_horizontal_gap=max_seam_gap,
                max_vertical_gap=max_seam_y_gap,
            )
            if not connected:
                skipped.append(
                    SkippedPair(
                        previous=paths[accepted_index],
                        current=paths[candidate_index],
                        shift_px=shift,
                        score=score,
                        reason=(
                            "seam disconnected"
                            if seam_horizontal_gap is not None
                            else "missing seam feature"
                        ),
                        seam_horizontal_gap_px=seam_horizontal_gap,
                        seam_vertical_gap_px=seam_vertical_gap,
                    )
                )
                continue

        max_blend_overlap(
            stitched=stitched,
            current_crop=crops[candidate_index],
            shift=shift,
            blend_width=blend_width,
        )
        new_strip = crops[candidate_index][:, -shift:]
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
            )
        )
        accepted_index = candidate_index

    if not pairs:
        raise RuntimeError("No image pairs passed the stitch checks")

    return stitched, pairs, skipped


def stitch_images(args: argparse.Namespace) -> StitchResult:
    paths = image_paths_from_inputs(args.input)
    images = load_images(paths)
    roi = args.roi
    if roi is None:
        roi = auto_roi(
            images,
            feature=args.feature,
            gray_threshold=args.gray_threshold,
            static_row_occupancy=args.static_row_occupancy,
            x_padding=args.x_padding,
            y_padding=args.y_padding,
        )

    crops = [crop_image(image, roi) for image in images]
    masks = [
        feature_mask(
            crop,
            feature=args.feature,
            gray_threshold=args.gray_threshold,
            static_row_occupancy=args.static_row_occupancy,
        )
        for crop in crops
    ]
    preserve_masks = [
        cleanup_preserve_mask(
            crop,
            feature=args.feature,
            gray_threshold=args.gray_threshold,
        )
        for crop in crops
    ]
    paths, crops, masks, reversed_order = choose_order(
        paths=paths,
        crops=crops,
        masks=masks,
        order=args.order,
        min_shift=args.min_shift,
        max_shift=args.max_shift,
    )
    if reversed_order:
        preserve_masks = list(reversed(preserve_masks))

    output_crops = crops
    if not args.keep_static:
        output_crops = apply_static_cleanup(
            crops,
            preserve_masks=preserve_masks,
            background=args.background,
        )

    stitched, pairs, skipped = stitch_crops(
        paths=paths,
        crops=output_crops,
        masks=masks,
        min_shift=args.min_shift,
        max_shift=args.max_shift,
        min_score=args.min_score,
        seam_check=args.seam_check,
        seam_window=args.seam_window,
        max_seam_gap=args.max_seam_gap,
        max_seam_y_gap=args.max_seam_y_gap,
        blend_width=args.blend_width,
        bridge_seams=not args.no_bridge_seams,
        bridge_width=args.bridge_width,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(stitched).save(args.output)
    return StitchResult(
        output_path=args.output,
        image_paths=paths,
        roi=roi,
        pairs=pairs,
        skipped=skipped,
        reversed_order=reversed_order,
        size=(stitched.shape[1], stitched.shape[0]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stitch Vevo rolling-window waveform PNG exports into one long image.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="Input image files, or one directory containing image files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output stitched PNG path.",
    )
    parser.add_argument(
        "--roi",
        type=parse_roi,
        help="Optional x,y,width,height crop. Default: auto-detect from waveform features.",
    )
    parser.add_argument(
        "--feature",
        choices=["auto", "blue", "gray"],
        default="auto",
        help="Feature used for alignment. Default: auto.",
    )
    parser.add_argument(
        "--order",
        choices=["auto", "given", "reverse"],
        default="auto",
        help="Input order handling. Default: auto chooses given or reverse by match score.",
    )
    parser.add_argument("--min-shift", type=int, default=20, help="Minimum shift to search. Default: 20.")
    parser.add_argument("--max-shift", type=int, default=260, help="Maximum shift to search. Default: 260.")
    parser.add_argument("--min-score", type=float, default=0.30, help="Minimum accepted pair score. Default: 0.30.")
    parser.add_argument(
        "--seam-check",
        choices=["auto", "on", "off"],
        default="auto",
        help="Only accept pairs whose tail/head seam is connected. Default: auto for blue curves.",
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
        default=15.0,
        help="Maximum vertical feature gap allowed at the seam. Default: 15.",
    )
    parser.add_argument(
        "--blend-width",
        type=int,
        default=8,
        help="Columns of overlap to max-blend before appending each new strip. Default: 8.",
    )
    parser.add_argument(
        "--no-bridge-seams",
        action="store_true",
        help="Disable drawing a short bridge between detected curve endpoints at each seam.",
    )
    parser.add_argument(
        "--bridge-width",
        type=int,
        default=3,
        help="Line width used for seam bridges. Default: 3.",
    )
    parser.add_argument("--x-padding", type=int, default=0, help="Auto ROI horizontal padding. Default: 0.")
    parser.add_argument("--y-padding", type=int, default=70, help="Auto ROI vertical padding. Default: 70.")
    parser.add_argument("--gray-threshold", type=int, default=18, help="Gray feature brightness threshold. Default: 18.")
    parser.add_argument(
        "--static-row-occupancy",
        type=float,
        default=0.35,
        help="Rows above this feature occupancy are treated as static and ignored. Default: 0.35.",
    )
    parser.add_argument(
        "--keep-static",
        action="store_true",
        help="Keep static overlays such as baseline, logo, and axis elements in stitched output.",
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
    try:
        result = stitch_images(args)
    except Exception as exc:
        print(f"error: {exc}")
        return 1
    print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
