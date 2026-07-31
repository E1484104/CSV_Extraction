from __future__ import annotations

import csv
from pathlib import Path

from config import IMAGE_SUFFIXES
from models import ProcessedImage


def image_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            path for path in input_path.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def output_path_for(input_image: Path, input_root: Path, output: Path) -> Path:
    if input_root.is_file():
        return output
    return output / f"{input_image.stem}.csv"


def write_csv(csv_path: Path, rows: list[dict[str, float]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["x_norm", "y_norm", "x_px", "y_px"]
    if rows and "x_value" in rows[0]:
        fieldnames.extend(["x_value", "y_value"])

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: f"{row[field]:.8g}" for field in fieldnames})


def format_processed_image(processed: ProcessedImage) -> str:
    result = processed.result
    color = ",".join(str(channel) for channel in result.target_color)
    return (
        f"{processed.image_path} -> {processed.csv_path} | "
        f"{len(result.rows)} points | target RGB {color} | "
        f"bbox x={result.roi.x + result.component.min_x}.."
        f"{result.roi.x + result.component.max_x}, "
        f"y={result.roi.y + result.component.min_y}.."
        f"{result.roi.y + result.component.max_y}"
    )


def print_processed_images(processed_images: list[ProcessedImage]) -> None:
    for processed in processed_images:
        print(format_processed_image(processed))
