from __future__ import annotations

import argparse
from pathlib import Path

from curve_extractor import extract_curve
from file_io import image_inputs, output_path_for, write_csv
from models import ProcessedImage


def process_image(
    image_path: Path,
    input_root: Path,
    output: Path,
    args: argparse.Namespace,
) -> ProcessedImage:
    csv_path = output_path_for(image_path, input_root, output)
    result = extract_curve(image_path, args)
    write_csv(csv_path, result.rows)
    return ProcessedImage(image_path=image_path, csv_path=csv_path, result=result)


def process_images(args: argparse.Namespace) -> list[ProcessedImage]:
    inputs = image_inputs(args.input)
    if not inputs:
        raise RuntimeError(f"No supported images found in {args.input}")

    return [
        process_image(
            image_path=image_path,
            input_root=args.input,
            output=args.output,
            args=args,
        )
        for image_path in inputs
    ]
