from __future__ import annotations

import sys

from cli import build_parser
from config import raw_data_path, stitched_images_path
from file_io import print_processed_images
from workflow import process_images


def apply_root_defaults(args) -> None:
    if args.input is None:
        args.input = stitched_images_path(args.root)
    if args.output is None:
        args.output = raw_data_path(args.root)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_root_defaults(args)

    try:
        processed_images = process_images(args)
        print_processed_images(processed_images)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
