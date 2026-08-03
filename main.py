from __future__ import annotations

import sys

from cli import build_parser
from file_io import print_processed_images
from workflow import process_images
from wav_envelope import format_wav_envelope_result, process_wav


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.input.is_file() and args.input.suffix.lower() == ".wav":
            wav_result = process_wav(args)
            print(format_wav_envelope_result(wav_result))
        else:
            processed_images = process_images(args)
            print_processed_images(processed_images)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
