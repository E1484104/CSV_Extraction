from __future__ import annotations

import sys

from cli import build_parser
from config import raw_data_path, stitched_images_path
from file_io import image_inputs, print_processed_images
from workflow import process_images
from wav_envelope import format_wav_envelope_result, process_wav, process_wavs, wav_inputs


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
        if args.input.is_file() and args.input.suffix.lower() == ".wav":
            wav_result = process_wav(args)
            print(format_wav_envelope_result(wav_result))
        elif args.input.is_dir():
            processed_images = []
            wav_results = []
            image_paths = image_inputs(args.input)
            wav_paths = wav_inputs(args.input)
            if image_paths:
                processed_images = process_images(args)
                print_processed_images(processed_images)
            if wav_paths:
                wav_results = process_wavs(args)
                for wav_result in wav_results:
                    print(format_wav_envelope_result(wav_result))
            if not processed_images and not wav_results:
                raise RuntimeError(f"No supported image or WAV files found in {args.input}")
        else:
            processed_images = process_images(args)
            print_processed_images(processed_images)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
