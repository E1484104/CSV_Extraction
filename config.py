from __future__ import annotations

from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Edit this one path for each experiment. Command-line --input/--output can
# still override the derived defaults in each tool.
ROOT_PATH = Path(r"../20260814/Test3")

RAW_IMAGES_DIRNAME = "Raw_Images"
STITCHED_IMAGES_DIRNAME = "Stitched_Images"
RAW_DATA_DIRNAME = "Raw_Data"
NORM_DATA_DIRNAME = "Norm_Data"
BPI_PROCESSED_DIRNAME = "BPI_Processed"


def raw_images_path(root_path: Path = ROOT_PATH) -> Path:
    return root_path / RAW_IMAGES_DIRNAME


def stitched_images_path(root_path: Path = ROOT_PATH) -> Path:
    return root_path / STITCHED_IMAGES_DIRNAME


def raw_data_path(root_path: Path = ROOT_PATH) -> Path:
    return root_path / RAW_DATA_DIRNAME


def norm_data_path(root_path: Path = ROOT_PATH) -> Path:
    return root_path / NORM_DATA_DIRNAME


def bpi_processed_path(root_path: Path = ROOT_PATH) -> Path:
    return root_path / BPI_PROCESSED_DIRNAME


RAW_IMAGES_PATH = raw_images_path()
STITCHED_IMAGES_PATH = stitched_images_path()
RAW_DATA_PATH = raw_data_path()
NORM_DATA_PATH = norm_data_path()
BPI_PROCESSED_PATH = bpi_processed_path()

# Defaults for main.py: stitched images -> raw CSV data.
INPUT_PATH = STITCHED_IMAGES_PATH
OUTPUT_PATH = RAW_DATA_PATH

# Defaults for vevo_stitcher.py: raw image series -> stitched PNGs.
STITCHER_INPUT_PATH = RAW_IMAGES_PATH
STITCHER_OUTPUT_PATH = STITCHED_IMAGES_PATH

# Defaults for ultrasound_normalizer.py: raw CSV data -> normalized CSV data.
NORMALIZER_INPUT_PATH = RAW_DATA_PATH
NORMALIZER_OUTPUT_PATH = NORM_DATA_PATH

DEBUG_IMAGE_PATH: Path | None = None
