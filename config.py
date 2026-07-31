from __future__ import annotations

from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Edit these paths for normal runs. Command-line --input/--output can still override them.
INPUT_PATH = Path(r"../WebPlotDigitizerTest.png")
OUTPUT_PATH = Path(r"../Test.csv")
DEBUG_IMAGE_PATH: Path | None = None
