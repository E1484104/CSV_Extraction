from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Roi:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class Component:
    pixels: np.ndarray
    area: int
    min_x: int
    max_x: int
    min_y: int
    max_y: int
    unique_x_count: int
    score: float


@dataclass(frozen=True)
class Calibration:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class ExtractResult:
    rows: list[dict[str, float]]
    target_color: tuple[int, int, int]
    roi: Roi
    component: Component


@dataclass(frozen=True)
class ProcessedImage:
    image_path: Path
    csv_path: Path
    result: ExtractResult
    plot_path: Path | None = None
