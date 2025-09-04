# histoslice/models.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class Settings(BaseModel):
    # Input / Output
    paths: List[Path]
    parent_dir: Path
    backend: Optional[str] = None  # None=automatic

    # Tile extraction
    level: int = 0
    width: int = 640
    height: Optional[int] = None
    overlap: float = 0.0
    max_background: float = 0.75
    in_bounds: bool = False

    # Tissue detection
    threshold: Optional[int] = None
    multiplier: float = 1.05
    tissue_level: Optional[int] = None
    max_dimension: int = 8192
    sigma: float = 1.0

    # Saving
    save_metrics: bool = False
    save_masks: bool = False
    save_thumbnails: bool = False
    overwrite: bool = False
    overwrite_unfinished: bool = False
    image_format: str = "jpeg"
    quality: int = Field(default=80, ge=0, le=100)
    num_workers: Optional[int] = None
    use_csv: bool = False

    @model_validator(mode="after")
    def _strict_ranges(self):
        if not (0.0 <= self.overlap <= 1.0):
            raise ValueError("--overlap must be in [0,1].")
        if not (0.0 < self.max_background < 1.0):
            raise ValueError("--max-background must be in (0,1).")
        if self.threshold is not None and not (0 <= self.threshold <= 255):
            raise ValueError("--threshold must be in [0,255].")
        return self
