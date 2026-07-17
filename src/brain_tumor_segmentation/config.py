"""Training and data configuration (cleaned from the original Colab notebook)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    """Hyperparameters and dataset paths used by the baseline pipeline."""

    train_images_path: Path
    train_masks_path: Path
    val_images_path: Path
    val_masks_path: Path

    input_shape: tuple[int, int, int] = (120, 120, 1)
    num_classes: int = 2
    batch_size: int = 16
    epochs: int = 20
    seed: int = 42

    # Subset for faster iteration (set to None to use the full split).
    max_samples: int | None = 500

    # Slice thickness used when converting mask area to volume (mL).
    slice_thickness_mm: float = 3.5

    extra: dict = field(default_factory=dict)

    @classmethod
    def from_data_root(cls, data_root: str | Path, **kwargs) -> "TrainConfig":
        """Build paths assuming ``train/images``, ``train/masks``, ``val/...`` layout."""
        root = Path(data_root)
        return cls(
            train_images_path=root / "train" / "images",
            train_masks_path=root / "train" / "masks",
            val_images_path=root / "val" / "images",
            val_masks_path=root / "val" / "masks",
            **kwargs,
        )
