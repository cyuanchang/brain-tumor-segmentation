#!/usr/bin/env python
"""Train the baseline U-Net on a BraTS-style image/mask directory layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from brain_tumor_segmentation.config import TrainConfig
from brain_tumor_segmentation.data import create_data_pipeline
from brain_tumor_segmentation.models import build_unet
from brain_tumor_segmentation.viz import plot_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root with train/images, train/masks, val/images, val/masks",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--full-dataset", action="store_true", help="Disable max_samples cap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for saved model weights",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_samples = None if args.full_dataset else args.max_samples
    config = TrainConfig.from_data_root(
        args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=max_samples,
        seed=args.seed,
    )

    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)

    train_dataset = create_data_pipeline(
        config.train_images_path,
        config.train_masks_path,
        config.input_shape,
        config.num_classes,
        config.seed,
        config.batch_size,
        config.max_samples,
    )
    val_dataset = create_data_pipeline(
        config.val_images_path,
        config.val_masks_path,
        config.input_shape,
        config.num_classes,
        config.seed,
        config.batch_size,
        config.max_samples,
    )

    model = build_unet(config.input_shape, config.num_classes)
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=config.epochs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "baseline_unet.keras"
    model.save(model_path)
    print(f"Saved model to {model_path}")

    plot_history(history, "loss")
    plot_history(history, "dice_accuracy")


if __name__ == "__main__":
    main()
