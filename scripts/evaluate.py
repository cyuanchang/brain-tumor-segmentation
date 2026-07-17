#!/usr/bin/env python
"""Evaluate a trained model: Dice on slices + patient-level volume agreement."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from brain_tumor_segmentation.config import TrainConfig
from brain_tumor_segmentation.data import create_data_pipeline
from brain_tumor_segmentation.evaluate import mean_tumor_dsc
from brain_tumor_segmentation.metrics import dice_accuracy
from brain_tumor_segmentation.viz import plot_volume_agreement
from brain_tumor_segmentation.volume import (
    calculate_volumes,
    get_list_of_patients_with_their_slices,
    list_sorted_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--full-dataset", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_samples = None if args.full_dataset else args.max_samples
    config = TrainConfig.from_data_root(
        args.data_root,
        max_samples=max_samples,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={"dice_accuracy": dice_accuracy},
        compile=False,
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

    # Re-attach metric only if needed for predict; DSC is computed offline.
    _, mean_dsc = mean_tumor_dsc(model, val_dataset, config.num_classes)
    print(f"Mean tumor Dice (validation subset): {mean_dsc:.4f}")

    images_list = list_sorted_images(config.val_images_path)
    patients = get_list_of_patients_with_their_slices(images_list)
    volumes_gt, volumes_pred = calculate_volumes(
        patients,
        model,
        config.val_images_path,
        config.val_masks_path,
        config.input_shape,
        config.slice_thickness_mm,
    )
    r = plot_volume_agreement(volumes_gt, volumes_pred)
    print(f"Pearson r (predicted vs GT volume): {r:.2f}")


if __name__ == "__main__":
    main()
