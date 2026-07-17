"""Patient-level tumor volume estimation from slice-wise masks."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.transform import resize


def get_list_of_patients_with_their_slices(images_list: list[str]) -> list[list]:
    """Group sorted slice filenames by patient ID (``BraTS18_XXXX`` prefix)."""
    patients_list: list[list] = []
    patient_id: str | list = []
    patient_slices: list[str] = []
    patient_name = "placeholder"

    for image_file in images_list:
        parts = image_file.split("_")
        new_patient_name = parts[0] + "_" + parts[1]
        if new_patient_name != patient_name:
            patients_list.append([patient_id, patient_slices])
            patient_id = new_patient_name
            patient_slices = []
            patient_name = new_patient_name
        patient_slices.append(image_file)

    return patients_list[1:]


def calculate_volumes(
    patients_list: list[list],
    trained_model: tf.keras.Model,
    images_path: str | Path,
    masks_path: str | Path,
    input_shape: tuple[int, int, int],
    slice_thickness_mm: float = 3.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate ground-truth and predicted tumor volumes (mL) per patient."""
    images_path = Path(images_path)
    masks_path = Path(masks_path)
    volumes_gt = np.zeros((len(patients_list), 1))
    volumes_pred = np.zeros((len(patients_list), 1))

    for k, patient_list in enumerate(patients_list):
        volume_gt = 0.0
        volume_pred = 0.0
        for mri_slice in patient_list[1]:
            image_path = images_path / mri_slice
            mask_path = masks_path / (mri_slice[:-4] + ".png")

            image = np.array(
                Image.open(image_path).resize((input_shape[1], input_shape[0]))
            ) / 255.0
            image = np.expand_dims(image, axis=(0, 3))

            mask_gt = (np.array(Image.open(mask_path)) / 50.0 > 0).astype(float)

            mask_pred = trained_model.predict(image, verbose=0)
            mask_pred = np.argmax(mask_pred, axis=-1)[0]
            mask_pred = (mask_pred > 0).astype(float)
            mask_pred = resize(mask_pred, mask_gt.shape, order=0)

            area_gt = np.sum(mask_gt == 1)
            area_pred = np.sum(mask_pred == 1)
            volume_gt += (area_gt * slice_thickness_mm) / 1000.0
            volume_pred += (area_pred * slice_thickness_mm) / 1000.0

        volumes_gt[k, 0] = volume_gt
        volumes_pred[k, 0] = volume_pred

    return volumes_gt, volumes_pred


def list_sorted_images(images_path: str | Path) -> list[str]:
    """Return sorted image filenames from a directory."""
    files = os.listdir(images_path)
    files.sort()
    return files
