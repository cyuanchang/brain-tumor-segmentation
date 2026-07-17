"""Offline evaluation helpers."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from brain_tumor_segmentation.metrics import evaluate_dsc_in_data


def convert_tf_data_to_np(dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Materialize a ``tf.data`` dataset into NumPy image/mask arrays."""
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for image, mask in dataset:
        images.append(image.numpy())
        masks.append(mask.numpy())
    return np.concatenate(images, axis=0), np.concatenate(masks, axis=0)


def predict_masks(model: tf.keras.Model, images: np.ndarray) -> np.ndarray:
    """Return hard class labels from softmax predictions."""
    preds = model.predict(images)
    return np.argmax(preds, axis=-1)


def mean_tumor_dsc(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    num_classes: int,
) -> tuple[np.ndarray, float]:
    """Evaluate mean Dice on tumor class(es) for a validation dataset."""
    x_val, y_val = convert_tf_data_to_np(dataset)
    y_val = np.argmax(y_val, axis=-1)
    y_pred = predict_masks(model, x_val)
    return evaluate_dsc_in_data(y_val, y_pred, num_classes)
