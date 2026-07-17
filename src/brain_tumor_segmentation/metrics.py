"""Segmentation metrics used during training and offline evaluation."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def dice_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor, epsilon: float = 1e-7) -> tf.Tensor:
    """Mean Dice over non-background classes (Keras metric-compatible)."""
    num_classes = y_true.shape[-1]
    y_true = tf.cast(y_true, tf.float32)
    y_pred_class = tf.one_hot(tf.argmax(y_pred, axis=-1), num_classes)

    y_true = y_true[..., 1:]
    y_pred_class = y_pred_class[..., 1:]

    intersection = tf.reduce_sum(y_true * y_pred_class, axis=[1, 2])
    union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred_class, axis=[1, 2])
    dice_scores = (2 * intersection + epsilon) / (union + epsilon)
    return tf.reduce_mean(dice_scores)


def get_dsc(a: np.ndarray, b: np.ndarray, epsilon: float = 1e-7) -> float:
    """Dice similarity coefficient between two binary arrays."""
    intersection = np.sum(a * b)
    return float((2 * intersection + epsilon) / (np.sum(a) + np.sum(b) + epsilon))


def evaluate_dsc_in_data(
    masks_gt: np.ndarray,
    masks_pred: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, float]:
    """Per-sample Dice scores, excluding background (class 0)."""
    num_samples = masks_gt.shape[0]
    dscs = np.zeros((num_samples, num_classes))
    for i in range(num_samples):
        for j in range(num_classes):
            dscs[i, j] = get_dsc(masks_gt[i] == j, masks_pred[i] == j)
    dscs = dscs[:, 1:]
    return dscs, float(np.mean(dscs))
