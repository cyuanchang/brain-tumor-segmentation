"""tf.data loading and preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf


def preprocess_data(
    image_path: tf.Tensor,
    mask_path: tf.Tensor,
    input_shape: tuple[int, int, int],
    num_classes: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Open, decode, binarize, resize, and one-hot encode an image/mask pair.

    Mask values in the BraTS-derived PNGs are ``{0, 50, 100, 150}``. This
    baseline collapses all non-zero classes into a single tumor class.
    """
    image = tf.io.read_file(image_path)
    mask = tf.io.read_file(mask_path)

    image = tf.image.decode_jpeg(image, channels=1)
    mask = tf.image.decode_png(mask, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Binary tumor vs background (optional: mask = tf.math.divide(mask, 50)).
    mask = tf.math.greater(mask, 0)
    mask = tf.cast(mask, tf.int32)

    image = tf.image.resize(image, (input_shape[0], input_shape[1]))
    mask = tf.image.resize(
        mask,
        (input_shape[0], input_shape[1]),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )
    mask = tf.one_hot(mask, depth=num_classes)
    mask = tf.squeeze(mask, axis=-2)
    return image, mask


def create_data_pipeline(
    images_path: str | Path,
    masks_path: str | Path,
    input_shape: tuple[int, int, int],
    num_classes: int,
    seed: int,
    batch_size: int,
    max_samples: int | None = 500,
) -> tf.data.Dataset:
    """Build a shuffled, batched, prefetched ``tf.data`` pipeline."""
    images = tf.data.Dataset.list_files(str(Path(images_path) / "*.jpg"), seed=seed)
    masks = tf.data.Dataset.list_files(str(Path(masks_path) / "*.png"), seed=seed)
    dataset = tf.data.Dataset.zip((images, masks))

    if max_samples is not None:
        dataset = dataset.take(max_samples)

    dataset = dataset.map(
        lambda x, y: preprocess_data(x, y, input_shape, num_classes),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    cardinality = dataset.cardinality()
    if cardinality == tf.data.UNKNOWN_CARDINALITY:
        shuffle_buffer = max_samples or 1024
    else:
        shuffle_buffer = int(cardinality.numpy())

    dataset = dataset.shuffle(buffer_size=max(shuffle_buffer, 1), seed=seed)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
