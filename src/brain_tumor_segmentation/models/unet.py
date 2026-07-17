"""Baseline 2-level U-Net used in the Capstone Colab notebook."""

from __future__ import annotations

import tensorflow as tf

from brain_tumor_segmentation.metrics import dice_accuracy


def build_unet(
    input_shape: tuple[int, int, int] = (120, 120, 1),
    num_classes: int = 2,
    compile_model: bool = True,
) -> tf.keras.Model:
    """Construct the baseline U-Net (encoder → bottleneck → decoder)."""
    input_layer = tf.keras.layers.Input(shape=input_shape)

    # Encoder — level 1
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(input_layer)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    skip_1 = x
    x = tf.keras.layers.MaxPooling2D(2, strides=2)(x)

    # Encoder — level 2
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    skip_2 = x
    x = tf.keras.layers.MaxPooling2D(2, strides=2)(x)

    # Bottleneck
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding="same")(x)

    # Decoder — level 2
    x = tf.keras.layers.concatenate([x, skip_2])
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding="same")(x)

    # Decoder — level 1
    x = tf.keras.layers.concatenate([x, skip_1])
    x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same")(x)

    output_layer = tf.keras.layers.Conv2D(num_classes, 1, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name="baseline_unet")

    if compile_model:
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[dice_accuracy],
        )
    return model
