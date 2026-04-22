from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE


def _collect_labeled_image_paths(data_dir: Path) -> tuple[list[str], list[int]]:
    covid_dir = data_dir / "COVID" / "images"
    normal_dir = data_dir / "Normal" / "images"

    if not covid_dir.exists() or not normal_dir.exists():
        raise ValueError("Dataset must contain COVID/images and Normal/images folders.")

    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    covid_paths = [str(p) for p in covid_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_ext]
    normal_paths = [str(p) for p in normal_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_ext]

    if not covid_paths or not normal_paths:
        raise ValueError("Both COVID/images and Normal/images must contain image files.")

    paths = covid_paths + normal_paths
    labels = [0] * len(covid_paths) + [1] * len(normal_paths)
    return paths, labels


def _decode_and_resize(path: tf.Tensor, label: tf.Tensor, image_size: tuple[int, int]):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def create_datasets(
    data_dir: Path,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42,
):
    paths, labels = _collect_labeled_image_paths(data_dir)

    rng = np.random.default_rng(seed)
    indices = np.arange(len(paths))
    rng.shuffle(indices)

    paths_arr = np.array(paths, dtype=object)[indices]
    labels_arr = np.array(labels, dtype=np.float32)[indices]

    split_index = int((1.0 - validation_split) * len(paths_arr))
    train_paths = paths_arr[:split_index].tolist()
    train_labels = labels_arr[:split_index].tolist()
    val_paths = paths_arr[split_index:].tolist()
    val_labels = labels_arr[split_index:].tolist()

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

    train_ds = train_ds.map(
        lambda p, y: _decode_and_resize(p, y, image_size),
        num_parallel_calls=AUTOTUNE,
    )
    val_ds = val_ds.map(
        lambda p, y: _decode_and_resize(p, y, image_size),
        num_parallel_calls=AUTOTUNE,
    )

    train_ds = train_ds.shuffle(2048, seed=seed).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


def build_model(input_shape=(224, 224, 3)) -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.04),
            tf.keras.layers.RandomZoom(0.08),
        ]
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train_and_save_model(
    data_dir: Path,
    output_model_path: Path,
    image_size=(224, 224),
    batch_size: int = 32,
    epochs: int = 6,
) -> Path:
    train_ds, val_ds = create_datasets(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
    )

    model = build_model(input_shape=(image_size[0], image_size[1], 3))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    if not output_model_path.exists():
        model.save(output_model_path)

    return output_model_path
