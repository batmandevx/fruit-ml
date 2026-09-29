"""tf.data input pipeline with augmentation. Images stay in [0, 255]; every
backbone includes its own rescaling layer."""
from pathlib import Path

import tensorflow as tf

from config import BATCH_SIZE, PROCESSED_DIR, SEED

AUTOTUNE = tf.data.AUTOTUNE


def build_augmenter():
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.25, fill_mode="reflect"),
            tf.keras.layers.RandomZoom((-0.15, 0.25), fill_mode="reflect"),
            tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="reflect"),
            tf.keras.layers.RandomBrightness(0.15, value_range=(0, 255)),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="augment",
    )


def load_split(split, img_size, shuffle=False, batch_size=BATCH_SIZE, data_dir=PROCESSED_DIR):
    return tf.keras.utils.image_dataset_from_directory(
        Path(data_dir) / split,
        labels="inferred",
        label_mode="int",
        image_size=(img_size, img_size),
        crop_to_aspect_ratio=True,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=SEED,
    )


def get_datasets(img_size, batch_size=BATCH_SIZE, one_hot=False, data_dir=PROCESSED_DIR):
    train = load_split("train", img_size, shuffle=True, batch_size=batch_size, data_dir=data_dir)
    val = load_split("val", img_size, batch_size=batch_size, data_dir=data_dir)
    test = load_split("test", img_size, batch_size=batch_size, data_dir=data_dir)
    class_names = train.class_names
    assert val.class_names == class_names == test.class_names
    n = len(class_names)
    enc = (lambda y: tf.one_hot(y, n)) if one_hot else (lambda y: y)

    augment = build_augmenter()
    train = train.map(lambda x, y: (tf.clip_by_value(augment(x, training=True), 0, 255), enc(y)),
                      num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val = val.map(lambda x, y: (x, enc(y))).cache().prefetch(AUTOTUNE)
    test = test.map(lambda x, y: (x, enc(y))).cache().prefetch(AUTOTUNE)
    return train, val, test, class_names
