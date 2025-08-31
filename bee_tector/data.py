"""
Dataset loading utilities for BeeTector.

This module provides functions to load Bombus image datasets into TensorFlow
`tf.data.Dataset` objects, ready for training, validation and testing.

Functions
---------
load_datasets(data_dir=FULL_DATA_DIR)
    Loads train, validation, and test datasets from a directory containing
    'train', 'val', and 'test' subfolders.
"""

import os
import random

import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

from bee_tector.config import FULL_DATA_DIR, IMAGE_SIZE, BATCH_SIZE, SEED


def load_datasets(data_dir=FULL_DATA_DIR):
    """
    Load train, validation, and test datasets from a directory.

    Parameters
    ----------
    data_dir : str, optional
        Path to the dataset root directory. Defaults to `FULL_DATA_DIR`.

    Returns
    -------
    train_ds : tf.data.Dataset
        Training dataset
    val_ds : tf.data.Dataset
        Validation dataset
    test_ds : tf.data.Dataset
        Test dataset
    """
    train_ds = image_dataset_from_directory(
        os.path.join(data_dir, 'train'),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    val_ds = image_dataset_from_directory(
        os.path.join(data_dir, 'val'),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    test_ds = image_dataset_from_directory(
        os.path.join(data_dir, 'test'),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_ds, val_ds, test_ds


def load_selected_classes(data_dir=FULL_DATA_DIR,
                          wanted_classes=None,
                          image_size=IMAGE_SIZE,
                          batch_size=BATCH_SIZE
                          ):
    """
    Load train/val/test datasets with the specified classes.

    Parameters
    ----------
    data_dir : str
        Root dataset directory
    wanted_classes : list[str]
        List of class folder names to include.
    image_size : tuple[int,int]
        Target image size for resizing.
    batch_size : int
        Batch size for datasets.

    Returns
    -------
    train_ds, val_ds, test_ds : tf.data.Dataset
    """
    train_ds = image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        image_size=image_size,
        batch_size=batch_size,
        class_names=wanted_classes,
        shuffle=True,
        seed=SEED
    )

    val_ds = image_dataset_from_directory(
        os.path.join(data_dir, "val"),
        image_size=image_size,
        batch_size=batch_size,
        class_names=wanted_classes,
        shuffle=True,
        seed=SEED
    )

    test_ds = image_dataset_from_directory(
        os.path.join(data_dir, "test"),
        image_size=image_size,
        batch_size=batch_size,
        class_names=wanted_classes,
        shuffle=False
    )

    return train_ds, val_ds, test_ds


def undersample_dataset(data_dir=FULL_DATA_DIR,
                    image_size=IMAGE_SIZE,
                    batch_size=BATCH_SIZE,
                    seed=SEED,
                    class_names=None):
    """
    Return a balanced training dataset by undersampling to the smallest class.
    If class_names is given, only those classes are included.

    Also returns a mapping {label_id: class_name}.
    """
    train_dir = os.path.join(data_dir, "train")
    all_classes = sorted(os.listdir(train_dir))
    classes = class_names or all_classes

    # collect files per class
    files_per_class = []
    for i, class_name in enumerate(classes):
        class_path = os.path.join(train_dir, class_name)
        files = [
            os.path.join(class_path, f) for f in os.listdir(class_path)
            if f.lower().endswith(".jpg")
        ]
        files_per_class.append((i, files))

    # find smallest class size
    min_count = min(len(f) for _, f in files_per_class)

    # undersample each class
    paths, labels = [], []
    for label, files in files_per_class:
        sample = random.sample(files, min_count)
        paths.extend(sample)
        labels.extend([label] * min_count)

    # build dataset
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size)
        return img, label

    ds = ds.shuffle(len(paths), seed=seed)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    id_to_class = {i: cname for i, cname in enumerate(classes)}

    return ds, id_to_class
