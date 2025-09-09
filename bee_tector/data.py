"""
This module provides functions to load image datasets into TensorFlow
tf.data.Dataset objects, ready for training, validation and testing.

Functions
---------
load_datasets()
    Loads train, val and test datasets
"""

import os

from tensorflow.keras.utils import image_dataset_from_directory

from bee_tector.config import SUBSPECIES_DATA_DIR, IMAGE_SIZE, BATCH_SIZE, SEED


def load_datasets(data_dir=SUBSPECIES_DATA_DIR):
    """
    Load train, validation, and test datasets from a directory.

    Parameters
    ----------
    data_dir : str, optional
        Path to the dataset root directory. Defaults to SUBSPECIES_DATA_DIR

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
