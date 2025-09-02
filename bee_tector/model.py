import numpy as np
import time

from colorama import Fore, Style

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow import keras
from keras import Sequential, Input, layers
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import regularizers
from keras.models import load_model

from bee_tector.config import IMAGE_SIZE, MODELS_DIR

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")



def initialize_model(shape=IMAGE_SIZE + (3,), num_classes=12):
    """
    Initialize a CNN classifier for Bombus images.

    Parameters
    ----------
    shape : tuple of int, optional
        Input image shape (height, width, channels).
        Defaults to IMAGE_SIZE + (3,) from config.
    num_classes : int, optional
        Number of output classes for classification. Default is 12.

    Returns
    -------
    keras.Model
        Uncompiled Keras Sequential model with Rescaling.
    """

    model = Sequential()

    model.add(Input(shape=shape))
    model.add(layers.Rescaling(1./255))  # RESCALE!

    model.add(layers.Conv2D(32, (4, 4), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    reg = regularizers.l2(1e-5)
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=reg))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation='relu', kernel_regularizer=reg))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(num_classes, activation='softmax'))

    print("✅ Model initialized")
    model.summary()

    return model


def compile_model(model, learning_rate=1e-4):
    """
    Compile the Neural Network

    Parameters
    ----------
    model : keras.Model
        Keras model returned by initialize_model.
    learning_rate : float, optional
        Learning rate for the Adam optimizer. Default is 1e-4.

    Returns
    -------
    keras.Model
        Compiled model.
    """
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    print("✅ Model compiled")

    return model


def train_model(
        model,
        train_ds,
        val_ds,
        patience=10,
        epochs=1000
    ):
    """
    Train a compiled Keras model.

    Parameters
    ----------
    model : keras.Model
        Compiled model to train.
    train_ds : tf.data.Dataset
        Training dataset.
    val_ds : tf.data.Dataset
        Validation dataset.
    patience : int, optional
        Early stopping patience in epochs. Default is 10.
    epochs : int, optional
        Maximum number of epochs. Default is 1000.

    Returns
    -------
    model : keras.Model
        Trained Keras model.
    history : keras.callbacks.History
        Training history object with loss/accuracy curves.
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[es],
        verbose=1
    )

    best_val_acc = np.max(history.history['val_accuracy'])
    print(f"✅ Model trained with best val accuracy: {best_val_acc:.3f}")

    return model, history


def evaluate_model(
        model,
        val_ds,
        test_ds,
    ):
    """
    Evaluate trained model on validation and test sets.

    Parameters
    ----------
    model : keras.Model
        Trained model.
    val_ds : tf.data.Dataset
        Validation dataset.
    test_ds : tf.data.Dataset
        Test dataset.

    Returns
    -------
    val_metrics : dict
        Dictionary of validation loss and metrics.
    test_metrics : dict
        Dictionary of test loss and metrics.
    """

    val_metrics  = model.evaluate(val_ds, verbose=0, return_dict=True)
    test_metrics = model.evaluate(test_ds, verbose=0, return_dict=True)

    print(f"Val  — {val_metrics}")
    print(f"Test — {test_metrics}")

    print(f"✅ Model evaluated")

    return val_metrics, test_metrics


def load_trained_model(model_name="baseline_model"):
    """
    Load a trained Keras model from disk.

    Parameters
    ----------
    model_name : str, optional
        File name (without extension). Default is "baseline_model".

    Returns
    -------
    keras.Model
        Loaded model.
    """
    path = MODELS_DIR / f"{model_name}.keras"
    model = load_model(path)
    print(f"✅ Model loaded")
    return model


def save_model(model, model_name):
    """
    Save a trained model to disk.

    Parameters
    ----------
    model : keras.Model
        Trained model to save.
    model_name : str, optional
        File name (without extension).
    """
    model.save(f"{MODELS_DIR}/{model_name}.keras")
    print(f"✅ Model saved")
