import numpy as np

from tensorflow import keras
from keras import Sequential, layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
from keras.models import load_model
from keras.applications import InceptionV3
from keras.optimizers.schedules import PiecewiseConstantDecay

from bee_tector.config import IMAGE_SIZE, MODELS_DIR



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
    aug = Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(100/360),
        layers.RandomZoom(0.1),
        layers.RandomShear(x_factor=[0.0, 0.3], y_factor=[0.0, 0.0]),
    ], name="augment")

    inc = InceptionV3(
        include_top=False,
        input_shape=shape,
        weights='imagenet'
    )
    inc.trainable = True

    model = Sequential([
        layers.Input(shape=shape),
        aug,
        layers.Rescaling(1./127.5, offset=-1.0),
        inc,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    print("✅ Model initialized")
    model.summary()

    return model


def compile_model(model, train_ds):
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
    steps = len(train_ds)

    schedule = PiecewiseConstantDecay(
        boundaries=[10*steps, 20*steps, 30*steps, 40*steps],
        values=[3e-3, 1e-3, 3e-4, 1e-4, 3e-5]
    )
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=SGD(learning_rate=schedule, momentum=0.9, nesterov=True),
        metrics=['accuracy']
    )

    print("✅ Model compiled")

    return model


def train_model(
        model,
        chkpt_model_name,
        train_ds,
        val_ds,
        patience=12,
        epochs=150
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


    ckpt = ModelCheckpoint(
        filepath=f"{MODELS_DIR}/{chkpt_model_name}.keras",
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    )

    es = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=patience,
        start_from_epoch=15,
        min_delta=1e-3,
        restore_best_weights=True,
    )


    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[ckpt, es],
        verbose=1
    )

    print("✅ Model trained")

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

    val_loss, val_acc = model.evaluate(val_ds)
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    print("✅ Model evaluated")

    return val_loss, val_acc, test_loss, test_acc


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
    print("✅ Model loaded")
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
    print("✅ Model saved")
