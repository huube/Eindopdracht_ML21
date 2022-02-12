import tensorflow as tf
from tensorflow.python.keras.layers import (
    Dense,
    Flatten,
    Input,
    Dropout,
    BatchNormalization,
    Reshape,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow import keras
from tensorflow_addons.layers import GELU
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
from loguru import logger
from pathlib import Path


import sys
import keras_tuner as kt
from kerastuner.tuners import RandomSearch, Hyperband
from kerastuner.engine.hyperparameters import HyperParameters as hp
from tensorflow.keras.models import Sequential

sys.path.append("..")

from definitions import get_project_root
from src.data.make_dataset import create_train_test_validation

from typing import Tuple, Optional, Union, Dict

x_train, x_valid, x_test, y_train, y_valid, y_test = create_train_test_validation()
x_train.shape, y_train.shape, x_valid.shape, y_valid.shape, x_test.shape, y_test.shape


def model_builder(hp):
    model = Sequential()

    model.add(Dense(23, activation="relu", input_shape=(23,)))

    for i in range(hp.Int("n_layers", 1, 10)):
        model.add(Dense(hp.Int(f"layer_{i}", 1, 200), activation="relu"))

    model.add(Dropout(hp.Float("dropout", 0.05, 0.4)))
    model.add(Dense(15, activation="softmax"))

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def tune_random_search():
    log_dir = "logs_kt_random"
    tensorboard = TensorBoard(log_dir=log_dir)

    tuner = RandomSearch(
        model_builder,
        objective="val_loss",
        max_trials=15,
        executions_per_trial=3,
        directory=log_dir,
    )

    tuner.search_space_summary()

    tuner.search(
        x=x_train,
        y=y_train,
        epochs=10,
        batch_size=64,
        callbacks=[tensorboard],
        validation_data=(x_valid, y_valid),
    )
