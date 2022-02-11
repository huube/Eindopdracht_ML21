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
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune import JupyterNotebookReporter
from ray import tune
from src.data import make_dataset
import ray

from typing import Tuple, Optional, Union, Dict

class BaseModel(tf.keras.Model):
    