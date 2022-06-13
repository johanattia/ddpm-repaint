"""Utilities functions"""


from typing import Union
import tensorflow as tf


def clone_initializer(initializer: tf.keras.initializers.Initializer):
    if isinstance(
        initializer,
        tf.keras.initializers.Initializer,
    ):
        return initializer.__class__.from_config(initializer.get_config())
    return initializer
