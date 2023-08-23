"""Some utilities"""


from typing import Optional, TypedDict
import tensorflow as tf


class ImageStepDict(TypedDict, total=False):
    image: tf.Tensor
    step: tf.Tensor
    label: Optional[tf.Tensor]


def clone_initializer(initializer: tf.keras.initializers.Initializer):
    if isinstance(
        initializer,
        tf.keras.initializers.Initializer,
    ):
        return initializer.__class__.from_config(initializer.get_config())
    return initializer


def get_input_shape(input_shape: tf.TensorShape):
    if len(input_shape) == 4:
        return tf.TensorShape(input_shape)[1:]
    elif len(input_shape) == 3:
        return tf.TensorShape(input_shape)
    else:
        raise ValueError("Input shape must be (B, H, W, C) or (B, C, H, W)")
