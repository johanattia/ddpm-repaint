"""Layers needed for PixelCNN++ model"""


from typing import Callable, Dict, Tuple, Union
import tensorflow as tf


class CELU(tf.keras.layers.Layer):
    """Concatenated ELU, analoguous of Concatenated ReLU (http://arxiv.org/abs/1603.05201)."""

    def __init__(self, alpha: float = 1.0, axis: int = -1, **kwargs):
        super().__init__(**kwargs)

        self._axis = axis
        self.concat = tf.keras.layers.Concatenate(axis=self._axis)

        self._alpha = float(alpha)
        self.elu = tf.keras.layers.Elu(alpha=self._alpha)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.elu(self.concat([inputs, -inputs]))

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({"alpha": self._alpha, "axis": self._axis})
        return config


class GatedResidualBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int]],
        strides: Tuple[int] = (1, 1),
        padding: str = "valid",
        dropout: float = 0.2,
        celu_axis: int = -1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self._filters = filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._dropout = dropout
        self._celu_axis = celu_axis

    def build(self, input_shape):
        pass

    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        pass
