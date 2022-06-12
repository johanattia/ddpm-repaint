"""Layers needed for U-Net & PixelCNN++ models"""


from typing import Callable, Dict, Tuple, Union

import tensorflow as tf
import tensorflow_addons as tfa


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


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int]],
        strides: Tuple[int] = (1, 1),
        padding: str = "valid",
        data_format: str = "channels_last",
        dilation_rate=(1, 1),
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        output_channel: int = None,
        conv_shortcut: bool = False,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self._output_channel = output_channel
        self._conv_shortcut = conv_shortcut
        self._dropout = dropout

        # self.swish = tf.nn.silu

    def build(self, input_shape: tf.TensorShape):
        self.normalization = tfa.layers.GroupNormalization()
        self.dropout = tf.keras.layers.Dropout(rate=self._dropout)
        self.conv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=(1, 1),
            groups=1,
            activation=None,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
        )

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        pass
