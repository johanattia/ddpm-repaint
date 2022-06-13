"""Layers needed for U-Net & PixelCNN++ models"""


from typing import Callable, Dict, Tuple, Union

import tensorflow as tf
import tensorflow_addons as tfa

from utils import clone_initializer


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
        output_channel: int = None,
        conv_shortcut: bool = False,
        kernel_size: Union[int, Tuple[int]] = (3, 3),
        strides: Tuple[int] = (1, 1),
        padding: str = "valid",
        data_format: str = "channels_last",
        dilation_rate: Tuple[int] = (1, 1),
        groups: int = 1,
        activation: Callable = tf.nn.silu,  # or tf.keras.activations.swish
        use_bias: bool = True,
        dropout: float = 0.2,
        kernel_initializer: Union[str, Callable] = "glorot_uniform",
        bias_initializer: Union[str, Callable] = "zeros",
        kernel_regularizer: Union[str, Callable] = None,
        bias_regularizer: Union[str, Callable] = None,
        activity_regularizer: Union[str, Callable] = None,
        kernel_constraint: Union[str, Callable] = None,
        bias_constraint: Union[str, Callable] = None,
        **kwargs,
    ):
        self.name = kwargs.pop("name", "residual_block")
        super().__init__(**kwargs)

        self._output_channel = output_channel
        self._conv_shortcut = conv_shortcut
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self._dropout = dropout
        self.dilation_rate = dilation_rate
        self.groups = groups
        self.activation = activation
        self.use_bias = use_bias

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.common_weights_params = dict(
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )

    def build(self, input_shape: tf.TensorShape):
        input_shape = tf.TensorShape(input_shape)
        if self.data_format == "channels_last":  # (B, H, W, C)
            channels = input_shape[-1]
        else:  # (B, C, H, W)
            channels = input_shape[1]
        if self._output_channel is None:
            self._output_channel = channels

        self.group_normalization1 = tfa.layers.GroupNormalization()
        self.group_normalization2 = tfa.layers.GroupNormalization()
        self.dropout = tf.keras.layers.Dropout(rate=self._dropout)

        self.conv_layers = [
            tf.keras.layers.Conv2D(
                filters=self._output_channel,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
                groups=self.groups,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                **self.common_weights_params,
                name=f"{self.name}_{i}",
            )
            for i in range(2)
        ]
        if self._output_channel:
            self.conv_layers.append(
                tf.keras.layers.Conv2D(
                    filters=self._output_channel,
                    kernel_size=self.kernel_size,
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate,
                    groups=self.groups,
                    activation=self.activation,
                    use_bias=self.use_bias,
                    kernel_initializer=clone_initializer(self.kernel_initializer),
                    bias_initializer=clone_initializer(self.bias_initializer),
                    **self.common_weights_params,
                    name=f"{self.name}_{2}",
                )
            )
        else:
            pass  # einsum ops

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        pass
