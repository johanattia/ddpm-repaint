"""Layers needed for U-Net & PixelCNN++ models"""


from multiprocessing.sharedctypes import Value
from typing import Callable, Dict, Tuple, Union

import tensorflow as tf
import tensorflow_addons as tfa

from utils import clone_initializer


def SinusoidalPositionEmbedding(embed_dim: int) -> tf.Tensor:
    # Reference: https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py#L90-L109
    def sinusoidal_embedding(steps: tf.Tensor) -> tf.Tensor:
        # input shape: (B, 1)
        batch_size = tf.shape(steps)[0]
        if steps.shape != tf.TensorShape([batch_size, 1]):
            raise ValueError

        half_dim = embed_dim // 2
        const = -tf.math.log(10000.0) / (half_dim - 1)
        const = tf.exp(
            tf.expand_dims(tf.range(half_dim, dtype=tf.float32), axis=0) * const
        )
        const = tf.cast(steps, dtype=tf.float32) * const
        step_embed = tf.concat([tf.sin(const), tf.cos(const)], axis=1)

        if embed_dim % 2 == 1:
            return tf.pad(step_embed, [[0, 0], [0, 1]])

        return step_embed  # output shape: (B, embed_dim)

    return tf.keras.layers.Lambda(sinusoidal_embedding)


class CELU(tf.keras.layers.Layer):
    def __init__(self, alpha: float = 1.0, axis: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.alpha = float(alpha)
        self.concat = tf.keras.layers.Concatenate(axis=self.axis)
        self.elu = tf.keras.layers.Elu(alpha=self.alpha)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.elu(self.concat([inputs, -inputs]))


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
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.common_weights_params = dict(
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
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
                kernel_initializer=clone_initializer(self._kernel_initializer),
                bias_initializer=clone_initializer(self._bias_initializer),
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
                    kernel_initializer=clone_initializer(self._kernel_initializer),
                    bias_initializer=clone_initializer(self._bias_initializer),
                    **self.common_weights_params,
                    name=f"{self.name}_{2}",
                )
            )
        else:
            pass  # einsum ops

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        pass
