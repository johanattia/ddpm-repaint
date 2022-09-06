"""U-Net layers"""


from typing import Callable, Dict, Tuple, Union
from matplotlib import units

import tensorflow as tf
import tensorflow_addons as tfa

from utils import clone_initializer


# TODO: Review groupnorm for ConvBlock & ResidualConv for channels_first


def PositionEmbedding(embed_dim: int):
    # Reference: https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py#L90-L109
    def _sinusoidal_embedding(
        steps: tfa.types.TensorLike,
    ) -> tfa.types.FloatTensorLike:
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

    return tf.keras.layers.Lambda(_sinusoidal_embedding)


class CELU(tf.keras.layers.Layer):
    def __init__(self, alpha: float = 1.0, axis: int = -1, **kwargs):
        super().__init__(**kwargs)
        if axis is None:
            raise ValueError(
                "`axis` of a CELU layer cannot be None, expecting an integer."
            )
        self._axis = axis
        self._alpha = float(alpha)

        self.elu = tf.keras.layers.Elu(alpha=self._alpha)
        self.concat = tf.keras.layers.Concatenate(axis=self._axis)

    def call(self, inputs: tfa.types.FloatTensorLike) -> tfa.types.FloatTensorLike:
        return self.elu(self.concat([inputs, -inputs]))

    def get_config(self) -> Dict:
        config = super(CELU, self).get_config()
        config.update({"alpha": self._alpha, "axis": self._axis})
        return config


class Upsample(tf.keras.layers.Layer):
    def __init__(
        self, data_format: str = "channels_last", use_conv: bool = True, **kwargs
    ):
        super().__init__(**kwargs)
        self.data_format = data_format
        self._use_conv = use_conv

    def build(self, input_shape: tf.TensorShape):
        input_shape = tf.TensorShape(input_shape)
        if len(input_shape) != 4:
            raise ValueError(
                f"""Input of an `Upsample` layer should correspond to a batch of images.
                Expected shape is either (B, H, W, C) or (B, C, H, W).  Received: {input_shape}.
                """
            )
        self.upsample = tf.keras.layers.UpSampling2D(
            size=(2, 2),
            data_format=self.data_format,
            interpolation="nearest",
        )

        if self._use_conv:
            if self.data_format == "channels_last":  # (B, H, W, C)
                channels = input_shape[-1]
            else:  # (B, C, H, W)
                channels = input_shape[1]

            self.conv = tf.keras.layers.Conv2D(
                filters=channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                data_format=self.data_format,
                dilation_rate=(1, 1),
                use_bias=True,
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                    scale=1.0, mode="fan_avg", distribution="uniform"
                ),
            )
        super().build(input_shape)

    def call(self, inputs: tfa.dtypes.FloatTensorLike):
        x = self.upsample(inputs)

        if self._use_conv:
            return self.conv(x)

        return x


def downsample_pool(data_format: str = "channels_last", channels: int = None):
    del channels
    return tf.keras.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        data_format=data_format,
    )


def downsample_conv(data_format: str = "channels_last", channels: int = 3):
    return tf.keras.layers.Conv2D(
        filters=channels,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        data_format=data_format,
        dilation_rate=(1, 1),
        use_bias=True,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"
        ),
    )


def downsample_pool_and_conv(data_format: str = "channels_last", channels: int = 3):
    return tf.keras.Sequential(
        [
            tf.keras.layers.AveragePooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                data_format=data_format,
            ),
            tf.keras.layers.Conv2D(
                filters=channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                data_format=data_format,
                dilation_rate=(1, 1),
                use_bias=True,
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                    scale=1.0, mode="fan_avg", distribution="uniform"
                ),
            ),
        ]
    )


class Downsample(tf.keras.layers.Layer):
    def __init__(
        self,
        downsample_fn: Callable[[str, int], tf.keras.layers.Layer],
        data_format: str = "channels_last",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.downsample_fn = downsample_fn
        self.data_format = data_format

    def build(self, input_shape: tf.TensorShape):
        input_shape = tf.TensorShape(input_shape)
        if len(input_shape) != 4:
            raise ValueError(
                f"""Input of a `Downsample` layer should correspond to a batch of images.
                Expected shape is either (B, H, W, C) or (B, C, H, W).  Received: {input_shape}.
                """
            )

        if self.data_format == "channels_last":  # (B, H, W, C)
            channels = input_shape[-1]
        else:  # (B, C, H, W)
            channels = input_shape[1]

        self.downsample = self.downsample_fn(
            data_format=self.data_format, channels=channels
        )
        super().build(input_shape)

    def call(self, inputs: tfa.dtypes.FloatTensorLike):
        return self.downsample(inputs)


class ConvBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int = None,
        kernel_size: Union[int, Tuple[int]] = (3, 3),
        strides: Tuple[int] = (1, 1),
        data_format: str = "channels_last",
        dilation_rate: Tuple[int] = (1, 1),
        use_bias: bool = True,
        dropout: float = 0.2,
        init_scale: float = 1.0,
        **kwargs,
    ):
        self.name = kwargs.pop("name", "ConvBlock")
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self._dropout = dropout
        self._init_scale = init_scale

    def build(self, input_shape: tf.TensorShape):
        input_shape = tf.TensorShape(input_shape)
        if len(input_shape) != 4:
            raise ValueError(
                f"""Input of a `ConvBlock` layer should correspond to a batch of images.
                Expected shape is either (B, H, W, C) or (B, C, H, W).  Received: {input_shape}.
                """
            )

        if self.data_format == "channels_last":  # (B, H, W, C)
            channels = input_shape[-1]
        else:  # (B, C, H, W)
            channels = input_shape[1]

        if self.filters is None:
            self.filters = channels

        if self._dropout:
            self.dropout = tf.keras.layers.Dropout(rate=self._dropout)

        self.conv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="same",
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            use_bias=self.use_bias,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=self._init_scale, mode="fan_avg", distribution="uniform"
            ),
        )
        self.normalize = tfa.layers.GroupNormalization()
        super().build(input_shape)

    def call(
        self, inputs: tfa.types.FloatTensorLike, training: bool = None
    ) -> tf.Tensor:
        x = tf.nn.silu(self.normalize(inputs))

        if self._dropout:
            x = self.dropout(x, training=training)

        return self.conv(x)

    def get_config(self) -> Dict:
        config = super(ConvBlock, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "data_format": self.data_format,
                "dilation_rate": self.dilation_rate,
                "use_bias": self.use_bias,
                "dropout": self._dropout,
                "init_scale": self._init_scale,
            }
        )
        return config


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        output_channel: int = None,
        kernel_size: Union[int, Tuple[int]] = (3, 3),
        strides: Tuple[int] = (1, 1),
        data_format: str = "channels_last",
        dilation_rate: Tuple[int] = (1, 1),
        conv_shortcut: bool = False,
        dropout: float = 0.2,
        **kwargs,
    ):
        self.name = kwargs.pop("name", "ResidualBlock")
        super().__init__(**kwargs)
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.strides = strides
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.conv_shortcut = conv_shortcut
        self.dropout = dropout

    def build(self, input_shape: tf.TensorShape):
        input_shape = tf.TensorShape(input_shape)

        if self.data_format == "channels_last":  # (B, H, W, C)
            channels = input_shape[-1]
        else:  # (B, C, H, W)
            channels = input_shape[1]

        if self.output_channel is None:
            self.output_channel = channels

        self.conv_block1 = ConvBlock(
            filters=self.output_channel,
            kernel_size=self.kernel_size,
            strides=self.strides,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            dropout=None,
        )
        self.conv_block2 = ConvBlock(
            filters=self.output_channel,
            kernel_size=self.kernel_size,
            strides=self.strides,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            dropout=self.dropout,
            init_scale=1e-10,
        )
        self.dense = tf.keras.layers.Dense(
            units=self.output_channel,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=1.0, mode="fan_avg", distribution="uniform"
            ),
        )
        # Projection of input tensor in output space before residual addition
        self.output_projection = None
        if self.output_channel != channels:
            if self.conv_shortcut:
                self.output_projection = tf.keras.layers.Conv2D(
                    filters=self.output_channel,
                    kernel_size=self.kernel_size,
                    strides=self.strides,
                    padding="same",
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate,
                    use_bias=True,
                    kernel_initializer=tf.keras.initializers.VarianceScaling(
                        scale=1.0, mode="fan_avg", distribution="uniform"
                    ),
                )
            else:
                if self.data_format == "channels_last":
                    equation = "abcd,de->abce"
                    output_shape = (None, None, self.output_channel)
                else:
                    equation = "abcd,be->aecd"
                    output_shape = (self.output_channel, None, None)

                self.output_projection = tf.keras.layers.experimental.EinsumDense(
                    equation,
                    output_shape=output_shape,
                    bias_axes="e",
                    kernel_initializer=tf.keras.initializers.VarianceScaling(
                        scale=1.0, mode="fan_avg", distribution="uniform"
                    ),
                )
        super().build(input_shape)

    def call(
        self,
        inputs: Tuple[tfa.types.FloatTensorLike, tfa.types.TensorLike],
        training: bool = None,
    ) -> tf.Tensor:
        x, step_embed = inputs

        h = self.conv_block1(x)
        h += self.dense(tf.nn.silu(step_embed))[:, tf.newaxis, tf.newaxis, :]
        h = self.conv_block2(x, training=training)

        if self.output_projection is not None:
            x = self.output_projection(x)

        return x + h

    def get_config(self) -> Dict:
        config = super(ResidualBlock, self).get_config()
        config.update(
            {
                "output_channel": self.output_channel,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "data_format": self.data_format,
                "dilation_rate": self.dilation_rate,
                "conv_shortcut": self.conv_shortcut,
                "dropout": self.dropout,
            }
        )
        return config


class AttentionBlock(tf.keras.layers.Layer):
    pass
