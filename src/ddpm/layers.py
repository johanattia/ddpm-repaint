"""Diffusion & U-Net layers"""


from typing import Callable, Dict, Iterable, Tuple, Union

import tensorflow as tf
from tensorflow.keras import layers

import tensorflow_addons as tfa
from tensorflow_addons import types  # import FloatTensorLike, TensorLike


# TODO:
# Review groupnorm for ConvBlock & ResidualConv for channels_first
# Review layers input : image + embedding


class NoiseScheduler(layers.Layer):
    def __init__(self, **kwargs):
        kwargs["trainable"] = False
        super().__init__(**kwargs)

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        return super().call(inputs, training)

    def get_config(self) -> Dict:
        config = super().get_config()
        return config


def PositionEmbedding(embed_dim: int):
    # Reference: https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py#L90-L109
    def _sinusoidal_embedding(
        steps: types.TensorLike,
    ) -> types.FloatTensorLike:
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

    return layers.Lambda(_sinusoidal_embedding)


class CELU(layers.Layer):
    def __init__(self, alpha: float = 1.0, axis: int = -1, **kwargs):
        super().__init__(**kwargs)
        if axis is None:
            raise ValueError(
                "`axis` of a CELU layer cannot be None, expecting an integer."
            )
        self.axis = axis
        self.alpha = float(alpha)
        self.elu = layers.ELU(alpha=self.alpha)
        self.concat = layers.Concatenate(axis=self.axis)

    def call(self, inputs: types.FloatTensorLike) -> types.FloatTensorLike:
        return self.elu(self.concat([inputs, -inputs]))

    def get_config(self) -> Dict:
        config = super(CELU, self).get_config()
        config.update({"alpha": self.alpha, "axis": self.axis})
        return config


class Upsample(layers.Layer):
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
        self.upsample = layers.UpSampling2D(
            size=(2, 2),
            data_format=self.data_format,
            interpolation="nearest",
        )

        if self._use_conv:
            if self.data_format == "channels_last":  # (B, H, W, C)
                channels = input_shape[-1]
            else:  # (B, C, H, W)
                channels = input_shape[1]

            self.conv = layers.Conv2D(
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

    def call(self, inputs: types.FloatTensorLike):
        x = self.upsample(inputs)

        if self._use_conv:
            return self.conv(x)

        return x


def downsample_pool(data_format: str = "channels_last", channels: int = None):
    del channels
    return layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        data_format=data_format,
    )


def downsample_conv(data_format: str = "channels_last", channels: int = 3):
    return layers.Conv2D(
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
            layers.AveragePooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                data_format=data_format,
            ),
            layers.Conv2D(
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


class Downsample(layers.Layer):
    def __init__(
        self,
        downsample_fn: Callable[[str, int], layers.Layer],
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

    def call(self, inputs: types.FloatTensorLike):
        return self.downsample(inputs)


class ConvBlock(layers.Layer):
    def __init__(
        self,
        filters: int = None,
        kernel_size: Union[int, Tuple[int]] = (3, 3),
        strides: Tuple[int] = (1, 1),
        data_format: str = "channels_last",
        dilation_rate: Tuple[int] = (1, 1),
        use_bias: bool = True,
        dropout: float = 0.2,
        groups: int = 32,
        init_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self._dropout = dropout
        self.groups = groups
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
            channels_axis = -1
        else:  # (B, C, H, W)
            channels_axis = 1
        channels = input_shape[channels_axis]

        if channels % self.groups != 0:
            raise ValueError(
                f"""Normalization `groups` must divide input channels. Received groups={self.groups}.
                while channels={channels}.
                """
            )
        self.group_norm = tfa.layers.GroupNormalization(
            groups=self.groups, axis=channels_axis
        )

        if self.filters is None:
            self.filters = channels

        if self._dropout:
            self.dropout = layers.Dropout(rate=self._dropout)

        self.conv = layers.Conv2D(
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
        super().build(input_shape)

    def call(self, inputs: types.FloatTensorLike, training: bool = None) -> tf.Tensor:
        x = tf.nn.silu(self.group_norm(inputs))

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
                "groups": self.groups,
                "init_scale": self._init_scale,
            }
        )
        return config


class ResidualBlock(layers.Layer):
    def __init__(
        self,
        output_channel: int = None,
        kernel_size: Union[int, Tuple[int]] = (3, 3),
        strides: Tuple[int] = (1, 1),
        data_format: str = "channels_last",
        dilation_rate: Tuple[int] = (1, 1),
        conv_shortcut: bool = False,
        dropout: float = 0.2,
        groups: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.strides = strides
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.conv_shortcut = conv_shortcut
        self.dropout = dropout
        self.groups = groups

    def build(self, input_shape: Union[tf.TensorShape, Iterable[tf.TensorShape]]):
        input_shape = tf.TensorShape(input_shape[0])

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
            groups=self.groups,
        )
        self.conv_block2 = ConvBlock(
            filters=self.output_channel,
            kernel_size=self.kernel_size,
            strides=self.strides,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            dropout=self.dropout,
            groups=self.groups,
            init_scale=1e-10,
        )
        self.dense = layers.Dense(
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
                self.output_projection = layers.Conv2D(
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

                self.output_projection = layers.experimental.EinsumDense(
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
        inputs: Tuple[types.FloatTensorLike],
        training: bool = None,
    ) -> tf.Tensor:
        x, step_embed = inputs  # x, step_embed = inputs["image"], inputs["step"]
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


class AttentionBlock(layers.Layer):
    def __init__(
        self,
        attention_channel: int = None,
        data_format: str = "channels_last",
        groups: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attention_channel = attention_channel
        self.data_format = data_format
        self.groups = groups
        self.attention_equation = None
        self.attention_output_equation = None

    def build(self, input_shape: tf.TensorShape):
        input_shape = tf.TensorShape(input_shape)
        if len(input_shape) != 4:
            raise ValueError(
                f"""Input of an `AttentionBlock` layer should correspond to a batch of images.
                Expected shape is either (B, H, W, C) or (B, C, H, W).  Received: {input_shape}.
                """
            )

        if self.data_format == "channels_last":  # (B, H, W, C)
            channels_axis = -1
        else:  # (B, C, H, W)
            channels_axis = 1
        channels = input_shape[channels_axis]

        if self.attention_channel is None:
            self.attention_channel = channels

        # Layers definition
        self.group_norm = tfa.layers.GroupNormalization(
            groups=self.groups, axis=channels_axis
        )

        if self.data_format == "channels_last":
            projection_equation = "bhwc,cae->bhwae"
            projection_shape = (None, None, self.attention_channel, 3)

            self.attention_equation = "bhwc,btlc->bhwtl"
            self.attention_output_equation = "bhwtl,btlc->bhwc"

            output_equation = "bhwc,cd->bhwd"
            output_shape = (None, None, channels)
        else:
            projection_equation = "bchw,cae->bahwe"
            projection_shape = (self.attention_channel, None, None, 3)

            self.attention_equation = "bchw,bctl->bhwtl"
            self.attention_output_equation = "bhwtl,bctl->bchw"

            output_equation = "bchw,cd->bdhw"
            output_shape = (channels, None, None)

        self.attention_projection = layers.experimental.EinsumDense(
            projection_equation,
            output_shape=projection_shape,
            bias_axes="ae",
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=1.0, mode="fan_avg", distribution="uniform"
            ),
        )
        self.output_dense = layers.experimental.EinsumDense(
            output_equation,
            output_shape=output_shape,
            bias_axes="d",
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=1.0, mode="fan_avg", distribution="uniform"
            ),
        )
        super().build(input_shape)

    def call(self, inputs: types.FloatTensorLike, training: bool = None):
        attention_tensors = self.attention_projection(
            self.group_norm(inputs)
        )  # (B, H, W, C, 3) or (B, C, H, W, 3)
        query, key, value = tf.unstack(
            attention_tensors, axis=-1
        )  # 3 * (B, H, W, C) or (B, C, H, W)

        attention_weights = tf.einsum(self.attention_equation, query, key) * (
            int(self.attention_channel) ** (-0.5)
        )
        batch, height, width = self._get_batch_shapes(inputs)
        attention_weights = tf.reshape(
            attention_weights, [batch, height, width, height * width]
        )
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)
        attention_weights = tf.reshape(
            attention_weights, [batch, height, width, height, width]
        )

        attention_output = tf.einsum(
            self.attention_output_equation, attention_weights, value
        )  # (B, H, W, C) or (B, C, H, W)
        attention_output = self.output_dense(attention_output)
        output = inputs + attention_output

        return output

    def _get_batch_shapes(self, inputs: types.FloatTensorLike) -> Tuple[int]:
        input_shape = tf.shape(inputs)
        if self.data_format == "channels_last":
            batch, height, width = input_shape[0], input_shape[1], input_shape[2]
        else:
            batch, height, width = input_shape[0], input_shape[2], input_shape[3]
        return batch, height, width

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update(
            {
                "attention_channel": self.attention_channel,
                "data_format": self.data_format,
                "groups": self.groups,
            }
        )
        raise config
