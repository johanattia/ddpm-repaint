"""U-Net layers"""


from typing import Callable, Dict, Iterable, List, Tuple, Union

import tensorflow as tf
from tensorflow.keras import layers

from generative_models.ddpm import utils


# TODO:
# Review PositionEmbedding : layer with in-memory embedding rather than lambda ?
# Review groupnorm for ResBlock for channels_first


def PositionEmbedding(embed_dim: int):
    # Reference: https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py#L90-L109
    def fourier_embed_fn(steps: tf.Tensor) -> tf.Tensor:
        # input shape: (B, 1)
        # batch_size = tf.shape(steps)[0]
        # if steps.shape != tf.TensorShape([batch_size, 1]):
        #     raise ValueError

        half_dim = embed_dim // 2
        freqs = -tf.math.log(10000.0) / (half_dim - 1)
        # freqs = tf.exp(
        #     tf.expand_dims(tf.range(half_dim, dtype=tf.float32), axis=0) * freqs
        # )
        freqs = tf.exp(tf.range(half_dim, dtype=tf.float32) * freqs)
        args = tf.cast(steps, dtype=tf.float32) * tf.expand_dims(freqs, axis=0)
        step_embed = tf.concat([tf.sin(args), tf.cos(args)], axis=1)

        if embed_dim % 2 == 1:
            return tf.pad(step_embed, [[0, 0], [0, 1]])

        return step_embed  # output shape: (B, embed_dim)

    return layers.Lambda(fourier_embed_fn)


class Upsample(layers.Layer):
    def __init__(
        self, data_format: str = "channels_last", use_conv: bool = True, **kwargs
    ):
        super().__init__(**kwargs)
        self.data_format = data_format
        self._use_conv = use_conv

        # Layers
        self.conv = None
        self.upsample = None

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
                kernel_initializer=utils.defaut_initializer(scale=1.0),
            )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
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
        kernel_initializer=utils.defaut_initializer(scale=1.0),
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
                kernel_initializer=utils.defaut_initializer(scale=1.0),
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

        # Layer/Op
        self.downsample = None

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

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.downsample(inputs)


class ResBlock(layers.Layer):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int]] = (3, 3),
        data_format: str = "channels_last",
        groups: int = 32,
        dropout: float = 0.2,
        output_channel: int = None,
        conv_shortcut: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # CONV LAYERS
        self.kernel_size = kernel_size
        self.data_format = data_format

        # OUTPUT LAYERS
        self.conv_shortcut = conv_shortcut
        self.output_channel = output_channel
        self.output_projection = None

        # DROPOUT & NORMALIZATION LAYERS
        self._dropout = dropout
        self.dropout = layers.Dropout(rate=self._dropout)

        self.groups = groups
        self.group_norm1 = layers.GroupNormalization(
            groups=self.groups, axis=-1 if data_format == "channels_last" else 1
        )
        self.group_norm2 = layers.GroupNormalization(
            groups=self.groups, axis=-1 if data_format == "channels_last" else 1
        )

    def build(self, input_shape: Union[tf.TensorShape, Iterable[tf.TensorShape]]):
        input_shape = tf.TensorShape(input_shape[0])
        if len(input_shape) != 4:
            raise ValueError(
                f"""Input of a `ConvBlock` layer should correspond to a batch of images.
                Expected shape is either (B, H, W, C) or (B, C, H, W).  Received: {input_shape}.
                """
            )
        channels = (
            input_shape[-1] if self.data_format == "channels_last" else input_shape[1]
        )
        if channels % self.groups != 0:
            raise ValueError(
                f"""Normalization `groups` must divide input channels. Received groups={self.groups}.
                while channels={channels}.
                """
            )
        if self.output_channel is None:
            self.output_channel = channels

        self.dense = layers.Dense(
            units=self.output_channel,
            use_bias=True,
            kernel_initializer=utils.defaut_initializer(1.0),
        )
        conv_kwargs = dict(
            filters=self.output_channel,
            kernel_size=self.kernel_size,
            padding="same",
            data_format=self.data_format,
            kernel_initializer=utils.defaut_initializer(scale=1.0),
        )
        self.conv1 = layers.Conv2D(**conv_kwargs)
        self.conv2 = layers.Conv2D(**conv_kwargs)

        if self.output_channel != channels:
            if not self.conv_shortcut:
                self.output_projection = layers.Conv2D(**conv_kwargs)
            else:
                if self.data_format == "channels_last":
                    equation = "abcd,de->abce"
                    output_shape = (None, None, self.output_channel)
                else:
                    equation = "abcd,be->aecd"
                    output_shape = (self.output_channel, None, None)

                self.output_projection = layers.EinsumDense(
                    equation,
                    output_shape=output_shape,
                    bias_axes="e",
                    kernel_initializer=utils.defaut_initializer(scale=1.0),
                )
        # else:
        #    self.output_projection = layers.Identity()

        super().build(input_shape)

    def call(self, inputs: Tuple[tf.Tensor], training: bool = None) -> tf.Tensor:
        x, step_embed = inputs

        h = tf.nn.silu(self.group_norm1(x))
        h = self.conv1(h)

        h += self.dense(tf.nn.silu(step_embed))[:, tf.newaxis, tf.newaxis, :]

        h = tf.nn.silu(self.group_norm2(h))
        h = self.dropout(h, training=training)
        h = self.conv2(h)

        if self.output_projection is not None:
            x = self.output_projection(x)

        return h + x  # h + self.output_projection(x)

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update(
            {
                "kernel_size": self.kernel_size,
                "data_format": self.data_format,
                "groups": self.groups,
                "dropout": self._dropout,
                "output_channel": self.output_channel,
                "conv_shortcut": self.conv_shortcut,
            }
        )
        return config


class SpatialAttention(layers.Layer):
    def __init__(self, data_format: str = "channels_last", groups: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.data_format = data_format

        self.groups = groups
        self.group_norm = layers.GroupNormalization(
            groups=self.groups, axis=-1 if data_format == "channels_last" else 1
        )

        self.scale = None
        self.attention_equation = None
        self.attention_output_equation = None

    def build(self, input_shape: tf.TensorShape):
        input_shape = tf.TensorShape(input_shape)
        if len(input_shape) != 4:
            raise ValueError(
                f"""Input of an `SpatialAttention` layer should correspond to a batch of images.
                Expected shape is either (B, H, W, C) or (B, C, H, W).  Received: {input_shape}.
                """
            )
        channels = (
            input_shape[-1] if self.data_format == "channels_last" else input_shape[1]
        )
        self.scale = int(channels) ** (-0.5)

        if self.data_format == "channels_last":
            projection_equation = "bhwc,cae->bhwae"
            projection_shape = (None, None, channels, 3)

            self.attention_equation = "bhwc,btlc->bhwtl"
            self.attention_output_equation = "bhwtl,btlc->bhwc"

            output_equation = "bhwc,cd->bhwd"
            output_shape = (None, None, channels)
        else:
            projection_equation = "bchw,cae->bahwe"
            projection_shape = (channels, None, None, 3)

            self.attention_equation = "bchw,bctl->bhwtl"
            self.attention_output_equation = "bhwtl,bctl->bchw"

            output_equation = "bchw,cd->bdhw"
            output_shape = (channels, None, None)

        self.attention_projection = layers.EinsumDense(
            projection_equation,
            output_shape=projection_shape,
            bias_axes="ae",
            kernel_initializer=utils.defaut_initializer(scale=1.0),
        )
        self.output_dense = layers.EinsumDense(
            output_equation,
            output_shape=output_shape,
            bias_axes="d",
            kernel_initializer=utils.defaut_initializer(scale=1.0),
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        input_shape = tf.shape(inputs)

        if self.data_format == "channels_last":
            batch, height, width = input_shape[0], input_shape[1], input_shape[2]
        else:
            batch, height, width = input_shape[0], input_shape[2], input_shape[3]

        # (B, H, W, C)/(B, C, H, W) => (B, H, W, C, 3)/(B, C, H, W, 3)
        attention_tensors = self.attention_projection(self.group_norm(inputs))

        # (B, H, W, C, 3)/(B, C, H, W, 3) => 3 * (B, H, W, C)/(B, C, H, W)
        query, key, value = tf.unstack(attention_tensors, axis=-1)

        attention_weights = tf.einsum(self.attention_equation, query, key) * self.scale
        attention_weights = tf.reshape(
            attention_weights, [batch, height, width, height * width]
        )
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)
        attention_weights = tf.reshape(
            attention_weights, [batch, height, width, height, width]
        )

        # (B, H, W, C) or (B, C, H, W)
        attention_output = tf.einsum(
            self.attention_output_equation, attention_weights, value
        )
        attention_output = self.output_dense(attention_output)

        return inputs + attention_output

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({"data_format": self.data_format, "groups": self.groups})
        return config


class ResAttentionBlock(layers.Layer):
    def __init__(
        self,
        data_format: str = "channels_last",
        groups: int = 32,
        dropout: float = 0.2,
        output_channel: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_format = data_format
        self.groups = groups
        self.dropout = dropout
        self.output_channel = output_channel

        self.resblock = ResBlock(
            data_format=data_format,
            dropout=dropout,
            output_channel=output_channel,
            name="resblock",
        )
        self.attention = SpatialAttention(
            data_format=data_format, groups=groups, name="spatial_attention"
        )

    def call(self, inputs: Tuple[tf.Tensor], training: bool = None) -> tf.Tensor:
        x, step_embed = inputs
        return self.attention(self.resblock(inputs=(x, step_embed), training=training))

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update(
            {
                "data_format": self.data_format,
                "groups": self.groups,
                "dropout": self.dropout,
                "output_channel": self.output_channel,
            }
        )
        return config


class UNetEncoderBlock(layers.Layer):
    def __init__(
        self,
        n_residual_blocks: int = 2,
        data_format: str = "channels_last",
        groups: int = 32,
        dropout: float = 0.2,
        output_channel: int = None,
        use_attention: bool = False,
        downsample: bool = True,
        use_conv: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_residual_blocks = n_residual_blocks
        self.data_format = data_format

        if use_attention and groups is None:
            raise ValueError(
                "`groups` arg should be a valid integer if `use_attention`=True."
            )
        self.groups = groups
        self.use_attention = use_attention

        self.dropout = dropout
        self.output_channel = output_channel

        self._downsample = downsample
        self.use_conv = use_conv

        self.downsample = None
        if self._downsample:
            self.downsample = Downsample(
                downsample_fn=downsample_conv if self.use_conv else downsample_pool,
                data_format=self.data_format,
                name=f"downsample",
            )

        for i in range(self.n_residual_blocks):
            if self.use_attention:
                setattr(
                    self,
                    f"block{i}",
                    ResAttentionBlock(
                        data_format=self.data_format,
                        groups=self.groups,
                        dropout=self.dropout,
                        output_channel=self.output_channel,
                        name=f"res_attention_block{i}",
                    ),
                )
            else:
                setattr(
                    self,
                    f"block{i}",
                    ResBlock(
                        data_format=self.data_format,
                        dropout=self.dropout,
                        output_channel=self.output_channel,
                        name=f"res_block{i}",
                    ),
                )

    def call(
        self, inputs: Iterable[tf.Tensor], training: bool = None
    ) -> List[tf.Tensor]:
        h, step_embed = inputs

        outputs = []

        for i in range(self.n_residual_blocks):
            h = getattr(self, f"block{i}")(inputs=(h, step_embed), training=training)
            outputs.append(h)

        if self._downsample:
            outputs.append(self.downsample(h))

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_residual_blocks": self.n_residual_blocks,
                "data_format": self.data_format,
                "groups": self.groups,
                "dropout": self.dropout,
                "output_channel": self.output_channel,
                "use_attention": self.use_attention,
                "downsample": self._downsample,
                "use_conv": self.use_conv,
            }
        )
        return config


class UNetDecoderBlock(layers.Layer):
    def __init__(
        self,
        n_residual_blocks: int = 2,
        data_format: str = "channels_last",
        groups: int = 32,
        dropout: float = 0.2,
        output_channel: int = None,
        use_attention: bool = False,
        upsample: bool = True,
        use_conv: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_residual_blocks = n_residual_blocks
        self.data_format = data_format

        if use_attention and groups is None:
            raise ValueError(
                "`groups` arg should be a valid integer if `use_attention`=True."
            )
        self.groups = groups
        self.use_attention = use_attention

        self.dropout = dropout
        self.output_channel = output_channel

        self._upsample = upsample
        self.use_conv = use_conv

        self.upsample = None
        if self._upsample:
            self.upsample = Upsample(
                data_format=self.data_format, use_conv=self.use_conv, name=f"upsample"
            )

        for i in range(self.n_residual_blocks + 1):
            if self.use_attention:
                setattr(
                    self,
                    f"block{i}",
                    ResAttentionBlock(
                        data_format=self.data_format,
                        groups=self.groups,
                        dropout=self.dropout,
                        output_channel=self.output_channel,
                        name=f"res_attention_block{i}",
                    ),
                )
            else:
                setattr(
                    self,
                    f"block{i}",
                    ResBlock(
                        data_format=self.data_format,
                        dropout=self.dropout,
                        output_channel=self.output_channel,
                        name=f"res_block{i}",
                    ),
                )

    def call(self, inputs: Iterable[tf.Tensor], training: bool = None) -> tf.Tensor:
        h, hidden_states, step_embed = inputs

        for i in range(self.n_residual_blocks + 1):
            h = tf.concat([h, hidden_states.pop()], axis=-1)
            h = getattr(self, f"block{i}")(inputs=(h, step_embed), training=training)

        if self._upsample:
            h = self.upsample(h)

        return h

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_residual_blocks": self.n_residual_blocks,
                "data_format": self.data_format,
                "groups": self.groups,
                "dropout": self.dropout,
                "output_channel": self.output_channel,
                "use_attention": self.use_attention,
                "upsample": self._upsample,
                "use_conv": self.use_conv,
            }
        )
        return config
