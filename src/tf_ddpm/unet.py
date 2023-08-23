"""U-Net Diffusion Model from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py"""


from typing import Dict, List, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers, layers

from tf_ddpm.base_diffusion import BaseDiffuser
from tf_ddpm.layers import (
    PositionEmbedding,
    ResBlock,
    Attention2D,
    UNetEncoderBlock,
    UNetDecoderBlock,
)


# TODO: model tests


class DiffusionUNet(BaseDiffuser):
    """Denoising Diffusion Probabilistic Model.

    Full DDPM implementation with U-Net, Diffusion/Noise Scheduler, training
    and sampling algorithms.
    """

    def __init__(
        self,
        image_shape: tf.TensorShape,
        data_format: str = "channels_last",
        class_conditioning: bool = False,
        n_classes: int = None,
        maxstep: int = 1000,
        hidden_units: int = 128,
        n_residual_blocks: int = 2,
        attention_resolutions: Tuple[int] = (16,),
        channels_multipliers: Tuple[int] = (1, 2, 2, 2),
        output_channel: int = 3,
        dropout: float = 0.1,
        use_conv: bool = True,
        groups: int = 8,
        **kwargs,
    ):
        # BaseDiffuser attributes
        super().__init__(
            image_shape=image_shape,
            data_format=data_format,
            class_conditioning=class_conditioning,
            n_classes=n_classes,
            maxstep=maxstep,
            **kwargs,
        )
        # U-Net attributes
        self.hidden_units = hidden_units
        self.n_residual_blocks = n_residual_blocks
        self.attention_resolutions = attention_resolutions
        self.channels_multipliers = channels_multipliers
        self.n_resolutions = len(channels_multipliers)
        self.output_channel = output_channel
        self.dropout = dropout
        self.use_conv = use_conv
        self.groups = groups

        # Timestep Layers
        self.step_embedding = PositionEmbedding(embed_dim=self.hidden_units)
        self.step_block = keras.Sequential(
            [
                layers.Dense(
                    units=4 * hidden_units,
                    activation="swish",
                    kernel_initializer=initializers.VarianceScaling(
                        scale=1.0, mode="fan_avg", distribution="uniform"
                    ),
                ),
                layers.Dense(
                    units=4 * self.hidden_units,
                    kernel_initializer=initializers.VarianceScaling(
                        scale=1.0, mode="fan_avg", distribution="uniform"
                    ),
                ),
            ],
            name="step_block",
        )

        # Class-conditioning Embedding Layer
        if self.class_conditioning:
            self.class_embedding = layers.Embedding(
                input_dim=n_classes, output_dim=hidden_units, name="class_embedding"
            )
        else:
            self.class_embedding = None

        # Downsampling Encoder Layers
        self.encoder_conv_0 = tf.keras.layers.Conv2D(
            filters=self.hidden_units,
            kernel_size=(3, 3),
            padding="same",
            data_format=self.data_format,
            kernel_initializer=initializers.VarianceScaling(
                scale=1.0, mode="fan_avg", distribution="uniform"
            ),
            name="encoder_conv_0",
        )

        self.encoder_channels_units: List[Tuple[int]] = []

        for i, multiplier in enumerate(self.channels_multipliers):
            channels_units = self.hidden_units * multiplier
            self.encoder_channels_units.append((i, channels_units))

            setattr(
                self,
                f"encoder_{i}_ch{channels_units}",
                UNetEncoderBlock(
                    n_residual_blocks=self.n_residual_blocks,
                    data_format=self.data_format,
                    groups=self.groups,
                    dropout=self.dropout,
                    output_channel=channels_units,
                    use_attention=(
                        self.image_shape[0] / (2**i) in self.attention_resolutions
                    ),
                    downsample=i != self.n_resolutions - 1,
                    use_conv=self.use_conv,
                    name=f"encoder_{i}_ch{channels_units}",
                ),
            )

        # Middle Block Layers
        self.middle_residual_block1 = ResBlock(
            data_format=self.data_format,
            groups=self.groups,
            dropout=self.dropout,
            name="middle_residual_block1",
        )
        self.middle_attention = Attention2D(
            data_format=self.data_format,
            groups=self.groups,
            name="middle_attention",
        )
        self.middle_residual_block2 = ResBlock(
            data_format=self.data_format,
            groups=self.groups,
            dropout=self.dropout,
            name="middle_residual_block2",
        )

        # Upsampling Decoder Layers
        self.decoder_channels_units: List[Tuple[int]] = []

        for i, multiplier in reversed(list(enumerate(self.channels_multipliers))):
            channels_units = self.hidden_units * multiplier
            self.decoder_channels_units.append((i, channels_units))

            setattr(
                self,
                f"decoder_{i}_ch{channels_units}",
                UNetDecoderBlock(
                    n_residual_blocks=self.n_residual_blocks + 1,
                    data_format=self.data_format,
                    groups=self.groups,
                    dropout=self.dropout,
                    output_channel=channels_units,
                    use_attention=(
                        self.image_shape[0] / (2**i) in self.attention_resolutions
                    ),
                    upsample=i != 0,
                    use_conv=self.use_conv,
                    name=f"decoder_{i}_ch{channels_units}",
                ),
            )

        # Output block
        output_layers = [
            layers.GroupNormalization(
                groups=groups, axis=-1 if data_format == "channels_last" else 1
            ),
            layers.Activation("swish"),
            layers.Conv2D(
                filters=output_channel,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                data_format=data_format,
                kernel_initializer=initializers.VarianceScaling(
                    scale=1.0, mode="fan_avg", distribution="uniform"
                ),
            ),
        ]
        self.output_block = keras.Sequential(output_layers, name="output_block")

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = None) -> tf.Tensor:
        # Timestep & Class Embedding
        step_embed = self.step_embedding(inputs["step"])

        if self.class_conditioning:
            step_embed += self.class_embedding(inputs["label"])

        step_embed = self.step_block(step_embed)

        # Downsampling Encoder
        h = self.encoder_conv_0(inputs["image"])
        hidden_states = [h]

        for i, channels_units in self.encoder_channels_units:
            block = f"encoder_{i}_ch{channels_units}"
            hidden_states += getattr(self, block)((h, step_embed), training=training)
            h = hidden_states[-1]

        # Middle Block
        h = self.middle_residual_block1((h, step_embed), training=training)
        h = self.middle_attention(h)
        h = self.middle_residual_block2((h, step_embed), training=training)

        # Upsampling Decoder
        for i, channels_units in self.decoder_channels_units:
            block = f"decoder_{i}_ch{channels_units}"
            h = getattr(self, block)((h, hidden_states, step_embed), training=training)

        # Output Block
        output = self.output_block(h, training=training)

        return output

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update(
            {
                "hidden_units": self.hidden_units,
                "n_residual_blocks": self.n_residual_blocks,
                "attention_resolutions": self.attention_resolutions,
                "channels_multipliers": self.channels_multipliers,
                "output_channel": self.output_channel,
                "dropout": self.dropout,
                "use_conv": self.use_conv,
                "groups": self.groups,
            }
        )
        return config
