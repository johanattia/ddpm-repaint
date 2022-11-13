"""U-Net Diffusion Model from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py"""

from tokenize import group
from typing import Dict, Iterable, List, Union

import tensorflow as tf
import tensorflow_addons as tfa

from .diffusion_model import DiffusionModel
from .layers import (
    PositionEmbedding,
    Upsample,
    downsample_conv,
    downsample_pool,
    downsample_pool_and_conv,
    Downsample,
    ConvBlock,
    ResidualBlock,
    AttentionBlock,
)

# from .utils import ImageStepDict


class DiffusionUNet(DiffusionModel):
    def __init__(
        self,
        image_shape: tf.TensorShape,
        hidden_channel: int,
        channels: List[int],
        attention_channel: int,
        n_residual_blocks: int,
        dropout: float,
        output_channel: int = 3,
        use_conv: bool = True,
        groups: int = 8,
        data_format: str = "channels_last",
        **kwargs
    ):
        super().__init__(image_shape=image_shape, data_format=data_format, **kwargs)
        self.hidden_channel = hidden_channel
        self.channels = channels
        self.attention_channel = attention_channel
        self.n_residual_blocks = n_residual_blocks
        self.dropout = dropout
        self.output_channel = output_channel
        self.use_conv = use_conv
        self.groups = groups

        # TIMESTEP LAYERS
        self.position_embedding = PositionEmbedding(embed_dim=self.hidden_channel)
        self.position_dense1 = tf.keras.layers.Dense(
            units=4 * self.hidden_channel,
            activation=tf.nn.silu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=1.0, mode="fan_avg", distribution="uniform"
            ),
        )
        self.position_dense2 = tf.keras.layers.Dense(
            units=4 * self.hidden_channel,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=1.0, mode="fan_avg", distribution="uniform"
            ),
        )

        # DOWNSAMPLING LAYERS

        # MIDDLE LAYERS
        self.residual_block1 = ResidualBlock(
            data_format=self.data_format, dropout=self.dropout, groups=self.groups
        )
        self.residual_block2 = ResidualBlock(
            data_format=self.data_format, dropout=self.dropout, groups=self.groups
        )
        self.attention_block = AttentionBlock(
            attention_channel=self.attention_channel,
            data_format=self.data_format,
            groups=self.groups,
        )

        # UPSAMPLING LAYERS

        # OUTPUT LAYERS
        self.output_normalization = tfa.layers.GroupNormalization(
            groups=self.groups, axis=-1 if self.data_format == "channels_last" else 1
        )
        self.output_conv = ConvBlock(filters=self.output_channel, init_scale=0.0)

    def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Args:
            inputs (Dict[str, tf.Tensor]): Inputs dictionary of tensors. Must have keys
            `image` and `step`.

        Returns:
            tf.Tensor: An image tensor with same shape as inputs["image"].
        """
        # TIMESTEP
        step_embeddings = self.position_embedding(inputs["step"])
        step_embeddings = self.position_dense2(self.position_dense1(step_embeddings))

        # DOWNSAMPLING

        # MIDDLE

        # UPSAMPLING

        # OUTPUT

        return
