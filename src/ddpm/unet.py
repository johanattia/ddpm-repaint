"""U-Net Diffusion Model from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py"""

from typing import Tuple
import tensorflow as tf

from .diffusion_model import DiffusionModel


class DiffusionUNet(DiffusionModel):
    def __init__(
        self, input_shape: tf.TensorShape, data_format: str = "channels_last", **kwargs
    ):
        super().__init__(input_shape, data_format, **kwargs)

    def call(self, input: Tuple[tf.Tensor]) -> tf.Tensor:
        raise NotImplementedError
