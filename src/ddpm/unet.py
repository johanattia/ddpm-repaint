"""U-Net Diffusion Model from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py"""

from typing import Dict
import tensorflow as tf

from .diffusion_model import DiffusionModel

# from .utils import ImageStepDict


class DiffusionUNet(DiffusionModel):
    def __init__(
        self, image_shape: tf.TensorShape, data_format: str = "channels_last", **kwargs
    ):
        super().__init__(image_shape=image_shape, data_format=data_format, **kwargs)

    def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        raise NotImplementedError
