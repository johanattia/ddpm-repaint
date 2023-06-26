"""RePaint model for image inpainting with Diffusion Models"""


from typing import Any, Dict
import tensorflow as tf
from src import ddpm


class RePaintDiffuser(tf.Module):
    """RePaint: Inpainting using Denoising Diffusion Probabilistic Models

    Resources:
    * https://www.tensorflow.org/api_docs/python/tf/Module
    * https://www.tensorflow.org/guide/intro_to_modules?hl=en
    * https://www.tensorflow.org/text/tutorials/transformer?hl=fr#run_inference
    """

    def __init__(
        self,
        diffuser: ddpm.unet.DiffusionUNet,
        resampling_steps: int = 20,
        name: str = None,
    ):
        super().__init__(name)
        self.diffuser = diffuser
        self.resampling_steps = resampling_steps

    def __call__(self, images: tf.Tensor, masks: tf.Tensor) -> tf.Tensor:
        return

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(**config)
