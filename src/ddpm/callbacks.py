"""Useful ImageGenerationCallback for training.
Reference: https://keras.io/examples/generative/gaugan/#gan-monitor-callback
"""


import matplotlib.pyplot as plt
import tensorflow as tf

from .utils import get_input_shape


class DiffusionSynthesisCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        image_rows: int = 4,
        image_cols: int = 4,
        image_width: int = 4,
        image_height: int = 4,
        epoch_interval: int = 5,
        verbose: int = 0,
    ):
        super().__init__()
        self.image_rows = image_rows
        self.image_cols = image_cols
        self.sampling_size = image_rows * image_cols
        self.image_width = image_width
        self.image_height = image_height
        self.epoch_interval = epoch_interval
        self.verbose = verbose

    def on_epoch_end(self, epoch: int):
        if epoch % self.epoch_interval == 0:
            images = self.model.sampling(
                sampling_size=self.sampling_size, verbose=self.verbose
            )
            self.imshow(images)

    def imshow(
        self,
        images: tf.Tensor = None,
        data_format: str = "channels_last",
        input_shape: tf.TensorShape = None,
    ):
        if self.model is not None:
            input_shape, data_format = self.model.input_shape, self.model.data_format
        else:
            if data_format not in ["channels_last", "channels_first"]:
                if data_format is None:
                    data_format = "channels_last"
                else:
                    raise ValueError(
                        """`data_format` must value `channels_last` or `channels_first`. 
                        Default behavior to `channels_last`.
                        """
                    )

            if input_shape is None:
                if images is not None:
                    input_shape = get_input_shape(tf.shape(images))
                else:
                    raise ValueError(
                        "Either `images` or `input_shape` must be non-empty valid tensors."
                    )
            else:
                input_shape = get_input_shape(input_shape)

        if images is None:
            images = tf.random.normal([self.sampling_size] + input_shape.as_list())

        fig = plt.figure(
            figsize=(
                self.image_cols * self.image_width,
                self.image_rows * self.image_height,
            )
        )

        for i in range(self.sampling_size):
            plt.subplot(self.image_rows, self.image_cols, i + 1)
            plt.imshow(
                tf.keras.preprocessing.image.array_to_img(
                    images[i], data_format=data_format
                )
            )
            plt.axis("off")
        plt.show()
