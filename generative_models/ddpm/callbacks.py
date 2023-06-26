"""Useful callbacks for training"""

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import preprocessing

from .utils import get_input_shape


# Reference: https://keras.io/examples/generative/gaugan/#gan-monitor-callback
class DiffusionSynthesisCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        image_rows: int = 4,
        image_cols: int = 4,
        image_width: int = 4,
        image_height: int = 4,
        epoch_interval: int = 5,
        save_path: Optional[str] = None,
        class_dict: Optional[Dict[int, str]] = None,
        verbose: int = 0,
    ):
        super().__init__()
        self.image_rows = image_rows
        self.image_cols = image_cols
        self.sampling_size = image_rows * image_cols
        self.image_width = image_width
        self.image_height = image_height
        self.epoch_interval = epoch_interval
        self.save_path = Path(save_path) if isinstance(save_path, str) else save_path
        self.save_images_ = save_path is not None
        self.class_dict = class_dict
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        del logs

        if epoch % self.epoch_interval == 0:
            input_kwargs = dict(sampling_size=self.sampling_size, verbose=self.verbose)

            if self.model.class_conditioning:
                labels = tf.random.uniform(
                    (self.sampling_size,),
                    maxval=self.model.n_classes,
                    dtype=tf.int32,
                )
                labels_names = [self.class_dict[i] for i in labels.numpy()]
                input_kwargs["label"] = labels
                print(
                    f"\nEpoch {epoch+1}: generating {self.sampling_size} samples with labels {labels}"
                )
            else:
                labels_names = None
                print(f"\nEpoch {epoch+1}: generating {self.sampling_size} samples")

            samples = self.model.generative_process(**input_kwargs)

            if self.save_images_:
                self.imshow(
                    samples,
                    save_fig=True,
                    filename=f"epoch{epoch+1}",
                    subtitles=labels_names,
                )
            else:
                self.imshow(samples)

    def imshow(
        self,
        samples: tf.Tensor = None,
        data_format: str = "channels_last",
        input_shape: tf.TensorShape = None,
        save_fig: bool = False,
        filename: Optional[str] = None,
        subtitles: Optional[Iterable[str]] = None,
    ):
        if self.model is not None:
            input_shape, data_format = self.model.image_shape, self.model.data_format
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
                if samples is not None:
                    input_shape = get_input_shape(tf.shape(samples))
                else:
                    raise ValueError(
                        "Either `images` or `input_shape` must be non-empty valid values."
                    )
            else:
                input_shape = get_input_shape(input_shape)

        if samples is None:
            samples = tf.random.normal([self.sampling_size] + input_shape)

        # https://stackoverflow.com/questions/25239933/how-to-add-a-title-to-each-subplot
        fig = plt.figure(
            figsize=(
                self.image_cols * self.image_width,
                self.image_rows * self.image_height,
            )
        )

        for i in range(self.sampling_size):
            plt.subplot(self.image_rows, self.image_cols, i + 1)
            subtitle = (
                f"Sample {i}"
                if subtitles is None
                else f"Sample {i}: {subtitles.get(i)}"
            )
            plt.title(subtitle)
            plt.imshow(
                preprocessing.image.array_to_img(samples[i], data_format=data_format)
            )
            plt.axis("off")

        if save_fig:
            plt.savefig(fname=self.save_path / filename)

        plt.show()
