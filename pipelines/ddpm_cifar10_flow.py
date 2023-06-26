import sys
from pathlib import Path

from metaflow import current, conda_base, FlowSpec, Parameter, step

import numpy as np
import tensorflow as tf

sys.path.append("..")
from generative_models import ddpm


class DiffusionFlow(FlowSpec):
    """Diffusion flow for Cifar10 dataset.

    Steps:
        [0] Start
        [1] Collect data
        [2] Train (and save weights)
        [3] Save (for serving)
        [4] Deploy
        [5] End
    """

    weights_registry = Parameter(
        name="weights_registry",
        help="Model weights registry",
    )
    models_registry = Parameter(
        name="models_registry",
        help="Serving models registry",
    )
    images_registry = Parameter(
        name="images_registry",
        help="Images registry for Diffusion Synthesis callback",
    )
    learning_rate = Parameter(
        name="learning_rate",
        default=1e-5,  # 2e-5
        help="Optimizer learning rate",
    )
    ema = Parameter(
        name="ema",
        default=True,
        help="Wether using Exponential Moving Average",
    )
    batch_size = Parameter(
        name="batch_size",
        default=16,
        help="Training batch size",
    )
    buffer_size = Parameter(
        name="buffer_size",
        default=64,
        help="Training buffer size",
    )
    epochs = Parameter(
        name="epochs",
        default=200,
        help="Number of epochs",
    )
    maxstep = Parameter(
        name="maxstep",
        default=1000,
        help="Number of diffusion steps",
    )

    @step
    def start(self):
        """Flow initialization"""
        self.weights_directory = Path(self.weights_registry) / f"run{current.run_id}"
        self.weights_directory.mkdir(exist_ok=True)

        self.models_directory = Path(self.models_registry) / f"run{current.run_id}"
        self.models_directory.mkdir(exist_ok=True)

        self.images_directory = Path(self.images_registry) / f"run{current.run_id}"
        self.images_directory.mkdir(exist_ok=True)

        self.diffuser_config = dict(
            image_shape=(32, 32, 3),
            class_conditioning=True,
            n_classes=10,
            maxstep=self.maxstep,
            hidden_units=128,
            n_residual_blocks=2,
            attention_resolutions=(16,),
            channels_multipliers=(1, 2, 2, 2),
            output_channel=3,
            groups=32,
            # maxstep=maxstep,
        )
        self.checkpoint_config = dict(
            filepath=self.weights_directory / "ckpt_weights.epoch{epoch:02d}.hdf5",
            monitor="loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="auto",
            save_freq="epoch",
        )
        self.synthesis_config = dict(
            image_rows=2,
            image_cols=3,
            image_width=2,
            image_height=2,
            epoch_interval=4,
            save_path=self.images_directory,
            verbose=1,
        )

        print(
            "DATASET: CIFAR10",
            "\n",
            f"ARTIFACTS - WEIGHTS REGISTRY: {self.weights_registry}",
            f"ARTIFACTS - WEIGHTS DIRECTORY: {self.weights_directory}",
            f"ARTIFACTS - MODELS REGISTRY: {self.models_registry}",
            f"ARTIFACTS - MODELS DIRECTORY: {self.models_directory}",
            f"ARTIFACTS - IMAGES REGISTRY: {self.images_registry}",
            f"ARTIFACTS - IMAGES DIRECTORY: {self.images_directory}",
            "\n",
            f"TRAINING - LEARNING RATE: {self.learning_rate}",
            f"TRAINING - EMA: {self.ema}",
            f"TRAINING - BATCH SIZE: {self.batch_size}",
            f"TRAINING - BUFFER SIZE: {self.buffer_size}",
            f"TRAINING - NB OF EPOCHS: {self.epochs}",
            f"TRAINING - MAXSTEP: {self.maxstep}"
            "\n",
            f"TRAINING - DIFFUSER CONFIG: OK",
            f"TRAINING - CHECKPOINT CONFIG: OK",
            f"TRAINING - SYNTHESIS CONFIG: OK",
            sep="\n",
        )

        self.next(self.collect_data)

    @step
    def collect_data(self):
        """Collect Cifar10 dataset"""

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        # See: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data
        self.class_mapping = {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck",
        }

        # Training data
        self.x_train = x_train
        self.y_train = y_train

        # Test data
        self.x_test = x_test
        self.y_test = y_test

        self.dataset_size = x_train.shape[0]

        self.next(self.train)

    @step
    def train(self):
        """Build TensorFlow Dataset & Training"""

        def preprocessing_func(img: tf.Tensor):
            img = img / 127.5 - 1.0
            img = tf.clip_by_value(img, -1.0, 1.0)
            img = tf.image.random_flip_left_right(img)
            return img

        x_train = tf.cast(self.x_train, dtype=tf.float32)
        y_train = tf.squeeze(self.y_train)

        training_dataset = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .map(
                lambda x, y: (preprocessing_func(x), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .batch(batch_size=self.batch_size)
            .shuffle(buffer_size=self.buffer_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        diffuser = ddpm.unet.DiffusionUNet.from_config(self.diffuser_config)

        if self.ema:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                use_ema=True,
                ema_momentum=0.9999,
                ema_overwrite_frequency=1,
            )
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        diffuser.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()],
        )

        STEPS_PER_EPOCH = int(np.ceil(self.dataset_size / self.batch_size))

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(self.checkpoint_config)
        synthesis_callback = ddpm.callbacks.DiffusionSynthesisCallback(
            self.synthesis_config
        )

        self.history = diffuser.fit(
            training_dataset,
            epochs=self.epochs,
            verbose=1,
            callbacks=[checkpoint_callback, synthesis_callback],
            steps_per_epoch=STEPS_PER_EPOCH,
        )

        self.best_epoch = checkpoint_callback.best_epoch
        self.best_weights_path = (
            self.weights_directory / f"ckpt_weights.epoch{self.best_epoch}.hdf5"
        )

        self.next(self.save)

    @step
    def save(self):
        """Save model for deployment purpose"""

        diffuser = ddpm.unet.DiffusionUNet.from_config(self.diffuser_config)
        diffuser.load_weights(self.best_weights_path)

        self.export_path = self.models_directory / f"cifar10_diffuser"

        # Serving model
        export_archive = tf.keras.export.ExportArchive()
        export_archive.track(diffuser)
        export_archive.add_endpoint(
            name="serve",
            fn=diffuser.serving_function,
            input_signature=diffuser.make_serving_signature(),
        )
        export_archive.write_out(self.export_path)

        self.next(self.deploy)

    @step
    def deploy(self):
        """Deployment using TensorFlow Serving?"""
        print("Deploy model to _____")

        self.next(self.end)

    @step
    def end(self):
        """Summarize training and saving"""
        print(
            f"BEST EPOCH: {self.best_epoch}",
            f"BEST WEIGHTS PATH: {self.best_weights_path}",
            f"SAVED (SERVING) MODEL PATH: {self.export_path}",
            sep="\n",
        )


if __name__ == "__main__":
    DiffusionFlow()
