"""Diffusion Model from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py"""


from typing import Dict, Iterable, Union

import tensorflow as tf
import tensorflow_probability as tfp

from .utils import get_input_shape


tfd = tfp.distributions
flatten = tf.keras.layers.Flatten()


class DiffusionModel(tf.keras.Model):
    def __init__(
        self, input_shape: tf.TensorShape, data_format: str = "channels_last", **kwargs
    ):
        super().__init__(**kwargs)
        self.input_shape = get_input_shape(input_shape)  # (H, W, C) or (C, H, W)
        self.flattened_shape = tf.reduce_prod(self.input_shape)

        if data_format is None:
            data_format = "channels_last"
        elif data_format not in [
            "channels_last",
            "channels_first",
        ]:
            raise ValueError(
                """`data_format` must value `channels_last` or `channels_first`. 
                Default behavior to `channels_last`.
                """
            )
        self.data_format = data_format

    def compile(
        self,
        maxstep: int = 1000,
        beta_min: float = 1e-4,
        beta_max: float = 0.02,
        optimizer: Union[str, tf.keras.optimizers.Optimizer] = "adam",
        metrics: Union[str, tf.keras.metrics.Metric] = None,
        loss_weights: Union[Iterable[float], Dict[str, float]] = None,
        weighted_metrics: Iterable[tf.keras.metrics.Metric] = None,
        run_eagerly: bool = False,
        steps_per_execution: int = 1,
        jit_compile: bool = False,
        **kwargs,
    ):
        self.maxstep = maxstep
        self.beta_min = beta_min
        self.beta_max = beta_max
        steps = tf.range(
            start=self.beta_min, limit=self.beta_max + 1, delta=1, dtype=tf.float32
        )
        self._beta_scheduler = (
            self.beta_min + steps * (self.beta_max - self.beta_min) / self.maxstep
        )
        self._alpha_scheduler = tf.math.cumprod(1 - self._beta_scheduler)
        super().compile(
            optimizer=optimizer,
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            jit_compile=jit_compile,
            **kwargs,
        )

    def get_alpha_step(self, steps: tf.Tensor) -> tf.Tensor:
        if not hasattr(self, "_alpha_scheduler"):
            raise AttributeError(
                "Must run `compile` method before using `get_alpha_step`."
            )
        return tf.gather(params=self._alpha_scheduler, indices=steps)

    def get_beta_step(self, steps: tf.Tensor) -> tf.Tensor:
        if not hasattr(self, "_beta_scheduler"):
            raise AttributeError(
                "Must run `compile` method before using `get_beta_step`."
            )
        return tf.gather(params=self._beta_scheduler, indices=steps)

    def sampling(self, sampling_size: int = 32, verbose: int = 1) -> tf.Tensor:
        # Add https://www.tensorflow.org/api_docs/python/tf/keras/utils/Progbar
        dist = tfd.MultivariateNormalDiag(
            loc=tf.zeros(shape=[self.flattened_shape], dtype=tf.float32)
        )
        x = dist.sample(sampling_size)  # (B, H * W * C)
        x = tf.reshape(x, [sampling_size] + self.input_shape.as_list())  # (B, H, W, C)
        raise NotImplementedError

    def train_step(
        self, data: Union[tf.Tensor, Iterable[tf.Tensor]]
    ) -> Dict[str, tf.Tensor]:
        # Input image tensor with shape (B, H, W, C)
        x, _, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        batch_size, input_dim = tf.shape(flatten(x))  # (B, H * W * C)

        # Define N(0,I) distribution for gaussian noise sampling
        dist = tfd.MultivariateNormalDiag(
            loc=tf.zeros(shape=[input_dim], dtype=tf.float32)
        )
        eps_target = dist.sample(batch_size)  # (B, H * W * C)

        steps = tf.random.uniform(
            shape=[batch_size, 1], minval=1, maxval=self.maxstep, dtype=tf.int32
        )  # (B, 1)
        alpha = self.get_alpha_step(steps)  # (B, 1)
        input = tf.math.sqrt(alpha) * x + tf.math.sqrt(1 - alpha) * tf.reshape(
            eps_target, shape=tf.shape(x)
        )  # (B, H, W, C)

        input_tuple = (input, steps)

        with tf.GradientTape() as tape:
            eps_pred = flatten(self(input_tuple, training=True))  # (B, H * W * C)
            loss = self.compute_loss(input_tuple, eps_target, eps_pred, sample_weight)

        self._validate_target_and_loss(eps_target, loss)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(input_tuple, eps_target, eps_pred, sample_weight)

    def get_config(self, include_compile: bool = False) -> Dict:
        base_config = super().get_config()
        config = {
            "input_shape": self.input_shape.as_list(),
            "data_format": self.data_format,
        }
        if include_compile:
            config.update(
                {
                    "compile": {
                        "maxstep": self.maxstep,
                        "beta_min": self.beta_min,
                        "beta_max": self.beta_max,
                    }
                }
            )
        return dict(base_config.items() + config.items())

    @classmethod
    def from_config(cls, config: Dict):
        compile = config.pop("compile", None)
        del compile
        return cls(**config)
