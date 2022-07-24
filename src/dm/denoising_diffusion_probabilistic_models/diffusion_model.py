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
        self._beta_min = beta_min
        self._beta_max = beta_max
        steps = tf.range(
            start=self._beta_min, limit=self._beta_max + 1, delta=1, dtype=tf.float32
        )
        alpha = 1 - (
            self._beta_min + steps * (self._beta_max - self._beta_min) / self.maxstep
        )
        self._alpha = tf.math.cumprod(alpha)
        loss_obj = tf.keras.losses.MeanSquaredError()
        super().compile(
            optimizer=optimizer,
            loss=loss_obj,
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            jit_compile=jit_compile,
            **kwargs,
        )

    def get_alpha_schedule(self, steps: tf.Tensor) -> tf.Tensor:
        if not self._is_compiled:
            raise AttributeError(
                "A `DiffusionModel` object must be compiled before use."
            )
        return tf.gather(params=self._alpha, indices=steps)

    def sampling(self, sampling_size: int = 32) -> tf.Tensor:
        dist = tfd.MultivariateNormalDiag(
            loc=tf.zeros(shape=[self.flattened_shape], dtype=tf.float32)
        )
        x = dist.sample(sampling_size)  # (B, H * W * C)
        x = tf.reshape(x, [sampling_size] + self.input_shape.as_list())  # (B, H, W, C)
        raise NotImplementedError

    def train_step(
        self, data: Union[tf.Tensor, Iterable[tf.Tensor]]
    ) -> Dict[str, tf.Tensor]:
        # Get input image tensor with shape (B, H, W, C)
        x, _, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        batch_size, input_dim = tf.shape(flatten(x))  # (B, H * W * C)

        # Define N(0,I) distribution then gaussian noise sampling
        dist = tfd.MultivariateNormalDiag(
            loc=tf.zeros(shape=[input_dim], dtype=tf.float32)
        )
        eps_target = dist.sample(batch_size)  # (B, H * W * C)

        steps = tf.random.uniform(
            shape=[batch_size, 1], minval=1, maxval=self.maxstep, dtype=tf.int32
        )  # (B, 1)
        alpha = self.get_alpha_schedule(steps)
        input = tf.math.sqrt(alpha) * x + tf.math.sqrt(1 - alpha) * tf.reshape(
            eps_target, shape=tf.shape(x)
        )  # (B, H, W, C)

        input_tuple = (input, steps)  # {"input": input, "timestep": step}

        with tf.GradientTape() as tape:
            eps_pred = flatten(self(input_tuple, training=True))  # (B, H * W * C)
            loss = self.compute_loss(input_tuple, eps_target, eps_pred, sample_weight)

        self._validate_target_and_loss(eps_target, loss)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(input_tuple, eps_target, eps_pred, sample_weight)

    def get_config(self) -> Dict:
        config = super().get_config()
        # config.update({})
        raise NotImplementedError  # config
