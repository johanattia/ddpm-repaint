"""Abstract Diffusion Model"""


from typing import Dict, Iterable, Union

import tensorflow as tf
import tensorflow_probability as tfp


tfd = tfp.distributions
flatten = tf.keras.layers.Flatten()


class DiffusionModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compile(
        self,
        maxstep: int = 1000,
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

    def get_alpha_schedule_(self, step: int) -> float:
        if not self._is_compiled:
            raise AttributeError("")
        raise NotImplementedError

    def sampling(
        self,
    ):
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

        step = tf.random.uniform(
            shape=[batch_size, 1], minval=1, maxval=self.maxstep, dtype=tf.int32
        )  # (B, 1)
        alpha = self.get_alpha_schedule_(step)
        input = tf.math.sqrt(alpha) * x + tf.math.sqrt(1 - alpha) * tf.reshape(
            eps_target, shape=tf.shape(x)
        )  # (B, H, W, C)

        input_dict = {"input": input, "timestep": step}

        with tf.GradientTape() as tape:
            eps_pred = flatten(self(input_dict, training=True))  # (B, H * W * C)
            loss = self.compute_loss(input_dict, eps_target, eps_pred, sample_weight)

        self._validate_target_and_loss(eps_target, loss)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(input_dict, eps_target, eps_pred, sample_weight)

    def get_config(self) -> Dict:
        config = super().get_config()
        # config.update({})
        raise NotImplementedError  # config
