"""Noise Scheduler as Layer"""


from typing import Dict, Iterable, Tuple, Union

import tensorflow as tf
from tensorflow.keras import layers

from src.tf_ddpm import utils


class DiffusionScheduler(layers.Layer):
    def __init__(
        self,
        image_shape: Union[Iterable[int], tf.TensorShape],
        maxstep: int = 1000,
        beta_min: float = 1e-4,
        beta_max: float = 0.02,
        dtype: tf.dtypes.DType = tf.float32,
        **kwargs,
    ):
        kwargs["trainable"] = False
        super().__init__(**kwargs)

        self.image_shape = utils.get_input_shape(image_shape)
        self.maxstep = maxstep
        self.beta_min = beta_min
        self.beta_max = beta_max

        beta_schedule = tf.linspace(start=beta_min, stop=beta_max, num=maxstep)
        self.beta_schedule = tf.Variable(
            initial_value=beta_schedule,
            trainable=False,
            name="beta_schedule",
            dtype=dtype,
        )

        alpha_schedule = 1.0 - beta_schedule
        self.alpha_schedule = tf.Variable(
            initial_value=alpha_schedule,
            trainable=False,
            name="alpha_schedule",
            dtype=dtype,
        )

        alpha_cumprod = tf.math.cumprod(alpha_schedule)
        self.alpha_cumprod = tf.Variable(
            initial_value=alpha_cumprod,
            trainable=False,
            name="alpha_cumprod",
            dtype=dtype,
        )

        alpha_cumprod_prev = tf.concat([[1.0], alpha_cumprod[:-1]], axis=0)
        self.alpha_cumprod_prev = tf.Variable(
            initial_value=alpha_cumprod_prev,
            trainable=False,
            name="alpha_cumprod_prev",
            dtype=dtype,
        )

        # Diffusion calculation: q(x_t | x_t-1)
        sqrt_alpha_schedule = tf.math.sqrt(alpha_schedule)
        self.sqrt_alpha_schedule = tf.Variable(
            initial_value=sqrt_alpha_schedule,
            trainable=False,
            name="sqrt_alpha_schedule",
            dtype=dtype,
        )

        sqrt_alpha_cumprod = tf.math.sqrt(alpha_cumprod)
        self.sqrt_alpha_cumprod = tf.Variable(
            initial_value=sqrt_alpha_cumprod,
            trainable=False,
            name="sqrt_alpha_cumprod",
            dtype=dtype,
        )

        one_minus_alpha_cumprod = 1.0 - alpha_cumprod
        self.one_minus_alpha_cumprod = tf.Variable(
            initial_value=one_minus_alpha_cumprod,
            trainable=False,
            name="one_minus_alpha_cumprod",
            dtype=dtype,
        )

        sqrt_one_minus_alpha_cumprod = tf.math.sqrt(one_minus_alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = tf.Variable(
            initial_value=sqrt_one_minus_alpha_cumprod,
            trainable=False,
            name="sqrt_one_minus_alpha_cumprod",
            dtype=dtype,
        )

        log_one_minus_alpha_cumprod = tf.math.log(one_minus_alpha_cumprod)
        self.log_one_minus_alpha_cumprod = tf.Variable(
            initial_value=log_one_minus_alpha_cumprod,
            trainable=False,
            name="log_one_minus_alpha_cumprod",
            dtype=dtype,
        )

        sqrt_inv_alpha_cumprod = tf.math.sqrt(1.0 / alpha_cumprod)
        self.sqrt_inv_alpha_cumprod = tf.Variable(
            initial_value=sqrt_inv_alpha_cumprod,
            trainable=False,
            name="sqrt_inv_alpha_cumprod",
            dtype=dtype,
        )

        sqrt_inv_m1_alphas_cumprod = tf.math.sqrt(
            1.0 / alpha_cumprod - 1
        )  # check if / -1 or /(-1) : self.sqrt_recipm1_alphas_cumprod
        self.sqrt_inv_m1_alphas_cumprod = tf.Variable(
            initial_value=sqrt_inv_m1_alphas_cumprod,
            trainable=False,
            name="sqrt_inv_m1_alphas_cumprod",
            dtype=dtype,
        )

        # Posterior variance: q(x_t-1 | x_t, x_0)
        posterior_variance = (
            beta_schedule * (1.0 - alpha_cumprod_prev) / one_minus_alpha_cumprod
        )
        self.posterior_variance = tf.Variable(
            initial_value=posterior_variance,
            trainable=False,
            name="posterior_variance",
            dtype=dtype,
        )

        posterior_log_variance_clipped = tf.math.log(
            tf.math.maximum(posterior_variance, 1e-20)
        )
        self.posterior_log_variance_clipped = tf.Variable(
            initial_value=posterior_log_variance_clipped,
            trainable=False,
            name="posterior_log_variance_clipped",
            dtype=dtype,
        )

        posterior_mean_coef1 = (
            beta_schedule * tf.math.sqrt(alpha_cumprod_prev) / one_minus_alpha_cumprod
        )
        self.posterior_mean_coef1 = tf.Variable(
            initial_value=posterior_mean_coef1,
            trainable=False,
            name="posterior_mean_coef1",
            dtype=dtype,
        )

        posterior_mean_coef2 = (
            (1.0 - alpha_cumprod_prev) * sqrt_alpha_schedule / one_minus_alpha_cumprod
        )
        self.posterior_mean_coef2 = tf.Variable(
            initial_value=posterior_mean_coef2,
            trainable=False,
            name="posterior_mean_coef2",
            dtype=dtype,
        )

    def call(self, inputs: Tuple[tf.Tensor, str]) -> tf.Tensor:
        schedule, steps = inputs
        schedule = getattr(self, schedule)

        sample_size, multiplier = tf.shape(steps)[0], self.image_shape.rank
        steps = tf.reshape(steps, shape=[sample_size] + multiplier * [1])

        return tf.gather(params=schedule, indices=steps)

    def mean(self, sample: tf.Tensor, steps: tf.Tensor):
        if tf.shape(sample)[0] != tf.shape(steps)[0]:
            raise ValueError("`sample` and `steps` args must have the same size.")

        return self(("sqrt_alpha_cumprod", steps)) * sample

    def variance(self, steps: tf.Tensor, log_variance: bool = False):
        if log_variance:
            return self(("log_one_minus_alpha_cumprod", steps))

        return self(("one_minus_alpha_cumprod", steps))

    def add_noise(
        self, sample: tf.Tensor, steps: tf.Tensor, eps: tf.Tensor
    ) -> tf.Tensor:
        """Add noise to x_0, see Eq (4) & Algorithm 1, https://arxiv.org/pdf/2006.11239.pdf"""
        sqrt_alpha_cumprod = self(("sqrt_alpha_cumprod", steps))
        sqrt_one_minus_alpha_cumprod = self(("sqrt_one_minus_alpha_cumprod", steps))

        return sqrt_alpha_cumprod * sample + sqrt_one_minus_alpha_cumprod * eps

    def remove_noise(
        self, noisy_sample: tf.Tensor, steps: tf.Tensor, eps: tf.Tensor
    ) -> tf.Tensor:
        """Retrieve x_0 from x_t: inverse of `add_noise` method"""  # Consistency Models?
        sqrt_inv_alpha_cumprod = self(("sqrt_inv_alpha_cumprod", steps))
        sqrt_inv_m1_alphas_cumprod = self(("sqrt_inv_m1_alphas_cumprod", steps))

        return sqrt_inv_alpha_cumprod * noisy_sample - sqrt_inv_m1_alphas_cumprod * eps

    def diffusion_process(self, sample: tf.Tensor, steps: tf.Tensor) -> tf.Tensor:
        """q(x_t | x_0), see Equation (4), https://arxiv.org/pdf/2006.11239.pdf"""
        shape = tf.shape(sample)
        batch_size = shape[0]

        if tf.TensorShape(shape) != [batch_size] + self.image_shape:
            raise ValueError(
                f"`sample` must have shape {self.image_shape}. Received {shape}"
            )

        eps = tf.random.normal(shape)
        noisy_sample = self.add_noise(sample=sample, steps=steps, eps=eps)

        return noisy_sample

    def diffusion_posterior(
        self, initial_sample: tf.Tensor, noisy_sample: tf.Tensor, steps: tf.Tensor
    ):
        """Mean and variance of the diffusion posterior q(x_t-1 | x_t, x_0)"""
        posterior_mean = (
            self(("posterior_mean_coef1", steps)) * initial_sample
            + self(("posterior_mean_coef2", steps)) * noisy_sample
        )
        posterior_variance = self(("posterior_variance", steps))
        posterior_log_variance_clipped = self(("posterior_log_variance_clipped", steps))

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update(
            {
                "image_shape": self.image_shape.as_list(),
                "maxstep": self.maxstep,
                "beta_min": self.beta_min,
                "beta_max": self.beta_max,
            }
        )
        return config
