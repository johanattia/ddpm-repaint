"""Diffusion Model from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py"""


from typing import Dict, Iterable, Union
import tensorflow as tf

from .utils import get_input_shape

# from .utils import ImageStepDict, get_input_shape
# import tensorflow_probability as tfp
# tfd = tfp.distributions
# flatten = tf.keras.layers.Flatten()


class DiffusionModel(tf.keras.Model):
    def __init__(
        self,
        image_shape: tf.TensorShape = (256, 256, 3),
        data_format: str = "channels_last",
        maxstep: int = 1000,
        beta_min: float = 1e-4,
        beta_max: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_shape = get_input_shape(image_shape)  # (H, W, C) or (C, H, W)
        # self.flattened_shape = tf.reduce_prod(self.image_shape)
        # self.gaussian_dist = tfd.MultivariateNormalDiag(
        #    loc=tf.zeros(shape=[self.flattened_shape], dtype=tf.float32)
        # )

        if data_format not in ["channels_last", "channels_first"]:
            if data_format is None:
                data_format = "channels_last"
            else:
                raise ValueError(
                    f"""`data_format` must value `channels_last` or `channels_first`. 
                    Received `{data_format}`, default behavior to `channels_last`.
                    """
                )
        self.data_format = data_format

        self.maxstep = maxstep
        self.beta_min = beta_min
        self.beta_max = beta_max

        self._beta_schedule = tf.linspace(
            start=self.beta_min, stop=self.beta_max, num=self.maxstep
        )
        self._alpha_schedule = 1.0 - self._beta_schedule
        self._alpha_bar_schedule = tf.math.cumprod(self._alpha_schedule)

    def get_beta_step(self, steps: tf.Tensor) -> tf.Tensor:
        return tf.gather(params=self._beta_schedule, indices=steps)

    def get_alpha_step(self, steps: tf.Tensor) -> tf.Tensor:
        return tf.gather(params=self._alpha_schedule, indices=steps)

    def get_alpha_bar_step(self, steps: tf.Tensor) -> tf.Tensor:
        return tf.gather(params=self._alpha_bar_schedule, indices=steps)

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = True) -> tf.Tensor:
        raise NotImplementedError(
            """Forward method `call` must be implemended in child class.
            `inputs` dict must have `image` and `step` keys.
            """
        )

    def train_step(
        self, data: Union[tf.Tensor, Iterable[tf.Tensor]]
    ) -> Dict[str, tf.Tensor]:
        # Input with shape (B, H, W, C) or (B, C, H, W)
        x, _, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        batch_size = tf.shape(x)[0]

        if tf.shape(x) != [batch_size] + self.image_shape:
            raise ValueError(f"Input must have shape {self.image_shape}")

        # Gaussian noise sampling (B, H, W, C)
        eps_target = tf.random.normal([batch_size] + self.image_shape)
        # eps_target = self.gaussian_dist.sample(batch_size)
        # eps = tf.reshape(eps_target, shape=[batch_size] + self.image_shape)

        # Prepare forward input (B, H, W, C)
        steps = tf.random.uniform(
            shape=[batch_size, 1], minval=1, maxval=self.maxstep, dtype=tf.int32
        )  # (B, 1)
        alpha = self.get_alpha_bar_step(steps)  # (B, 1)
        input = tf.math.sqrt(alpha) * x + tf.math.sqrt(1.0 - alpha) * eps_target

        # input_tuple = (input, steps)
        # input_dict: ImageStepDict = {"image": input, "step": steps}
        input_dict = {"image": input, "step": steps}

        with tf.GradientTape() as tape:
            # Run forward pass.
            eps_pred = self(input_dict, training=True)  # (B, H, W, C)
            loss = self.compute_loss(input_dict, eps_target, eps_pred, sample_weight)
        self._validate_target_and_loss(eps_target, loss)

        # Backpropagation: run backward pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(input_dict, eps_target, eps_pred, sample_weight)

    def _denoising_step(self, noise: tf.Tensor, step: int) -> tf.Tensor:
        """Denoising step.

        Args:
            noise (tf.Tensor): _description_
            step (int): _description_

        Returns:
            tf.Tensor: _description_
        """
        # Sample gaussian noise
        sampling_size = tf.shape(noise)[0]
        z = (
            tf.random.normal([sampling_size] + self.image_shape)
            if step > 1
            else tf.zeros([sampling_size] + self.image_shape)
        )

        # Ste formating + get diffusion schedules
        steps = tf.expand_dims(tf.repeat(step, [sampling_size]), axis=1)

        sigma = tf.math.sqrt(self.get_beta_step(steps))
        alpha_steps = self.get_alpha_step(steps)
        alpha_bar_steps = self.get_alpha_bar_step(steps)

        const1 = 1.0 / tf.math.sqrt(alpha_steps)
        const2 = (1.0 - alpha_steps) / tf.math.sqrt(1.0 - alpha_bar_steps)

        # Denoising step transformation
        # input_tuple = (noise, steps)
        # input_dict: ImageStepDict = {"image": noise, "step": steps}
        input_dict = {"image": noise, "step": steps}
        noise = const1 * (noise - const2 * self(input_dict, training=False)) + sigma * z

        return noise

    def diffusion_sampling(
        self, sampling_size: int = 32, verbose: int = 1
    ) -> tf.Tensor:
        """Sequential sampling procedure.

        Args:
            sampling_size (int, optional): _description_. Defaults to 32.
            verbose (int, optional): _description_. Defaults to 1.

        Returns:
            tf.Tensor: _description_
        """
        # Sample gaussian noise
        # noise = self.gaussian_dist.sample(sampling_size)  # (B, H * W * C)
        # noise = tf.reshape(noise, [sampling_size] + self.image_shape)  # (B, H, W, C)
        noise = tf.random.normal([sampling_size] + self.image_shape)

        # Initialize step and progress bar
        step = tf.Variable(0, dtype=tf.int32)
        progbar = tf.keras.utils.Progbar(target=self.maxstep, verbose=verbose)

        # Diffusion sampling with sequential denoising
        # TODO: compare performances and syntax with tf.while_loop
        while step < self.maxstep:
            noise = self._denoising_step(noise=noise, step=step)
            progbar.update(step, finalize=False)
            step.assign_add(1)

        # Progress bar finalization + output generated images
        progbar.update(step, finalize=True)

        return noise

    def interpolate(self):
        # Reference:https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils.py#L258-L299
        raise NotImplementedError

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update(
            {
                "image_shape": self.image_shape.as_list(),
                "data_format": self.data_format,
                "maxstep": self.maxstep,
                "beta_min": self.beta_min,
                "beta_max": self.beta_max,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict):
        return cls(**config)
