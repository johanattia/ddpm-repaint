"""Diffusion Model based on https://github.com/hojonathanho/diffusion/"""


from typing import Any, Dict, Iterable, List, Union

import tensorflow as tf
from tensorflow import keras

from tf_ddpm.scheduler import DiffusionScheduler
from tf_ddpm.utils import get_input_shape


class BaseDiffuser(keras.Model):
    """Base class for Denoising Diffusion Probabilistic Model.
    U-Net architecture must be implemented as a child class.
    """

    def __init__(
        self,
        image_shape: Union[Iterable[int], tf.TensorShape],
        data_format: str = "channels_last",
        maxstep: int = 1000,
        beta_min: float = 1e-4,
        beta_max: float = 0.02,
        class_conditioning: bool = False,
        n_classes: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Image attributes
        self.image_shape = get_input_shape(image_shape)  # (H, W, C) or (C, H, W)

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

        # Scheduler attributes
        self.maxstep = maxstep
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.diffusion_scheduler = DiffusionScheduler(
            image_shape=self.image_shape,
            maxstep=maxstep,
            beta_min=beta_min,
            beta_max=beta_max,
        )

        # Class conditioning attributes
        self.class_conditioning = class_conditioning
        if class_conditioning and not isinstance(n_classes, int):
            raise TypeError(
                f"`n_classes` arg should be an integer > 1. Received {n_classes}."
            )
        self.n_classes = n_classes

        # Serving attributes
        self.serving_function = self.make_serving_function(force=True)
        self.serving_signature = self.make_serving_signature()

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = None) -> tf.Tensor:
        """_summary_

        Args:
            inputs (Dict[str, tf.Tensor]): _description_
            training (bool, optional): _description_.
                Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            tf.Tensor: _description_
        """
        raise NotImplementedError(
            """`call` method must be implemented in a child class. Expected `inputs` argument 
            is a dict of tensors with mandatory keys `image` and `steps`. In case of class 
            conditioning modeling, a key `label` is also expected.
            """
        )

    def train_step(self, data: Iterable[tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Training procedure, Algorithm 1, https://arxiv.org/pdf/2006.11239.pdf.

        Args:
            data (Iterable[tf.Tensor]): _description_

        Returns:
            Dict[str, tf.Tensor]: _description_
        """
        sample, label, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        batch_size = tf.shape(sample)[0]

        steps = tf.random.uniform(
            shape=[batch_size, 1], maxval=self.maxstep, dtype=tf.int32
        )

        with tf.GradientTape() as tape:
            # Sample Gaussian noise
            eps_target = tf.random.normal([batch_size] + self.image_shape.as_list())

            # Add noise: q(x_t | x_0)
            noisy_sample = self.diffusion_scheduler.add_noise(
                sample=sample, steps=steps, eps=eps_target
            )
            # Predict noise with optional class conditioning
            eps_pred = self(
                {"image": noisy_sample, "step": steps, "label": label}, training=True
            )
            # Mean squared error calculation
            loss = self.compiled_loss(
                eps_target, eps_pred, sample_weight, regularization_losses=self.losses
            )

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(eps_target, eps_pred, sample_weight)
        return self.get_metrics_result()

    def reverse_step(
        self, noisy_sample: tf.Tensor, steps: tf.Tensor, label: tf.Tensor = None
    ) -> tf.Tensor:
        """Reverse (generative) step from denoising U-Net

        Args:
            noisy_sample (tf.Tensor): _description_
            steps (tf.Tensor): _description_
            label (tf.Tensor, optional): _description_. Defaults to None.

        Returns:
            tf.Tensor: _description_
        """
        sampling_size, multiplier = tf.shape(noisy_sample)[0], self.image_shape.rank

        # Predicted distribution parameters from diffusion posterior
        predicted_output = self(
            {"image": noisy_sample, "step": steps, "label": label}, training=False
        )
        predicted_initial = self.diffusion_scheduler.remove_noise(
            noisy_sample=noisy_sample, steps=steps, eps=predicted_output
        )
        predicted_initial = tf.clip_by_value(predicted_initial, -1.0, 1.0)

        (
            predicted_mean,
            _,
            posterior_log_variance,
        ) = self.diffusion_scheduler.diffusion_posterior(
            initial_sample=predicted_initial, noisy_sample=noisy_sample, steps=steps
        )

        # Sample from predicted Gaussian distribution
        noise = tf.random.normal([sampling_size] + self.image_shape.as_list())

        variance_mask = 1 - tf.cast(tf.equal(steps, 0), tf.float32)
        variance_mask = tf.reshape(variance_mask, [sampling_size] + multiplier * [1])

        denoised_sample = (
            predicted_mean
            + variance_mask * tf.exp(0.5 * posterior_log_variance) * noise
        )

        return denoised_sample

    def generate(
        self,
        sampling_size: int,
        label: tf.Tensor = None,
        output_trajectory: bool = False,
        verbose: int = 1,
    ) -> Union[tf.Tensor, List[tf.Tensor]]:
        """_summary_

        Args:
            sampling_size (int): _description_
            label (tf.Tensor, optional): _description_.
                Defaults to None.
            output_trajectory (bool, optional): _description_.
                Defaults to False.
            verbose (int, optional): _description_.
                Defaults to 1.

        Raises:
            TypeError: _description_
            ValueError: _description_

        Returns:
            Union[tf.Tensor, List[tf.Tensor]]: _description_
        """
        if self.class_conditioning:
            if label is None:
                raise TypeError(
                    """`label` must be a valid iterable/tensor of length 1 or 
                    `sampling_size`. Received None.
                    """
                )
            else:
                label_size = tf.shape(label)[0]

                if (label_size != 1) and (label_size != sampling_size):
                    raise ValueError(
                        "`label` length must be equal to 1 or `sampling_size`"
                    )
                elif label_size == 1:
                    label = tf.repeat(label, repeats=[sampling_size], axis=0)

        if output_trajectory:
            sample = [tf.random.normal([sampling_size] + self.image_shape)]
        else:
            sample = tf.random.normal([sampling_size] + self.image_shape)

        iteration = 0
        progbar = keras.utils.Progbar(target=self.maxstep, verbose=verbose)

        for step in reversed(range(self.maxstep)):
            steps = tf.expand_dims(tf.repeat(step, [sampling_size]), axis=1)

            if output_trajectory:
                sample.append(
                    self.reverse_step(noisy_sample=sample[-1], steps=steps, label=label)
                )
            else:
                sample = self.reverse_step(
                    noisy_sample=sample, steps=steps, label=label
                )

            progbar.update(iteration, finalize=False)
            iteration += 1

        progbar.update(iteration, finalize=True)

        return sample

    def inpaint(
        self,
        image: tf.Tensor,
        mask: tf.Tensor,
        resampling_steps: int,
        label: tf.Tensor = None,
        output_trajectory: bool = False,
        verbose: int = 1,
    ) -> Union[tf.Tensor, List[tf.Tensor]]:
        """_summary_

        Args:
            image (tf.Tensor): _description_
            mask (tf.Tensor): _description_
            label (Iterable[int], optional): _description_.
                Defaults to None.
            output_trajectory (bool, optional): _description_.
                Defaults to False.
            verbose (int, optional): _description_.
                Defaults to 1.

        Returns:
            Union[tf.Tensor, List[tf.Tensor]]: _description_
        """
        return

    def make_serving_function(self, force: bool = False):
        """Make generative function for serving"""
        if self.serving_function is not None and not force:
            return self.serving_function

        if self.class_conditioning:

            def serving_function(labels):
                sampling_size = tf.shape(labels)[0]
                sample = tf.random.normal([sampling_size] + self.image_shape.as_list())
                step = tf.constant(self.maxstep - 1, dtype=tf.int32)

                cond = lambda step, sample: tf.greater_equal(step, 0)
                body = lambda step, sample: (
                    step - 1,
                    self.reverse_step(
                        noisy_sample=sample,
                        steps=tf.expand_dims(tf.repeat(step, [sampling_size]), axis=1),
                        label=labels,
                    ),
                )
                _, sample = tf.while_loop(
                    cond,
                    body,
                    loop_vars=(step, sample),
                    shape_invariants=(step.get_shape(), sample.get_shape()),
                )
                return sample

        else:

            def serving_function(sampling_size):
                sample = tf.random.normal([sampling_size] + self.image_shape.as_list())
                step = tf.constant(self.maxstep - 1, dtype=tf.int32)

                cond = lambda step, sample: tf.greater_equal(step, 0)
                body = lambda step, sample: (
                    step - 1,
                    self.reverse_step(
                        noisy_sample=sample,
                        steps=tf.expand_dims(tf.repeat(step, [sampling_size]), axis=1),
                    ),
                )
                _, sample = tf.while_loop(
                    cond,
                    body,
                    loop_vars=(step, sample),
                    shape_invariants=(step.get_shape(), sample.get_shape()),
                )
                return sample

        return serving_function

    def make_serving_signature(self):
        """Make input signature of serving-purpose generative function"""
        if self.class_conditioning:
            return [
                tf.TensorSpec(
                    shape=[
                        None,
                    ],
                    dtype=tf.int32,
                )
            ]

        return [tf.TensorSpec(shape=[], dtype=tf.int32)]

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "image_shape": self.image_shape.as_list(),
                "data_format": self.data_format,
                "class_conditioning": self.class_conditioning,
                "n_classes": self.n_classes,
                "maxstep": self.maxstep,
                "beta_min": self.beta_min,
                "beta_max": self.beta_max,
            }
        )
        return config
