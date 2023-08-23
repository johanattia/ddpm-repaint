from tf_ddpm import base_diffusion, callbacks, layers, scheduler, unet, utils


CONFIG_ADAM_32x32 = dict(
    learning_rate=2e-4,
    use_ema=True,
    ema_momentum=0.9999,
    ema_overwrite_frequency=1,
)

CONFIG_ADAM_64x64 = dict(
    learning_rate=2e-4,
    use_ema=True,
    ema_momentum=0.9999,
    ema_overwrite_frequency=1,
)

CONFIG_ADAM_256x256 = dict(
    learning_rate=2e-5,
    use_ema=True,
    ema_momentum=0.9999,
    ema_overwrite_frequency=1,
)
