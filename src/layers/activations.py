"""Useful activation functions/layers"""


from typing import Dict

import tensorflow as tf
from tensorflow.keras import layers

from tensorflow_addons import types


class CELU(layers.Layer):
    def __init__(self, alpha: float = 1.0, axis: int = -1, **kwargs):
        super().__init__(**kwargs)
        if axis is None:
            raise ValueError(
                "`axis` of a CELU layer cannot be None, expecting an integer."
            )
        self.axis = axis
        self.alpha = float(alpha)
        self.elu = layers.ELU(alpha=self.alpha)
        self.concat = layers.Concatenate(axis=self.axis)

    def call(self, inputs: types.FloatTensorLike) -> types.FloatTensorLike:
        return self.elu(self.concat([inputs, -inputs]))

    def get_config(self) -> Dict:
        config = super(CELU, self).get_config()
        config.update({"alpha": self.alpha, "axis": self.axis})
        return config
