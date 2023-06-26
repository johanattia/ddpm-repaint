"""Feed-forward net/multi-layer perceptron"""


from typing import Callable, Iterable, Union
import tensorflow as tf


def FFN(
    output_units: int = None,
    hidden_units: Iterable[int] = [],
    output_activation: Union[str, Callable] = None,
    hidden_activation: Union[str, Callable] = None,
    use_bias: bool = True,
    dropout: float = None,
    **kwargs,
):
    name = kwargs.pop("name", "FFN")
    layers = []
    for units in hidden_units:
        if dropout is not None:
            layers.extend(
                [
                    tf.keras.layers.Dense(
                        units=units,
                        activation=hidden_activation,
                        use_bias=use_bias,
                        **kwargs,
                    ),
                    tf.keras.layers.Dropout(rate=dropout),
                ]
            )
        else:
            layers.append(
                tf.keras.layers.Dense(
                    units=units,
                    activation=hidden_activation,
                    use_bias=use_bias,
                    **kwargs,
                ),
            )
    if output_units is not None:
        layers.append(
            tf.keras.layers.Dense(
                units=output_units,
                activation=output_activation,
                use_bias=use_bias,
                **kwargs,
            )
        )
    return tf.keras.Sequential(layers, name)
