import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import layers as kl

from utils import iter_layers


class BaseModel(keras.Model):
    def build(self, inp_shape):
        # Build constituent layers when model is built
        inputs = keras.Input(inp_shape[1:])
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
    def to_functional(self, inp):
        # inputs = keras.Input(shape=input_shape)
        outputs = inp
        for layer in iter_layers(self):
            outputs = layer(outputs)
        return keras.Model(name=self.name, inputs=inp, outputs=outputs)


class SmallClassifier(BaseModel):
    def __init__(
            self, name=None, inp_size=20, output_activation="sigmoid",
            *args, num_outputs=1, **kwargs):
        super().__init__(name=name, *args, **kwargs)

        self.dense_1 = kl.Dense(inp_size, activation="relu", dtype=kwargs.get("dtype", tf.float32))
        self.dense_2 = kl.Dense(num_outputs, activation=output_activation, dtype=kwargs.get("dtype", tf.float32))

    def call(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)


class LargeClassifier(BaseModel):
    def __init__(self, name=None, enc_in_size=40, enc_out_size=20, num_outputs=1):
        super().__init__(name=name)

        self.encode = keras.Sequential(
            [kl.Dense(enc_in_size, activation="relu") for _ in range(3)]
            + [kl.Dense(enc_out_size, activation="relu")]
        )
        self.classify = SmallClassifier(inp_size=enc_out_size, num_outputs=num_outputs)

    def call(self, x):
        encoding = self.encode(x)
        return self.classify(encoding)
    

def tiny_model_func(input_shape, num_outputs=1):
    inputs = keras.Input(shape=input_shape)
    x = kl.Dense(10)(inputs)
    x = kl.ReLU()(x)
    x = kl.Dense(num_outputs)(x)
    return keras.Model(inputs=inputs, outputs=x)


def create_model(num_inputs=20):
    sm_inverter = SmallClassifier(
        "sm_inverter", inp_size=20, num_outputs=20
    )
    return sm_inverter.to_functional(keras.Input((num_inputs,)))


def make_large_classifier(enc_in_size=40, enc_out_size=20):
    return LargeClassifier(
        enc_in_size=enc_in_size,
        enc_out_size=enc_out_size
    ).to_functional(keras.Input((enc_in_size,)))
