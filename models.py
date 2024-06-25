import tensorflow.keras as keras
from model_splitter import SmallClassifier, LargeClassifier


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
