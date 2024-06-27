import os
import copy
import warnings
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from typing import Callable, List

from tinymlgen import port as get_c_code
from tensorflow.keras import layers

from nnom.scripts.nnom_utils import generate_model

SIZE_UNITS = ['B', "KB", "MB", "GB", "TB"]
KiB = 1024
DEFAULT_OUTPUT_FOLDER = "outputs"


def clone_layer(layer, seed=1337):
    # https://www.tensorflow.org/api_docs/python/tf/keras/models/clone_model
    config = layer.get_config()
    if seed is not None and "seed" in config:
        config["seed"] = seed
    new_layer = layer.__class__.from_config(config)
    # if hasattr(layer, "weights"):
    #     new_layer.set_weights(copy.deepcopy(layer.weights))
    return new_layer


def is_input_layer(layer):
    return "input" in layer.name


def check_split(model, segments, inp):
    expected = model(inp).numpy()
    segments_result = inp
    for segment in segments:
        segments_result = segment(segments_result)
    return np.array_equal(expected, segments_result)


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

        self.dense_1 = layers.Dense(inp_size, activation="relu", dtype=kwargs.get("dtype", tf.float32))
        self.dense_2 = layers.Dense(num_outputs, activation=output_activation, dtype=kwargs.get("dtype", tf.float32))

    def call(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)


class LargeClassifier(BaseModel):
    def __init__(self, name=None, enc_in_size=40, enc_out_size=20, num_outputs=1):
        super().__init__(name=name)

        self.encode = keras.Sequential(
            [layers.Dense(enc_in_size, activation="relu") for _ in range(3)]
            + [layers.Dense(enc_out_size, activation="relu")]
        )
        self.classify = SmallClassifier(inp_size=enc_out_size, num_outputs=num_outputs)

    def call(self, x):
        encoding = self.encode(x)
        return self.classify(encoding)


def iter_layers(
        model: keras.Model | keras.Sequential,
        contains_layers=lambda layer: isinstance(
            layer, keras.Model | keras.Sequential
        )):
    """
    Yields sublayers in model

    Parameters
    ----------
    model : keras.Model | keras.Sequential
        Model with layers to iterate over
    contains_layers : TYPE, optional
        Used to check if layer contains sub layers that should be yielded
        instead of the layer itself.
        Can be set to lambda: False to
        iterate over only the top-level layers of the model (model.layers).
        The default is
        lambda layer: isinstance(layer, keras.Model | keras.Sequential).

    Yields
    ------
    tf.Module
        Specific type of yielded elements will depend on contains_layers
        function (Will not yield Model or Sequential with default function).

    """

    for layer in model.layers:
        if contains_layers(layer):
            yield from iter_layers(layer)
            continue
        yield layer


def calc_model_size(model: keras.Model, units="KB"):
    """
    Calculcate size of a keras.Model

    Parameters
    ----------
    model : keras.Model
        model to calculate size of.
    units : str, optional
        units of output size. The default is "KB".

    Returns
    -------
    float
        size of the model in units of `units`.

    """

    div = KiB ** SIZE_UNITS.index(units.upper())
    return sum(p.size * p.itemsize for p in model.get_weights()) / div


def model_wrap(layers: tf.Module | list | tuple, suppress_warnings=False):
    """
    Wrap tf.Modules in keras.Model for saving with tinymlgen.port or
    tf.lite.TFLiteConverter.from_keras_model

    Parameters
    ----------
    layers : tf.Module | list | tuple
        module or iterable of modules to wrap in a keras Model.
        If module is iterable, it must contain at least 1 module
        Requires that the module is built (module.built == True)

    Returns
    -------
    model : keras.Model
        model that when called with an input, returns the same output as
        module(input).

    """

    if isinstance(layers, tf.Module):
        layers = [layers]

    # Build inputs
    inp_shape = layers[0].input_shape
    if is_input_layer(layers[0]):  # Ignore input layer
        inp_shape = inp_shape[0]
        layers = layers[1:]
    if not suppress_warnings and not layers:
        warnings.warn("Wrapping single Input layer: pointless wrap", RuntimeWarning)
    inputs = keras.Input(inp_shape[1:])

    # Build outputs
    outputs = inputs
    for layer in layers:
        # Copy layers to avoid disconnected graph error
        outputs = clone_layer(layer)(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Copy weights from original layers to new model
    num_model_layers = len(model.layers) - 1  # Model layers - input layer
    num_inp_layers = len(layers)
    assert num_model_layers == num_inp_layers, f"{num_model_layers} != {num_inp_layers}"
    for i, layer in enumerate(layers):
        if not hasattr(layer, "weights"):
            continue
        model.layers[i + 1].set_weights(layer.weights)

    return model


def split_by_num_segments(num_segments: int):
    """
    Create splitter for use in make_c_code function.
    Splitter will split model into `num_segments` equal segments of
    len(model.layers) // num_segments.
    len(model.layers) // num_segments must be non-zero

    Parameters
    ----------
    num_segments : int
        Number of segments to split model into.

    Returns
    -------
    function
        Function that when called with a model, returns a list the number of layers
        in each segment to split the model into.

    """

    def splitter(model: keras.Model):
        all_model_layers = list(iter_layers(model))
        num_layers = len(all_model_layers)
        layers_per_segment = num_layers / num_segments
        assert layers_per_segment >= 1, f"Not enough layers for {num_segments} segements"
        target_segment_len = round(layers_per_segment)
        segments = [0]

        i = 0
        while i < num_layers:
            segment_completion = 0
            while (
                segment_completion != target_segment_len
                and i < num_layers
            ):
                # Input layers do not count torward segment completion
                segment_completion += int(
                    not is_input_layer(all_model_layers[i])
                )
                segments[-1] += 1
                i += 1
            if len(segments) < num_segments:
                segments.append(0)

        # Redistribute layers
        j = num_segments - 1
        while j > 0 and segments[j] < 1:
            segments[j - 1] -= 1
            segments[j] += 1
            j -= 1
        return segments

    return splitter


def split_by_size(target_max_size: int | float):
    """
    Create splitter for use in make_c_code function.

    Parameters
    ----------
    target_max_size : int | float
        size (in KB) of segments to split model into.

    Returns
    -------
    function
        Function that when called with a model, returns a list the number of layers
        in each segment to split the model into.

    """

    def contains_input_layer(layers):
        return len(list(filter(lambda l: is_input_layer(l), layers))) > 0

    def splitter(model: keras.Model):
        segment_lengths = []
        current_segment_layers = []
        all_layers = list(iter_layers(model))

        for i, layer in enumerate(all_layers):
            current_segment_layers.append(layer)
            if len(current_segment_layers) == 1 and is_input_layer(layer):
                continue
            segment = model_wrap(current_segment_layers)
            segment_size = calc_model_size(segment, units="KB")

            if segment_size >= target_max_size or i == len(all_layers) - 1:
                next_segment_layers = []
                if (
                    segment_size > target_max_size
                    and len(current_segment_layers) > 1
                    and not (
                        len(current_segment_layers) == 2
                        and contains_input_layer(current_segment_layers)
                    )
                ):
                    # Move last layer in segment to next segment if segment too large
                    last_added_layer = current_segment_layers.pop(-1)
                    next_segment_layers.append(last_added_layer)

                segment_lengths.append(len(current_segment_layers))
                # Put remaing layer in segment of its own
                if i == len(all_layers) - 1 and next_segment_layers:
                    segment_lengths.append(1)
                current_segment_layers = next_segment_layers

        return segment_lengths

    return splitter


def save_tflite_model(model, file_name):
    """
    Export model to tflite file

    Parameters
    ----------
    model : keras.Model
        model to export to tflite file.
    file_name : str
        full path of save file to create without file extension.

    Returns
    -------
    None.

    """

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open(file_name + ".tflite", 'wb') as file:
        file.write(tflite_model)


def save_tinymlgen_model(model, file_name):
    """
    Export model to c code for use with EloquentML

    Parameters
    ----------
    model : keras.Model
        model to export to c_code.
    file_name : str
        full path of save file to create without file extension.

    Returns
    -------
    None.

    """

    c_code = get_c_code(model)

    with open(file_name + ".h", "w+", encoding="utf-8") as file:
        file.write(c_code)


def save_nnom_model(model, file_path, num_samples=1000):
    # TODO: Adapt to accept representative dataset other than random
    parent, file_name = os.path.split(file_path)
    nnom_path = os.path.join(parent, "nnom")
    os.makedirs(nnom_path, exist_ok=True)
    generate_model(
        model,
        np.random.rand(num_samples, *model.input_shape[1:]),
        name=os.path.join(nnom_path, f"{file_name}.h")
    )


def split_model(
        model: keras.Model,
        splitter: int | Callable[..., List[int]],
        output_folder=DEFAULT_OUTPUT_FOLDER,
        saver=None):
    """
    Splits model into segments derived from `splitter` and saves the segments
    Requires that all model layers are built (layer.built == True)

    Parameters
    ----------
    model : keras.Model
        Model to split and save.
    splitter : TYPE, optional
        Used to split model into segments. takes keras.Model as argument and
        returns a list of ints (segment sizes).
        The sum of segment sizes should be equal to the number of layers in the model.
        The default is split_by_size(1).
    output_folder : str, optional
        Folder to save output in. The default is ''.
    saver : function, optional
        Used to save model segments, takes model and file_path without extention.
        The default is save_tinymlgen_model.

    Returns
    ----------
    segments : list of keras.Model
        segments of model created and saved

    """

    if type(splitter) is int:
        splitter = split_by_num_segments(splitter)

    segment_sizes = splitter(model)

    segment_indicies = [
        (prev_sum := sum(segment_sizes[:i]), prev_sum + segment_sizes[i])
        for i in range(len(segment_sizes))
    ]

    if saver:
        save_root = os.path.join(output_folder, model.name)
        os.makedirs(save_root, exist_ok=True)

    segments = []
    all_model_layers = list(iter_layers(model))
    for i in range(len(segment_indicies)):
        start, end = segment_indicies[i]
        segment_layers = all_model_layers[start:end]
        segment = model_wrap(segment_layers)

        if saver:
            file_name = os.path.join(save_root, f"{model.name}_{i}")
            saver(segment, file_name)
        segments.append(segment)

    test_input = np.random.rand(1, *segments[0].input_shape[1:])
    if not check_split(model, segments, test_input):
        raise Exception(
            f"Result of split model on {test_input} does not match model."
        )

    return segments


def tiny_model_func(input_shape, num_outputs=1):
    inputs = keras.Input(shape=input_shape)
    x = layers.Dense(10)(inputs)
    x = layers.ReLU()(x)
    x = layers.Dense(num_outputs)(x)
    return keras.Model(inputs=inputs, outputs=x)

if __name__ == "__main__":
    small_classifier = SmallClassifier()
    # initialize model with input
    small_classifier(keras.Input((1, 30)))
    split_model(small_classifier, 2)

    large_classifier = LargeClassifier()
    large_classifier(keras.Input((1, 50)))
    split_model(large_classifier, split_by_size(13))
