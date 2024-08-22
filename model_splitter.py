import os
import copy
import warnings
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from typing import Callable, List

from tinymlgen import port as get_c_code
from tensorflow.keras import layers as kl

from nnom.scripts.nnom_utils import generate_model

SIZE_UNITS = ['B', "KB", "MB", "GB", "TB"]
KiB = 1024
DEFAULT_OUTPUT_FOLDER = "outputs"
def is_input_layer(layer: kl.Layer):
    """
    Check if layer is an input layer

    Parameters
    ----------
    layer : kl.Layer

    Returns
    -------
    bool
        True if layer is an input layer, False otherwise.

    """
    return "input" in layer.name


def get_input_list(model: keras.Model | kl.Layer):
    """
    Return list of model/layer's inputs

    Parameters
    ----------
    model : keras.Model | kl.Layer

    Returns
    -------
    inputs : list
        List of keras_tensors.

    """
    inputs = model.input
    if not isinstance(inputs, list):
        inputs = [inputs]
    return inputs


def is_activation_layer(layer: kl.Layer):
    """
    Check if layer is an activation layer

    Parameters
    ----------
    layer : kl.Layer

    Returns
    -------
    bool
        True if layer is an activatin layer, False otheriwse.

    """
    cls_str = str(type(layer))
    return any(
        activ_indicator in cls_str
        for activ_indicator in [
            "layers.activation",
            "layers.core.activation"
        ]
    )


def is_branching_model(model: keras.Model):
    """
    Check if there are multiple input branches in a model

    Parameters
    ----------
    model : keras.Model
        Tensorflow funcitonal model.

    Returns
    -------
    bool
        True if model contains branches/parallel steps,
        False otherwise (purely sequential).

    """
    return len([
        layer for layer in iter_layers(model)
        if is_input_layer(layer)
    ]) > 1


def clone_layer(layer, seed=1337, **config_vars):
    """
    Clone tensorflow layer

    Parameters
    ----------
    layer : kl.Layer
    seed : int, optional
        Random seed. The default is 1337.
    config_vars : dict
        Config parameters to override when cloning layer.

    Returns
    -------
    new_layer : kl.Layer
        Clone of input layer.

    """
    # https://www.tensorflow.org/api_docs/python/tf/keras/models/clone_model
    config = layer.get_config()
    config.update(config_vars)
    if seed is not None and "seed" in config:
        config["seed"] = seed
    new_layer = layer.__class__.from_config(config)
    return new_layer


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


def iter_layers(
        model: keras.Model | keras.Sequential,
        contains_layers=lambda layer: isinstance(
            layer, keras.Model | keras.Sequential
        ),
        skip_input_layers=False):
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
        if skip_input_layers and is_input_layer(layer):
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
    last_non_activ_name = None
    for layer in layers:
        assert not is_input_layer(layer), "Input layers are only accepted as first layer in `layers`"
        is_activ_layer = is_activation_layer(layer)

        # Copy layers to avoid disconnected graph error
        config_update = {}
        if not is_activ_layer:
            config_update["activation"] = None
        layer_clone = clone_layer(layer, **config_update)
        outputs = layer_clone(outputs)

        # Forcibly set activation input name to last layer
        # Otherwise, when activations from keras.layer.activation are present in the
        # source layers, a cloned activation layer's input name will be a placeholder
        # (unsure why) instead of the name of the last layer, which will break NNoM
        if is_activ_layer:
            layer_clone.input._name = last_non_activ_name
            continue
        else:
            last_non_activ_name = layer_clone.name

        # NNoM does not support activation in layer, so make separate activation layer
        if (activation := layer.get_config().get("activation")) and activation != "linear":
            if isinstance(activation, str):
                activation = keras.layers.Activation(activation)
            outputs = activation(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Copy weights from original layers to new model
    weights_dict = {layer.name: layer.get_weights() for layer in layers}
    for layer in model.layers:
        if is_input_layer(layer) or layer.name not in weights_dict:
            continue
        layer.set_weights(weights_dict[layer.name])

    return model


def group_layers(layers, independent_activations=False):
    groups = []
    current_group = []
    for layer in layers:
        if is_input_layer(layer):
            if current_group:
                groups.append(current_group)
                current_group = []
            current_group.append(layer)
        elif not independent_activations and is_activation_layer(layer):
            groups[-1].append(layer)
        else:
            current_group.append(layer)
            groups.append(current_group)
            current_group = []
    if current_group:
        groups.append(current_group)
    return groups


def split_by_num_segments(num_segments: int, independent_activations=False):
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
        grouped_layers = group_layers(
            all_model_layers,
            independent_activations=independent_activations
        )
        layers_per_segment = len(all_model_layers) / num_segments
        smallest_group = min(grouped_layers, key=len)
        assert layers_per_segment >= len(smallest_group), (
            "Not enough layers for {num_segments} segements when "
            f"smallest group is size {len(smallest_group)}"
        )

        while len(grouped_layers) > num_segments:
            sm_idx = grouped_layers.index(smallest_group)
            neighbors = []
            if sm_idx > 0:
                neighbors.append(sm_idx - 1)
            if sm_idx < len(grouped_layers) - 1:
                neighbors.append(sm_idx + 1)
            fuse_idx = min(neighbors, key=lambda i: len(grouped_layers[i]))
            lower_idx, upper_idx = sorted([sm_idx, fuse_idx])
            grouped_layers[lower_idx].extend(grouped_layers.pop(upper_idx))
            smallest_group = min(grouped_layers, key=len)

        segments = [len(group) for group in grouped_layers]
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
        return len(list(filter(is_input_layer, layers))) > 0

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


def save_tflite_model(model, file_name, _last_saver_result=None):
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


def save_tinymlgen_model(model, file_name, _last_saver_result=None):
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


def get_nnom_saver(init_test_set=None, num_samples=1000):
    def save(model, file_path, x_test=None):
        parent, file_name = os.path.split(file_path)
        nnom_path = os.path.join(parent, "nnom")
        os.makedirs(nnom_path, exist_ok=True)
        if type(x_test) is type(None):
            x_test = (
                init_test_set
                if type(init_test_set) is not type(None)
                else np.random.rand(num_samples, *model.input_shape[1:])
            )
        generate_model(
            model, x_test,
            name=os.path.join(nnom_path, f"{file_name}.h")
        )
        return model(x_test).numpy()
    return save


def save_nnom_model(model, file_path, x_test=None, num_samples=1000):
    return get_nnom_saver(x_test, num_samples=num_samples)(
        model, file_path
    )


def addr(obj: object):
    return hex(id(obj))


def get_prev_layer(keras_tensor):
    return keras_tensor._keras_history.layer


def segment_branching_model(model: keras.Model):
    blocks = []
    nodes = {}
    seen = set()

    def find_block_by_tail(tail_name):
        for block in blocks:
            if tail_name == block[-1].name:
                return block
        return None

    for i, layer in enumerate(iter_layers(model)):
        if addr(layer) in seen:
            continue
        if is_input_layer(layer):
            blocks.append([layer])
            continue
        inputs = layer.input
        outputs = layer.output
        single_input = not isinstance(inputs, list)
        single_output = not isinstance(outputs, list)

        ## Extend existsing block
        if single_input and single_output:
            input_name = get_prev_layer(inputs).name
            target_block = find_block_by_tail(input_name)
            assert target_block
            target_block.append(layer)
            seen.add(addr(layer))
            continue

        ## Create node and new blocks for each output
        if single_input:
            inputs = [inputs]
        try:
            node_input_names = tuple(
                find_block_by_tail(get_prev_layer(inp).name)[0].name
                for inp in inputs
            )
        except TypeError as e:
            print("Inputs to layers accepting multiple inputs must be the output of a block.")
            raise e
        node_output_names = []
        # Search remaining layers for layers that use one of current layer's output as input
        for search_layer in model.layers[i + 1:]:
            search_layer_inp = search_layer.input
            if (
                isinstance(search_layer_inp, list)
                or get_prev_layer(search_layer_inp).name != layer.name
            ):
                continue
            block_start_layer = search_layer
            node_output_names.append(block_start_layer.name)
            blocks.append([block_start_layer])
            seen.add(addr(block_start_layer))

        node_name = (node_input_names, tuple(node_output_names))
        nodes[node_name] = layer
        seen.add(addr(layer))
    return {block[0].name: block for block in blocks}, nodes


def lateral_input_split(model: keras.Model, keras_input: keras.Input):
    input_shape = list(keras_input.shape)
    assert input_shape[-1] % 2 == 0
    input_shape[-1] = keras_input.shape[-1] // 2
    split_inputs = [keras.Input(input_shape), keras.Input(input_shape)]

    model_layers = list(iter_layers(model, skip_input_layers=True))
    split_layer = model_layers[0]
    layer_parts = [
        keras.Model(
            inputs=inp,
            outputs=clone_layer(
                split_layer,
                name=split_layer.name + f"_{i}",
                activation=None)(inp))
        for i, inp in enumerate(split_inputs)
    ]
    outputs = kl.Add()([m.output for m in layer_parts])
    activation = split_layer.get_config().get("activation")
    if activation:
        if isinstance(activation, str):
            activation = getattr(keras.activations, activation)
        outputs = activation(outputs)

    weights_dict = {}
    for layer in model_layers[1:]:
        weights_dict[layer.name] = layer.weights
        outputs = clone_layer(layer)(outputs)

    new_model = keras.Model(inputs=split_inputs, outputs=outputs)
    # Copy weights
    for i, sub_model in enumerate(layer_parts):
        copy_kernel, copy_bias = split_layer.weights
        sub_model.layers[-1].set_weights([
            copy_kernel[input_shape[-1] * i: input_shape[-1] * (i + 1)],
            copy_bias / 2
        ])
    for layer in iter_layers(new_model):
        if layer.name in weights_dict:
            layer.set_weights(weights_dict[layer.name])
    return new_model


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

    if isinstance(splitter, int):
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
    last_saver_result = None
    for i, (start, end) in enumerate(segment_indicies):
        segment_layers = all_model_layers[start:end]
        segment = model_wrap(segment_layers)

        if saver:
            file_name = os.path.join(save_root, f"{model.name}_{i}")
            last_saver_result = saver(segment, file_name, last_saver_result)
        segments.append(segment)

    test_input = np.random.rand(1, *segments[0].input_shape[1:])
    if not check_split(model, segments, test_input):
        raise Exception(
            f"Result of split model on {test_input} does not match model."
        )

    return segments


def tiny_model_func(input_shape, num_outputs=1):
    inputs = keras.Input(shape=input_shape)
    x = kl.Dense(10)(inputs)
    x = kl.ReLU()(x)
    x = kl.Dense(num_outputs)(x)
    return keras.Model(inputs=inputs, outputs=x)

if __name__ == "__main__":
    small_classifier = SmallClassifier()
    # initialize model with input
    small_classifier(keras.Input((1, 30)))
    split_model(small_classifier, 2)

    large_classifier = LargeClassifier()
    large_classifier(keras.Input((1, 50)))
    split_model(large_classifier, split_by_size(13))
