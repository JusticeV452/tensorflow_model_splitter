import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers as kl
from utils import (
    clone_layer, is_input_layer, calc_model_size,
    iter_layers, group_layers, model_wrap
)

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

    def splitter(layers: list | tuple | keras.Model):
        if isinstance(layers, keras.Model):
            layers = list(iter_layers(layers))
        grouped_layers = group_layers(
            layers,
            independent_activations=independent_activations
        )
        layers_per_segment = len(layers) / num_segments
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

    def splitter(layers: list | tuple | keras.Model):
        segment_lengths = []
        current_segment_layers = []

        if isinstance(layers, keras.Model):
            layers = list(iter_layers(layers))

        for i, layer in enumerate(layers):
            current_segment_layers.append(layer)
            if len(current_segment_layers) == 1 and is_input_layer(layer):
                continue
            segment = model_wrap(current_segment_layers)
            segment_size = calc_model_size(segment, units="KB")

            # Continue extending segment if there are more segments
            # and current segment is smaller than requested
            if segment_size < target_max_size and i != len(layers) - 1:
                continue

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
            if i == len(layers) - 1 and next_segment_layers:
                segment_lengths.append(1)
            current_segment_layers = next_segment_layers

        return segment_lengths

    return splitter


def lateral_input_split(model: keras.Model, keras_input: keras.Input):
    """
    Split the input layer of a simple sequential, functional keras model

    Parameters
    ----------
    model : keras.Model
        Functional purely sequential model.
    keras_input : keras.Input
        Size of input that will be split.

    Returns
    -------
    new_model : keras.Model
        Model with two inputs, the first input being the first half of the
        original model's input. The second input being the second half

    """
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
                activation=None
            )(inp))
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
