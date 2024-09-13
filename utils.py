import warnings
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras import layers as kl

SIZE_UNITS = ['B', "KB", "MB", "GB", "TB"]
KiB = 1024


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


def group_layers(layers: list | keras.Model, independent_activations=False):
    if isinstance(layers, keras.Model):
        layers = list(iter_layers(layers))

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


def get_connection_key(node_name, connections):
    for inputs, outputs in connections:
        if node_name in outputs:
            return inputs, outputs
    return None


def get_parent_result(
        node_name, connections, intermediate_results,
        default_func=lambda node_name: None):
    connection_key = get_connection_key(node_name, connections)
    if connection_key:
        inputs, outputs = connection_key
        parent_results = [
            intermediate_results[parent_name]
            for parent_name in inputs
        ]
        merge_func = (
            (lambda x: connections[connection_key](x).numpy())
            if len(inputs) > 1 else lambda x: x[0]
        )
        combined_parent_result = merge_func(parent_results)
    else:
        combined_parent_result = default_func(node_name)
    return combined_parent_result


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
        assert not is_input_layer(layer), (
            "Input layers are only accepted as first layer in `layers`"
        )
        is_activ_layer = is_activation_layer(layer)

        # Copy layers to avoid disconnected graph error
        config_update = {}
        if getattr(layer, "weights", []):
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


def quantization_error(model, quantized_input, quantized_output, output_shift, input_shift=7):
    """
    Determine error of quantized nnom model running on MCU
    """
    dequantized_inp_arr = np.array(
        quantized_input
    ).reshape(1, *model.layers[0].input.shape[1:]) / (2 ** input_shift)
    dequantized_expected = model(dequantized_inp_arr).numpy().flatten()
    suboptimal_quantizations = {}
    total_error = 0
    for i, (q, ex_de_q) in enumerate(zip(quantized_output, dequantized_expected)):
        best_q = min([q - 1, q, q + 1], key=lambda n: abs(ex_de_q - n / (2 ** output_shift)))
        if q != best_q:
            suboptimal_quantizations[i] = (q, best_q)
        total_error += abs(q / (2 ** output_shift) - ex_de_q)
    return total_error / len(quantized_output), suboptimal_quantizations
