import os
from datetime import datetime
from typing import Callable, List, Dict

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tinymlgen import port as get_c_code

from utils import iter_layers, get_parent_result, model_wrap
from nnom_utils import is_input_layer, generate_model

from segmented_model import get_segment_ids, SegmentedModel

DEFAULT_OUTPUT_FOLDER = "outputs"


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


def save_tflite_model(model, save_root, segment_id, _last_saver_result=None):
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
    _, save_name = os.path.split(save_root)
    file_name = os.path.join(save_root, f"{save_name}_{segment_id}.tflite")
    # Save the model.
    with open(file_name, 'wb') as file:
        file.write(tflite_model)
    return file_name, None


def save_tinymlgen_model(model, save_root, segment_id, _last_saver_result=None):
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
    _, save_name = os.path.split(save_root)
    file_name = os.path.join(save_root, f"{save_name}_{segment_id}.h")
    with open(file_name, "w+", encoding="utf-8") as file:
        file.write(c_code)
    return file_name, None


def get_nnom_saver(init_test_set=None, num_samples=1000):
    def save(model, save_root, segment_id, x_test=None):
        _, save_name = os.path.split(save_root)
        nnom_path = os.path.join(save_root, "nnom")
        os.makedirs(nnom_path, exist_ok=True)
        if isinstance(x_test, type(None)):
            x_test = (
                init_test_set
                if not isinstance(init_test_set, type(None))
                else np.random.rand(num_samples, *model.input_shape[1:])
            )
        weights_path = os.path.join(nnom_path, f"{save_name}_{segment_id}.h")
        generate_model(model, x_test, name=weights_path)
        return weights_path, model(x_test).numpy()
    return save


def save_nnom_model(model, save_root, segment_id, x_test=None, num_samples=1000):
    return get_nnom_saver(x_test, num_samples=num_samples)(
        model, save_root, segment_id
    )


def split_model(
        model: keras.Model | SegmentedModel,
        splitter: int | Callable[..., List[int]] | Dict[str, int | Callable[..., List[int]]],
        output_folder=DEFAULT_OUTPUT_FOLDER,
        save_name=None,
        saver=None):
    """
    Splits model into segments derived from `splitter` and saves the segments
    Requires that all model layers are built (layer.built == True)

    Parameters
    ----------
    model : keras.Model | SegmentedModel
        Model to split and save.
    splitter : int | func | dict, optional
        Used to split model into segments.
        - int, splits blocks in the model into `splitter` segments
        - func, takes keras.Model as argument and returns a list of ints (segment sizes).
        - dict, dictionary mapping block names to splitter int/func
        The sum of segment sizes must be equal to the number of layers in the model.
    output_folder : str, optional
        Folder to save output in. The default is ''.
    save_name : str, optional
        Folder created in output folder that saver will save outputs to.
        The default is model-mm-dd-yy-hh-mm-ss
    saver : function, optional
        Used to save model segments, takes model and file_path without extention.
        The default is None.

    Returns
    ----------
    blocks : dict {str: list of keras.Model}
        segments of model created and saved

    """

    orig_model = model
    if isinstance(model, keras.Model):
        save_name = model.name if save_name is None else save_name
        model = SegmentedModel(model)

    save_name = (
        "model_" + datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        if save_name is None else save_name
    )

    blocks = {}
    upload_info = {}
    saver_results = {}
    nodes, connections = model.extend(splitter)
    node_ids = get_segment_ids(nodes.keys(), connections)

    if saver:
        save_root = os.path.join(output_folder, save_name)
        os.makedirs(save_root, exist_ok=True)

    # Generate intermediate results
    for node_name, node_id in node_ids.items():
        segment = nodes[node_name]
        if isinstance(segment, list):
            segment = model_wrap(segment)
        connection_key, parent_result = get_parent_result(node_name, connections, saver_results)
        save_path, saver_results[node_name] = saver(
            segment, save_root, node_id, parent_result
        )
        blocks[node_name] = segment
        input_thresholds = []
        if connection_key:
            inputs, _ = connection_key
            input_sizes = [saver_results[inp_name][0].size for inp_name in inputs]
            input_thresholds = [
                sum(input_sizes[:i + 1]) for i in range(len(input_sizes))
            ]
        upload_info[node_id] = {
            "save_path": save_path,
            "reduce_type": (
                "MULT"
                if "merging.multiply" in str(type(connections.get(connection_key)))
                else "ADD"
            ),
            "input_thresholds": input_thresholds,
            "row_size": int(node_id.split('_')[-2].split('-')[-1])
        }

    result_model = SegmentedModel(blocks, connections)
    assert result_model.func_eq(orig_model)
    return upload_info, result_model
