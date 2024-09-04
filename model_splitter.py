import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from datetime import datetime
from typing import Callable, List, Dict

from tinymlgen import port as get_c_code

from utils import is_input_layer, iter_layers, get_parent_result, model_wrap
from nnom_utils import generate_model

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


def check_split(model, segments, inp):
    expected = model(inp).numpy()
    segments_result = inp
    for segment in segments:
        segments_result = segment(segments_result)
    return np.array_equal(expected, segments_result)


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


def get_nnom_saver(init_test_set=None, num_samples=1000):
    def save(model, save_root, segment_id, x_test=None):
        _, save_name = os.path.split(save_root)
        nnom_path = os.path.join(save_root, "nnom")
        os.makedirs(nnom_path, exist_ok=True)
        if type(x_test) is type(None):
            x_test = (
                init_test_set
                if type(init_test_set) is not type(None)
                else np.random.rand(num_samples, *model.input_shape[1:])
            )
        generate_model(
            model, x_test,
            name=os.path.join(nnom_path, f"{save_name}_{segment_id}.h")
        )
        return model(x_test).numpy()
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
        parent_result = get_parent_result(node_name, connections, saver_results)
        saver_results[node_name] = saver(
            segment, save_root, node_id, parent_result
        )
        blocks[node_name] = segment

    result_model = SegmentedModel(blocks, connections)
    assert result_model.func_eq(orig_model)
    return result_model
