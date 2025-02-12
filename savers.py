import os

import tensorflow as tf
from tinymlgen import port as get_c_code
from nnom.scripts.nnom_utils import generate_model

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


def get_nnom_saver(save_root, nnom_subdir=True):
    def save(segment_id, model, x_test):
        assert not isinstance(x_test, type(None)), (
            "A reference dataset is required for NNoM quantization"
        )
        weights_path = None
        if save_root:
            _, save_name = os.path.split(save_root)
            nnom_dir = save_root
            if nnom_subdir:
                nnom_dir = os.path.join(save_root, "nnom")
            os.makedirs(nnom_dir, exist_ok=True)
            weights_path = os.path.join(nnom_dir, f"{save_name}_{segment_id}.h")
        return weights_path is not None, generate_model(model, x_test, name=weights_path)
    return save
