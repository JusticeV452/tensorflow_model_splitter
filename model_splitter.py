import os
import tensorflow as tf
import tensorflow.keras as keras

from tinymlgen import port as get_c_code
from tensorflow.keras import layers

SIZE_UNITS = ['B', "KB", "MB", "GB", "TB"]
KiB = 1024


class SmallClassifier(keras.Model):    
    def __init__(self, name=None):
        super().__init__(name=name)

        self.dense_1 = layers.Dense(20, activation="relu")
        self.dense_2 = layers.Dense(1, activation="sigmoid")
        
    def build(self, inp):
        # Build constituent layers when model is built
        self.dense_1(keras.Input(inp[1:]))
        out_shape1 = keras.Input(self.dense_1.output_shape[1:])
        self.dense_2(out_shape1)
        
    def call(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)


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


def model_wrap(module: tf.Module | list | tuple):
    """
    Wrap a tf.Module in keras.Model for saving with tinymlgen.port or 
    tf.lite.TFLiteConverter.from_keras_model

    Parameters
    ----------
    module : tf.Module | list | tuple
        module or iterable of modules to wrap in a keras Model.
        If module is iterable, it must contain at least 1 module
        Requires that the module is built (module.built == True)

    Returns
    -------
    model : keras.Model
        model that when called with an input, returns the same output as
        module(input).

    """
    
    if type(module) in [list, tuple] and len(module) > 1:
        # Build sequential
        layers = module
        module = keras.Sequential(layers)        
        module_inp = keras.Input(layers[0].input_shape[1:])
        module(module_inp)
    elif type(module) in [list, tuple]:
        # List of 1 element can be treated as module
        module = module[0]
    
    # Create model with output of module(input)
    inputs = keras.Input(module.input_shape[1:])
    outputs = module(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
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
        layers_per_segment = len(model.layers) // num_segments
        assert layers_per_segment
        remaining_layers = len(model.layers) % num_segments
        segments = [layers_per_segment for _ in range(num_segments)]
        segments[-1] += remaining_layers
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
    
    def splitter(model: keras.Model):
        segment_lengths = []
        current_segment_layers = []
        
        for i, layer in enumerate(model.layers):
            current_segment_layers.append(layer)
            segment = model_wrap(current_segment_layers)
            segment_size = calc_model_size(segment, units="KB")
            
            if segment_size >= target_max_size or i == len(model.layers) - 1:
                next_segment_layers = []
                if (segment_size > target_max_size
                    and len(current_segment_layers) > 1):
                    # Move last layer in segment to next segment if segment too large
                    last_added_layer = current_segment_layers.pop(-1)
                    next_segment_layers.append(last_added_layer)
                    
                segment_lengths.append(len(current_segment_layers))
                # Put remaing layer in segment of its own
                if i == len(model.layers) - 1 and next_segment_layers:
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


def split_model(model: keras.Model, splitter=split_by_size(1), output_folder='', saver=save_tinymlgen_model):
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
    
    segment_sizes = splitter(model)
    
    segment_indicies = [
        (prev_sum := sum(segment_sizes[:i]), prev_sum + segment_sizes[i])
        for i in range(len(segment_sizes))
    ]
    
    save_root = os.path.join(output_folder, model.name)
    os.makedirs(save_root, exist_ok=True)
    
    segments = []
    for i in range(len(segment_indicies)):
        start, end = segment_indicies[i]
        segment_layers = model.layers[start:end]
        segment = model_wrap(segment_layers)        
        
        file_name = os.path.join(save_root, f"{model.name}_{i}")
        saver(segment, file_name)
        segments.append(segment)
    
    return segments


if __name__ == "__main__":
    classifier = SmallClassifier()
    # initialize model with input
    classifier(keras.Input((1, 30)))
    split_model(classifier)
