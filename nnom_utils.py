"""
Derived from nnom.scripts.nnom_utils
"""

import os
import io
import warnings
import scipy.stats

import numpy as np
import tensorflow as tf

import tensorflow.keras as keras
from tensorflow.keras import layers as kl
from tensorflow.keras.models import Model
from nnom.scripts.nnom_utils import (
    is_shift_layer, is_shift_fixed, convert_to_x4_q7_weights, fuse_bn_to_conv,
    generate_weights as nnom_generate_weights, generate_model,
    layers_output_ranges as nnom_layers_output_ranges
)

from utils import is_input_layer


def get_int_bits(min_value: float, max_value: float):
    """
    Determine the number of bits needed to represent a set of values

    Parameters
    ----------
    min_value : float
        The smallest value in a set of values.
    max_value : float
        The largest value in a set of values.

    Returns
    -------
    int
        The number of bits needed to represent a set of values with a minimum
        of `min_value` and maxmum of `max_value`.

    """
    return int(np.ceil(np.log2(max([abs(min_value), abs(max_value), 1e-10]))))


def pad_filter_sizes(*filter_sizes, pad_val=1, shape=2):
    padded_sizes = []
    for f_size in filter_sizes:
        if type(f_size) is int:
            f_size = [f_size]
        padded_sizes.append(
            # Extend shape with pad_val if len(f_size) < shape
            (*(pad_val,) * (shape - len(f_size)), *f_size) if len(f_size) < shape else tuple(f_size)
        )
    return padded_sizes


def flatten(L):
    if not isinstance(L, list | tuple):
        return [L]
    return sum([flatten(el) for el in L], start=[])


def to_transposed_x4_q7_weights(weights: np.array):
    transposed_wts = np.transpose(weights)
    return convert_to_x4_q7_weights(np.reshape(
        transposed_wts,
        (transposed_wts.shape[0], transposed_wts.shape[1], 1, 1)
    ))


def dec_bits_by_kld(layer: kl.Layer, features, dec_bits: int, verbose: bool=False):
    max_val = features.max()
    min_val = features.min()
    abs_max = max(abs(max_val), abs(min_val))
    small_var = 1e-5
    bins = np.arange(-abs_max, abs_max, abs_max / 2048 * 2)
    q_bins = np.arange(-abs_max, abs_max, abs_max / 256 * 2)
    flat_hist = np.histogram(features.flatten(), bins=bins)[0]
    kl_loss = []
    kl_shifts = []
    for shift in range(4):
        t = 2 ** (dec_bits + shift)     # 2-based threshold
        act = np.round(features.flatten() * t)
        act = act / t
        act = np.clip(act, -128 / t, 127 / t)
        act = np.histogram(act, bins=q_bins)[0]
        act_hist = np.zeros(2047)
        chunk = int(2048 / 256)
        for i in range(int(255)):
            none_zero = np.count_nonzero(flat_hist[i * chunk:(i + 1) * chunk])
            if none_zero == 0:
                continue
            for j in range(chunk):
                act_hist[i * chunk + j] = (
                    act[i] / none_zero
                    if flat_hist[i * chunk + j] != 0
                    else 0
                )
        flat_hist[flat_hist == 0] = small_var
        act_hist[act_hist == 0] = small_var
        kl = scipy.stats.entropy(flat_hist, act_hist)
        kl_loss.append(kl)
        kl_shifts.append(dec_bits + shift)

    # set the dec_bit to the KLD results
    new_dec = kl_shifts[np.argmin(kl_loss)]

    if verbose:
        print("KLD loss:", kl_loss)
        print("KLD shift:", kl_shifts)
    if verbose and dec_bits != new_dec:
        print(layer.name, "is using KLD method, original shift:", dec_bits, "KLD results:", new_dec)

    dec_bits = new_dec
    return dec_bits


def make_initial_shift_list(
        model: keras.Model, x_test: np.array,
        quantize_method: str="max_min", verbose: bool=False):
    shift_list = {}
    last_layer = None
    
    def get_features(model, inp):
        if verbose:
            return model.predict(inp)
        return model(inp).numpy()

    # FIXME: only support one input
    model_layers = model.layers
    if not is_input_layer(model.layers[0]):
        model_layers = [model.input] + model_layers

    for layer in model_layers: # layer loop
        if is_input_layer(layer):
            features = x_test
        # batch_normalization will need to be handled differently, since we are fusing the weight to its predecessor.
        # sigmoid and tanh are different, their shift is fixed to 7
        elif is_shift_layer(layer) or "batch_normalization" in layer.name:
            layer_model = Model(inputs=model.input, outputs=layer.output)
            features = get_features(layer_model, x_test)
        # Otherwise leave the features not changed, so this layer shift will be the same
        # as its inputs
        
        #  calculate no saturation shift
        max_val = features.max()
        min_val = features.min()
        int_bits = get_int_bits(min_val, max_val)
        dec_bits = 7 - int_bits

        # saturation shift, using KLD method
        # Ref: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
        if (
            "kld" in quantize_method
            and not is_shift_fixed(layer)
            and not is_input_layer(layer)
            and "dense" not in layer.name
        ): # test, also do not use kld in input layer
            dec_bits = dec_bits_by_kld(layer, features, dec_bits, verbose=verbose)

        if verbose:
            print(layer.name, "max value:", max_val, "min value:", min_val, "dec bit:", dec_bits)
        
        # record the shift
        shift_name = layer.name
        if isinstance(model.input, tf.Tensor) and not is_input_layer(model.layers[0]):
            shift_name = shift_name.split(':')[0]
        shift_list[shift_name] = dec_bits

        if "batch_normalization" in layer.name:
            # use the bn layer shift to update the last layer.
            shift_list[last_layer.name] = dec_bits
        last_layer = layer
    return shift_list


def layers_output_ranges(model, x_test, quantize_method="max_min", calibrate_size=1000, verbose=False):
    # limit the test data size
    np.random.shuffle(x_test)
    if x_test.shape[0] > calibrate_size:
        x_test = x_test[:calibrate_size]
    
    shift_list = make_initial_shift_list(
        model, x_test,
        quantize_method=quantize_method, verbose=verbose
    )

    layer_dict = {}
    for layer in model.layers:
        layer_dict[layer.name] = layer

    def get_iname(layer):
        return layer.name.split('/')[0]

    def update_previous_layer_shift(init_layer, Qmin, check_input=True):        
        layers = init_layer.input if check_input else init_layer
        if not isinstance(layers, list):
            layers = [init_layer]

        for layer in layers:
            if is_input_layer(layer):
                continue
            iname = get_iname(layer)
            shift_list[iname] = Qmin
            if not is_shift_layer(layer_dict[iname]):
                update_previous_layer_shift(layer_dict[iname], Qmin)

    for layer in reversed(model.layers[1:]):
        if not isinstance(layer.input, list):
            continue

        # detemine Qmin
        Qmin = shift_list[get_iname(layer.input[0])]
        for inp in layer.input:
            Qmin = min(Qmin, shift_list[get_iname(inp)])

        for inp in layer.input:
            update_previous_layer_shift(inp, Qmin, check_input=False)

        if verbose:
            print(
                f"Set shift {Qmin} for the input of {layer.name}:",
                f"{[inp.name.split('/')[0] for inp in layer.input]}"
            )
        # update current layer's shift only when we cannot change the shift
        if not is_shift_layer(layer) or Qmin < shift_list[layer.name]:
            shift_list[layer.name] = Qmin

    if verbose:
        print("shift list:", shift_list)
    return shift_list


def generate_weights(
        model: keras.Model, x_test: np.array=None, quantize_method="max_min",
        calibrate_size=1000, format="hwc", verbose=False):
    # Quantize weights to 8-bits using (min,max) and write to file
    f = io.StringIO()
    f.write('#include "nnom.h"\n\n')
    
    if type(x_test) is type(None):
        shift_list = None
    else:
        shift_list = layers_output_ranges(
            model, x_test, quantize_method=quantize_method,
            calibrate_size=calibrate_size, verbose=verbose
        )

    layer_quantize_info = {}
    layer_weights = {}
    for layer in model.layers:
        if not layer.weights:
            continue

        # before merging bn layer, check if the bn is "legally" after Conv
        if (
            "batch_normalization" in layer.name
            and "conv" not in layer.inbound_nodes[0].inbound_layers.name
        ):
            raise Exception(
                "Currently only support batch_normalization after conv",
                layer.name, layer._inbound_nodes[0].inbound_layers[0].name
            )

        # try to fuse BN layer to convolutional
        if (
            "conv" in layer.name
            and layer.outbound_nodes
            and "batch_normalization" in layer.outbound_nodes[0].outbound_layer.name
        ):
            fuse_bn_to_conv(layer)

        # generate weights and bias now
        weight_dec_shift = 0
        if verbose:
            print('weights for layer', layer.name)
        
        layer_quantize_info[layer.name] = {}
        layer_weights[layer.name] = {}
        for var in layer.weights:
            var_name = str(var.name)
            is_kernel = "kernel" in var_name
            if not is_kernel and "bias" not in var_name:
                continue
            
            var_values = var.numpy()
            min_value = np.min(var_values)
            max_value = np.max(var_values)

            int_bits = get_int_bits(min_value, max_value)
            dec_bits = 7 - int_bits

            if verbose:
                print(f" {'weight' if is_kernel else 'bias'}:", var_name)
                print("  original shape: ", var_values.shape)
                print("  dec bit", dec_bits)
            
            bSameAsKernel = False
            if is_shift_layer(layer):
                assert shift_list, f"Layer {layer.name} is classified as a shift layer so shift_list is required."
                inp = layer.input.name.replace(':', '/').split('/')[0]
                input_encoding = shift_list[inp]
                if is_kernel:
                    weight_dec_shift = dec_bits
                else:
                    shift = input_encoding + weight_dec_shift - dec_bits
                    if shift < 0:
                        bSameAsKernel = True
            
            if shift_list is None or bSameAsKernel:
                # check if bias shift > weight shift, then reduce bias shift to weight shift
                if is_kernel:
                    weight_dec_shift = dec_bits
                elif dec_bits > weight_dec_shift:
                    dec_bits = weight_dec_shift
                if verbose:
                    print("  new dec bit", dec_bits)

            layer_quantize_info[layer.name][var_name] = {
                "min": min_value,
                "max": max_value,
                "data_width": dec_bits
            }

            # convert to [-128,128) or int8
            var_values = np.round(var_values * 2 ** dec_bits)
            layer_weights[layer.name][int(not is_kernel)] = var_values
            var_name = var_name.replace('/', '_').replace(':', '_')
            f.write("#define " + var_name.upper() + " {")
            
            # CHW format
            if "chw" in format:
                if is_kernel and "dense" in var_name:
                    transposed_wts = to_transposed_x4_q7_weights(var_values)
                # all other kernels, bias stay the same
                else:
                    transposed_wts = var_values
            # HWC format
            else:
                if len(var_values.shape) == 3:  # 1D convolution layer weights
                    transposed_wts = np.transpose(var_values, (2, 0, 1))
                elif len(var_values.shape) == 4:  # 2D convolution layer weights
                    transposed_wts = np.transpose(var_values, (3, 0, 1, 2))
                elif is_kernel and "dense" in var_name:
                    # fully connected layer weights or biases of any layer
                    # test, use opt weight reorder
                    transposed_wts = to_transposed_x4_q7_weights(var_values)
                else:
                    transposed_wts = np.transpose(var_values)
            if verbose:
                print("  reshape to:", transposed_wts.shape)
            
            f.write(np.array2string(
                transposed_wts.flatten(),
                separator=", ",
                threshold=transposed_wts.size,
                formatter={"all": lambda x: str(int(x))}
            ).strip("[]"))
            # transposed_wts.tofile(f, sep=", ", format="%d")
            f.write("}\n\n")
            f.write(f"#define {var_name.upper()}_SHIFT ({dec_bits})\n\n")
            if not is_kernel:
                f.write("\n")
    return f, layer_weights, layer_quantize_info, shift_list

def quantize_model(model, x_test):
    shift_list = layers_output_ranges(model, x_test)
    
def generate_model(model, x_test, name='weights.h', format='hwc', quantize_method='max_min', verbose=False):
    f, *_, shift_list = generate_weights(model, x_test=x_test, format=format, quantize_method=quantize_method, verbose=verbose)
    
    model_layers = model.layers
    if not is_input_layer(model.layers[0]):
        model_layers = [model.input] + model_layers

    def get_iname(layer):
        return layer.name.replace(':', '/').split('/')[0]
    
    def to_cpp_var_name(layer_name):
        return layer_name.upper().replace('/', '_').replace(':', '_')
    
    def is_skipable_layer(layer):
        # FIXME: add more that could be skiped
        # flatten layer can be skipped in HWC but have to present in CHW
        return (
            "lambda" in layer.name
            or "dropout" in layer.name
            or "batch_normalization" in layer.name
            or ("flatten" in layer.name and "chw" not in format)
        )

    f.write('\n/* output encoding for each layer */\n')
    for layer in model_layers:
        iname = get_iname(layer)
        f.write('#define %s_OUTPUT_SHIFT %s\n'%(iname.upper(), shift_list[iname]))
    
    f.write('\n/* bias shift and output shift for each layer */\n')
    for layer in model_layers:
        if not is_shift_layer(layer):
            continue
        iname = layer.name.upper()
        if (
                len(layer.weights) == 2
                and "kernel" in layer.weights[0].name
                and "bias" in layer.weights[1].name
            ):
            kernel, bias = layer.weights
            kname = to_cpp_var_name(kernel.name)
            bname = to_cpp_var_name(bias.name)
            inp = get_iname(layer.input).upper()
            f.write('#define {0}_OUTPUT_RSHIFT ({1}_OUTPUT_SHIFT+{2}_SHIFT-{0}_OUTPUT_SHIFT)\n'.format(
                    iname, inp, kname))
            f.write('#define {0}_BIAS_LSHIFT   ({1}_OUTPUT_SHIFT+{2}_SHIFT-{3}_SHIFT)\n'.format(
                    iname, inp, kname, bname))
            f.write('#if {0}_OUTPUT_RSHIFT < 0\n#error {0}_OUTPUT_RSHIFT must be bigger than 0\n#endif\n'.format(iname))
            f.write('#if {0}_BIAS_LSHIFT < 0\n#error {0}_BIAS_RSHIFT must be bigger than 0\n#endif\n'.format(iname))
        # add, sub
        elif "add" in layer.name or "subtract" in layer.name:
            # only consider the first, they have been set to same in out_put_range()
            inp = get_iname(layer.input[0]).upper()
            f.write('#define {0}_OUTPUT_RSHIFT ({1}_OUTPUT_SHIFT-{0}_OUTPUT_SHIFT)\n'.format(
                    iname, inp))
            f.write('#if {0}_OUTPUT_RSHIFT < 0\n#error {0}_OUTPUT_RSHIFT must be bigger than 0\n#endif\n'.format(iname))
        # mult is different, Q3.4 * Q3.4 = Q6.8. if mult out is Q4.3, then shift (Q.4+q.4)-Q.3=5. Am I right?
        elif "multiply" in layer.name:
            inp = get_iname(layer.input[0]).upper()
            f.write(f"#define {iname}_OUTPUT_RSHIFT ({inp}_OUTPUT_SHIFT*2-{iname}_OUTPUT_SHIFT)\n")
            f.write(f"#if {iname}_OUTPUT_RSHIFT < 0\n#error {iname}_OUTPUT_RSHIFT must be bigger than 0\n#endif\n")

    ID = 0
    LI = {}    
    f.write('\n/* weights for each layer */\n')
    for layer_id, layer in enumerate(model_layers):
        if is_skipable_layer(layer):
            inp = get_iname(layer.input)
            LI[layer.name] = (LI[inp][0], layer)
        else:
            if isinstance(model.input, tf.Tensor) and not is_input_layer(model.layers[0]):
                LI[layer.name.split(':')[0]] = (ID, layer)
            else:
                LI[layer.name] = (ID, layer)
            ID += 1

        if is_input_layer(layer) or not layer.weights:
            continue
        for var in layer.weights:
            var_name = to_cpp_var_name(var.name)
            if "KERNEL" in var_name:
                f.write(f"static const int8_t {layer.name}_weights[] = {var_name};\n")
                f.write('static const nnom_weight_t %s_w = { (const void*)%s_weights, %s_OUTPUT_RSHIFT};\n' % (layer.name, layer.name, layer.name.upper()))
            elif "BIAS" in var_name:
                f.write(f"static const int8_t {layer.name}_bias[] = {var_name};\n")
                f.write('static const nnom_bias_t %s_b = { (const void*)%s_bias, %s_BIAS_LSHIFT};\n' % (layer.name, layer.name, layer.name.upper()))
    
    f.write("\n/* nnom model */\n")
    # FIXME: now only support one input and one output
    sz = 1
    for d in model.input.shape[1:]:
        sz *= d
    f.write(f"const int INPUT_LENGTH = {sz};\n")
    f.write("static int8_t nnom_input_data[INPUT_LENGTH];\n")
    sz = 1
    for d in model.output.shape[1:]:
        sz *= d
    f.write(f"const int OUTPUT_LENGTH = {sz};\n")
    f.write("static int8_t nnom_output_data[OUTPUT_LENGTH];\n")
    f.write("static nnom_model_t* nnom_model_create(void)\n{\n")
    f.write("\tstatic nnom_model_t model;\n")
    
    if ID > 32:
        f.write(f"\tnnom_layer_t ** layer = malloc(sizeof(nnom_layer_t *)*{ID + 1});\n")
        f.write("\tif(NULL == layer) return NULL;\n")
    else:
        f.write(f"\tnnom_layer_t* layer[{ID + 1}];\n")
    
    f.write("\n\tnew_model(&model);\n\n")
    for layer in model_layers:
        if is_skipable_layer(layer):
            continue
        #FIXME: need a better solution to seperate the input 'tensor' from other layers
        if isinstance(model.input, tf.Tensor) and not is_input_layer(model.layers[0]):
            layer_id, _ = LI[layer.name.split(':')[0]]
        else:
            layer_id, _ = LI[layer.name]
        try:
            inp = get_iname(getattr(layer, "input", None))
        except AttributeError:
            inp = ""
        cfg = getattr(layer, "get_config", lambda: None)()

        if "input" in layer.name:
            try:
                inshape = layer.input_shape[0][1:] # new changes in tf2?
            except:
                inshape = layer.shape[1:]
            if len(inshape) == 1:  # 1-D input
                f.write('\tlayer[%d] = Input(shape(%d,1,1), nnom_input_data);\n' % (layer_id, inshape[0]))
            elif len(inshape) == 2:  # 1-D input
                f.write('\tlayer[%d] = Input(shape(1,%d,%d), nnom_input_data);\n' % (layer_id, inshape[0], inshape[1]))
            else:
                f.write('\tlayer[%d] = Input(shape%s, nnom_input_data);\n' % (layer_id, inshape))

        # convolutional
        elif "conv" in layer.name:
            is_depthwise = "depthwise" in layer.name
            num_filters = 1 if is_depthwise else cfg["filters"]
            conv_type = "Conv2D"
            if is_depthwise:
                conv_type = "DW_" + conv_type
            
            # Expand kernel, stride, and dilation for 1D conv
            kernel, stride, dilation = pad_filter_sizes(
                cfg['kernel_size'], cfg['strides'], cfg['dilation_rate']
            )
            f.write(
                f"\tlayer[{layer_id}] = model.hook("
                + f"{conv_type}({num_filters}, kernel{kernel}, "
                + f"stride{stride}, dilation{dilation}, "
                + f"PADDING_{cfg['padding']}, &{layer.name}_w, "
                + f"&{layer.name}_b), layer[{LI[inp][0]}]);\n"
            )

        # activations
        elif "activation" in layer.name:
            activ_name = cfg["activation"]
            if activ_name in ["tanh", "sigmoid"]:
                f.write(f"\tlayer[{layer_id}] = model.active(act_{activ_name}({inp.upper()}_OUTPUT_SHIFT), layer[{LI[inp][0]}]);\n")
            elif activ_name in ["softmax", "relu"]:
                func_name = "Softmax" if activ_name == "softmax" else "act_relu"
                func_type = "hook" if activ_name == "softmax" else "active"
                f.write(f"\tlayer[{layer_id}] = model.{func_type}({func_name}(), layer[{LI[inp][0]}]);\n")
            elif activ_name != "linear":
                raise Exception(f"{activ_name} activation is unsupported.")
        # ReLU
        elif "re_lu" in layer.name:
            f.write(f"\tlayer[{layer_id}] = model.active(act_relu(), layer[{LI[inp][0]}]);\n")

        # pooling
        elif "pooling" in layer.name:
            pooling_type = "Avg" if "average" in layer.name else layer.name[:3].capitalize()
            if "global" in layer.name:
                # a global avg pool before softmax can be replace by sumpool in MCU (recommend)
                if pooling_type == "Avg" and layer == model.layers[-2] and "Softmax" in model.layers[-1].output.name:
                    if verbose:
                        print(layer.name, 'has been replaced by GlobalSumPool()')
                    f.write(f"\tlayer[{layer_id}] = model.hook(GlobalSumPool(), layer[{LI[inp][0]}]);\n")
                else:
                    f.write(f"\tlayer[{layer_id}] = model.hook(Global{pooling_type}Pool(), layer[{LI[inp][0]}]);\n")
            else:
                # Expand 1D Pooling params
                pool_size, strides = pad_filter_sizes(cfg["pool_size"], cfg["strides"])
                padding = cfg["padding"].upper()
                f.write(
                    f"\tlayer[{layer_id}] = model.hook("
                    + f"{pooling_type}Pool("
                    + f"kernel{pool_size}, stride{strides}, PADDING_{padding}"
                    + f"), layer[{LI[inp][0]}]);\n"
                )
        elif "up_sampling" in layer.name:
            size = pad_filter_sizes(cfg["size"])[0]
            f.write(f"\tlayer[{layer_id}] = model.hook(UpSample(kernel{size}), layer[{LI[inp][0]}]);\n")

        # Zero padding / Cropping
        elif "zero_padding" in layer.name or "cropping" in layer.name:
            is_padding = "zero_padding" in layer.name
            config_var = "padding" if is_padding else "cropping"
            func_name = "ZeroPadding" if is_padding else "Cropping"
            border_size = pad_filter_sizes(flatten(cfg[config_var]), pad_val=0, shape=4)[0]
            f.write(f"\tlayer[{layer_id}] = model.hook({func_name}(border{border_size}), layer[{LI[inp][0]}]);\n")

        # others
        elif "flatten" in layer.name: # flatten is needed in CHW backend but not needed in HWC
            f.write(f"\tlayer[{layer_id}] = model.hook(Flatten(), layer[{LI[inp][0]}]);\n")
        elif any(merge_name in layer.name for merge_name in ["concatenate", "add", "subtract", "multiply"]):
            inps = [get_iname(input) for input in layer.input]
            inX = sum([f" ,layer[{LI[inp][0]}]" for inp in inps], start="")
            if "concatenate" in layer.name:
                f.write(f"\tlayer[{layer_id}] = model.mergex(Concat({cfg['axis']}), {len(inps)}{inX});\n")
            else:
                func_name = "Mult" if "multiply" in layer.name else layer.name[:3].capitalize()
                if func_name == "Mult":
                    warnings.warn("Warning mutiply is under testing")
                f.write(
                    f"\tlayer[{layer_id}] = model.mergex("
                    + f"{func_name}({layer.name.upper()}_OUTPUT_RSHIFT), {len(inps)}{inX});\n"
                )
        elif "dense" in layer.name:
            f.write(
                f"\tlayer[{layer_id}] = model.hook("
                + f"Dense({cfg['units']}, &{layer.name}_w, &{layer.name}_b), layer[{LI[inp][0]}]);\n"
            )
        elif "softmax" in layer.name:
            f.write(f"\tlayer[{layer_id}] = model.hook(Softmax(), layer[{LI[inp][0]}]);\n")
        else:
            raise Exception("unsupported layer", layer.name, layer)
			
        # # temporary fixed for activations attached into layers in construction
        # def is_activation_attached(layer):
        #     if(("Softmax" in layer.output.name and "softmax" not in layer.name)or
        #     ("Relu" in layer.output.name and "re_lu" not in layer.name) or
        #     ("Sigmoid" in layer.output.name and "sigmoid" not in layer.name) or
        #     ("Tanh" in layer.output.name and "tanh" not in layer.name)):
        #         return True
        #     return False
        # if "input" not in layer.name and is_activation_attached(layer):
        #     inp = layer.output.name.replace(':', '/').split('/')[0]
        #     cfg = layer.get_config()
        #     if(cfg['activation'] == 'relu'):
        #         f.write('\tlayer[%s] = model.active(act_relu(), layer[%s]);\n'%(id, LI[inp][0]))
        #     if(cfg['activation'] == 'tanh'):
        #         f.write('\tlayer[%s] = model.active(act_tanh(%s_OUTPUT_SHIFT), layer[%s]);\n'%(id, inp.upper(), LI[inp][0]))
        #     if(cfg['activation'] == 'sigmoid'):
        #         f.write('\tlayer[%s] = model.active(act_sigmoid(%s_OUTPUT_SHIFT), layer[%s]);\n'%(id, inp.upper(), LI[inp][0]))
        #     elif(cfg['activation'] == 'softmax'):
        #         f.write('\tlayer[%s] = model.hook(Softmax(), layer[%s]);\n'%(id, LI[inp][0]))

    # FIXME, test later.
    if (
        "softmax" in layer.name
        or len(layer.output.shape) == 2
        or ("activation" in layer.name and layer.get_config()["activation"] == "softmax")
    ):
        out_shape = (layer.output.shape[1], 1, 1)
    elif len(layer.output.shape) == 4:
        out_shape = layer.output.shape[1:]
    elif len(layer.output.shape) == 3:
        out_shape = (1, layer.output.shape[1], layer.output.shape[2])
    else:
        raise Exception("unsupported output shape of the last layer", layer.name, layer)
    f.write(f"\tlayer[{layer_id + 1}] = model.hook(Output(shape{out_shape}, nnom_output_data), layer[{layer_id}]);\n")
    f.write(f"\tmodel_compile(&model, layer[0], layer[{layer_id + 1}]);\n")
    if ID > 32:
        f.write("\tfree(layer);\n")
    f.write("\treturn &model;\n}\n")
    save_root, _ = os.path.split(name)
    with open(os.path.join(save_root, ".shift_list"), 'w') as file:
        file.write(str(shift_list))
    with open(name, 'w+', encoding="utf-8") as file:
        file.write(f.getvalue())
