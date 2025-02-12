import os
import sys
import json
import shutil
import tempfile
import itertools
from textwrap import dedent

import pyduinocli
import numpy as np

from segmented_model import SegmentedModel
from multi_threading import multi_thread

arduino = pyduinocli.Arduino()


def unique_permutations(iterable, get_uid=str):
    seen_ids = set()
    for perm in itertools.permutations(iterable):
        unique_id = get_uid(perm)
        if unique_id in seen_ids:
            continue
        seen_ids.add(unique_id)
        yield unique_id, perm


def get_config_weights(device_config):
    if not device_config["saved_at_path"]:
        return device_config["weights"]
    with open(device_config["weights"], 'r', encoding="utf-8") as f:
        return f.read()


def save_config_weights(device_config, save_path):
    if device_config["saved_at_path"]:
        shutil.copy(device_config["weights"], save_path)
        return
    with open(save_path, "w+", encoding="utf-8") as f:
        f.write(device_config["weights"])


def get_config_weights_size(device_config):
    return len(get_config_weights(device_config).encode("utf-8"))


def get_c_arr_str(iterable):
    return '{' + ", ".join([str(el) for el in iterable]) + '}'


def arduino_compile(
        board, device_config, project_path, use_tempdir=True,
        run_compilation=True, full_output=False, keep_tempdir=False):
    project_name = os.path.split(project_path)[-1]
    if use_tempdir:
        temp_dir = tempfile.TemporaryDirectory()
        shutil.copytree(
            project_path, temp_dir.name,
            dirs_exist_ok=True
        )
        temp_project_name = os.path.split(temp_dir.name)[-1]
        os.rename(
            os.path.join(temp_dir.name, f"{project_name}.ino"),
            os.path.join(temp_dir.name, f"{temp_project_name}.ino")
        )
        project_path = temp_dir.name
        project_name = temp_project_name
    weights_dest_path = os.path.join(project_path, "weights.h")
    save_config_weights(device_config, weights_dest_path)
    # Update macros
    macros = board["optional_macros"]
    with open(os.path.join(project_path, "macros.h"), "w+", encoding="utf-8") as f:
        macros_str = ""
        macros_str += ("#ifndef MACROS_H\n#define MACROS_H\n#include \"Arduino.h\"\n\n")
        macros_str += ("#define NULL_ID 255\n")
        for macro in macros:
            macros_str += (f"#define {macro.upper()}\n")
        if (is_input_node := not device_config["input_thresholds"]):
            macros_str += ("#define INPUT_NODE\n")
        if (is_output_node := device_config["out_device"] == "NULL_ID"):
            macros_str += ("#define OUTPUT_NODE\n")
        if "SLOT_MALLOC" not in macros:
            macros_str += (
                f"#define SEND_BUFFER_SIZE {device_config['send_buffer_size']}\n")
            macros_str += (
                f"#define RECEIVE_BUFFER_SIZE {device_config['receive_buffer_size']}\n")
        macros_str += (f"#define {device_config['reduce_type']}_REDUCE\n")
        num_thresholds = len(device_config["input_thresholds"])
        connect_devices = list(device_config["receive_order"])
        if not is_output_node:
            connect_devices.append(device_config["out_device"])
        num_devices = len(connect_devices)
        thresholds_str = get_c_arr_str(device_config["input_thresholds"])
        receive_order_str = get_c_arr_str(device_config["receive_order"])
        connect_str = get_c_arr_str(connect_devices)
        macros_str += (dedent(f"""
            constexpr int DEVICE_ID = {device_config['node_id']};
            constexpr int NUM_DEVICES = {num_devices};
            constexpr int NUM_THRESHOLDS = {num_thresholds};
            constexpr bool IS_MULTI_INPUT = NUM_THRESHOLDS > 0;
            constexpr int INPUT_THRESHOLDS[] = {thresholds_str};
            constexpr bool IS_RECURRENT = false;
            constexpr uint8_t RECEIVE_ORDER[] = {receive_order_str};
            constexpr uint8_t CONNECT[] = {connect_str};
            constexpr uint8_t OUT_DEVICE = {device_config['out_device']};
            constexpr bool IS_INPUT_NODE = {str(is_input_node).lower()};
            constexpr bool IS_OUTPUT_NODE = {str(is_output_node).lower()};\n"""
        ))
        macros_str += ("#endif")
        f.write(macros_str)

    shutil.copy(board["firmware_path"], os.path.join(
        project_path, "firmware.h"))
    if not run_compilation:
        if use_tempdir:
            temp_dir.cleanup()
        return
    try:
        result = arduino.compile(
            project_path, fqbn=board["fqbn"],
            board_options=board.get("options")
        )
        if not full_output:
            result = result["result"]
    except pyduinocli.errors.arduinoerror.ArduinoError as arduino_error:
        error_vars = vars(arduino_error)
        stdout = json.loads(error_vars["result"]["__stdout"])
        error_str = stdout["compiler_err"]
        if not error_str:
            error_str = stdout["compiler_out"]
        print(error_str)
        sys.exit(1)
    finally:
        if use_tempdir and not keep_tempdir:
            temp_dir.cleanup()
    if keep_tempdir:
        return temp_dir, result
    return result


def compiled_code_util(
        boards, upload_info, project_path="template_project",
        num_threads=1, compile_size_history=None):
    def get_storage_util(dev_and_weights):
        board, device_config = dev_and_weights
        history_key = board["save_key"]

        # Use size of previously compiled sketch to estimate storage utilization
        if compile_size_history and history_key in compile_size_history:
            last_weights_size, reference_util = compile_size_history[history_key]
            return (get_config_weights_size(device_config) / last_weights_size) * reference_util

        compiler_out = arduino_compile(
            board, device_config, project_path
        )["compiler_out"]
        storage_util = int(
            compiler_out[compiler_out.find('(') + 1:compiler_out.find('%')])

        # Save utilization for approximating compile size of other configurations
        if compile_size_history:
            weights_size = get_config_weights_size(device_config)
            compile_size_history[history_key] = (weights_size, storage_util)
        return storage_util

    return multi_thread(
        get_storage_util, zip(boards, upload_info.values()),
        num_threads=num_threads, preserve_order=True
    )


def automatic_device_assigment(
        model, boards, project_path="template_project", num_threads=1,
        sub_threads=1, compile_size_approx=True):
    seg_model = SegmentedModel(model)
    if len(boards) < len(seg_model.nodes):
        raise Exception(
            f"Too few boards selected to run model segments - Only {len(boards)} selected while the minimum number of segments is {len(seg_model.nodes)}"
        )
    if len(seg_model.nodes) == 1:
        seg_model = seg_model.extend(len(boards))

    # Bisect largest node until number of nodes equal number of boards
    while len(seg_model.nodes) < len(boards):
        largest_node = max(seg_model.nodes, key=lambda node_name: len(
            seg_model.nodes[node_name]))
        orig_num_nodes = len(seg_model.nodes)
        seg_model = seg_model.extend({largest_node: 2})
        if not len(seg_model.nodes) > orig_num_nodes:
            raise Exception(
                f"Too many boards selected - Only {len(seg_model.nodes)} segments can be created while the number of selected boards is {len(boards)}"
            )

    upload_info = seg_model.save()
    compile_size_history = {} if compile_size_approx else None

    def calculate_scores(item):
        _, board_perm = item
        board_pairs = [
            (board, device_config)
            for board, device_config in zip(board_perm, upload_info.values())
        ]
        return (board_pairs, compiled_code_util(
            board_perm, upload_info, project_path=project_path,
            num_threads=sub_threads, compile_size_history=compile_size_history
        ))

    scores = multi_thread(
        calculate_scores,
        unique_permutations(
            boards,
            get_uid=lambda perm: tuple(board["save_key"] for board in perm)
        ),
        num_threads=num_threads
    )
    best_distrib = min(scores, key=lambda entry: np.array(entry[1]).std())
    for score in scores:
        if score is best_distrib:
            continue
        for compile_result in score[1][0]:
            os.rmdir(compile_result["builder_result"]["build_path"])
    return best_distrib
