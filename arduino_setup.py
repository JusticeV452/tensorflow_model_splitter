import os
import re
import sys
import ast
import json
import keras
import shutil
import argparse
import inquirer
import pyduinocli
import importlib.util

from device_assignment import automatic_device_assigment, arduino_compile

NNOM_DIR = "nnom"
DEFAULT_PROJECT_PATH = "SETML_Arduino"
INO_TEMPLATE_PATH = "arduino_template.ino"
CONFIG_PATH = "config.json"
CRC8_DEF_PATTERN = r"[a-zA-Z_\d]+ crc8\(.*?, .*?\) {"


def load_json(file_path, binary=False, encoding="utf-8", default=lambda: {}):
    if not os.path.exists(file_path):
        return default()
    with open(file_path, f"r{'b' if binary else ''}", encoding=encoding) as file:
        return json.load(file)


def save_json(obj, file_path, binary=False, encoding="utf-8"):
    with open(file_path, f"w{'b' if binary else ''}+", encoding=encoding) as file:
        json.dump(obj, file)


def make_board_options(board_list_results):
    board_options = {}
    results = board_list_results["result"]

    # Board list under "detected_ports" in Windows
    if "detected_ports" in results:
        results = results["detected_ports"]

    for result in results:
        if not result["port"].get("properties"):
            continue
        board = result["port"]
        board_props = board.pop("properties")
        board.update(board_props)
        display_str = board["address"] + f" (vid: {board['vid']}, pid: {board['pid']})".lower()
        if "matching_boards" in result:
            first_matching = result["matching_boards"][0]
            board["fqbn"] = first_matching["fqbn"]
            board["name"] = first_matching["name"]
            display_str = board["name"] + " - " + display_str
        board["display_name"] = display_str
        board["save_key"] = f"{board.get('vid')}-{board.get('pid')}".lower()
        board_options[display_str] = board
    return board_options


def find_board(boards, _allow_missing=False, **kwargs):
    for board in boards:
        if all(val == board[key] for key, val in kwargs.items()):
            return board
    if not _allow_missing:
        attr_str = ", ".join([
            f"{key}={json.dumps(val)}" for key, val in kwargs.items()
        ])
        raise Exception(f"No board found with {attr_str}")


def load_py_file(python_file_path, module_name=None):
    module_name = (
        os.path.split(python_file_path)[-1].split('.')[0]
        if module_name is None else module_name
    )
    spec = importlib.util.spec_from_file_location(
        module_name, python_file_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_text_input(prompt, var_name="request", path_type=None):
    inquirer_class = inquirer.Path if path_type else inquirer.Text
    kwargs = {"message": prompt}
    if path_type:
        kwargs["path_type"] = path_type
    return inquirer.prompt([inquirer_class(var_name, **kwargs)])[var_name]


def get_model(python_file_path, model_creator_func, *args, **kwargs):
    model_gen_lib = load_py_file(python_file_path)
    model_gen_func = getattr(model_gen_lib, model_creator_func)
    return model_gen_func(*args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Model Distributer',
        description='Splits a model into parts based on the number of devices selected'
    )
    parser.add_argument('-ppth', '--project_path', type=str, default=DEFAULT_PROJECT_PATH)
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('-md', '--module_path', type=str, default="models.py")
    parser.add_argument('-g', '--gen_func_name', type=str, default="create_model")
    parser.add_argument('-garg', '--gen_func_kwargs', type=str, nargs='*', default=[])
    parser.add_argument('-mn', '--model_name', type=str)
    parser.add_argument('-w', '--weights_path', type=str)
    parser.add_argument('-p', '--ports', type=str, nargs='*')
    args = parser.parse_args()

    project_path = args.project_path
    os.makedirs(project_path, exist_ok=True)

    # Copy nnom library to project
    nnom_port_dir = os.path.join(NNOM_DIR, "port")
    nnom_inc_dir = os.path.join(NNOM_DIR, "inc")
    nnom_src_dir = os.path.join(NNOM_DIR, "src")
    if not os.path.exists(nnom_port_dir):
        shutil.copytree(nnom_port_dir, project_path, dirs_exist_ok=True)
    if not os.path.exists(nnom_inc_dir):
        shutil.copytree(nnom_inc_dir, project_path, dirs_exist_ok=True)
    if not os.path.exists(nnom_src_dir):
        shutil.copytree(
            nnom_src_dir, os.path.join(project_path, "src"),
            dirs_exist_ok=True
        )
    _, project_name = os.path.split(project_path)

    config = load_json(CONFIG_PATH)
    cli_path = config.get("cli_path", "arduino-cli")
    arduino = pyduinocli.Arduino(cli_path)
    config["cli_path"] = cli_path
    config.setdefault("fqbns", {})
    config.setdefault("device_names", {})

    board_list = arduino.board.list()
    board_options = make_board_options(board_list)
    if args.ports:
        target_boards = [
            find_board(board_options.values(), address=port)
            for port in args.ports
        ]
    else:
        chosen_boards = inquirer.prompt([inquirer.Checkbox(
            "board_targets",
            message="Select boards to split model over",
            choices=board_options,
        )])["board_targets"]
        target_boards = [board_options[choice] for choice in chosen_boards]
    assert target_boards, "No boards selected."

    # Validate board fqbns
    temp_fqbns = {}
    board_details = {}
    for board in target_boards:
        fqbn = board.get("fqbn")
        if not fqbn:
            fqbn = config.get("fqbns", {}).get(
                board["save_key"], temp_fqbns.get(board["save_key"])
            )
        if not fqbn:
            fqbn = get_text_input(
                f"No FQBN found for {board['display_name']}. Please enter FQBN"
            )
        board["fqbn"] = fqbn
        temp_fqbns[board["save_key"]] = board["fqbn"]
        if fqbn not in board_details:
            board_details[fqbn] = arduino.board.details(fqbn)["result"]
        fqbn_config_options = board_details[fqbn].get("config_options")
        for option in fqbn_config_options:
            if "part number" not in option["option_label"]:
                continue
            pnum_options = {
                part_entry["value_label"]: part_entry["value"]
                for part_entry in option["values"]
            }
            selected_part = inquirer.prompt([inquirer.List(
                "pnum",
                message=f"Select a board part number for {board['display_name']}",
                choices=pnum_options
            )])["pnum"]
            board.setdefault("options", {})
            board["options"][option["option"]] = pnum_options[selected_part]
            break
        board["name"] = board.get("options", {}).get(
            "pnum", board.get("name", board_details[board["fqbn"]]["name"].replace(' ', '_'))
        )
        firmware_path = f"devices/{board['name']}/firmware.h"
        while not os.path.exists(firmware_path):
            firmware_path = get_text_input(
                f"No firmware.h found for {board['name']} at {firmware_path}. Please provide a path",
                path_type=inquirer.Path.FILE
            )
        board["firmware_path"] = firmware_path

    # Select global macros (macros for all boards)
    sending_choices = {
        "Default - Only available device interaction is sending": [],
        "Enable pings - Enable sending pings between devices": ["ENABLE_PING"],
        "Enable hold requests - Devices can request that senders stop sending": ["ENABLE_HOLD"]
    }
    sending_macros = sending_choices[inquirer.prompt([inquirer.List(
        "sending_macros",
        message="Select device sending capabilities",
        choices=sending_choices
    )])["sending_macros"]]

    # Macro select
    for board in target_boards:
        macro_choices = [
            "STATIC_SLOT_ASSIGN",
            "UPDATE_ON_END",
            "DEBUG_PRINT",
            "HARDWARE_CRC",
            "SLOT_MALLOC",
            "TRACK_STATS",
            "TRACK_SEND_STATS",
            "TRACK_INTERLEAVING"
        ]
        default_macros = [
            "STATIC_SLOT_ASSIGN",
            "UPDATE_ON_END",
            "DEBUG_PRINT"
        ]
        with open(board["firmware_path"], 'r', encoding="utf-8") as f:
            firmware_content = f.read()
        if re.findall(CRC8_DEF_PATTERN, firmware_content):
            default_macros.append("HARDWARE_CRC")
        board["optional_macros"] = sending_macros + inquirer.prompt([inquirer.Checkbox(
            "macros",
            message=f"Select macros to use for {board['display_name']}",
            choices=macro_choices, default=default_macros
        )])["macros"]

    # Make model args: python file, func that returns model, model_weights
    if isinstance(args.model_path, str) and os.path.exists(args.model_path):
        model = keras.saving.load_model(args.model_path)
    else:
        gen_func_kwargs = {}
        for arg_val_pair in args.gen_func_kwargs:
            arg_name, val = arg_val_pair.split('=', 1)
            gen_func_kwargs[arg_name] = ast.literal_eval(val)

        model = get_model(
            args.module_path, args.gen_func_name, **gen_func_kwargs
        )
        if args.weights_path:
            model.load_weights(args.weights_path)

    # Split model with nnom
    board_pairs = automatic_device_assigment(model, target_boards)[0]

    # copy each weight set into project dir and run
    for i, (board, device_config) in enumerate(board_pairs):
        board_name = board["name"]
        temp_dir, compile_result = arduino_compile(
            board, device_config, project_path, keep_tempdir=True
        )

        # arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:samd:mkr1000 MyFirstSketch
        print(board["address"])
        uploader_path = f"devices/{board_name}/uploader.py"
        if os.path.exists(uploader_path):
            load_py_file(uploader_path).upload(temp_dir.name, board, compile_result)
        else:
            arduino.upload(temp_dir.name, port=board["address"], fqbn=board.get("fqbn"))
        temp_dir.cleanup()

        # If successful, save config
        config["fqbns"][board["save_key"]] = board["fqbn"]
        # if device_name_def:
        #     config["device_names"][board["save_key"]] = device_name_def
        save_json(config, CONFIG_PATH)
