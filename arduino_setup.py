import os
import sys
import ast
import json
import shutil
import argparse
import inquirer
import pyduinocli
import importlib.util

from model_splitter import split_model, save_nnom_model, DEFAULT_OUTPUT_FOLDER

NNOM_DIR = "nnom"
DEFAULT_PROJECT_PATH = "SETML_Arduino"
INO_TEMPLATE_PATH = "arduino_template.ino"
CONFIG_PATH = "config.json"
DEVICE_NAME_DEFS = {"0x0483-0x374b": "STM32C0116_DK"}


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
        board["display_name"] = display_str
        board["save_key"] = f"{board.get('vid')}-{board.get('pid')}".lower()
        board_options[display_str] = board
    return board_options


def load_py_file(module_name, python_file_path):
    spec = importlib.util.spec_from_file_location(
        module_name, python_file_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_model(python_file_path, model_creator_func, *args, **kwargs):
    model_gen_lib = load_py_file("model_gen", python_file_path)
    model_gen_func = getattr(model_gen_lib, model_creator_func)
    return model_gen_func(*args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Model Distributer',
        description='What the program does',
        epilog='Text at the bottom of help'
    )
    parser.add_argument('-ppth', '--project_path', type=str, default=DEFAULT_PROJECT_PATH)
    parser.add_argument('-m', '--module_path', type=str, default="models.py")
    parser.add_argument('-g', '--gen_func_name', type=str, default="create_model")
    parser.add_argument('-garg', '--gen_func_kwargs', type=str, nargs='*', default=[])
    parser.add_argument('-mn', '--model_name', type=str)
    parser.add_argument('-w', '--weights_path', type=str)

    parser.add_argument('-rid', '--root_id', type=int, default=0)
    parser.add_argument('-tid', '--tail_id', type=int)
    parser.add_argument('-sd', '--start_device', type=int, default=0)
    args = parser.parse_args()

    project_path = args.project_path
    os.makedirs(project_path, exist_ok=True)

    # Copy nnom library to project
    shutil.copytree(os.path.join(NNOM_DIR, "port"), project_path, dirs_exist_ok=True)
    shutil.copytree(os.path.join(NNOM_DIR, "inc"), project_path, dirs_exist_ok=True)
    shutil.copytree(
        os.path.join(NNOM_DIR, "src"), os.path.join(project_path, "src"),
        dirs_exist_ok=True
    )
    _, project_name = os.path.split(project_path)
    ino_path = os.path.join(project_path, f"{project_name}.ino")

    config = load_json(CONFIG_PATH)
    cli_path = config.get("cli_path", "arduino-cli")
    arduino = pyduinocli.Arduino(cli_path)
    config["cli_path"] = cli_path
    config.setdefault("fqbns", {})
    config.setdefault("device_names", {})

    board_list = arduino.board.list()
    board_options = make_board_options(board_list)
    answer = inquirer.prompt([inquirer.Checkbox(
        "board_targets",
        message="Select boards to split model over",
        choices=board_options,
    )])
    target_boards = [board_options[choice] for choice in answer["board_targets"]]

    # Validate board fqbns
    temp_fqbns = {}
    for board in target_boards:
        if board.get("fqbn"):
            continue

        board["fqbn"] = config.get("fqbns", {}).get(
            board["save_key"], temp_fqbns.get(board["save_key"])
        )

        if board.get("fqbn"):
            continue
        board["fqbn"] = inquirer.prompt([inquirer.Text(
            "fqbn",
            message=f"No FQBN found for {board['display_name']}. Please enter FQBN"
        )])["fqbn"]
        temp_fqbns[board["save_key"]] = board["fqbn"]

    # Make model args: python file, func that returns model, model_weights
    gen_func_kwargs = {}
    for arg_val_pair in args.gen_func_kwargs:
        arg_name, val = arg_val_pair.split('=', 1)
        gen_func_kwargs[arg_name] = ast.literal_eval(val)

    model = get_model(args.module_path, args.gen_func_name, **gen_func_kwargs)
    if args.weights_path:
        model.load_weights(args.weights_path)
    if args.model_name:
        model._name = args.model_name

    # Split model with nnom
    split_model(model, len(target_boards), saver=save_nnom_model)

    # copy each weight set into project dir and run
    for i, board in enumerate(target_boards):
        device_id = args.root_id + args.start_device + i
        weights_path = os.path.join(
            DEFAULT_OUTPUT_FOLDER,
            model.name, "nnom",
            f"{model.name}_{device_id}.h"
        )
        weights_dest_path = os.path.join(project_path, "weights.h")
        print(weights_path, "->", weights_dest_path)
        shutil.copy(weights_path, weights_dest_path)

        # Update variables in ino file
        shutil.copy(INO_TEMPLATE_PATH, ino_path)
        with open(ino_path, 'r+', encoding="utf-8") as file:
            content = file.read()
            tail_id = len(target_boards) - 1 if args.tail_id is None else args.tail_id
            device_name_def = config.get("device_names", {}).get(
                board["save_key"], DEVICE_NAME_DEFS.get(board["save_key"], "")
            )
            content = content.replace(
                "{{DEVICE_NAME_DEF}}",
                ("#define " if device_name_def else "") + device_name_def
            ).replace(
                "{{ROOT_ID}}", str(args.root_id)
            ).replace(
                "{{TAIL_ID}}", str(tail_id)
            ).replace(
                "{{DEVICE_ID}}", str(device_id)
            )
        with open(ino_path, 'w+', encoding="utf-8") as file:
            file.write(content)

        arduino.compile(project_path, fqbn=board.get("fqbn"))

        # arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:samd:mkr1000 MyFirstSketch
        print(board["address"])
        arduino.upload(project_path, port=board["address"], fqbn=board.get("fqbn"))

        # If successful, save config
        config["fqbns"][board["save_key"]] = board["fqbn"]
        if device_name_def:
            config["device_names"][board["save_key"]] = device_name_def
        save_json(config, CONFIG_PATH)
