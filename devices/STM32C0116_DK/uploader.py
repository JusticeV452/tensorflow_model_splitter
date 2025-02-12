import os
import platform
import subprocess

OS_NAME = platform.system()
if OS_NAME == "Windows":
    STM32CP_CLI = "STM32_Programmer_CLI.exe"
    UPLOADER_SUBDIR = "STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin"
    STM_UPLOADER_PATH = os.path.join(
        os.environ.get("ProgramW6432"), UPLOADER_SUBDIR, STM32CP_CLI
    )
    if not os.path.exists(STM_UPLOADER_PATH):
        STM_UPLOADER_PATH = os.path.join(
            os.environ.get("ProgramFiles(x86)"), UPLOADER_SUBDIR, STM32CP_CLI
        )
    if not os.path.exists(STM_UPLOADER_PATH):
        raise Exception("Uploader not found: ")
elif OS_NAME == "Linux":
    STM32CP_CLI = "STM32_Programmer.sh"
elif OS_NAME == "Darwin":
    STM32CP_CLI = "STM32_Programmer_CLI"
else:
    raise Exception("Unrecognized platform")

DEFAULT_ADDRESS_OFFSET = "0x8000000"

def upload(sketch_folder, board, compile_result):
    sketch_name = os.path.split(sketch_folder)[-1]
    bin_name = f"{sketch_name}.ino.bin"
    bin_path = os.path.join(compile_result["builder_result"]["build_path"], bin_name)
    assert bin_path, "Compiled binary for {sketch_name} not found in {bin_path}"
    serial_number = f"sn={sn}" if (sn := board.get("serialNumber")) else ''
    subprocess.run([
        STM_UPLOADER_PATH,
        "--connect", "port=SWD", "mode=UR", serial_number,
        "--download", bin_path, DEFAULT_ADDRESS_OFFSET,
        "--start", DEFAULT_ADDRESS_OFFSET
    ])
