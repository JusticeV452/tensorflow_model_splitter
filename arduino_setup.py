import os
import sys
import shutil

NNOM_DIR = "nnom"
DEFAULT_PROJECT_PATH = "SETML_Arduino"

if __name__ == "__main__":
    project_path = DEFAULT_PROJECT_PATH
    if len(sys.argv) == 2:
        project_path = sys.argv[1]
    os.makedirs(project_path, exist_ok=True)

    # Copy nnom library to project
    shutil.copytree(os.path.join(NNOM_DIR, "port"), project_path, dirs_exist_ok=True)
    shutil.copytree(os.path.join(NNOM_DIR, "inc"), project_path, dirs_exist_ok=True)
    shutil.copytree(
        os.path.join(NNOM_DIR, "src"), os.path.join(project_path, "src"),
        dirs_exist_ok=True
    )
    _, project_name = os.path.split(project_path)
    shutil.copy("test_nnom.ino", os.path.join(project_path, f"{project_name}.ino"))
