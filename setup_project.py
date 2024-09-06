import os
import subprocess

if __name__ == "__main__":
    if os.path.exists("nnom"):
        print("nnom dir exists, skipping repo clone.")
    else:
        subprocess.run([
            "git", "clone",
            "https://github.com/majianjia/nnom.git",
            "--branch", "v0.4.3", "nnom"
        ], check=True)
    subprocess.run([
        "pip", "install", "-r",
        "requirements.txt"
    ], check=True)
