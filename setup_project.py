import subprocess

if __name__ == "__main__":
    subprocess.run([
        "git", "clone",
        "https://github.com/majianjia/nnom.git",
        "--branch", "v0.4.3", "nnom"
    ], check=True)
    subprocess.run([
        "pip", "install", "-r",
        "requirements.txt"
    ], check=True)
