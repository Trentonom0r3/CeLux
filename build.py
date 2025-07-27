import os
import subprocess

VERSION = "0.6.2"

def main():
    os.environ["CELUX_VERSION"] = VERSION
    subprocess.run(["python", "cpusetup.py", "bdist_wheel"], check=True)

if __name__ == "__main__":
    main()
