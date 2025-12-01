import subprocess
import sys


def main():
    subprocess.run([sys.executable, "-m", "scripts.build_all"])
    subprocess.run([sys.executable, "-m", "scripts.upload_all", "--tag", "v3.0.0", "--dry-run"])


if __name__ == "__main__":
    main()
