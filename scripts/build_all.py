import sys
import platform
import subprocess

OS_NAME = platform.system()

BUILD_PLATFORMS = {
    "Windows": [
        "cpu",
        "cuda",
        "vulkan",
    ],
    "Linux": [
        "cpu",
        # "cuda",
        "vulkan",
    ],
    "Darwin": [
        "cpu",
        "metal",
    ],
}

print(f"Detected OS: {OS_NAME}")
platforms = BUILD_PLATFORMS.get(OS_NAME, ["cpu"])
print(f"Platforms to build for: {platforms}")


def main():
    for platform_name in platforms:
        cmd = [sys.executable, "-m", "scripts.build", "--platform", platform_name]

        print(f"Running build script: {cmd}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise Exception(f"Build failed for platform: {platform_name}")


if __name__ == "__main__":
    main()
