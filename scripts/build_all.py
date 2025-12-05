import sys
import platform
import subprocess

OS_NAME = platform.system()

BUILD_PLATFORMS = {
    "Windows": [
        ("cpu", "x64"),
        ("cuda", "x64"),
        ("vulkan", "x64"),
        ("vulkan", "arm64"),
    ],
    "Linux": [
        ("cpu", "x64"),
        ("cuda", "x64"),
        ("vulkan", "x64"),
    ],
    "Darwin": [
        ("metal", "arm64"),
    ],
}

print(f"Detected OS: {OS_NAME}")
platforms = BUILD_PLATFORMS.get(OS_NAME, ["cpu"])
print(f"Platforms to build for: {platforms}")


def main():
    for platform_name, arch in platforms:
        cmd = [sys.executable, "-m", "scripts.build", "--platform", platform_name, "--arch", arch]

        print(f"Running build script: {cmd}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise Exception(f"Build failed for platform: {platform_name}")


if __name__ == "__main__":
    main()
