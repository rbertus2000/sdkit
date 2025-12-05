import os
import re
import subprocess


def check_environment():
    """Check if Vulkan SDK is set up."""
    try:
        result = subprocess.run(["vulkaninfo", "--version"], capture_output=True, text=True)
        if not result.stderr:
            print("Vulkan environment detected.")
            return True
    except:
        pass
    print("Vulkan SDK not found. Please install Vulkan SDK.")
    return False


def get_compile_flags(target_any):
    """Get compile flags for Vulkan."""
    flags = ["-DSD_VULKAN=ON"]

    if target_any.startswith("win-arm64"):
        vulkan_sdk_path = os.environ["VULKAN_SDK"].replace("\\", "/")

        flags.append("-G Ninja")
        flags.append("-DCMAKE_TOOLCHAIN_FILE=cmake/arm64-windows-llvm.cmake")
        flags.append(f"-DVulkan_LIBRARY={vulkan_sdk_path}/Lib-ARM64/vulkan-1.lib")

    return flags


def get_platform_name():
    """Get platform name for Vulkan."""
    return "vulkan"


def get_env(target_any):
    """Get environment variables for Vulkan build."""
    if target_any.startswith("win-arm64"):
        env = os.environ.copy()

        # Prevent ggml-vulkan's cmake from picking up MSVC tools
        env["PATH"] = re.sub(r"Tools\\MSVC\\[\d\.]+\\bin\\Hostx64\\x64", r"Tools\\Llvm\\x64\\bin", env["PATH"])

        return env

    return None
