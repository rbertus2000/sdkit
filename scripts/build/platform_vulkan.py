import os
import platform
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


def get_compile_flags():
    """Get compile flags for Vulkan."""
    return ["-DSD_VULKAN=ON"]


def get_platform_name():
    """Get platform name for Vulkan."""
    return "vulkan"


def get_env():
    """Get environment variables for Vulkan build."""
    env = os.environ.copy()
    if platform.system() == "Windows":
        # Override this to prevent Vulkan from using cl.exe instead of clang.exe
        env["PATH"] = env["PATH"].replace("Tools\\MSVC\\14.41.34120\\bin\\Hostx64\\x64", "")
        env["PATH"] += ";C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\Llvm\\x64\\bin"

    return env
