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


def get_compile_flags():
    """Get compile flags for Vulkan."""
    return ["-DSD_VULKAN=ON"]


def get_platform_name():
    """Get platform name for Vulkan."""
    return "vulkan"


def get_manifest_data():
    """Get additional manifest data for Vulkan."""
    try:
        result = subprocess.run(["vulkaninfo"], capture_output=True, text=True)
        if result.returncode == 0:
            # Parse version from output like "Vulkan Instance Version: 1.3.268"
            output = result.stdout + result.stderr
            output = output[:400]
            match = re.search(r"Vulkan Instance Version:\s*(\d+\.\d+\.\d+)", output)
            if match:
                version = match.group(1)
                return {"vulkan_version": version}
        else:
            print(f"DEBUG: vulkaninfo failed with return code {result.returncode}")
    except Exception as e:
        print(f"Exception: {e}")
        pass
    return {}
