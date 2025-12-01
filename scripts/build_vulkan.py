#!/usr/bin/env python3

import subprocess


def check_environment():
    """Check if Vulkan SDK is set up."""
    try:
        result = subprocess.run(["vulkaninfo", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
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
