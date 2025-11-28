#!/usr/bin/env python3

import os
import subprocess


def check_environment():
    """Check if Vulkan SDK is set up."""
    vulkan_sdk = os.environ.get("VULKAN_SDK")
    if vulkan_sdk and os.path.exists(vulkan_sdk):
        print("Vulkan SDK detected.")
        return True
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
