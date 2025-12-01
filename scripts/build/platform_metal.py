import platform


def check_environment():
    """Check if Metal is available (macOS only)."""
    if platform.system() == "Darwin":
        print("Metal environment detected (macOS).")
        return True
    print("Metal is only available on macOS.")
    return False


def get_compile_flags():
    """Get compile flags for Metal."""
    return ["-DSD_METAL=ON"]


def get_platform_name():
    """Get platform name for Metal."""
    return "metal"
