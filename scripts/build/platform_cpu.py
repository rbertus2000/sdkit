def check_environment():
    """CPU build always available."""
    print("CPU build environment ready.")
    return True


def get_compile_flags(target_any):
    """Get compile flags for CPU."""
    return []


def get_platform_name():
    """Get platform name for CPU."""
    return "cpu"
