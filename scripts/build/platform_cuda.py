import re
import subprocess


def check_environment():
    """Check if CUDA environment is set up."""
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0 and "nvcc" in result.stdout:
            print("CUDA environment detected.")
            return True
    except:
        pass
    print("CUDA environment not found. Please install CUDA toolkit.")
    return False


def get_compile_flags():
    """Get compile flags for CUDA."""
    return ["-DSD_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=90;89;86;80;75"]


def get_platform_name():
    """Get platform name based on CUDA version."""
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            # Extract version from output like "Cuda compilation tools, release 12.1, V12.1.105"
            match = re.search(r"release (\d+)\.(\d+)", result.stdout)
            if match:
                major = int(match.group(1))
                return f"cu{major}"
    except:
        pass
    return "cuda"  # fallback
