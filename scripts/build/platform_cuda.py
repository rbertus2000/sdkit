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
    return ["-DSD_CUDA=ON"]


def get_variants():
    """Get list of CUDA compute capability variants to build.

    Returns a list of dicts, each with 'name' and 'compile_flags' keys.
    """
    return [
        {"name": "sm75", "compile_flags": ["-DCMAKE_CUDA_ARCHITECTURES=75"]},
        {"name": "sm80", "compile_flags": ["-DCMAKE_CUDA_ARCHITECTURES=80"]},
        {"name": "sm86", "compile_flags": ["-DCMAKE_CUDA_ARCHITECTURES=86"]},
        {"name": "sm89", "compile_flags": ["-DCMAKE_CUDA_ARCHITECTURES=89"]},
        {"name": "sm90", "compile_flags": ["-DCMAKE_CUDA_ARCHITECTURES=90"]},
        {"name": "sm100", "compile_flags": ["-DCMAKE_CUDA_ARCHITECTURES=100"]},
        {"name": "sm120", "compile_flags": ["-DCMAKE_CUDA_ARCHITECTURES=120"]},
    ]


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
