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
    """Get platform name for CUDA."""
    return "cuda"


def get_manifest_data():
    """Get additional manifest data for CUDA."""
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            # Parse version from output like "nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c) 2005-2023 NVIDIA Corporation\nBuilt on Tue_Jun_13_19:16:58_PDT_2023\nCuda compilation tools, release 12.2, V12.2.91\nBuild cuda_12.2.r12.2/compiler.32965470_0"
            for line in result.stdout.split("\n"):
                if "release" in line:
                    version = line.split("release ")[1].split(",")[0]
                    return {"cuda_version": version}
    except:
        pass
    return {}
