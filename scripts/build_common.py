import os
import subprocess
import hashlib
import sys
import json
import platform
import shutil
import tarfile


def get_os():
    """Get OS name for target triple."""
    system = platform.system()
    if system == "Windows":
        return "win"
    elif system == "Darwin":
        return "mac"
    elif system == "Linux":
        return "linux"
    else:
        return system.lower()


def get_arch():
    """Get architecture for target triple."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x64"
    elif machine in ("arm64", "aarch64"):
        return "arm64"
    else:
        return machine


def configure_cmake(build_dir, source_dir, options=[]):
    """Configure CMake with given options."""
    options = options + ["-DCMAKE_BUILD_TYPE=Release"]
    os.makedirs(build_dir, exist_ok=True)
    cmake_cmd = ["cmake", "-S", source_dir, "-B", build_dir] + options
    print(f"Configuring CMake: {' '.join(cmake_cmd)}")

    result = subprocess.run(cmake_cmd)
    if result.returncode != 0:
        print("CMake configure failed")
        sys.exit(1)


def build_cmake(build_dir):
    """Build the project using CMake."""
    cmake_cmd = ["cmake", "--build", build_dir, "--config", "Release"]
    print(f"Building: {' '.join(cmake_cmd)}")
    result = subprocess.run(cmake_cmd)
    if result.returncode != 0:
        print("Build failed")
        sys.exit(1)


def get_release_files(build_dir):
    """Get the list of all distributable release files (executables and shared libraries)."""
    bin_dir = os.path.join(build_dir, "bin")
    lib_dir = os.path.join(build_dir, "lib")
    files = []
    # Collect all files in bin (executables and dlls)
    if os.path.exists(bin_dir):
        for root, dirs, filenames in os.walk(bin_dir):
            for filename in filenames:
                files.append(os.path.join(root, filename))
    # Collect shared libraries in lib (.so, .dylib, .dll)
    if os.path.exists(lib_dir):
        for root, dirs, filenames in os.walk(lib_dir):
            for filename in filenames:
                if filename.endswith((".so", ".dylib", ".dll")):
                    files.append(os.path.join(root, filename))
    return files


def compute_sha256(file_path):
    """Compute SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def prepare_build(build_subdir, check_func):
    """Prepare build directories and check prerequisites."""
    project_root = os.getcwd()
    build_root = os.path.join(project_root, "build")
    build_dir = os.path.join(build_root, build_subdir)

    os.makedirs(build_root, exist_ok=True)

    cmake_file = os.path.join(project_root, "CMakeLists.txt")
    if not os.path.exists(cmake_file):
        print("CMakeLists.txt not found in the current directory.")
        sys.exit(1)

    if not check_func():
        sys.exit(1)

    return project_root, build_dir


def get_target_triple(get_platform_name_func):
    """Get the target triple for the build."""
    platform_name = get_platform_name_func()
    os_name = get_os()
    arch_name = get_arch()
    target_triple = f"{os_name}-{arch_name}-{platform_name}"
    return target_triple


def build_project_cmake(build_dir, project_root, options):
    """Configure and build the project using CMake."""
    configure_cmake(build_dir, project_root, options)
    build_cmake(build_dir)


def collect_and_move_artifacts(build_dir, target_triple):
    """Collect release files, compress each to .tar.gz with maximum compression, move to release_artifacts with prefixed names, and create manifest."""
    release_files = get_release_files(build_dir)

    release_artifacts_dir = os.path.join(build_dir, "release_artifacts")
    os.makedirs(release_artifacts_dir, exist_ok=True)

    manifest = {"files": {}}
    for file_path in release_files:
        basename = os.path.basename(file_path)
        new_name = f"{target_triple}-{basename}"
        tar_gz_name = f"{new_name}.tar.gz"
        tar_gz_path = os.path.join(release_artifacts_dir, tar_gz_name)
        with tarfile.open(tar_gz_path, "w:gz", compresslevel=9) as tar:
            tar.add(file_path, arcname=basename)
        sha256 = compute_sha256(tar_gz_path)
        uri = tar_gz_name
        manifest["files"][new_name] = {"sha256": sha256, "uri": uri}

    return release_artifacts_dir, manifest


def write_manifest(release_artifacts_dir, target_triple, manifest):
    """Write the manifest to the release_artifacts directory and print it."""
    manifest_path = os.path.join(release_artifacts_dir, f"{target_triple}-manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"Target triple: {target_triple}")
    print("Build manifest:")
    print(json.dumps(manifest, indent=4))


def build_project(check_func, get_compile_flags_func, get_platform_name_func):
    """Common build logic for all backends."""
    target_triple = get_target_triple(get_platform_name_func)
    project_root, build_dir = prepare_build(target_triple, check_func)
    options = get_compile_flags_func()
    build_project_cmake(build_dir, project_root, options)
    release_artifacts_dir, manifest = collect_and_move_artifacts(build_dir, target_triple)
    write_manifest(release_artifacts_dir, target_triple, manifest)

    print("Release artifacts are located in:", release_artifacts_dir)
