import sys
import argparse
from . import common

from . import platform_cuda  # noqa: F401
from . import platform_vulkan  # noqa: F401
from . import platform_cpu  # noqa: F401
from . import platform_metal  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description="Build the project for a specific platform.")
    parser.add_argument(
        "--platform", required=True, choices=["cuda", "vulkan", "cpu", "metal"], help="The platform to build for"
    )
    args = parser.parse_args()

    platform = args.platform
    try:
        module = globals()[f"platform_{platform}"]
    except KeyError:
        print(f"Module platform_{platform} not found.")
        sys.exit(1)

    common.build_project(module.check_environment, module.get_compile_flags, module.get_platform_name)


if __name__ == "__main__":
    main()
