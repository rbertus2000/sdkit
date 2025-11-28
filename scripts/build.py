#!/usr/bin/env python3

import sys
import argparse
import build_common

import build_cuda  # noqa: F401
import build_vulkan  # noqa: F401
import build_cpu  # noqa: F401
import build_metal  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description="Build the project for a specific platform.")
    parser.add_argument(
        "--platform", required=True, choices=["cuda", "vulkan", "cpu", "metal"], help="The platform to build for"
    )
    args = parser.parse_args()

    platform = args.platform
    try:
        module = globals()[f"build_{platform}"]
    except KeyError:
        print(f"Module build_{platform} not found.")
        sys.exit(1)

    build_common.build_project(module.check_environment, module.get_compile_flags, module.get_platform_name)


if __name__ == "__main__":
    main()
