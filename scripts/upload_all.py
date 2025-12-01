import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def find_target_triples() -> List[str]:
    """Find all target triples with release_artifacts in the build folder."""
    build_dir = Path("build")

    if not build_dir.exists():
        print(f"Build directory not found: {build_dir}")
        return []

    target_triples = []
    for item in build_dir.iterdir():
        if item.is_dir():
            artifacts_dir = item / "release_artifacts"
            if artifacts_dir.exists() and artifacts_dir.is_dir():
                target_triples.append(item.name)

    return target_triples


def main():
    parser = argparse.ArgumentParser(
        description="Upload release artifacts to GitHub releases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--tag", required=True, help="GitHub release tag (e.g., v1.0.0)")
    parser.add_argument(
        "--repo",
        default="easydiffusion/sdkit",
        help="GitHub repository in format owner/repo (default: easydiffusion/sdkit)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without actually uploading")
    parser.add_argument(
        "--force", action="store_true", help="Force upload all files, even if they already exist with matching hashes"
    )

    args = parser.parse_args()

    # Find all target triples
    target_triples = find_target_triples()
    if not target_triples:
        print("No target triples with release_artifacts found in build/")
        sys.exit(1)

    print(f"Found {len(target_triples)} target triples:")
    for triple in target_triples:
        print(f"  - {triple}")

    # Process each target triple
    for target_triple in target_triples:
        print(f"\nUploading for {target_triple}...")

        cmd = [sys.executable, "-m", "scripts.upload"]
        cmd += ["--tag", args.tag, "--target-triple", target_triple, "--repo", args.repo]

        if args.dry_run:
            cmd.append("--dry-run")
        if args.force:
            cmd.append("--force")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Error uploading {target_triple}")
            sys.exit(1)

    print("\nAll uploads completed.")


if __name__ == "__main__":
    main()
