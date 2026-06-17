# Step 1b: COLMAP SfM reconstruction (GPU Version)

import os
import shutil
import sys
from pathlib import Path

from Script_1A_COLMAP_CPU import main as run_s1_colmap


LOCAL_CUDA_COLMAP = (
    Path(__file__).resolve().parent
    / "Tools"
    / "colmap-cuda-manifest"
    / "vcpkg_installed"
    / "x64-windows"
    / "tools"
    / "colmap"
    / "colmap.exe"
)


# Parameter settings
SETTINGS = {
    "input": "Input/images",
    "output": "Output",
    "colmap_cuda_path": LOCAL_CUDA_COLMAP,
    "max_size": 1600,
    "overlap": 10,
    "camera_sharing": "per_folder",
    "vocab_tree_path": "",
    "vocab_num_images": 50,
    "max_bundle_errors": 25,
    "check_only": False,
    "reuse_prepared_images": False,
    "force_gpu_argument": True,
}


def get_argument_value(arguments, name):
    """Return the value passed after an argument name, if present."""
    for index, argument in enumerate(arguments):
        if argument == name and index + 1 < len(arguments):
            return arguments[index + 1]
        prefix = f"{name}="
        if argument.startswith(prefix):
            return argument[len(prefix):]
    return None


def has_argument(arguments, name):
    prefix = f"{name}="
    return name in arguments or any(argument.startswith(prefix) for argument in arguments)


def add_setting_argument(arguments, name, value):
    if has_argument(arguments, name) or value in (None, ""):
        return arguments
    return [*arguments, name, str(value)]


def add_setting_flag(arguments, name, enabled):
    if has_argument(arguments, name) or not enabled:
        return arguments
    return [*arguments, name]


def apply_settings(arguments):
    # Keep command-line values strongest, but let the top SETTINGS block control
    # the normal Run-button workflow.
    for option_name, setting_name in [
        ("--input", "input"),
        ("--output", "output"),
        ("--max_size", "max_size"),
        ("--overlap", "overlap"),
        ("--camera_sharing", "camera_sharing"),
        ("--vocab_tree_path", "vocab_tree_path"),
        ("--vocab_num_images", "vocab_num_images"),
        ("--max_bundle_errors", "max_bundle_errors"),
    ]:
        arguments = add_setting_argument(arguments, option_name, SETTINGS[setting_name])

    arguments = add_setting_flag(arguments, "--check_only", SETTINGS["check_only"])
    arguments = add_setting_flag(
        arguments,
        "--reuse_prepared_images",
        SETTINGS["reuse_prepared_images"],
    )
    return arguments


def resolved_colmap_path(colmap_path):
    if not colmap_path:
        return None

    if colmap_path.lower() == "colmap":
        found = shutil.which("colmap")
        return Path(found) if found else None

    return Path(colmap_path)


def validate_cuda_colmap(arguments):
    if "--help" in arguments or "-h" in arguments:
        return arguments

    colmap_path = get_argument_value(arguments, "--colmap_path")
    env_colmap_path = os.environ.get("COLMAP_CUDA_PATH")

    if colmap_path is None and env_colmap_path:
        arguments = [*arguments, "--colmap_path", env_colmap_path]
        colmap_path = env_colmap_path

    configured_cuda_colmap = SETTINGS["colmap_cuda_path"]
    if colmap_path is None and configured_cuda_colmap:
        configured_cuda_colmap = Path(configured_cuda_colmap)
        if configured_cuda_colmap.exists():
            arguments = [*arguments, "--colmap_path", str(configured_cuda_colmap)]
            colmap_path = str(configured_cuda_colmap)

    if colmap_path is None:
        raise SystemExit(
            "GPU COLMAP needs a CUDA-enabled COLMAP executable.\n"
            "Install a CUDA COLMAP build and make it available in one of these ways:\n"
            "  1. Set COLMAP_CUDA_PATH to the CUDA COLMAP executable.\n"
            "  2. Pass --colmap_path <path-to-cuda-colmap>.\n"
            "  3. Put a local CUDA build in Tools\\colmap-cuda-manifest.\n\n"
            "Example:\n\n"
            "  python Script_1B_COLMAP_GPU.py "
            "--colmap_path <path-to-cuda-colmap>"
        )

    candidate = resolved_colmap_path(colmap_path)
    if candidate is None or not candidate.exists():
        raise SystemExit(f"COLMAP executable not found: {colmap_path}")

    if "nocuda" in str(candidate).lower():
        raise SystemExit(
            "This looks like the non-CUDA COLMAP build:\n"
            f"  {candidate}\n\n"
            "Use a CUDA-enabled COLMAP executable for GPU feature extraction "
            "and matching."
        )

    return arguments


def main():
    arguments = apply_settings(sys.argv[1:])
    arguments = validate_cuda_colmap(arguments)

    if SETTINGS["force_gpu_argument"] and "--use_gpu" not in arguments:
        arguments.append("--use_gpu")

    sys.argv = [sys.argv[0], *arguments]
    run_s1_colmap()


if __name__ == "__main__":
    main()
