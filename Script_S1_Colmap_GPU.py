# GPU launcher for COLMAP reconstruction

import os
import shutil
import sys
from pathlib import Path

from Script_S1_Colmap_CPU import main as run_s1_colmap


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


def get_argument_value(arguments, name):
    """Return the value passed after an argument name, if present."""
    for index, argument in enumerate(arguments):
        if argument == name and index + 1 < len(arguments):
            return arguments[index + 1]
        prefix = f"{name}="
        if argument.startswith(prefix):
            return argument[len(prefix):]
    return None


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

    if colmap_path is None and LOCAL_CUDA_COLMAP.exists():
        arguments = [*arguments, "--colmap_path", str(LOCAL_CUDA_COLMAP)]
        colmap_path = str(LOCAL_CUDA_COLMAP)

    if colmap_path is None:
        raise SystemExit(
            "GPU COLMAP needs a CUDA-enabled COLMAP executable.\n"
            "Install a CUDA COLMAP build, then run for example:\n\n"
            "  python Script_S1_Colmap_GPU.py "
            "--colmap_path C:\\path\\to\\cuda\\colmap.exe\n\n"
            "The current default project install is the non-CUDA build at "
            "C:\\Users\\Gebruiker\\colmap-x64-windows-nocuda\\COLMAP.bat."
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
    arguments = validate_cuda_colmap(sys.argv[1:])

    if "--use_gpu" not in arguments:
        arguments.append("--use_gpu")

    sys.argv = [sys.argv[0], *arguments]
    run_s1_colmap()


if __name__ == "__main__":
    main()
