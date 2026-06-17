# Step 1a: COLMAP SfM reconstruction (CPU Version)

import argparse
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent


# Supported image types

SUPPORTED_IMAGES = [
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
]


# Parameter settings
SETTINGS = {
    "input": "Input/images",
    "output": "Output",
    "colmap_path": "colmap",
    "max_size": 1600,
    "overlap": 10,
    "camera_sharing": "per_folder",
    "vocab_tree_path": "",
    "vocab_num_images": 50,
    "use_gpu": False,
    "max_bundle_errors": 25,
    "check_only": False,
    "reuse_prepared_images": False,
}


# Run terminal command

def run_command(command, max_bundle_errors=None):

    print("\nRunning:")
    print(" ".join(command))

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    bundle_error_count = 0

    bundle_warning_printed = False

    bundle_error_text = [
        "linear solver failure",
        "matrix not positive definite",
        "bundle adjustment failed",
    ]

    for line in process.stdout:

        print(line, end="")

        lower_line = line.lower()

        if max_bundle_errors is not None and any(
            text in lower_line for text in bundle_error_text
        ):

            bundle_error_count += 1

            if (
                bundle_error_count >= max_bundle_errors
                and not bundle_warning_printed
            ):

                bundle_warning_printed = True

                print("\nCOLMAP WARNING")
                print(
                    f"Bundle adjustment warning limit reached: "
                    f"{bundle_error_count}/{max_bundle_errors}"
                )
                print("S1 will keep running, but the reconstruction may be unstable.")

    return_code = process.wait()

    if return_code != 0:

        raise subprocess.CalledProcessError(return_code, command)


# Find the COLMAP executable

def resolve_colmap_path(colmap_path):

    explicit_path = Path(colmap_path)

    if explicit_path.exists():
        if explicit_path.is_file():
            return str(explicit_path)

        if colmap_path != "colmap":
            raise FileNotFoundError(
                f"COLMAP path points to a folder, not an executable: {explicit_path}"
            )

    installed_path = shutil.which(colmap_path)

    if installed_path:
        return installed_path

    env_colmap_path = os.environ.get("COLMAP_PATH")
    if env_colmap_path and Path(env_colmap_path).exists():
        return str(Path(env_colmap_path).resolve())

    if colmap_path == "colmap":
        home = Path.home()
        executable_names = ["colmap.exe", "COLMAP.bat", "colmap"]
        candidate_folders = [
            SCRIPT_DIR / "Tools" / "colmap" / "bin",
            SCRIPT_DIR / "Tools" / "colmap",
            SCRIPT_DIR / "Tools" / "colmap-cuda-manifest" / "vcpkg_installed" / "x64-windows" / "tools" / "colmap",
            home / "colmap-x64-windows-nocuda",
            home / "colmap-x64-windows-cuda",
            home / "COLMAP",
            home / "colmap",
        ]
        for folder in candidate_folders:
            for executable_name in executable_names:
                candidate = folder / executable_name
                if candidate.exists():
                    print(f"Using COLMAP installation: {candidate}")
                    return str(candidate)

    raise FileNotFoundError(
        "COLMAP executable not found.\n"
        "Install COLMAP and make it available in one of these ways:\n"
        "  1. Add colmap to PATH, then rerun the script.\n"
        "  2. Set COLMAP_PATH to the full COLMAP executable path.\n"
        "  3. Pass --colmap_path <path-to-colmap> when running the script.\n"
        "  4. Put a local install in Tools\\colmap."
    )


def resolve_project_path(path_value):
    path = Path(path_value)

    if path.is_absolute():
        return path

    return SCRIPT_DIR / path


# Sort images by folder and by the number in the file name

def image_sort_key(image_path, input_folder):

    relative_path = image_path.relative_to(input_folder)

    numbers = re.findall(r"\d+", relative_path.stem)

    sequence_number = int(numbers[-1]) if numbers else 999999999

    return (
        tuple(part.lower() for part in relative_path.parent.parts),
        sequence_number,
        relative_path.name.lower(),
    )


# Find all input images

def get_images(input_folder):

    if not input_folder.exists():
        raise FileNotFoundError(
            f"Input folder not found: {input_folder}. "
            "Put your raw images there or pass --input with the correct folder."
        )

    images = []

    for file in input_folder.rglob("*"):

        if file.is_file() and file.suffix.lower() in SUPPORTED_IMAGES:

            images.append(file)

    images.sort(key=lambda image_path: image_sort_key(image_path, input_folder))

    if len(images) == 0:

        raise FileNotFoundError(
            f"No images found in {input_folder}. "
            f"Supported image types: {', '.join(SUPPORTED_IMAGES)}"
        )

    return images


# Prepare images for COLMAP

def prepare_images(input_folder, images_folder, max_size):

    images = get_images(input_folder)

    print(f"\nFound {len(images)} images")
    print("Input image folders:")

    folder_counts = Counter(
        image_path.relative_to(input_folder).parent for image_path in images
    )

    for folder, count in sorted(
        folder_counts.items(),
        key=lambda item: (tuple(str(part).lower() for part in item[0].parts), item[1]),
    ):
        print(f"  {folder}: {count}")

    print("\nPreparing images...")

    sequence_indexes = {}

    for i, image_path in enumerate(images, start=1):

        relative_path = image_path.relative_to(input_folder)

        parent_folder = relative_path.parent

        sequence_indexes[parent_folder] = sequence_indexes.get(parent_folder, 0) + 1

        new_name = f"{sequence_indexes[parent_folder]:06d}_{image_path.name}"

        output_path = images_folder / parent_folder / new_name

        os.makedirs(output_path.parent, exist_ok=True)

        with Image.open(image_path) as image:

            # Convert transparent images to RGB for COLMAP

            if image.mode in ("RGBA", "LA"):

                background = Image.new(
                    "RGB",
                    image.size,
                    (255, 255, 255)
                )

                background.paste(
                    image,
                    mask=image.split()[-1]
                )

                image = background

            else:

                image = image.convert("RGB")

            width, height = image.size

            scale = min(
                max_size / width,
                max_size / height
            )

            # Resize only if image is too large

            if scale < 1:

                new_width = int(width * scale)

                new_height = int(height * scale)

                image = image.resize(
                    (new_width, new_height),
                    Image.Resampling.LANCZOS
                )

            image.save(output_path)

        if i % 50 == 0 or i == len(images):

            print(f"Prepared {i}/{len(images)} images")

    print("Finished preparing images")


# Main pipeline

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=SETTINGS["input"]
    )

    parser.add_argument(
        "--output",
        type=str,
        default=SETTINGS["output"]
    )

    parser.add_argument(
        "--colmap_path",
        default=SETTINGS["colmap_path"]
    )

    parser.add_argument(
        "--max_size",
        type=int,
        default=SETTINGS["max_size"]
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=SETTINGS["overlap"]
    )

    parser.add_argument(
        "--camera_sharing",
        choices=["single", "per_folder", "per_image"],
        default=SETTINGS["camera_sharing"],
        help="Use per_folder when input images are split into capture folders."
    )

    parser.add_argument(
        "--vocab_tree_path",
        type=str,
        default=SETTINGS["vocab_tree_path"],
        help="Optional COLMAP vocabulary tree for matching across image folders."
    )

    parser.add_argument(
        "--vocab_num_images",
        type=int,
        default=SETTINGS["vocab_num_images"]
    )

    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=SETTINGS["use_gpu"]
    )

    parser.add_argument(
        "--max_bundle_errors",
        type=int,
        default=SETTINGS["max_bundle_errors"],
        help="Print a warning if COLMAP gives this many bundle adjustment solver warnings."
    )

    parser.add_argument(
        "--check_only",
        action="store_true",
        default=SETTINGS["check_only"],
        help="Print resolved paths and image count without preparing images or running COLMAP."
    )

    parser.add_argument(
        "--reuse_prepared_images",
        action="store_true",
        default=SETTINGS["reuse_prepared_images"],
        help="Reuse an existing prepared image folder when continuing a failed S1 run."
    )

    args = parser.parse_args()

    input_folder = resolve_project_path(args.input)

    output_folder = resolve_project_path(args.output)

    images_folder = output_folder / "resized_images"

    colmap_folder = output_folder / "colmap"

    sparse_folder = colmap_folder / "sparse"

    database_path = colmap_folder / "database.db"

    colmap_path = resolve_colmap_path(args.colmap_path)

    print(f"\nRunning S1 script: {Path(__file__).resolve()}")
    print(f"Input folder: {input_folder}")
    print(f"Prepared image folder: {images_folder}")
    print(f"COLMAP workspace: {colmap_folder}")
    print(f"COLMAP executable: {colmap_path}")
    print(f"Camera sharing: {args.camera_sharing}")
    print(f"Bundle adjustment warning limit: {args.max_bundle_errors}")

    if args.check_only:

        if args.reuse_prepared_images:
            print(f"Prepared images found: {len(get_images(images_folder))}")
        else:
            print(f"Input images found: {len(get_images(input_folder))}")
        return

    if database_path.exists():
        raise FileExistsError(
            f"COLMAP database already exists: {database_path}. "
            "Use a new --output folder or delete the existing COLMAP workspace."
        )

    reuse_prepared_images = False

    if images_folder.exists() and any(images_folder.rglob("*")):
        if args.reuse_prepared_images:
            reuse_prepared_images = True
        else:
            raise FileExistsError(
                f"Prepared image folder already contains files: {images_folder}. "
                "Use a new --output folder for a clean S1 run, or pass "
                "--reuse_prepared_images to continue from the prepared images."
            )

    # Create folders

    os.makedirs(images_folder, exist_ok=True)

    os.makedirs(sparse_folder, exist_ok=True)

    if reuse_prepared_images:
        prepared_count = len(get_images(images_folder))
        input_count = len(get_images(input_folder))

        if prepared_count != input_count:
            raise RuntimeError(
                f"Prepared image folder is incomplete: {prepared_count} prepared "
                f"images, but {input_count} input images. Delete {images_folder} "
                "and rerun S1 without --reuse_prepared_images."
            )

        print(f"\nReusing prepared images: {prepared_count}")
    else:
        prepare_images(
            input_folder=input_folder,
            images_folder=images_folder,
            max_size=args.max_size,
        )

    # COLMAP feature extraction

    camera_options = {
        "single": ["--ImageReader.single_camera", "1"],
        "per_folder": ["--ImageReader.single_camera_per_folder", "1"],
        "per_image": [],
    }[args.camera_sharing]

    feature_command = [
        colmap_path,
        "feature_extractor",

        "--database_path",
        str(database_path),

        "--image_path",
        str(images_folder),

        "--SiftExtraction.use_gpu",
        "1" if args.use_gpu else "0"
    ]

    feature_command.extend(camera_options)

    run_command(feature_command)

    # COLMAP sequential matching

    run_command([
        colmap_path,
        "sequential_matcher",

        "--database_path",
        str(database_path),

        "--SequentialMatching.overlap",
        str(args.overlap),

        "--SiftMatching.use_gpu",
        "1" if args.use_gpu else "0"
    ])

    # Optional matching between different capture folders

    if args.vocab_tree_path:

        vocab_tree_path = resolve_project_path(args.vocab_tree_path)

        run_command([
            colmap_path,
            "vocab_tree_matcher",

            "--database_path",
            str(database_path),

            "--VocabTreeMatching.vocab_tree_path",
            str(vocab_tree_path),

            "--VocabTreeMatching.num_images",
            str(args.vocab_num_images),

            "--SiftMatching.use_gpu",
            "1" if args.use_gpu else "0"
        ])

    # COLMAP sparse reconstruction

    run_command([
        colmap_path,
        "mapper",

        "--database_path",
        str(database_path),

        "--image_path",
        str(images_folder),

        "--output_path",
        str(sparse_folder)
    ], max_bundle_errors=args.max_bundle_errors)

    # Finished

    print("\nDone")

    print("\nBrush Inputs:")

    print(f"Images Folder: {images_folder}")

    print(f"COLMAP Folder: {colmap_folder}")


if __name__ == "__main__":

    main()
