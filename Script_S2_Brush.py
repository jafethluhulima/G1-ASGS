# Brush 3DGS generation from output

import argparse
import shutil
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


# Run terminal command

def run_command(command):

    print("\nRunning:")
    print(" ".join(command))

    subprocess.run(command, check=True)


# Create folders if they do not exist

def make_folder(path):

    path.mkdir(parents=True, exist_ok=True)


# Find the Brush executable

def resolve_brush_path(brush_path):

    explicit_path = Path(brush_path)

    if explicit_path.exists():
        return str(explicit_path)

    installed_path = shutil.which(brush_path)

    if installed_path:
        return installed_path

    local_build = Path.home() / "brush" / "brush-main" / "target" / "release" / "brush.exe"

    if brush_path == "brush" and local_build.exists():
        print(f"Using Brush installation: {local_build}")
        return str(local_build)

    raise FileNotFoundError(
        "Brush executable not found. Provide its location with --brush_path."
    )


def resolve_project_path(path_value):
    path = Path(path_value)

    if path.is_absolute():
        return path

    return SCRIPT_DIR / path


# Find one sparse COLMAP model from S1

def find_sparse_model(colmap_folder, model_id):

    sparse_folder = colmap_folder / "sparse"

    model_folder = sparse_folder / model_id

    needed_files = [
        "cameras.bin",
        "images.bin",
        "points3D.bin",
    ]

    if all((model_folder / file).exists() for file in needed_files):
        return model_folder

    raise FileNotFoundError(
        f"COLMAP sparse model {model_id} not found in {sparse_folder}"
    )


# Create the dataset folder Brush expects

def brush_dataset_is_clean(dataset_folder):

    if not dataset_folder.exists():

        return False

    extra_sparse_models = [
        file
        for file in (dataset_folder / "sparse").rglob("cameras.bin")
        if file.parent.name != "0"
    ]

    return (
        (dataset_folder / "images").exists()
        and (dataset_folder / "sparse" / "0" / "cameras.bin").exists()
        and len(extra_sparse_models) == 0
    )


def create_brush_dataset(images_folder, colmap_folder, dataset_folder, model_id):

    if dataset_folder.exists():

        if brush_dataset_is_clean(dataset_folder):
            print(f"\nReusing Brush dataset: {dataset_folder}")
            return

        raise FileExistsError(
            f"Brush dataset exists but is not clean for this S2 script: {dataset_folder}. "
            "Use a new --dataset folder for a clean Brush run."
        )

    sparse_model = find_sparse_model(colmap_folder, model_id)

    if not images_folder.exists():
        raise FileNotFoundError(f"S1 images folder not found: {images_folder}")

    print(f"\nCreating Brush dataset: {dataset_folder}")

    shutil.copytree(
        images_folder,
        dataset_folder / "images",
    )

    make_folder(dataset_folder / "sparse")

    shutil.copytree(
        sparse_model,
        dataset_folder / "sparse" / "0",
    )

    print(f"Using COLMAP sparse model: {sparse_model}")
    print("Finished creating Brush dataset")


# Run Brush training

def find_latest_export(export_folder):

    exports = list(export_folder.glob("*.ply"))

    if len(exports) == 0:

        return None

    exports.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    return exports[0]


def find_expected_export(export_folder, export_name, steps):

    names_to_check = [
        export_name.replace("{iter}", str(steps)),
        export_name.replace("{iter}", f"{steps:04d}"),
        export_name.replace("{iter}", f"{steps:06d}"),
    ]

    for name in names_to_check:

        export_path = export_folder / name

        if export_path.exists():

            return export_path

    return None

def run_brush(
    brush_path,
    dataset_folder,
    steps,
    export_every,
    export_folder,
    export_name,
    final_ply,
    with_viewer,
):

    if not dataset_folder.exists():
        raise FileNotFoundError(f"Brush dataset not found: {dataset_folder}")

    if not (dataset_folder / "images").exists():
        raise FileNotFoundError(f"Brush images folder not found: {dataset_folder / 'images'}")

    if not any((dataset_folder / "sparse").rglob("cameras.bin")):
        raise FileNotFoundError(f"Brush sparse COLMAP model not found: {dataset_folder / 'sparse'}")

    command = [
        brush_path,
        str(dataset_folder),
        "--total-train-iters",
        str(steps),
        "--export-every",
        str(export_every),
        "--export-path",
        str(export_folder),
        "--export-name",
        export_name,
    ]

    if with_viewer:
        command.append("--with-viewer")

    print("\nStarting Brush training")

    try:

        run_command(command)

    except subprocess.CalledProcessError as error:

        print("\nBrush stopped with an error.")
        print(f"Exit code: {error.returncode}")
        print("S2 will still copy the latest export if one exists.")

    expected_export = find_expected_export(export_folder, export_name, steps)

    if expected_export is None:

        expected_export = find_latest_export(export_folder)

    if expected_export is None:

        raise FileNotFoundError(f"No Brush exports found in {export_folder}")

    make_folder(final_ply.parent)
    shutil.copy2(expected_export, final_ply)

    print(f"\nCopied Brush export: {expected_export}")
    print(f"\nFinal 3DGS PLY for S3: {final_ply}")


# Main pipeline

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default="Output",
        help="S1 output folder containing resized_images and colmap folders."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="Output/brush_dataset",
        help="Brush dataset folder to create or reuse."
    )

    parser.add_argument(
        "--brush_path",
        default="brush"
    )

    parser.add_argument(
        "--model_id",
        default="0",
        help="COLMAP sparse model folder to use, usually 0."
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Number of Brush training steps."
    )

    parser.add_argument(
        "--export_every",
        type=int,
        default=500,
        help="Export a PLY every N training steps."
    )

    parser.add_argument(
        "--export_folder",
        type=str,
        default="Output/brush_exports"
    )

    parser.add_argument(
        "--export_name",
        default="export_{iter}.ply"
    )

    parser.add_argument(
        "--final_ply",
        type=str,
        default=None,
        help="Clear final PLY name used by S3."
    )

    parser.add_argument(
        "--with_viewer",
        action="store_true",
        help="Open the Brush viewer during training."
    )

    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Print resolved paths without creating the dataset or running Brush."
    )

    args = parser.parse_args()

    input_folder = resolve_project_path(args.input)

    images_folder = input_folder / "resized_images"

    colmap_folder = input_folder / "colmap"

    dataset_folder = resolve_project_path(args.dataset)

    export_folder = resolve_project_path(args.export_folder).resolve()

    if args.final_ply:
        final_ply = resolve_project_path(args.final_ply).resolve()
    else:
        final_ply = (
            SCRIPT_DIR / "Output/viewer_ready" / f"gaussian_splats_{args.steps}_steps.ply"
        ).resolve()

    brush_path = resolve_brush_path(args.brush_path)

    print(f"\nRunning S2 script: {Path(__file__).resolve()}")
    print(f"S1 output folder: {input_folder}")
    print(f"S1 images folder: {images_folder}")
    print(f"S1 COLMAP folder: {colmap_folder}")
    print(f"COLMAP sparse model: {args.model_id}")
    print(f"Brush dataset folder: {dataset_folder}")
    print(f"Brush executable: {brush_path}")
    print(f"Training steps: {args.steps}")
    print(f"Export every: {args.export_every}")
    print(f"Export folder: {export_folder}")
    print(f"Final PLY: {final_ply}")

    if args.check_only:
        if brush_dataset_is_clean(dataset_folder):
            print(f"Existing Brush dataset is ready: {dataset_folder}")
            return

        print(f"S1 images exist: {images_folder.exists()}")
        print(f"S1 sparse model: {find_sparse_model(colmap_folder, args.model_id)}")
        return

    create_brush_dataset(
        images_folder=images_folder,
        colmap_folder=colmap_folder,
        dataset_folder=dataset_folder,
        model_id=args.model_id,
    )

    run_brush(
        brush_path=brush_path,
        dataset_folder=dataset_folder,
        steps=args.steps,
        export_every=args.export_every,
        export_folder=export_folder,
        export_name=args.export_name,
        final_ply=final_ply,
        with_viewer=args.with_viewer,
    )

    print("\nS2 Brush pipeline done")


if __name__ == "__main__":

    main()
