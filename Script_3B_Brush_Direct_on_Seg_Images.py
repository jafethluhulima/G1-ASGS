# Step 3B: Train Brush on segmented images directly

import argparse
import os
import shutil
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def restart_with_project_venv():
    venv_python = SCRIPT_DIR / ".venv_v3" / "Scripts" / "python.exe"
    if not venv_python.exists():
        return

    current_python = Path(sys.executable).resolve()
    target_python = venv_python.resolve()
    if str(current_python).lower() == str(target_python).lower():
        return

    print(f"Restarting with project Python: {target_python}", flush=True)
    os.execv(str(target_python), [str(target_python), str(Path(__file__).resolve()), *sys.argv[1:]])


restart_with_project_venv()

from pipeline_core.brush import create_brush_dataset
from pipeline_core.brush import resolve_brush_path
from pipeline_core.brush import run_brush
from pipeline_core.final_shared import make_segmented_training_images


# EDIT THESE SETTINGS FOR A NORMAL RUN.
# Command-line arguments with the same names can still override these values.
SETTINGS = {
    "s1_output": Path("Output"),
    "segmentation_folder": Path("Output/2_segmentation_2D"),
    "brush_path": "brush",
    "steps": 8000,
    "unknown_mode": "original",
    "blend": 1.0,
    "jpeg_quality": 95,
}


def resolve_project_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def remove_dataset_if_requested(dataset_folder, overwrite_dataset):
    if not dataset_folder.exists() or not overwrite_dataset:
        return

    resolved_dataset = dataset_folder.resolve()
    resolved_output = (SCRIPT_DIR / "Output").resolve()
    if resolved_output not in resolved_dataset.parents:
        raise RuntimeError(f"Refusing to delete dataset outside Output: {resolved_dataset}")

    print(f"Removing old Brush dataset: {dataset_folder}")
    shutil.rmtree(dataset_folder)


def count_masks(segmentation_folder):
    return len(list((segmentation_folder / "masks").rglob("*.png")))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Use existing Final 2D segmentation masks to make segmented images, "
            "then train Brush directly on those segmented images."
        )
    )
    parser.add_argument("--s1_output", type=Path, default=SETTINGS["s1_output"])
    parser.add_argument("--segmentation_folder", type=Path, default=SETTINGS["segmentation_folder"])
    parser.add_argument(
        "--segmented_images",
        type=Path,
        default=Path("Output/3B_segmented_images"),
    )
    parser.add_argument("--dataset", type=Path, default=Path("Output/3B_segmented_brush_dataset"))
    parser.add_argument("--brush_path", default=SETTINGS["brush_path"])
    parser.add_argument("--model_id", default="0")
    parser.add_argument("--steps", type=int, default=SETTINGS["steps"])
    parser.add_argument("--export_every", type=int, default=500)
    parser.add_argument("--export_folder", type=Path, default=Path("Output/3B_segmented_brush_exports"))
    parser.add_argument("--export_name", default="segmented_{iter}.ply")
    parser.add_argument(
        "--final_ply",
        type=Path,
        default=Path("Output/3B_segmented_brush_model/final_segmented_brush_10000_steps.ply"),
    )
    parser.add_argument("--unknown_mode", choices=["original", "color"], default=SETTINGS["unknown_mode"])
    parser.add_argument("--blend", type=float, default=SETTINGS["blend"])
    parser.add_argument("--jpeg_quality", type=int, default=SETTINGS["jpeg_quality"])
    parser.add_argument("--overwrite_images", action="store_true", default=False)
    parser.add_argument("--overwrite_dataset", action="store_true", default=False)
    parser.add_argument("--images_only", action="store_true", default=False)
    parser.add_argument("--skip_training", action="store_true", default=False)
    parser.add_argument("--with_viewer", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be positive.")
    if args.export_every < 1:
        raise ValueError("--export_every must be positive.")
    if not 0 <= args.blend <= 1:
        raise ValueError("--blend must be between 0 and 1.")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg_quality must be between 1 and 100.")

    s1_output = resolve_project_path(args.s1_output)
    images_folder = s1_output / "resized_images"
    colmap_folder = s1_output / "colmap"
    segmentation_folder = resolve_project_path(args.segmentation_folder)
    segmented_images = resolve_project_path(args.segmented_images)
    dataset_folder = resolve_project_path(args.dataset)
    export_folder = resolve_project_path(args.export_folder).resolve()
    final_ply = resolve_project_path(args.final_ply).resolve()

    if not images_folder.exists():
        raise FileNotFoundError(f"S1 resized images not found: {images_folder}")
    if not colmap_folder.exists():
        raise FileNotFoundError(f"S1 COLMAP folder not found: {colmap_folder}")
    mask_count = count_masks(segmentation_folder)

    print("\nStep 3B: Brush directly on segmented images")
    print(f"S1 images: {images_folder}")
    print(f"S1 COLMAP: {colmap_folder}")
    print(f"2D segmentation folder: {segmentation_folder}")
    print(f"2D masks found: {mask_count}")
    print(f"Segmented images: {segmented_images}")
    print(f"Segmented Brush dataset: {dataset_folder}")
    print(f"Segmented Brush final PLY: {final_ply}")

    if args.dry_run:
        if mask_count == 0:
            print(f"Dry-run warning: no masks found yet in {segmentation_folder / 'masks'}")
        print("Dry run only. No files changed.")
        return

    if mask_count == 0:
        raise FileNotFoundError(
            f"No 2D segmentation masks found in: {segmentation_folder / 'masks'}"
        )

    make_segmented_training_images(
        image_folder=images_folder,
        masks_folder=segmentation_folder,
        segmented_folder=segmented_images,
        unknown_mode=args.unknown_mode,
        blend=args.blend,
        overwrite=args.overwrite_images,
        jpeg_quality=args.jpeg_quality,
        copy_missing_masks=False,
    )

    if args.images_only:
        print("\nSegmented images done. Brush training skipped because --images_only was used.")
        return

    remove_dataset_if_requested(dataset_folder, args.overwrite_dataset)
    create_brush_dataset(
        images_folder=segmented_images,
        colmap_folder=colmap_folder,
        dataset_folder=dataset_folder,
        model_id=args.model_id,
    )

    if args.skip_training:
        print("\nSegmented Brush dataset is ready. Brush training skipped because --skip_training was used.")
        return

    brush_path = resolve_brush_path(args.brush_path)
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

    print("\nStep 3B done")


if __name__ == "__main__":
    main()
