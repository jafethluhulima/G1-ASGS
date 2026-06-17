# Step 3A: 2D to 3D segmentation lifting using Brush 

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
from pipeline_core.final_shared import lift_masks_to_splats


# Parameter settings
SETTINGS = {
    "s1_output": Path("Output"),
    "segmentation_folder": Path("Output/2_segmentation_2D"),
    "dataset": Path("Output/3A_regular_brush_dataset"),
    "brush_path": "brush",
    "model_id": "0",
    "steps": 12000,
    "export_every": 500,
    "export_folder": Path("Output/3A_regular_brush_exports"),
    "export_name": "regular_{iter}.ply",
    "final_ply": Path("Output/3A_regular_brush_model/final_regular_brush_10000_steps.ply"),
    "semantic_output_folder": Path("Output/3A_semantic_lifted_model"),
    "lift_every_nth": 5,
    "min_opacity": 0.05,
    "min_pixel_confidence": 0.50,
    "min_views": 3,
    "min_vote_ratio": 0.60,
    "no_depth_weighting": False,
    "depth_weight_power": 1.0,
    "min_depth_weight": 0.25,
    "max_depth_weight": 4.0,
    "overwrite_dataset": False,
    "skip_brush": False,
    "skip_lift": False,
    "with_viewer": False,
    "dry_run": False,
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
            "Train a normal realistic Brush 3DGS from S1 output, then lift existing "
            "Final 2D segmentation masks onto that 3DGS."
        )
    )
    parser.add_argument("--s1_output", type=Path, default=SETTINGS["s1_output"])
    parser.add_argument("--segmentation_folder", type=Path, default=SETTINGS["segmentation_folder"])
    parser.add_argument("--dataset", type=Path, default=SETTINGS["dataset"])
    parser.add_argument("--brush_path", default=SETTINGS["brush_path"])
    parser.add_argument("--model_id", default=SETTINGS["model_id"])
    parser.add_argument("--steps", type=int, default=SETTINGS["steps"])
    parser.add_argument("--export_every", type=int, default=SETTINGS["export_every"])
    parser.add_argument("--export_folder", type=Path, default=SETTINGS["export_folder"])
    parser.add_argument("--export_name", default=SETTINGS["export_name"])
    parser.add_argument(
        "--final_ply",
        type=Path,
        default=SETTINGS["final_ply"],
    )
    parser.add_argument(
        "--semantic_output_folder",
        type=Path,
        default=SETTINGS["semantic_output_folder"],
    )
    parser.add_argument("--lift_every_nth", type=int, default=SETTINGS["lift_every_nth"])
    parser.add_argument("--min_opacity", type=float, default=SETTINGS["min_opacity"])
    parser.add_argument("--min_pixel_confidence", type=float, default=SETTINGS["min_pixel_confidence"])
    parser.add_argument("--min_views", type=int, default=SETTINGS["min_views"])
    parser.add_argument("--min_vote_ratio", type=float, default=SETTINGS["min_vote_ratio"])
    parser.add_argument("--no_depth_weighting", action="store_true", default=SETTINGS["no_depth_weighting"])
    parser.add_argument("--depth_weight_power", type=float, default=SETTINGS["depth_weight_power"])
    parser.add_argument("--min_depth_weight", type=float, default=SETTINGS["min_depth_weight"])
    parser.add_argument("--max_depth_weight", type=float, default=SETTINGS["max_depth_weight"])
    parser.add_argument("--overwrite_dataset", action="store_true", default=SETTINGS["overwrite_dataset"])
    parser.add_argument("--skip_brush", action="store_true", default=SETTINGS["skip_brush"])
    parser.add_argument("--skip_lift", action="store_true", default=SETTINGS["skip_lift"])
    parser.add_argument("--with_viewer", action="store_true", default=SETTINGS["with_viewer"])
    parser.add_argument("--dry_run", action="store_true", default=SETTINGS["dry_run"])
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be positive.")
    if args.export_every < 1:
        raise ValueError("--export_every must be positive.")
    if args.lift_every_nth < 1:
        raise ValueError("--lift_every_nth must be at least 1.")
    if args.min_views < 1:
        raise ValueError("--min_views must be at least 1.")
    for name in ("min_opacity", "min_pixel_confidence", "min_vote_ratio"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name} must be between 0 and 1.")

    s1_output = resolve_project_path(args.s1_output)
    images_folder = s1_output / "resized_images"
    colmap_folder = s1_output / "colmap"
    segmentation_folder = resolve_project_path(args.segmentation_folder)
    dataset_folder = resolve_project_path(args.dataset)
    export_folder = resolve_project_path(args.export_folder).resolve()
    final_ply = resolve_project_path(args.final_ply).resolve()
    semantic_output_folder = resolve_project_path(args.semantic_output_folder)

    if not images_folder.exists():
        raise FileNotFoundError(f"S1 resized images not found: {images_folder}")
    if not colmap_folder.exists():
        raise FileNotFoundError(f"S1 COLMAP folder not found: {colmap_folder}")
    mask_count = count_masks(segmentation_folder)

    print("\nStep 3A: regular Brush then 3D lifting")
    print(f"S1 images: {images_folder}")
    print(f"S1 COLMAP: {colmap_folder}")
    print(f"2D segmentation folder: {segmentation_folder}")
    print(f"2D masks found: {mask_count}")
    print(f"Regular Brush dataset: {dataset_folder}")
    print(f"Regular Brush final PLY: {final_ply}")
    print(f"Lifted semantic output: {semantic_output_folder}")

    if args.dry_run:
        if mask_count == 0:
            print(f"Dry-run warning: no masks found yet in {segmentation_folder / 'masks'}")
        print("Dry run only. No files changed.")
        return

    if mask_count == 0:
        raise FileNotFoundError(
            f"No 2D segmentation masks found in: {segmentation_folder / 'masks'}"
        )

    if not args.skip_brush:
        remove_dataset_if_requested(dataset_folder, args.overwrite_dataset)
        create_brush_dataset(
            images_folder=images_folder,
            colmap_folder=colmap_folder,
            dataset_folder=dataset_folder,
            model_id=args.model_id,
        )
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

    if not args.skip_lift:
        lift_masks_to_splats(
            masks_output_folder=segmentation_folder,
            colmap_model_folder=dataset_folder / "sparse" / args.model_id,
            splat_ply=final_ply,
            semantic_output_folder=semantic_output_folder,
            every_nth=args.lift_every_nth,
            min_opacity=args.min_opacity,
            min_pixel_confidence=args.min_pixel_confidence,
            min_views=args.min_views,
            min_vote_ratio=args.min_vote_ratio,
            depth_weighting=not args.no_depth_weighting,
            depth_weight_power=args.depth_weight_power,
            min_depth_weight=args.min_depth_weight,
            max_depth_weight=args.max_depth_weight,
        )

    print("\nStep 3A done")


if __name__ == "__main__":
    main()
