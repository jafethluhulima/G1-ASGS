# Step 2: 2D Image Segmentation

import argparse
import os
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

from pipeline_core.final_shared import DEFAULT_GROUNDING_MODEL
from pipeline_core.final_shared import DEFAULT_MODEL
from pipeline_core.final_shared import DEFAULT_PROMPT
from pipeline_core.final_shared import DEFAULT_SAM_MODEL
from pipeline_core.final_shared import generate_masks
from pipeline_core.final_shared import preview_run


# Parameter settings
SETTINGS = {
    "image_folder": Path("Output/resized_images"),
    "output_folder": Path("Output/2_segmentation_2D"),
    "model": DEFAULT_MODEL,
    "grounding_model": DEFAULT_GROUNDING_MODEL,
    "sam_model": DEFAULT_SAM_MODEL,
    "prompt": DEFAULT_PROMPT,
    "device": "auto",
    "box_threshold": 0.25,
    "text_threshold": 0.20,
    "limit": 0,
    "every_nth": 1,
    "overwrite": False,
    "skip_previews": False,
    "dry_run": False,
}


def resolve_project_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def main():
    parser = argparse.ArgumentParser(
        description="Create combined SegFormer + DINO/SAM 2D segmentation masks only."
    )
    parser.add_argument("--image_folder", type=Path, default=SETTINGS["image_folder"])
    parser.add_argument("--output_folder", type=Path, default=SETTINGS["output_folder"])
    parser.add_argument("--model", default=SETTINGS["model"])
    parser.add_argument("--grounding_model", default=SETTINGS["grounding_model"])
    parser.add_argument("--sam_model", default=SETTINGS["sam_model"])
    parser.add_argument("--prompt", default=SETTINGS["prompt"])
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=SETTINGS["device"])
    parser.add_argument("--box_threshold", type=float, default=SETTINGS["box_threshold"])
    parser.add_argument("--text_threshold", type=float, default=SETTINGS["text_threshold"])
    parser.add_argument("--limit", type=int, default=SETTINGS["limit"])
    parser.add_argument("--every_nth", type=int, default=SETTINGS["every_nth"])
    parser.add_argument("--overwrite", action="store_true", default=SETTINGS["overwrite"])
    parser.add_argument("--skip_previews", action="store_true", default=SETTINGS["skip_previews"])
    parser.add_argument("--dry_run", action="store_true", default=SETTINGS["dry_run"])
    args = parser.parse_args()

    if args.limit < 0:
        raise ValueError("--limit cannot be negative.")
    if args.every_nth < 1:
        raise ValueError("--every_nth must be at least 1.")
    for name in ("box_threshold", "text_threshold"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name} must be between 0 and 1.")

    image_folder = resolve_project_path(args.image_folder)
    output_folder = resolve_project_path(args.output_folder)

    if not image_folder.exists():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    print("\nStep 2: combined 2D segmentation")
    print(f"Image folder: {image_folder}")
    print(f"Segmentation output: {output_folder}")

    if args.dry_run:
        preview_run(image_folder, output_folder, args.limit, args.every_nth)
        return

    generate_masks(
        image_folder=image_folder,
        output_folder=output_folder,
        model_name=args.model,
        grounding_model_name=args.grounding_model,
        sam_model_name=args.sam_model,
        prompt=args.prompt,
        requested_device=args.device,
        limit=args.limit,
        every_nth=args.every_nth,
        overwrite=args.overwrite,
        write_previews=not args.skip_previews,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    print("\nStep 2 done")


if __name__ == "__main__":
    main()
