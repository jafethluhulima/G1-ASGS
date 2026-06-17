# 3DGS Semantic Segmentation Pipeline

This project turns ordered indoor photos into a COLMAP reconstruction, produces 2D semantic masks, and creates a semantic 3D Gaussian Splat result using Brush.

Run all commands from the project root.

## What The Pipeline Produces

- A resized image set and COLMAP sparse reconstruction in `Output/resized_images` and `Output/colmap`.
- 2D semantic masks, confidence rasters, previews, a legend, and a manifest in `Output/2_segmentation_2D`.
- Either a realistic Brush model with semantic labels lifted onto the splats, or a Brush model trained directly from segmented images.

## Folder Layout

```text
Input/images
```

Place the raw source photos here. Supported formats are `.jpg`, `.jpeg`, `.png`, `.tif`, and `.tiff`. Subfolders are allowed and are useful when images come from separate capture sequences.

```text
Output
```

All generated data is written here. The pipeline creates the subfolders it needs. Existing S1 COLMAP data is not overwritten automatically.

```text
pipeline_core
```

Shared implementation code for labels, mask generation, Brush dataset preparation, and 2D-to-3D lifting. You normally do not run these files directly.

```text
Tools
```

Optional local tool installs, including a CUDA COLMAP build if one has been placed in the expected `Tools/colmap-cuda-manifest` location.

## Prerequisites

- Windows 10 or Windows 11, 64-bit.
- Python 3.11 is recommended for the AI packages.
- COLMAP is required for step 1.
- A CUDA-enabled COLMAP build is recommended when using `Script_1B_COLMAP_GPU.py`.
- Brush is required for step 3A and 3B. Put `brush` on `PATH` or pass `--brush_path <path_to_brush.exe>`.
- For GPU acceleration: recent NVIDIA driver, CUDA Toolkit, Microsoft Visual C++ Redistributable, and a CUDA-compatible PyTorch install.

See `Software Dependencies and Requirements.docx` for the fuller dependency checklist.

## Python Environment

Create and activate the project virtual environment before running the segmentation or Brush stages. Stages 2 and 3 automatically restart with `.venv_v3` when that environment exists.

```powershell
python -m venv .venv_v3
.\.venv_v3\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch first using the command recommended by the PyTorch website for your CUDA version. For CUDA 12.4:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Then install the remaining Python packages used by the scripts:

```powershell
pip install numpy pillow opencv-python transformers
```

## Run Order

Use either the CPU or GPU COLMAP script, not both for the same output folder.

### 1A. COLMAP Reconstruction On CPU

```powershell
python Script_1A_COLMAP_CPU.py
```

Default input: `Input/images`  
Default output: `Output/resized_images` and `Output/colmap`

Useful checks and options:

```powershell
python Script_1A_COLMAP_CPU.py --check_only
python Script_1A_COLMAP_CPU.py --input Input/images --output Output --camera_sharing per_folder
```

### 1B. COLMAP Reconstruction With CUDA COLMAP

```powershell
python Script_1B_COLMAP_GPU.py
```

This wrapper calls the CPU script with GPU feature extraction and matching enabled, after validating that a CUDA COLMAP executable is available.

It looks for `COLMAP_CUDA_PATH`, `--colmap_path`, or the local `Tools/colmap-cuda-manifest` install.

```powershell
python Script_1B_COLMAP_GPU.py --colmap_path C:\path\to\colmap.exe
```

### 2. Combined 2D Segmentation

```powershell
python Script_2_Segmentation_2D.py
```

This runs SegFormer for structural classes and GroundingDINO plus SAM for the unified sign class.

Default input: `Output/resized_images`  
Default output: `Output/2_segmentation_2D`

Default classes written into the mask PNGs:

- `unknown`
- `floor`
- `wall`
- `ceiling`
- `sign`
- `door`

The sign class includes signs, arrows, text-like sign content, and medical crosses.

Useful options:

```powershell
python Script_2_Segmentation_2D.py --dry_run
python Script_2_Segmentation_2D.py --device cuda --limit 25 --overwrite
python Script_2_Segmentation_2D.py --every_nth 2 --skip_previews
```

### 3A. Train Realistic Brush, Then Lift 2D Labels To 3D

```powershell
python Script_3A_Brush_2D_to_3D_Lift.py
```

This path trains a normal photorealistic Brush model from the COLMAP output, then projects the 2D masks onto the Gaussian splats using the COLMAP camera poses.

Each splat receives votes from the camera views in which it is visible. The final semantic class is assigned with confidence filtering, view-count filtering, and majority voting.

Main default outputs:

```text
Output/3A_regular_brush_dataset
Output/3A_regular_brush_exports
Output/3A_regular_brush_model/final_regular_brush_10000_steps.ply
Output/3A_semantic_lifted_model
```

Useful options:

```powershell
python Script_3A_Brush_2D_to_3D_Lift.py --dry_run
python Script_3A_Brush_2D_to_3D_Lift.py --steps 12000 --with_viewer
python Script_3A_Brush_2D_to_3D_Lift.py --skip_brush --final_ply Output/3A_regular_brush_model/final_regular_brush_10000_steps.ply
```

### 3B. Train Brush Directly On Segmented Images

```powershell
python Script_3B_Brush_Direct_on_Seg_Images.py
```

This path does not run segmentation again. It uses the masks from step 2 to recolor the resized images, builds a Brush dataset from those segmented images, and trains Brush on that dataset.

Main default outputs:

```text
Output/3B_segmented_images
Output/3B_segmented_brush_dataset
Output/3B_segmented_brush_exports
Output/3B_segmented_brush_model/final_segmented_brush_10000_steps.ply
```

Useful options:

```powershell
python Script_3B_Brush_Direct_on_Seg_Images.py --dry_run
python Script_3B_Brush_Direct_on_Seg_Images.py --images_only --overwrite_images
python Script_3B_Brush_Direct_on_Seg_Images.py --steps 8000 --with_viewer
```

## Which Stage 3 Option To Use

- Use 3A when you want a normal photorealistic 3DGS result plus separate semantic labels lifted onto it.
- Use 3B when you want the visible splat colors themselves to be semantic colors, because Brush is trained on the recolored segmentation images.
- Both paths require successful step 1 output and step 2 masks.

## Script Reference

- `Script_1A_COLMAP_CPU.py`: prepares/resizes input images and runs COLMAP on CPU.
- `Script_1B_COLMAP_GPU.py`: validates a CUDA COLMAP executable, then runs step 1 with GPU flags.
- `Script_2_Segmentation_2D.py`: creates combined SegFormer + GroundingDINO/SAM masks.
- `Script_3A_Brush_2D_to_3D_Lift.py`: trains regular Brush and lifts 2D semantic masks onto the 3D splats.
- `Script_3B_Brush_Direct_on_Seg_Images.py`: creates segmented training images and trains Brush directly on them.

## Reruns And Existing Outputs

- Step 1 refuses to reuse an existing COLMAP database. Use a new `--output` folder or remove the old COLMAP workspace before rerunning from scratch.
- If resized images already exist and the previous step 1 run failed later, rerun step 1 with `--reuse_prepared_images`.
- Step 2 reuses existing masks unless `--overwrite` is passed.
- Step 3A and 3B can rebuild Brush datasets with `--overwrite_dataset`. Dataset deletion is restricted to folders inside `Output`.
- Use `--dry_run` on steps 2, 3A, and 3B to verify resolved paths before writing files.

## Troubleshooting

- COLMAP executable not found: add COLMAP to `PATH`, set `COLMAP_PATH` or `COLMAP_CUDA_PATH`, pass `--colmap_path`, or place a local install under `Tools`.
- CUDA requested but unavailable: check `nvidia-smi`, confirm the NVIDIA driver, and install the PyTorch build that matches your CUDA setup.
- No images found: check that photos are under `Input/images` and use one of the supported image extensions.
- No masks found in step 3: run step 2 first and confirm `Output/2_segmentation_2D/masks` contains PNG files.
- Brush not found: add `brush.exe` to `PATH` or pass `--brush_path <path_to_brush.exe>`.

## Recommended Minimal Workflow

```powershell
python Script_1B_COLMAP_GPU.py
python Script_2_Segmentation_2D.py --device cuda
python Script_3A_Brush_2D_to_3D_Lift.py
```

If CUDA COLMAP is not available, replace the first command with:

```powershell
python Script_1A_COLMAP_CPU.py
```

If you prefer a directly color-segmented splat, replace the final command with:

```powershell
python Script_3B_Brush_Direct_on_Seg_Images.py
```
