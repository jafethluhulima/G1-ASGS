# Semantic segmentation and semantic labelling

import argparse
import csv
import json
import os
import re
import shutil
import struct
import sys
import time
from datetime import date
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

    print(f"Restarting S3 with project Python: {target_python}", flush=True)
    os.execv(
        str(target_python),
        [str(target_python), str(Path(__file__).resolve()), *sys.argv[1:]],
    )


restart_with_project_venv()

from PIL import Image
from PIL import ImageDraw


SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

DEFAULT_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
SH_C0 = 0.2820948


# compact semantic label scheme with RGB colors

LABELS = {
    0: {"name": "unknown", "color": [100, 100, 100]},
    1: {"name": "floor", "color": [55, 180, 75]},
    2: {"name": "wall", "color": [65, 105, 225]},
    3: {"name": "ceiling", "color": [245, 205, 65]},
    4: {"name": "door_window", "color": [220, 70, 70]},
}

LEGEND_PADDING = 14
LEGEND_ROW_HEIGHT = 30
LEGEND_SWATCH_SIZE = 20
LEGEND_WIDTH = 210

COLMAP_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
}


def natural_key(path):
    """Sort image paths so frame 2 comes before frame 10."""

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.as_posix())
    ]


def get_images(image_folder):
    images = [
        path
        for path in image_folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGES
    ]
    images.sort(key=natural_key)
    return images


def model_slug(model_name):
    slug = model_name.strip().replace("/", "_")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", slug)
    return slug.strip("._-") or "semantic_model"


def resolve_project_path(path_value):
    path = Path(path_value)

    if path.is_absolute():
        return path

    return SCRIPT_DIR / path


def dated_model_output_folder(model_name, stage_name):
    run_date = date.today().strftime("%Y%m%d")
    return SCRIPT_DIR / "Output" / f"{run_date}_{model_slug(model_name)}_{stage_name}"


def newest_ply_in_folder(folder):
    ply_files = list(folder.glob("*.ply"))

    if len(ply_files) == 0:
        return None

    ply_files.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    return ply_files[0]


def resolve_splat_ply(splat_ply):
    if splat_ply.exists():
        return splat_ply

    newest_ply = newest_ply_in_folder(splat_ply.parent)

    if newest_ply:
        print(f"Default PLY not found: {splat_ply}")
        print(f"Using newest PLY in the same folder: {newest_ply}")
        return newest_ply

    return splat_ply


def target_label_for_ade_label(label_name):
    """Reduce ADE20K labels to compact indoor classes."""

    normalized = label_name.lower().replace("-", " ").replace("_", " ")

    if "door" in normalized or "window" in normalized:
        return 4
    if "ceiling" in normalized:
        return 3
    if re.search(r"\bwall\b", normalized):
        return 2
    if re.search(r"\bfloor\b|\bflooring\b", normalized):
        return 1
    return 0


def load_semantic_dependencies():
    try:
        import numpy as np
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Semantic mask generation requires numpy, torch and transformers. "
            f"The current Python is: {sys.executable}\n"
            "Use the project virtual environment instead, for example:\n\n"
            f"  {SCRIPT_DIR / '.venv_v3' / 'Scripts' / 'python.exe'} "
            f"{SCRIPT_DIR / 'Script_S3_Segmentation.py'}\n\n"
            "Or create/install the environment with: pip install -r requirements.txt"
        ) from error

    return np, torch, AutoImageProcessor, AutoModelForSemanticSegmentation


def choose_device(torch, requested_device):
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU.")

    return requested_device


def load_cached_or_online(model_class, model_name):
    try:
        return model_class.from_pretrained(model_name, local_files_only=True)
    except OSError:
        return model_class.from_pretrained(model_name)


def write_label_scheme(output_folder, model_name):
    output_folder.mkdir(parents=True, exist_ok=True)

    scheme = {
        "model": model_name,
        "labels": {str(label_id): properties for label_id, properties in LABELS.items()},
        "notes": [
            "door_window combines ADE20K door and window classes.",
            "unknown contains all categories outside the compact structural scheme.",
        ],
    }

    with (output_folder / "label_scheme.json").open("w", encoding="utf-8") as file:
        json.dump(scheme, file, indent=2)

    legend = create_legend()
    legend.save(output_folder / "legend.png")


def create_legend():
    height = (2 * LEGEND_PADDING) + (len(LABELS) * LEGEND_ROW_HEIGHT) + 26
    legend = Image.new("RGB", (LEGEND_WIDTH, height), (255, 255, 255))
    draw = ImageDraw.Draw(legend)
    draw.text((LEGEND_PADDING, LEGEND_PADDING), "Semantic classes", fill=(20, 20, 20))

    top = LEGEND_PADDING + 28
    for label_id, properties in LABELS.items():
        color = tuple(properties["color"])
        row_top = top + (label_id * LEGEND_ROW_HEIGHT)
        draw.rectangle(
            (
                LEGEND_PADDING,
                row_top,
                LEGEND_PADDING + LEGEND_SWATCH_SIZE,
                row_top + LEGEND_SWATCH_SIZE,
            ),
            fill=color,
            outline=(40, 40, 40),
        )
        name = properties["name"].replace("_", " / ")
        draw.text(
            (LEGEND_PADDING + LEGEND_SWATCH_SIZE + 10, row_top + 3),
            name,
            fill=(20, 20, 20),
        )

    return legend


def colorize_mask(np, rgb_image, mask):
    colors = np.array(
        [LABELS[label_id]["color"] for label_id in sorted(LABELS)],
        dtype=np.uint8,
    )
    color_mask = colors[mask]
    preview = rgb_image.copy()
    selected = mask != 0
    preview[selected] = (
        (0.45 * rgb_image[selected]) + (0.55 * color_mask[selected])
    ).astype(np.uint8)
    preview_image = Image.fromarray(preview, mode="RGB")
    legend = create_legend()
    margin = 12
    canvas_width = preview_image.width + legend.width + (3 * margin)
    canvas_height = max(preview_image.height, legend.height) + (2 * margin)
    canvas = Image.new("RGB", (canvas_width, canvas_height), (245, 245, 245))
    canvas.paste(preview_image, (margin, margin))
    canvas.paste(legend, (preview_image.width + (2 * margin), margin))
    return canvas


# generate semantic masks from the images used for reconstruction

def generate_masks(
    image_folder,
    output_folder,
    model_name,
    requested_device,
    limit,
    every_nth,
    overwrite,
    write_previews,
):
    np, torch, processor_class, model_class = load_semantic_dependencies()

    images = get_images(image_folder)[::every_nth]
    if limit:
        images = images[:limit]

    if not images:
        raise FileNotFoundError(f"No supported images found in: {image_folder}")

    device = choose_device(torch, requested_device)
    print(f"Loading semantic segmentation model: {model_name}")
    print(f"Using device: {device}")

    processor = load_cached_or_online(processor_class, model_name)
    model = load_cached_or_online(model_class, model_name).to(device).eval()
    source_to_target = {
        int(source_id): target_label_for_ade_label(label_name)
        for source_id, label_name in model.config.id2label.items()
    }

    write_label_scheme(output_folder, model_name)

    masks_folder = output_folder / "masks"
    confidence_folder = output_folder / "confidence"
    previews_folder = output_folder / "previews"
    output_folders = [masks_folder, confidence_folder]
    if write_previews:
        output_folders.append(previews_folder)
    for folder in output_folders:
        folder.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    processed = 0
    skipped = 0

    with (output_folder / "mask_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(["image", "mask", "confidence", "preview"])

        for image_number, image_path in enumerate(images, start=1):
            relative_png = image_path.relative_to(image_folder).with_suffix(".png")
            mask_path = masks_folder / relative_png
            confidence_path = confidence_folder / relative_png
            preview_path = previews_folder / relative_png if write_previews else None

            writer.writerow(
                [
                    str(image_path.relative_to(image_folder)),
                    str(mask_path.relative_to(output_folder)),
                    str(confidence_path.relative_to(output_folder)),
                    str(preview_path.relative_to(output_folder)) if preview_path else "",
                ]
            )

            if mask_path.exists() and not overwrite:
                skipped += 1
                continue

            target_paths = [mask_path, confidence_path]
            if preview_path:
                target_paths.append(preview_path)
            for target_path in target_paths:
                target_path.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(image_path) as opened_image:
                image = opened_image.convert("RGB")

            inputs = processor(images=image, return_tensors="pt")
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

            with torch.inference_mode():
                logits = model(**inputs).logits
                logits = torch.nn.functional.interpolate(
                    logits,
                    size=(image.height, image.width),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                source_mask = logits.argmax(dim=0).cpu().numpy()
                source_confidence = (
                    logits.softmax(dim=0).amax(dim=0).mul(255).byte().cpu().numpy()
                )

            mask = np.zeros(source_mask.shape, dtype=np.uint8)
            for source_id, target_id in source_to_target.items():
                if target_id:
                    mask[source_mask == source_id] = target_id

            confidence = np.where(mask != 0, source_confidence, 0).astype(np.uint8)

            Image.fromarray(mask, mode="L").save(mask_path)
            Image.fromarray(confidence, mode="L").save(confidence_path)
            if preview_path:
                preview = colorize_mask(np, np.asarray(image), mask)
                preview.save(preview_path)
            processed += 1

            if image_number % 25 == 0 or image_number == len(images):
                print(f"Processed {image_number}/{len(images)} images")

    elapsed = time.perf_counter() - started
    print(
        f"Semantic masks complete: {processed} processed, {skipped} reused "
        f"in {elapsed:.1f} seconds."
    )
    print(f"Outputs: {output_folder}")


# read the COLMAP camera model and trained Brush splats

def read_exact(file, byte_count):
    data = file.read(byte_count)
    if len(data) != byte_count:
        raise EOFError("Unexpected end of COLMAP binary file.")
    return data


def read_colmap_cameras(np, path):
    cameras = {}
    with path.open("rb") as file:
        camera_count = struct.unpack("<Q", read_exact(file, 8))[0]
        for _ in range(camera_count):
            camera_id, model_id = struct.unpack("<ii", read_exact(file, 8))
            width, height = struct.unpack("<QQ", read_exact(file, 16))
            if model_id not in COLMAP_CAMERA_MODELS:
                raise ValueError(f"Unsupported COLMAP camera model ID: {model_id}")
            model_name, parameter_count = COLMAP_CAMERA_MODELS[model_id]
            params = np.array(
                struct.unpack(f"<{parameter_count}d", read_exact(file, 8 * parameter_count)),
                dtype=np.float64,
            )
            cameras[camera_id] = {
                "model": model_name,
                "width": int(width),
                "height": int(height),
                "params": params,
            }
    return cameras


def quaternion_to_rotation(np, qvec):
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [
                1 - (2 * qy * qy) - (2 * qz * qz),
                (2 * qx * qy) - (2 * qw * qz),
                (2 * qx * qz) + (2 * qw * qy),
            ],
            [
                (2 * qx * qy) + (2 * qw * qz),
                1 - (2 * qx * qx) - (2 * qz * qz),
                (2 * qy * qz) - (2 * qw * qx),
            ],
            [
                (2 * qx * qz) - (2 * qw * qy),
                (2 * qy * qz) + (2 * qw * qx),
                1 - (2 * qx * qx) - (2 * qy * qy),
            ],
        ],
        dtype=np.float32,
    )


def read_colmap_images(np, path):
    images = []
    with path.open("rb") as file:
        image_count = struct.unpack("<Q", read_exact(file, 8))[0]
        for _ in range(image_count):
            image_id = struct.unpack("<i", read_exact(file, 4))[0]
            qvec = struct.unpack("<4d", read_exact(file, 32))
            tvec = np.array(struct.unpack("<3d", read_exact(file, 24)), dtype=np.float32)
            camera_id = struct.unpack("<i", read_exact(file, 4))[0]
            name_bytes = bytearray()
            while True:
                byte = read_exact(file, 1)
                if byte == b"\0":
                    break
                name_bytes.extend(byte)
            point_count = struct.unpack("<Q", read_exact(file, 8))[0]
            file.seek(point_count * 24, 1)
            images.append(
                {
                    "id": image_id,
                    "name": name_bytes.decode("utf-8"),
                    "camera_id": camera_id,
                    "rotation": quaternion_to_rotation(np, qvec),
                    "translation": tvec,
                }
            )
    images.sort(key=lambda image: natural_key(Path(image["name"])))
    return images


def read_brush_ply(np, path):
    header_lines = []
    property_names = []
    vertex_count = None
    reading_vertex_properties = False

    with path.open("rb") as file:
        while True:
            raw_line = file.readline()
            if not raw_line:
                raise ValueError("PLY header does not contain end_header.")
            header_lines.append(raw_line)
            line = raw_line.decode("ascii").strip()
            if line == "format binary_little_endian 1.0":
                pass
            elif line.startswith("format "):
                raise ValueError("Only binary little-endian PLY files are supported.")
            elif line.startswith("element vertex "):
                vertex_count = int(line.split()[2])
                reading_vertex_properties = True
            elif line.startswith("element "):
                reading_vertex_properties = False
            elif reading_vertex_properties and line.startswith("property "):
                _, data_type, name = line.split()
                if data_type != "float":
                    raise ValueError(f"Unsupported vertex PLY property type: {data_type}")
                property_names.append(name)
            elif line == "end_header":
                body_offset = file.tell()
                break

    required_fields = {"x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"}
    if not required_fields.issubset(property_names):
        raise ValueError("PLY is missing required Brush Gaussian properties.")

    dtype = np.dtype([(name, "<f4") for name in property_names])
    vertices = np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=body_offset,
        shape=(vertex_count,),
    )
    positions = np.column_stack((vertices["x"], vertices["y"], vertices["z"]))
    return b"".join(header_lines), vertices, positions.astype(np.float32, copy=False)


def project_points(np, positions, camera, image, usable_splats):
    camera_points = positions @ image["rotation"].T + image["translation"]
    z = camera_points[:, 2]
    valid = usable_splats & (z > 1e-4)
    indexes = np.flatnonzero(valid)
    if indexes.size == 0:
        return indexes, indexes, indexes, indexes

    x = camera_points[indexes, 0] / z[indexes]
    y = camera_points[indexes, 1] / z[indexes]
    params = camera["params"]

    if camera["model"] == "SIMPLE_RADIAL":
        radial = 1.0 + params[3] * ((x * x) + (y * y))
        u = params[0] * x * radial + params[1]
        v = params[0] * y * radial + params[2]
    elif camera["model"] == "SIMPLE_PINHOLE":
        u = params[0] * x + params[1]
        v = params[0] * y + params[2]
    elif camera["model"] == "PINHOLE":
        u = params[0] * x + params[2]
        v = params[1] * y + params[3]
    elif camera["model"] == "RADIAL":
        radius_squared = (x * x) + (y * y)
        radial = 1.0 + params[3] * radius_squared + params[4] * radius_squared**2
        u = params[0] * x * radial + params[1]
        v = params[0] * y * radial + params[2]
    elif camera["model"] == "OPENCV":
        radius_squared = (x * x) + (y * y)
        radial = 1.0 + params[4] * radius_squared + params[5] * radius_squared**2
        distorted_x = x * radial + (2 * params[6] * x * y) + params[7] * (
            radius_squared + 2 * x * x
        )
        distorted_y = y * radial + params[6] * (
            radius_squared + 2 * y * y
        ) + (2 * params[7] * x * y)
        u = params[0] * distorted_x + params[2]
        v = params[1] * distorted_y + params[3]
    else:
        raise ValueError(f"Unsupported projection camera model: {camera['model']}")

    finite_inside = (
        np.isfinite(u)
        & np.isfinite(v)
        & np.isfinite(z[indexes])
        & (u >= -0.5)
        & (u <= camera["width"] - 0.5)
        & (v >= -0.5)
        & (v <= camera["height"] - 0.5)
    )
    indexes = indexes[finite_inside]
    u = np.rint(u[finite_inside]).astype(np.int32)
    v = np.rint(v[finite_inside]).astype(np.int32)
    depths = z[indexes]
    inside = (
        (u >= 0)
        & (u < camera["width"])
        & (v >= 0)
        & (v < camera["height"])
    )
    return indexes[inside], u[inside], v[inside], depths[inside]


def write_semantic_ply(np, input_path, header, vertices, labels, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rest_fields = [name for name in vertices.dtype.names if name.startswith("f_rest_")]
    colors = np.array(
        [LABELS[label_id]["color"] for label_id in sorted(LABELS)],
        dtype=np.float32,
    ) / 255.0
    dc_colors = (colors - 0.5) / SH_C0

    with output_path.open("wb") as output_file:
        output_file.write(header)
        chunk_size = 100000
        for start in range(0, len(vertices), chunk_size):
            end = min(start + chunk_size, len(vertices))
            chunk = vertices[start:end].copy()
            chunk_labels = labels[start:end]
            chunk["f_dc_0"] = dc_colors[chunk_labels, 0]
            chunk["f_dc_1"] = dc_colors[chunk_labels, 1]
            chunk["f_dc_2"] = dc_colors[chunk_labels, 2]
            for field in rest_fields:
                chunk[field] = 0
            chunk.tofile(output_file)


# lift the 2D semantic votes onto the 3D Gaussian splats

def lift_masks_to_splats(
    masks_output_folder,
    colmap_model_folder,
    splat_ply,
    semantic_output_folder,
    every_nth,
    min_opacity,
    min_pixel_confidence,
    min_views,
    min_vote_ratio,
):
    try:
        import numpy as np
    except ModuleNotFoundError as error:
        raise SystemExit("3D lifting requires numpy.") from error

    cameras_path = colmap_model_folder / "cameras.bin"
    images_path = colmap_model_folder / "images.bin"
    if not cameras_path.exists() or not images_path.exists():
        raise FileNotFoundError(f"COLMAP binary model not found: {colmap_model_folder}")
    if not splat_ply.exists():
        raise FileNotFoundError(f"Brush PLY not found: {splat_ply}")

    print("\nLoading COLMAP cameras and Brush splats for 3D lifting...")
    cameras = read_colmap_cameras(np, cameras_path)
    images = read_colmap_images(np, images_path)
    header, vertices, positions = read_brush_ply(np, splat_ply)
    opacity = 1.0 / (1.0 + np.exp(-np.clip(vertices["opacity"], -80, 80)))
    usable_splats = (opacity >= min_opacity) & np.isfinite(positions).all(axis=1)

    available_views = []
    for image in images[::every_nth]:
        relative_mask = Path(image["name"]).with_suffix(".png")
        mask_path = masks_output_folder / "masks" / relative_mask
        confidence_path = masks_output_folder / "confidence" / relative_mask
        if mask_path.exists() and confidence_path.exists():
            available_views.append((image, mask_path, confidence_path))

    if not available_views:
        raise FileNotFoundError(
            f"No mask files matched the COLMAP image names in: {masks_output_folder}"
        )

    print(f"Splats: {len(vertices):,}; opacity-filtered splats: {usable_splats.sum():,}")
    print(f"Registered images: {len(images):,}; lifting views: {len(available_views):,}")

    votes = np.zeros((len(vertices), len(LABELS)), dtype=np.uint16)
    confidence_sums = np.zeros((len(vertices), len(LABELS)), dtype=np.float32)
    threshold = int(round(min_pixel_confidence * 255))
    started = time.perf_counter()

    for view_number, (image, mask_path, confidence_path) in enumerate(
        available_views, start=1
    ):
        camera = cameras[image["camera_id"]]
        indexes, u, v, depths = project_points(
            np, positions, camera, image, usable_splats
        )
        if indexes.size:
            pixel_ids = v.astype(np.int64) * camera["width"] + u
            order = np.lexsort((depths, pixel_ids))
            sorted_pixels = pixel_ids[order]
            first_at_pixel = np.concatenate(
                ([True], sorted_pixels[1:] != sorted_pixels[:-1])
            )
            visible = order[first_at_pixel]
            visible_indexes = indexes[visible]
            visible_u = u[visible]
            visible_v = v[visible]

            mask = np.asarray(Image.open(mask_path), dtype=np.uint8)
            confidence = np.asarray(Image.open(confidence_path), dtype=np.uint8)
            view_labels = mask[visible_v, visible_u]
            view_confidence = confidence[visible_v, visible_u]
            accepted = (view_labels != 0) & (view_confidence >= threshold)
            accepted_indexes = visible_indexes[accepted]
            accepted_labels = view_labels[accepted]
            accepted_confidence = view_confidence[accepted].astype(np.float32) / 255.0
            np.add.at(votes, (accepted_indexes, accepted_labels), 1)
            np.add.at(
                confidence_sums,
                (accepted_indexes, accepted_labels),
                accepted_confidence,
            )

        if view_number % 25 == 0 or view_number == len(available_views):
            print(f"Lifted {view_number}/{len(available_views)} camera views")

    structural_votes = votes[:, 1:]
    best_labels = structural_votes.argmax(axis=1).astype(np.uint8) + 1
    row_indexes = np.arange(len(vertices))
    best_votes = votes[row_indexes, best_labels]
    total_votes = structural_votes.sum(axis=1)
    vote_ratio = np.divide(
        best_votes,
        total_votes,
        out=np.zeros(len(vertices), dtype=np.float32),
        where=total_votes > 0,
    )
    mean_confidence = np.divide(
        confidence_sums[row_indexes, best_labels],
        best_votes,
        out=np.zeros(len(vertices), dtype=np.float32),
        where=best_votes > 0,
    )
    reliable = (best_votes >= min_views) & (vote_ratio >= min_vote_ratio)
    labels = np.zeros(len(vertices), dtype=np.uint8)
    labels[reliable] = best_labels[reliable]

    semantic_output_folder.mkdir(parents=True, exist_ok=True)
    create_legend().save(semantic_output_folder / "legend.png")
    semantic_ply = semantic_output_folder / "semantic_splats_colored.ply"
    supersplat_ply = semantic_output_folder / "semantic_splats_supersplat_compatible.ply"
    write_semantic_ply(np, splat_ply, header, vertices, labels, semantic_ply)
    shutil.copy2(semantic_ply, supersplat_ply)

    np.savez_compressed(
        semantic_output_folder / "semantic_splat_labels.npz",
        labels=labels,
        votes=votes,
        mean_confidence=mean_confidence,
        vote_ratio=vote_ratio,
    )
    counts = np.bincount(labels, minlength=len(LABELS))
    summary = {
        "source_ply": str(splat_ply),
        "semantic_ply": str(semantic_ply),
        "supersplat_ply": str(supersplat_ply),
        "total_splats": int(len(vertices)),
        "views_used": int(len(available_views)),
        "parameters": {
            "every_nth_view": every_nth,
            "min_opacity": min_opacity,
            "min_pixel_confidence": min_pixel_confidence,
            "min_views": min_views,
            "min_vote_ratio": min_vote_ratio,
        },
        "class_counts": {
            LABELS[label_id]["name"]: int(counts[label_id]) for label_id in LABELS
        },
    }
    with (semantic_output_folder / "lifting_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    elapsed = time.perf_counter() - started
    print(f"\n3D semantic lifting completed in {elapsed:.1f} seconds.")
    for label_id, properties in LABELS.items():
        count = int(counts[label_id])
        print(f"  {properties['name']}: {count:,} splats ({count / len(vertices):.1%})")
    print(f"Semantic PLY: {semantic_ply}")
    print(f"SuperSplat-ready semantic PLY: {supersplat_ply}")


def preview_run(image_folder, output_folder, limit, every_nth):
    images = get_images(image_folder)[::every_nth]
    if limit:
        images = images[:limit]

    if not images:
        raise FileNotFoundError(f"No supported images found in: {image_folder}")

    print("V3 compact semantic segmentation")
    print(f"Input folder: {image_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Images selected: {len(images)}")
    print("Labels: unknown, floor, wall, ceiling, door_window")
    for path in images[:5]:
        print(f"  {path.relative_to(image_folder)}")


# main

def main():
    parser = argparse.ArgumentParser(
        description="Generate compact indoor semantic masks and lift them onto 3DGS splats."
    )
    parser.add_argument(
        "--image_folder",
        type=Path,
        default=Path("Output/brush_dataset/images"),
        help="Prepared images used by COLMAP and Brush.",
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        help="Folder for semantic masks, confidence maps, and previews. "
        "Default includes today's date and the selected model name.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face ADE20K semantic segmentation model.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N selected images; 0 processes all images.",
    )
    parser.add_argument(
        "--every_nth",
        type=int,
        default=1,
        help="Process every Nth image, useful for an initial quick inspection.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate mask files that already exist.",
    )
    parser.add_argument(
        "--skip_previews",
        action="store_true",
        help="Save masks and confidence maps without large overlay previews.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show the selected inputs and labels without loading a model.",
    )
    parser.add_argument(
        "--generate_masks",
        action="store_true",
        help="Generate 2D masks before lifting. Kept for explicit full-pipeline runs.",
    )
    parser.add_argument(
        "--lift_to_3d",
        action="store_true",
        help="Lift existing 2D masks onto a Brush Gaussian PLY model.",
    )
    parser.add_argument(
        "--masks_only",
        action="store_true",
        help="Only generate 2D masks and skip 3D semantic lifting.",
    )
    parser.add_argument(
        "--colmap_model_folder",
        type=Path,
        default=Path("Output/brush_dataset/sparse/0"),
        help="COLMAP sparse model containing cameras.bin and images.bin.",
    )
    parser.add_argument(
        "--splat_ply",
        type=Path,
        default=Path("Output/viewer_ready/gaussian_splats_3000_steps.ply"),
        help="Final Brush PLY to receive semantic colors.",
    )
    parser.add_argument(
        "--semantic_output_folder",
        type=Path,
        help="Output folder for semantic PLY and lifted label data. "
        "Default includes today's date and the selected model name.",
    )
    parser.add_argument(
        "--lift_every_nth",
        type=int,
        default=5,
        help="Use every Nth registered camera view during lifting.",
    )
    parser.add_argument(
        "--min_opacity",
        type=float,
        default=0.05,
        help="Ignore very transparent splats when estimating visible surfaces.",
    )
    parser.add_argument(
        "--min_pixel_confidence",
        type=float,
        default=0.50,
        help="Minimum 2D prediction confidence accepted as a 3D vote.",
    )
    parser.add_argument(
        "--min_views",
        type=int,
        default=3,
        help="Minimum agreeing camera votes required for a semantic splat label.",
    )
    parser.add_argument(
        "--min_vote_ratio",
        type=float,
        default=0.60,
        help="Minimum fraction of votes supporting a splat's winning class.",
    )
    args = parser.parse_args()

    if args.output_folder is None:
        args.output_folder = dated_model_output_folder(args.model, "semantic_2d")

    if args.semantic_output_folder is None:
        args.semantic_output_folder = dated_model_output_folder(args.model, "semantic_3d")

    args.image_folder = resolve_project_path(args.image_folder)
    args.output_folder = resolve_project_path(args.output_folder)
    args.colmap_model_folder = resolve_project_path(args.colmap_model_folder)
    args.splat_ply = resolve_project_path(args.splat_ply)
    args.semantic_output_folder = resolve_project_path(args.semantic_output_folder)

    args.splat_ply = resolve_splat_ply(args.splat_ply)

    if args.masks_only:
        run_masks = True
        run_lifting = False
    elif args.lift_to_3d and not args.generate_masks:
        run_masks = False
        run_lifting = True
    else:
        run_masks = True
        run_lifting = True

    if args.limit < 0:
        raise ValueError("--limit cannot be negative.")
    if args.every_nth < 1:
        raise ValueError("--every_nth must be at least 1.")
    if args.lift_every_nth < 1:
        raise ValueError("--lift_every_nth must be at least 1.")
    if args.min_views < 1:
        raise ValueError("--min_views must be at least 1.")
    for name in ("min_opacity", "min_pixel_confidence", "min_vote_ratio"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name} must be between 0 and 1.")
    if run_masks and not args.image_folder.exists():
        raise FileNotFoundError(f"Image folder not found: {args.image_folder}")

    if args.dry_run:
        preview_run(
            image_folder=args.image_folder,
            output_folder=args.output_folder,
            limit=args.limit,
            every_nth=args.every_nth,
        )
        return

    if run_masks:
        generate_masks(
            image_folder=args.image_folder,
            output_folder=args.output_folder,
            model_name=args.model,
            requested_device=args.device,
            limit=args.limit,
            every_nth=args.every_nth,
            overwrite=args.overwrite,
            write_previews=not args.skip_previews,
        )

    if run_lifting:
        lift_masks_to_splats(
            masks_output_folder=args.output_folder,
            colmap_model_folder=args.colmap_model_folder,
            splat_ply=args.splat_ply,
            semantic_output_folder=args.semantic_output_folder,
            every_nth=args.lift_every_nth,
            min_opacity=args.min_opacity,
            min_pixel_confidence=args.min_pixel_confidence,
            min_views=args.min_views,
            min_vote_ratio=args.min_vote_ratio,
        )


if __name__ == "__main__":
    main()
