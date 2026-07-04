#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Version v6.1.

Three-dimensional porous-matrix analysis from sequential two-dimensional slices.

Default expected input:
    structures/1.png
    structures/2.png
    ...
    structures/n.png

The script also accepts common image extensions such as jpg, jpeg, tif, tiff,
bmp, and webp, provided that the files are numerically ordered.

By default this script:
    - automatically crops the central image region, ignoring borders, titles,
      scale bars, axis labels, numbers, and text around the figure;
    - attempts to automatically segment the pore phase;
    - can remove an overlaid black grid before segmentation;
    - saves debug images to inspect cropping, grid cleaning, and the final mask.

For red/blue images, typically use:
    --mode blue
because blue = pore and red = matrix/solid.

If the mask is inverted, use --invert.
If automatic segmentation does not work, use --mode blue, red, nonblack,
bright, dark, green, or color.

Outputs:
    output/slice_properties.csv
    output/volume_properties.csv
    output/pore_size_distribution.csv
    output/permeability.csv
    output/morphohydraulic_indices.csv
    output/morphohydraulic_summary.csv
    output/morphohydraulic_signature.json
    output/permeability_methods.csv
    output/permeability_methods_summary.csv
    output/permeability_method_signature.json
    output/*.png with plots and visualizations

Model:
    - Binary 2D images stacked into a 3D volume.
    - Pore voxel = 1.
    - Solid voxel = 0.
    - Percolation by 3D connected components.
    - Relative permeability by a hydraulic resistor network on pore voxels.
    - Approximate local conductance: g_ij ~ r_ij^4 / L_ij.
      This is a discrete-network approximation of Poiseuille flow.
    - Morphohydraulic indices: directional connected porosity, connectivity
      efficiency, dead porosity, transportability, anisotropy, dominant flow
      direction, permeability yield, percolation gap, redundancy, flow entropy,
      effective participation, and hydraulic bottleneck.
    - Multiple permeability estimators: voxel resistor network, Kozeny--Carman,
      connected Kozeny--Carman, local-radius capillary model, median-throat
      capillary model, and bottleneck-throat capillary model.
    - It also accepts a binary .npz volume directly, avoiding PNG/colormap loss.
"""

import argparse
import re
import json
import math
import os
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
from PIL import Image

import scipy.ndimage as ndi
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import matplotlib.pyplot as plt


# ============================================================
# General utilities
# ============================================================

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def numeric_key(path: Path):
    """
    Sort files by the number in the filename.

    Ex:
        1.png, 2.png, 10.png
        f1.png, f2.png, f10.png

    If there is no number in the filename, place the file at the end.
    """
    nums = re.findall(r"\d+", path.stem)
    if nums:
        return (0, int(nums[-1]), path.name.lower())
    return (1, path.name.lower())


def image_number(path: Path):
    """
    Return the last number found in the filename.
    Ex: 1.png -> 1, f012.png -> 12.
    """
    nums = re.findall(r"\d+", path.stem)
    if not nums:
        return None
    return int(nums[-1])


def ensure_output_folder(output_folder):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder


def save_json(path, data):
    def convert(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=convert)


# ============================================================
# Input reading and segmentation
# ============================================================

def list_sequential_images(
    folder,
    prefix="",
    ext="",
    strict_sequence=True,
):
    """
    List sequential images inside `folder`.

    New pattern:
        structures/1.png
        structures/2.png
        structures/3.png
        ...

    Also works with other common image extensions.

    Parameters:
        prefix:
            Optional prefix.
            - "" read any numbered image: 1.png, 2.png, ...
            - "f" read f1.png, f2.png, ...
        ext:
            Optional extension without dot.
            - "" read png, jpg, jpeg, tif, tiff, bmp e webp
            - "png" read only PNG
        strict_sequence:
            If True, require the exact sequence 1, 2, 3, ..., n without gaps.
    """
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    if not folder.is_dir():
        raise NotADirectoryError(f"The path is not a folder: {folder}")

    if ext:
        allowed_exts = {"." + ext.lower().lstrip(".")}
    else:
        allowed_exts = SUPPORTED_IMAGE_EXTENSIONS

    candidates = []
    for file in folder.iterdir():
        if not file.is_file():
            continue
        if file.suffix.lower() not in allowed_exts:
            continue
        if prefix and not file.stem.startswith(prefix):
            continue
        if image_number(file) is None:
            continue
        candidates.append(file)

    files = sorted(candidates, key=numeric_key)

    if not files:
        if ext:
            pattern_desc = f"{prefix or ''}*.{ext}"
        else:
            pattern_desc = f"{prefix or ''}*"
        raise FileNotFoundError(
            f"No numbered image found in {folder} with approximate pattern: {pattern_desc}"
        )

    # Ensure that two files do not share the same number, for example 1.png and 1.jpg.
    numbers = [image_number(file) for file in files]
    duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
    if duplicates:
        dup_desc = ", ".join(str(n) for n in duplicates)
        raise ValueError(
            "More than one image has the same number in the sequence: "
            f"{dup_desc}. Remove duplicates such as 1.png and 1.jpg existing at the same time."
        )

    if strict_sequence:
        expected = list(range(1, len(files) + 1))
        if numbers != expected:
            found = ", ".join(str(n) for n in numbers[:30])
            if len(numbers) > 30:
                found += ", ..."
            raise ValueError(
                "Images must follow the exact sequence 1, 2, 3, ..., n. "
                f"Numbers found: {found}"
            )

    return files


def parse_roi(roi_text):
    """
    Read a manual ROI in the x1,y1,x2,y2 format.

    Coordinates follow the image convention:
        x = column, y = row
        x1,y1 = upper-left corner
        x2,y2 = lower-right corner, exclusive
    """
    if roi_text is None or str(roi_text).strip() == "":
        return None

    parts = re.split(r"[,;\s]+", str(roi_text).strip())
    parts = [p for p in parts if p != ""]
    if len(parts) != 4:
        raise ValueError(
            "Invalid manual ROI. Use the x1,y1,x2,y2 format. "
            "Example: --roi 120,80,980,840"
        )

    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in parts]
    except ValueError as exc:
        raise ValueError(
            "Invalid manual ROI. Use only numbers: x1,y1,x2,y2"
        ) from exc

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            "Invalid manual ROI: x2 must be greater than x1 and y2 must be greater than y1."
        )

    return (x1, y1, x2, y2)


def clamp_roi(roi, width, height):
    if roi is None:
        return None

    x1, y1, x2, y2 = roi
    x1 = max(0, min(width, int(x1)))
    x2 = max(0, min(width, int(x2)))
    y1 = max(0, min(height, int(y1)))
    y2 = max(0, min(height, int(y2)))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"ROI outside image bounds: {roi}, size={width}x{height}"
        )

    return (x1, y1, x2, y2)


def crop_array_by_roi(arr, roi):
    if roi is None:
        return arr
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = clamp_roi(roi, width=w, height=h)
    return arr[y1:y2, x1:x2].copy()


def estimate_background_rgb(rgb):
    """
    Estimate the external background color from the image corners.
    This helps remove white/gray borders and text outside the central figure.
    """
    h, w = rgb.shape[:2]
    patch_h = max(5, min(50, h // 20 if h >= 20 else h))
    patch_w = max(5, min(50, w // 20 if w >= 20 else w))

    corners = [
        rgb[:patch_h, :patch_w],
        rgb[:patch_h, -patch_w:],
        rgb[-patch_h:, :patch_w],
        rgb[-patch_h:, -patch_w:],
    ]
    samples = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
    return np.median(samples, axis=0)


def largest_component_bbox(mask):
    labels, n_labels = ndi.label(mask)
    if n_labels == 0:
        return None

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest_label = int(np.argmax(counts))
    if largest_label == 0 or counts[largest_label] == 0:
        return None

    ys, xs = np.where(labels == largest_label)
    if len(xs) == 0:
        return None

    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    return (x1, y1, x2, y2)


def auto_crop_roi(
    arr,
    background_threshold=18,
    padding=0,
    min_area_fraction=0.02,
    close_size=7,
    prefer_colored=True,
    color_saturation_threshold=60,
    color_value_threshold=50,
):
    """
    Automatically detect the useful image region.

    For images exported by software with:
        - red matrix,
        - blue pores,
        - axes/numbers/grid in black,
        - white external background,

    the best crop is the box containing only the colored pixels
    saturated pixels, i.e., red + blue. This automatically removes:
        - axis numbers,
        - white frames,
        - text,
        - external black marks.

    If there are not enough colored pixels, the method falls back to the
    older mode, based on the difference relative to the corner background.
    """
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("auto_crop_roi expects an RGB/RGBA image.")

    rgb = arr[..., :3].astype(np.int16)
    h, w = rgb.shape[:2]

    if h < 2 or w < 2:
        return (0, 0, w, h)

    # ------------------------------------------------------------------
    # 1) Preferred crop by colored region.
    #    This is the colorrect method for red/blue screenshots:
    #    red and blue are data; black/white/gray are grid/axes/background.
    # ------------------------------------------------------------------
    if prefer_colored:
        r = rgb[..., 0]
        g = rgb[..., 1]
        b = rgb[..., 2]

        maxc = np.maximum.reduce([r, g, b])
        minc = np.minimum.reduce([r, g, b])
        saturation = maxc - minc

        colored = (
            (saturation >= int(color_saturation_threshold))
            & (maxc >= int(color_value_threshold))
        )

        # Remove isolated colored speckles outside the figure, if present.
        if close_size and close_size > 1:
            colored_clean = ndi.binary_opening(colored, structure=np.ones((3, 3), dtype=bool))
            colored_clean = ndi.binary_closing(
                colored_clean,
                structure=np.ones((int(close_size), int(close_size)), dtype=bool),
            )
        else:
            colored_clean = colored

        # IMPORTANT:
        # Do not use the box of ALL colored pixels, because the colorbar
        # from the software is also colored and lies separately below the figure.
        # Instead, we use only the LARGEST connected colored component.
        # The closing operation above merges regions separated by thin
        # grid lines, but the colorbar remains separated by a white band.
        bbox = largest_component_bbox(colored_clean)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            if area >= min_area_fraction * h * w:
                pad = int(padding)
                return clamp_roi(
                    (x1 - pad, y1 - pad, x2 + pad, y2 + pad),
                    width=w,
                    height=h,
                )

        # Fallback within the colored method: if for some reason there is no
        # large component, use the box of colored pixels as before.
        ys, xs = np.where(colored_clean)
        if len(xs) > 0:
            x1 = int(xs.min())
            x2 = int(xs.max()) + 1
            y1 = int(ys.min())
            y2 = int(ys.max()) + 1

            area = (x2 - x1) * (y2 - y1)
            if area >= min_area_fraction * h * w:
                pad = int(padding)
                return clamp_roi(
                    (x1 - pad, y1 - pad, x2 + pad, y2 + pad),
                    width=w,
                    height=h,
                )

    # ------------------------------------------------------------------
    # 2) Fallback: crop by difference relative to the corner background.
    #    Useful for grayscale images or another palette.
    # ------------------------------------------------------------------
    bg = estimate_background_rgb(rgb)
    diff = np.max(np.abs(rgb - bg.reshape(1, 1, 3)), axis=2)
    content = diff > int(background_threshold)

    if close_size and close_size > 1:
        structure = np.ones((int(close_size), int(close_size)), dtype=bool)
        content_clean = ndi.binary_closing(content, structure=structure)
        content_clean = ndi.binary_fill_holes(content_clean)
    else:
        content_clean = content

    bbox = largest_component_bbox(content_clean)

    if bbox is None:
        return (0, 0, w, h)

    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    if area < min_area_fraction * h * w:
        return (0, 0, w, h)

    pad = int(padding)
    roi = (x1 - pad, y1 - pad, x2 + pad, y2 + pad)
    return clamp_roi(roi, width=w, height=h)


def shrink_roi_fraction(roi, trim_fraction, width, height):
    """
    Shrink the ROI by a fraction of its width/height.
    Useful when the central figure itself still has an internal frame with text.
    """
    roi = clamp_roi(roi, width=width, height=height)
    x1, y1, x2, y2 = roi
    if trim_fraction <= 0:
        return roi

    trim_fraction = float(trim_fraction)
    if trim_fraction >= 0.45:
        raise ValueError("--crop-trim-fraction must be smaller than 0.45")

    dx_trim = int(round((x2 - x1) * trim_fraction))
    dy_trim = int(round((y2 - y1) * trim_fraction))
    return clamp_roi(
        (x1 + dx_trim, y1 + dy_trim, x2 - dx_trim, y2 - dy_trim),
        width=width,
        height=height,
    )


def resolve_crop_roi_for_image(
    image_path,
    crop="auto",
    roi_text="",
    crop_padding=0,
    crop_bg_threshold=18,
    crop_trim_fraction=0.0,
):
    """
    Resolve the ROI that will be used to crop the image.
    """
    arr = np.asarray(Image.open(image_path).convert("RGBA"))
    h, w = arr.shape[:2]

    if crop == "none":
        roi = (0, 0, w, h)
    elif crop == "manual":
        roi = parse_roi(roi_text)
        if roi is None:
            raise ValueError("Use --roi x1,y1,x2,y2 when --crop manual is selected.")
        roi = clamp_roi(roi, width=w, height=h)
    elif crop == "auto":
        roi = auto_crop_roi(
            arr,
            background_threshold=crop_bg_threshold,
            padding=crop_padding,
        )
    else:
        raise ValueError("--crop must be auto, manual, or none")

    roi = shrink_roi_fraction(roi, crop_trim_fraction, width=w, height=h)
    return roi



def detect_grid_overlay_mask(
    arr,
    dark_threshold=25,
    line_fraction=0.35,
    dilate=1,
):
    """
    Detect dark grid/axis lines overlaid on the figure by the software.

    The detection searches for horizontal/vertical lines with a large fraction of dark pixels.
    This avoids treating the black grid as a real solid phase.
    """
    if arr.ndim != 3 or arr.shape[2] < 3:
        return np.zeros(arr.shape[:2], dtype=bool)

    rgb = arr[..., :3].astype(np.int16)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    dark = gray <= float(dark_threshold)

    h, w = dark.shape
    if h < 5 or w < 5:
        return np.zeros((h, w), dtype=bool)

    row_fraction = dark.mean(axis=1)
    col_fraction = dark.mean(axis=0)

    row_lines = row_fraction >= float(line_fraction)
    col_lines = col_fraction >= float(line_fraction)

    # Keep only the dark pixels on these lines; do not erase local dark spots.
    grid = dark & (row_lines[:, None] | col_lines[None, :])

    if dilate and int(dilate) > 0:
        size = 2 * int(dilate) + 1
        grid = ndi.binary_dilation(grid, structure=np.ones((size, size), dtype=bool))

    return grid.astype(bool)


def remove_grid_overlay(
    arr,
    dark_threshold=25,
    line_fraction=0.35,
    dilate=1,
):
    """
    Remove the black grid by replacing its pixels with the nearest non-grid color.
    This is a visual/geometric colorrection for screenshots exported with an overlaid grid.
    """
    grid = detect_grid_overlay_mask(
        arr,
        dark_threshold=dark_threshold,
        line_fraction=line_fraction,
        dilate=dilate,
    )

    if not grid.any():
        return arr, grid

    processed = arr.copy()
    # distance_transform_edt with return_indices returns, for each True pixel in the mask,
    # the index of the nearest False pixel.
    indices = ndi.distance_transform_edt(
        grid,
        return_distances=False,
        return_indices=True,
    )
    processed[grid] = arr[tuple(ind[grid] for ind in indices)]
    return processed, grid

def segment_array(
    arr,
    mode="auto",
    green_min=80,
    green_margin=40,
    blue_min=80,
    blue_margin=40,
    red_min=80,
    red_margin=40,
    threshold=128,
    black_threshold=12,
    white_threshold=245,
    color_min=25,
    invert=False,
):
    """
    Segment an already cropped RGBA/RGB array.

    Returns:
        True  = pore
        False = solid

    Useful modes:
        auto      : tries green, blue, red, then color/saturation, then Otsu.
        blue      : blue pores. Use for images with blue pores and red matrix.
        red       : red pores. Use if the pore phase is red.
        green     : green pores.
        color     : colored/saturated pixels are pore.
        nonblack  : everything that is not black is pore. Good for images with black background and bright/colored pores.
        nonwhite  : everything that is not white is pore. Good for white backgrounds.
        bright    : bright pixels are pore.
        dark      : dark pixels are pore.
        alpha     : pixels visible through the alpha channel are pore.
    """
    if arr.ndim != 3:
        raise ValueError("The image must be RGB/RGBA.")

    if arr.shape[2] == 3:
        alpha = np.full(arr.shape[:2], 255, dtype=np.uint8)
        rgb_arr = arr
    else:
        alpha = arr[..., 3]
        rgb_arr = arr[..., :3]

    rgb = rgb_arr.astype(np.int16)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    a = alpha.astype(np.int16)

    gray = (0.299 * r + 0.587 * g + 0.114 * b)
    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    saturation = maxc - minc

    green_mask = (g >= green_min) & (g > r + green_margin) & (g > b + green_margin)
    blue_mask = (b >= blue_min) & (b > r + blue_margin) & (b > g + blue_margin)
    red_mask = (r >= red_min) & (r > g + red_margin) & (r > b + red_margin)
    color_mask = (saturation >= color_min) & (gray > black_threshold) & (gray < white_threshold)

    if mode == "green":
        mask = green_mask

    elif mode == "blue":
        mask = blue_mask

    elif mode == "red":
        mask = red_mask

    elif mode == "color":
        mask = color_mask

    elif mode == "nonblack":
        mask = gray > black_threshold

    elif mode == "nonwhite":
        mask = gray < white_threshold

    elif mode == "bright":
        mask = gray >= threshold

    elif mode == "dark":
        mask = gray <= threshold

    elif mode == "alpha":
        mask = a > threshold

    elif mode == "auto":
        green_fraction = float(green_mask.mean())
        blue_fraction = float(blue_mask.mean())
        red_fraction = float(red_mask.mean())
        color_fraction = float(color_mask.mean())

        # In colored binary images, choose a valid dominant color first.
        # For the typical case: blue = pore, red = matrix, use --mode blue to avoid ambiguity.
        if 0.0005 <= green_fraction <= 0.95:
            mask = green_mask
        elif 0.0005 <= blue_fraction <= 0.95 and red_fraction > 0.0005:
            mask = blue_mask
        elif 0.0005 <= red_fraction <= 0.95 and blue_fraction <= 0.0005:
            mask = red_mask
        elif 0.0005 <= color_fraction <= 0.95:
            mask = color_mask
        else:
            gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
            t = otsu_threshold(gray_u8)
            bright_mask = gray >= t
            dark_mask = gray < t

            bright_fraction = float(bright_mask.mean())
            dark_fraction = float(dark_mask.mean())

            # By default, choose the non-empty minority class.
            # This prevents the solid background from becoming entirely porous.
            candidates = []
            if 0.0005 <= bright_fraction <= 0.9995:
                candidates.append((bright_fraction, bright_mask))
            if 0.0005 <= dark_fraction <= 0.9995:
                candidates.append((dark_fraction, dark_mask))

            if candidates:
                candidates.sort(key=lambda item: item[0])
                mask = candidates[0][1]
            else:
                mask = bright_mask

    else:
        raise ValueError(f"Unknown segmentation mode: {mode}")

    # Ignore transparent pixels, if any.
    mask = mask & (a > 0)

    if invert:
        mask = ~mask
        mask = mask & (a > 0)

    return mask.astype(bool)


def segment_image(
    image_path,
    mode="auto",
    green_min=80,
    green_margin=40,
    blue_min=80,
    blue_margin=40,
    red_min=80,
    red_margin=40,
    threshold=128,
    black_threshold=12,
    white_threshold=245,
    color_min=25,
    invert=False,
    crop_roi=None,
    return_cropped=False,
    remove_grid=True,
    grid_dark_threshold=25,
    grid_line_fraction=0.35,
    grid_dilate=1,
):
    """
    Open, crop, and segment an image.
    """
    img = Image.open(image_path).convert("RGBA")
    arr = np.asarray(img)

    if crop_roi is not None:
        arr = crop_array_by_roi(arr, crop_roi)

    arr_for_segmentation = arr
    if remove_grid:
        arr_for_segmentation, grid_mask = remove_grid_overlay(
            arr_for_segmentation,
            dark_threshold=grid_dark_threshold,
            line_fraction=grid_line_fraction,
            dilate=grid_dilate,
        )

    mask = segment_array(
        arr_for_segmentation,
        mode=mode,
        green_min=green_min,
        green_margin=green_margin,
        blue_min=blue_min,
        blue_margin=blue_margin,
        red_min=red_min,
        red_margin=red_margin,
        threshold=threshold,
        black_threshold=black_threshold,
        white_threshold=white_threshold,
        color_min=color_min,
        invert=invert,
    )

    if return_cropped:
        return mask.astype(bool), arr

    return mask.astype(bool)

def otsu_threshold(gray):
    """
    Simple Otsu implementation to avoid an additional dependency.
    """
    hist, bin_edges = np.histogram(gray.ravel(), bins=256, range=(0, 255))
    hist = hist.astype(float)

    total = gray.size
    sum_total = np.dot(np.arange(256), hist)

    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    threshold = 128

    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue

        w_f = total - w_b
        if w_f == 0:
            break

        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f

        var_between = w_b * w_f * (m_b - m_f) ** 2

        if var_between > max_var:
            max_var = var_between
            threshold = t

    return threshold


def load_volume(
    folder,
    prefix="",
    ext="",
    strict_sequence=True,
    mode="auto",
    downsample=1,
    green_min=80,
    green_margin=40,
    blue_min=80,
    blue_margin=40,
    red_min=80,
    red_margin=40,
    threshold=128,
    black_threshold=12,
    white_threshold=245,
    color_min=25,
    invert=False,
    crop="auto",
    roi="",
    crop_padding=0,
    crop_bg_threshold=18,
    crop_trim_fraction=0.0,
    crop_each_slice=False,
    remove_grid=True,
    grid_dark_threshold=25,
    grid_line_fraction=0.35,
    grid_dilate=1,
    debug_output_folder=None,
    debug_first_n=3,
):
    files = list_sequential_images(
        folder,
        prefix=prefix,
        ext=ext,
        strict_sequence=strict_sequence,
    )

    fixed_crop_roi = None
    if crop_each_slice:
        fixed_crop_roi = None
    else:
        fixed_crop_roi = resolve_crop_roi_for_image(
            files[0],
            crop=crop,
            roi_text=roi,
            crop_padding=crop_padding,
            crop_bg_threshold=crop_bg_threshold,
            crop_trim_fraction=crop_trim_fraction,
        )

    masks = []
    base_shape = None
    used_rois = []

    if debug_output_folder is not None:
        debug_output_folder = Path(debug_output_folder)
        debug_output_folder.mkdir(parents=True, exist_ok=True)

    for idx, file in enumerate(files, start=1):
        if crop_each_slice:
            crop_roi = resolve_crop_roi_for_image(
                file,
                crop=crop,
                roi_text=roi,
                crop_padding=crop_padding,
                crop_bg_threshold=crop_bg_threshold,
                crop_trim_fraction=crop_trim_fraction,
            )
        else:
            crop_roi = fixed_crop_roi

        mask, cropped_arr = segment_image(
            file,
            mode=mode,
            green_min=green_min,
            green_margin=green_margin,
            blue_min=blue_min,
            blue_margin=blue_margin,
            red_min=red_min,
            red_margin=red_margin,
            threshold=threshold,
            black_threshold=black_threshold,
            white_threshold=white_threshold,
            color_min=color_min,
            invert=invert,
            crop_roi=crop_roi,
            return_cropped=True,
            remove_grid=remove_grid,
            grid_dark_threshold=grid_dark_threshold,
            grid_line_fraction=grid_line_fraction,
            grid_dilate=grid_dilate,
        )

        if downsample > 1:
            mask = mask[::downsample, ::downsample]

        if base_shape is None:
            base_shape = mask.shape
        elif mask.shape != base_shape:
            raise ValueError(
                f"Image {file} generated a mask with shape {mask.shape}, but expected {base_shape}. "
                "All slices must have the same size. "
                "Use --crop manual --roi x1,y1,x2,y2 or disable --crop-each-slice."
            )

        masks.append(mask)
        used_rois.append(tuple(int(v) for v in crop_roi) if crop_roi is not None else None)

        if debug_output_folder is not None and idx <= int(debug_first_n):
            # debug_crop = raw crop of the central figure, still with the software grid, if present.
            Image.fromarray(cropped_arr[..., :3]).save(debug_output_folder / f"debug_crop_slice_{idx}.png")

            # debug_clean = crop used for segmentation after attempting to remove black grid/axis lines.
            if remove_grid:
                clean_arr, grid_mask = remove_grid_overlay(
                    cropped_arr,
                    dark_threshold=grid_dark_threshold,
                    line_fraction=grid_line_fraction,
                    dilate=grid_dilate,
                )
            else:
                clean_arr = cropped_arr
                grid_mask = np.zeros(cropped_arr.shape[:2], dtype=bool)

            Image.fromarray(clean_arr[..., :3]).save(debug_output_folder / f"debug_clean_slice_{idx}.png")
            Image.fromarray((grid_mask.astype(np.uint8) * 255)).save(debug_output_folder / f"debug_grid_slice_{idx}.png")

            # debug_mask = final binary mask. White = pore; black = solid.
            Image.fromarray((mask.astype(np.uint8) * 255)).save(debug_output_folder / f"debug_mask_slice_{idx}.png")

    # Volume axes:
    # z, y, x
    volume = np.stack(masks, axis=0).astype(bool)


    return volume, files, used_rois


def load_npz_volume(
    npz_file,
    key="pores",
    order="xyz",
    solid_is_true=False,
    downsample=1,
):
    """
    Read a binary volume directly from a .npz file.

    Internal convention of this analyzer:
        volume[z, y, x] = True  -> pore
        volume[z, y, x] = False -> solid

    Accepted conventions in --npz-order:
        xyz -> file is stored as pores[x, y, z], same as the snake generator
        zyx -> file is already stored as volume[z, y, x]

    Use --npz-solid-is-true if True means solid in the file.
    """
    npz_path = Path(npz_file)
    if not npz_path.exists():
        raise FileNotFoundError(f" .npz file not found: {npz_file}")

    data = np.load(npz_path)
    if key not in data.files:
        raise KeyError(
            f"Key '{key}' not found in {npz_file}. "
            f"Available keys: {list(data.files)}"
        )

    arr = np.asarray(data[key])
    if arr.ndim != 3:
        raise ValueError(f"The key '{key}' must be 3D; received shape: {arr.shape}")

    volume = arr.astype(bool)
    if solid_is_true:
        volume = ~volume

    order = str(order).lower().strip()
    if order == "xyz":
        # file[x,y,z] -> volume[z,y,x]
        volume = np.transpose(volume, (2, 1, 0))
    elif order == "zyx":
        volume = volume.copy()
    else:
        raise ValueError("--npz-order must be 'xyz' or 'zyx'")

    if downsample and int(downsample) > 1:
        ds = int(downsample)
        # Keep all z slices and downsample only y/x, as in the image workflow.
        volume = volume[:, ::ds, ::ds]

    files = [Path(f"npz_slice_{i + 1:04d}") for i in range(volume.shape[0])]
    used_rois = [None for _ in range(volume.shape[0])]
    return volume.astype(bool), files, used_rois


# ============================================================
# 2D slice analysis
# ============================================================

def component_stats_2d(mask, dx=1.0, dy=1.0):
    """
    Compute 2D statistics for one slice.
    """
    structure_2d = ndi.generate_binary_structure(2, 2)  # 8-connectivity
    labels, n_labels = ndi.label(mask, structure=structure_2d)

    total_pixels = mask.size
    pore_pixels = int(mask.sum())
    porosity = pore_pixels / total_pixels if total_pixels > 0 else 0.0

    if n_labels == 0:
        return {
            "porosity_2d": porosity,
            "n_pores_2d": 0,
            "largest_pore_area_px": 0,
            "largest_pore_fraction": 0.0,
            "mean_area_px": 0.0,
            "median_area_px": 0.0,
            "mean_radius_eq": 0.0,
            "median_radius_eq": 0.0,
            "percolates_x_2d": False,
            "percolates_y_2d": False,
        }, pd.DataFrame()

    counts = np.bincount(labels.ravel())
    counts[0] = 0

    areas_px = counts[counts > 0]
    areas_phys = areas_px * dx * dy
    radii_eq = np.sqrt(areas_phys / np.pi)

    largest = int(areas_px.max())
    largest_fraction = largest / pore_pixels if pore_pixels > 0 else 0.0

    left_labels = set(np.unique(labels[:, 0])) - {0}
    right_labels = set(np.unique(labels[:, -1])) - {0}
    top_labels = set(np.unique(labels[0, :])) - {0}
    bottom_labels = set(np.unique(labels[-1, :])) - {0}

    percolates_x = len(left_labels.intersection(right_labels)) > 0
    percolates_y = len(top_labels.intersection(bottom_labels)) > 0

    props = {
        "porosity_2d": porosity,
        "n_pores_2d": int(n_labels),
        "largest_pore_area_px": largest,
        "largest_pore_fraction": largest_fraction,
        "mean_area_px": float(np.mean(areas_px)),
        "median_area_px": float(np.median(areas_px)),
        "mean_radius_eq": float(np.mean(radii_eq)),
        "median_radius_eq": float(np.median(radii_eq)),
        "percolates_x_2d": bool(percolates_x),
        "percolates_y_2d": bool(percolates_y),
    }

    pore_df = pd.DataFrame({
        "area_px": areas_px,
        "area": areas_phys,
        "radius_eq": radii_eq,
        "diameter_eq": 2.0 * radii_eq,
    })

    return props, pore_df


def analyze_slices(volume, files, output_folder, dx=1.0, dy=1.0):
    rows = []
    all_pores = []

    for z in range(volume.shape[0]):
        props, pore_df = component_stats_2d(volume[z], dx=dx, dy=dy)
        props["slice_index"] = z + 1
        props["filename"] = files[z].name
        rows.append(props)

        if not pore_df.empty:
            pore_df["slice_index"] = z + 1
            pore_df["filename"] = files[z].name
            all_pores.append(pore_df)

    slice_df = pd.DataFrame(rows)
    slice_df = slice_df[
        [
            "slice_index",
            "filename",
            "porosity_2d",
            "n_pores_2d",
            "largest_pore_area_px",
            "largest_pore_fraction",
            "mean_area_px",
            "median_area_px",
            "mean_radius_eq",
            "median_radius_eq",
            "percolates_x_2d",
            "percolates_y_2d",
        ]
    ]

    slice_df.to_csv(output_folder / "slice_properties.csv", index=False)

    if all_pores:
        pore_df_all = pd.concat(all_pores, ignore_index=True)
    else:
        pore_df_all = pd.DataFrame()

    pore_df_all.to_csv(output_folder / "pore_size_distribution.csv", index=False)

    return slice_df, pore_df_all


# ============================================================
# 3D analysis: porosity, clusters, and percolation
# ============================================================

def label_volume(volume, connectivity=1):
    """
    connectivity:
        1 = 6-connectivity
        2 = 18-connectivity
        3 = 26-connectivity
    """
    structure = ndi.generate_binary_structure(3, connectivity)
    labels, n_labels = ndi.label(volume, structure=structure)
    return labels, n_labels


def percolation_3d(labels):
    """
    Check percolation along the x, y, and z axes.

    Volume has axes:
        z, y, x
    """
    if labels.max() == 0:
        return {
            "percolates_x_3d": False,
            "percolates_y_3d": False,
            "percolates_z_3d": False,
            "percolatesting_labels_x": [],
            "percolatesting_labels_y": [],
            "percolatesting_labels_z": [],
        }

    left = set(np.unique(labels[:, :, 0])) - {0}
    right = set(np.unique(labels[:, :, -1])) - {0}

    top = set(np.unique(labels[:, 0, :])) - {0}
    bottom = set(np.unique(labels[:, -1, :])) - {0}

    front = set(np.unique(labels[0, :, :])) - {0}
    back = set(np.unique(labels[-1, :, :])) - {0}

    px = sorted(left.intersection(right))
    py = sorted(top.intersection(bottom))
    pz = sorted(front.intersection(back))

    return {
        "percolates_x_3d": len(px) > 0,
        "percolates_y_3d": len(py) > 0,
        "percolates_z_3d": len(pz) > 0,
        "percolatesting_labels_x": [int(v) for v in px],
        "percolatesting_labels_y": [int(v) for v in py],
        "percolatesting_labels_z": [int(v) for v in pz],
    }


def volume_cluster_stats(volume, labels, n_labels, dx=1.0, dy=1.0, dz=1.0):
    total_voxels = volume.size
    pore_voxels = int(volume.sum())

    porosity = pore_voxels / total_voxels if total_voxels > 0 else 0.0

    if n_labels == 0:
        return {
            "porosity_3d": porosity,
            "n_clusters_3d": 0,
            "largest_cluster_voxels": 0,
            "largest_cluster_fraction_of_pores": 0.0,
            "mean_cluster_volume": 0.0,
            "median_cluster_volume": 0.0,
            "mean_equiv_radius_3d": 0.0,
            "median_equiv_radius_3d": 0.0,
        }

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    volumes_vox = counts[counts > 0]

    voxel_volume = dx * dy * dz
    volumes_phys = volumes_vox * voxel_volume

    # Equivalent sphere radius in 3D:
    # V = 4/3 pi r^3
    radii_eq = ((3.0 * volumes_phys) / (4.0 * np.pi)) ** (1.0 / 3.0)

    largest = int(volumes_vox.max())
    largest_fraction = largest / pore_voxels if pore_voxels > 0 else 0.0

    return {
        "porosity_3d": float(porosity),
        "n_clusters_3d": int(n_labels),
        "largest_cluster_voxels": largest,
        "largest_cluster_fraction_of_pores": float(largest_fraction),
        "mean_cluster_volume": float(np.mean(volumes_phys)),
        "median_cluster_volume": float(np.median(volumes_phys)),
        "mean_equiv_radius_3d": float(np.mean(radii_eq)),
        "median_equiv_radius_3d": float(np.median(radii_eq)),
    }


# ============================================================
# Euclidean distance and throat/local-radius distribution
# ============================================================

def distance_radius_distribution(volume, dx=1.0, dy=1.0, dz=1.0):
    """
    Compute the distance from each pore voxel to the nearest solid.
    This approximates a local pore radius.

    spacing in scipy follows the axis order:
        z, y, x
    """
    if volume.sum() == 0:
        return np.array([]), np.zeros_like(volume, dtype=float)

    dist = ndi.distance_transform_edt(volume, sampling=(dz, dy, dx))
    radii = dist[volume]
    return radii, dist


def fit_power_law_tail(radii, min_percentile=50):
    """
    Simple fit to test fractal/Apollonian-like behavior.

    Uses the cumulative distribution:
        N(r >= R) ~ R^(-D)

    Returns an approximate D from the log-log slope.

    Note:
        this is only an approximate statistical diagnostic,
        not proof of fractality.
    """
    radii = np.asarray(radii)
    radii = radii[np.isfinite(radii)]
    radii = radii[radii > 0]

    if len(radii) < 20:
        return {
            "power_law_D_estimate": np.nan,
            "power_law_r2": np.nan,
            "power_law_n_points": int(len(radii)),
        }

    rmin = np.percentile(radii, min_percentile)
    tail = np.sort(radii[radii >= rmin])

    if len(tail) < 20:
        return {
            "power_law_D_estimate": np.nan,
            "power_law_r2": np.nan,
            "power_law_n_points": int(len(tail)),
        }

    unique_r = np.unique(tail)
    if len(unique_r) < 5:
        return {
            "power_law_D_estimate": np.nan,
            "power_law_r2": np.nan,
            "power_law_n_points": int(len(unique_r)),
        }

    # N(r >= R)
    R = unique_r
    N = np.array([np.sum(tail >= rr) for rr in R], dtype=float)

    x = np.log(R)
    y = np.log(N)

    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Como N ~ R^(-D), slope = -D
    D_est = -slope

    return {
        "power_law_D_estimate": float(D_est),
        "power_law_r2": float(r2),
        "power_law_n_points": int(len(R)),
    }


# ============================================================
# Approximate tortuosity by BFS in the pore cluster
# ============================================================

def shortest_path_length_binary(volume, axis):
    """
    Compute the shortest path in pore voxels between opposite faces.
    Use a simple BFS.

    axis:
        0 = z
        1 = y
        2 = x

    Returns:
        shortest path as number of steps, or None.
    """
    if volume.sum() == 0:
        return None

    shape = volume.shape
    visited = np.zeros(shape, dtype=bool)
    dist = np.full(shape, -1, dtype=np.int32)

    q = deque()

    if axis == 0:
        starts = np.argwhere(volume[0, :, :])
        for y, x in starts:
            idx = (0, y, x)
            visited[idx] = True
            dist[idx] = 0
            q.append(idx)

        def is_target(idx):
            return idx[0] == shape[0] - 1

    elif axis == 1:
        starts = np.argwhere(volume[:, 0, :])
        for z, x in starts:
            idx = (z, 0, x)
            visited[idx] = True
            dist[idx] = 0
            q.append(idx)

        def is_target(idx):
            return idx[1] == shape[1] - 1

    elif axis == 2:
        starts = np.argwhere(volume[:, :, 0])
        for z, y in starts:
            idx = (z, y, 0)
            visited[idx] = True
            dist[idx] = 0
            q.append(idx)

        def is_target(idx):
            return idx[2] == shape[2] - 1

    else:
        raise ValueError("axis must be 0, 1, or 2")

    neighbors = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]

    while q:
        z, y, x = q.popleft()
        if is_target((z, y, x)):
            return int(dist[z, y, x])

        for dz_, dy_, dx_ in neighbors:
            nz, ny, nx = z + dz_, y + dy_, x + dx_

            if (
                0 <= nz < shape[0]
                and 0 <= ny < shape[1]
                and 0 <= nx < shape[2]
                and volume[nz, ny, nx]
                and not visited[nz, ny, nx]
            ):
                visited[nz, ny, nx] = True
                dist[nz, ny, nx] = dist[z, y, x] + 1
                q.append((nz, ny, nx))

    return None


def tortuosity_estimates(volume, dx=1.0, dy=1.0, dz=1.0):
    """
    Approximate tortuosity:
        tau = shortest pore path / straight length
    """
    results = {}

    axes = {
        "z": (0, dz, volume.shape[0] - 1),
        "y": (1, dy, volume.shape[1] - 1),
        "x": (2, dx, volume.shape[2] - 1),
    }

    for name, (axis, spacing, straight_steps) in axes.items():
        path_steps = shortest_path_length_binary(volume, axis)
        if path_steps is None or straight_steps <= 0:
            tau = np.nan
            path_length = np.nan
        else:
            path_length = path_steps * spacing
            straight_length = straight_steps * spacing
            tau = path_length / straight_length if straight_length > 0 else np.nan

        results[f"shortest_path_{name}"] = (
            None if path_steps is None else int(path_steps)
        )
        results[f"tortuosity_{name}"] = float(tau) if np.isfinite(tau) else np.nan

    return results


# ============================================================
# Permeability by hydraulic resistor network
# ============================================================

def extract_percolatesting_subvolume(volume, labels, direction):
    """
    Keep only clusters that percolateste in the desired direction.
    This reduces the linear system and avoids spending time on isolated pores.

    direction:
        "x", "y" ou "z"
    """
    perc = percolation_3d(labels)

    if direction == "x":
        labs = perc["percolatesting_labels_x"]
    elif direction == "y":
        labs = perc["percolatesting_labels_y"]
    elif direction == "z":
        labs = perc["percolatesting_labels_z"]
    else:
        raise ValueError("direction must be x, y, or z")

    if not labs:
        return np.zeros_like(volume, dtype=bool)

    return np.isin(labels, labs)



def _axis_for_direction(direction):
    axis_map = {"z": 0, "y": 1, "x": 2}
    if direction not in axis_map:
        raise ValueError("direction must be x, y, or z")
    return axis_map[direction]


def _positive_neighbor_specs(dx=1.0, dy=1.0, dz=1.0):
    """
    Unique positive neighbors to count edges without duplication.
    Volume in z,y,x order.
    """
    return [
        ((0, 0, 1), dx),
        ((0, 1, 0), dy),
        ((1, 0, 0), dz),
    ]


def _domain_length_and_area(shape, direction, dx=1.0, dy=1.0, dz=1.0):
    nz, ny, nx = shape
    if direction == "x":
        return (nx - 1) * dx, ny * dy * nz * dz
    if direction == "y":
        return (ny - 1) * dy, nx * dx * nz * dz
    if direction == "z":
        return (nz - 1) * dz, nx * dx * ny * dy
    raise ValueError("direction must be x, y, or z")


def _percolatesting_labels_for_direction(labels, direction):
    perc = percolation_3d(labels)
    if direction == "x":
        return perc["percolatesting_labels_x"]
    if direction == "y":
        return perc["percolatesting_labels_y"]
    if direction == "z":
        return perc["percolatesting_labels_z"]
    raise ValueError("direction must be x, y, or z")


def extract_percolatesting_subvolume(volume, labels, direction):
    """
    Keep only clusters that percolateste in the desired direction.
    This reduces the linear system and avoids spending time on isolated pores.

    direction:
        "x", "y" ou "z"
    """
    labs = _percolatesting_labels_for_direction(labels, direction)
    if not labs:
        return np.zeros_like(volume, dtype=bool)
    return np.isin(labels, labs)


def safe_div(num, den, eps=1e-12):
    if den is None or not np.isfinite(den) or abs(den) <= eps:
        return np.nan
    return num / (den + eps)


def edge_throat_statistics(volume, dist, dx=1.0, dy=1.0, dz=1.0, min_radius=None):
    """
    Throat statistics r_ij = min(r_i, r_j) in a 6-connected pore network.
    Use only positive edges to avoid duplicate connections.
    """
    volume = volume.astype(bool)
    if min_radius is None:
        min_radius = min(dx, dy, dz) * 0.5

    if volume.sum() == 0:
        return {
            "n_edges_network": 0,
            "mean_throat_radius": np.nan,
            "median_throat_radius": np.nan,
            "throat_radius_p10": np.nan,
            "throat_radius_p50": np.nan,
            "mean_throat_radius4": np.nan,
            "bottleneck_index_network": np.nan,
        }

    nz, ny, nx = volume.shape
    radii = []
    for (dz_, dy_, dx_), _length in _positive_neighbor_specs(dx=dx, dy=dy, dz=dz):
        # Pairs in non-wrapped slices
        src = volume[0:nz-dz_ if dz_ else nz, 0:ny-dy_ if dy_ else ny, 0:nx-dx_ if dx_ else nx]
        dst = volume[dz_:nz if dz_ else nz, dy_:ny if dy_ else ny, dx_:nx if dx_ else nx]
        edge_mask = src & dst
        if not np.any(edge_mask):
            continue

        r1 = dist[0:nz-dz_ if dz_ else nz, 0:ny-dy_ if dy_ else ny, 0:nx-dx_ if dx_ else nx][edge_mask]
        r2 = dist[dz_:nz if dz_ else nz, dy_:ny if dy_ else ny, dx_:nx if dx_ else nx][edge_mask]
        rr = np.minimum(r1, r2)
        rr = np.maximum(rr.astype(float), min_radius)
        radii.append(rr)

    if not radii:
        return {
            "n_edges_network": 0,
            "mean_throat_radius": np.nan,
            "median_throat_radius": np.nan,
            "throat_radius_p10": np.nan,
            "throat_radius_p50": np.nan,
            "mean_throat_radius4": np.nan,
            "bottleneck_index_network": np.nan,
        }

    radii = np.concatenate(radii)
    p10 = float(np.percentile(radii, 10))
    p50 = float(np.percentile(radii, 50))
    return {
        "n_edges_network": int(radii.size),
        "mean_throat_radius": float(np.mean(radii)),
        "median_throat_radius": p50,
        "throat_radius_p10": p10,
        "throat_radius_p50": p50,
        "mean_throat_radius4": float(np.mean(radii ** 4)),
        "bottleneck_index_network": float(p10 / (p50 + 1e-12)) if p50 > 0 else np.nan,
    }


def flux_distribution_metrics(edge_fluxes, edge_radii=None, eps=1e-30):
    """
    Compute flow entropy, effective participation, and bottleneck based on active edges.

    v6.1 note:
        The hydraulic flow in SI units may be on the order of 1e-16 to 1e-12.
        A fixed threshold of 1e-12 removed all active fluxes in some cases,
        producing n_edges_active_solver=0 and NaN entropy even when Q was valid.
        Now the threshold is adaptive: max(eps, q_max * 1e-12).
    """
    q_original = np.asarray(edge_fluxes, dtype=float)
    finite_mask = np.isfinite(q_original)
    q = q_original[finite_mask]

    if q.size == 0:
        return {
            "n_edges_total_solver": 0,
            "n_edges_active_solver": 0,
            "flux_sum_abs": 0.0,
            "flux_entropy": np.nan,
            "flux_participation": np.nan,
            "bottleneck_index_flow": np.nan,
        }

    q = np.abs(q)
    q_max = float(np.max(q)) if q.size else 0.0
    active_eps = max(float(eps), q_max * 1e-12)

    active_mask = q > active_eps
    q_active = q[active_mask]

    if q_active.size == 0:
        return {
            "n_edges_total_solver": int(q.size),
            "n_edges_active_solver": 0,
            "flux_sum_abs": 0.0,
            "flux_entropy": np.nan,
            "flux_participation": np.nan,
            "bottleneck_index_flow": np.nan,
        }

    q_sum = float(np.sum(q_active))
    weights = q_active / (q_sum + float(eps))
    entropy_raw = -float(np.sum(weights * np.log(weights + float(eps))))

    if q_active.size > 1:
        entropy_norm = entropy_raw / np.log(q_active.size)
    else:
        entropy_norm = 0.0

    participation = np.exp(entropy_raw) / q_active.size

    bottleneck_flow = np.nan
    if edge_radii is not None:
        r_original = np.asarray(edge_radii, dtype=float)
        if r_original.size == q_original.size:
            r = r_original[finite_mask]
            r_active = r[active_mask]
            r_active = r_active[np.isfinite(r_active)]
            if r_active.size > 0:
                p10 = np.percentile(r_active, 10)
                p50 = np.percentile(r_active, 50)
                bottleneck_flow = float(p10 / (p50 + float(eps))) if p50 > 0 else np.nan

    return {
        "n_edges_total_solver": int(q.size),
        "n_edges_active_solver": int(q_active.size),
        "flux_sum_abs": q_sum,
        "flux_entropy": float(entropy_norm),
        "flux_participation": float(participation),
        "bottleneck_index_flow": bottleneck_flow,
    }


def solve_voxel_resistor_permeability(
    volume,
    dist,
    direction="x",
    dx=1.0,
    dy=1.0,
    dz=1.0,
    mu=1.0,
    max_nodes=300000,
    min_radius=None,
    collect_flow_metrics=True,
):
    """
    Solve pressure in a connected pore-voxel network.

    Each pore voxel is a node.
    6-connected neighbors are edges.
    Pressure:
        inlet = 1
        outlet = 0

    Approximate conductance:
        g_ij = pi * r_ij^4 / (8 * mu * L_ij)

    Result:
        k_rel = mu * Q * L / (A * DeltaP)

    In addition to Q and k, this version also computes flow metrics:
        H_q,d      = normalized flow entropy
        Pi_q,d     = effective flow participation
        B_d_flux   = bottleneck based on the throats of active edges
    """
    volume = volume.astype(bool)

    if min_radius is None:
        min_radius = min(dx, dy, dz) * 0.5

    shape = volume.shape
    nz, ny, nx = shape
    neighbor_specs = _positive_neighbor_specs(dx=dx, dy=dy, dz=dz)
    L_domain, A_cross = _domain_length_and_area(shape, direction, dx=dx, dy=dy, dz=dz)

    empty_metrics = {
        "n_edges_total_solver": 0,
        "n_edges_active_solver": 0,
        "flux_sum_abs": np.nan,
        "flux_entropy": np.nan,
        "flux_participation": np.nan,
        "bottleneck_index_flow": np.nan,
    }

    if volume.sum() == 0:
        return {
            "direction": direction,
            "percolates": False,
            "n_solver_nodes": 0,
            "Q": 0.0,
            "k_relative": 0.0,
            "status": "no pores",
            **empty_metrics,
        }

    n_nodes = int(volume.sum())
    if n_nodes > max_nodes:
        return {
            "direction": direction,
            "percolates": True,
            "n_solver_nodes": n_nodes,
            "Q": np.nan,
            "k_relative": np.nan,
            "status": (
                f"system too large for direct solver: {n_nodes} nodes. "
                f"Use a larger --downsample or increase --max-nodes."
            ),
            **empty_metrics,
        }

    node_id = -np.ones(shape, dtype=np.int64)
    coords = np.argwhere(volume)
    node_id[volume] = np.arange(len(coords))

    if direction == "x":
        inlet_coords = np.argwhere(volume[:, :, 0])
        outlet_coords = np.argwhere(volume[:, :, -1])
        inlet_ids = set(int(node_id[z, y, 0]) for z, y in inlet_coords)
        outlet_ids = set(int(node_id[z, y, nx - 1]) for z, y in outlet_coords)
    elif direction == "y":
        inlet_coords = np.argwhere(volume[:, 0, :])
        outlet_coords = np.argwhere(volume[:, -1, :])
        inlet_ids = set(int(node_id[z, 0, x]) for z, x in inlet_coords)
        outlet_ids = set(int(node_id[z, ny - 1, x]) for z, x in outlet_coords)
    elif direction == "z":
        inlet_coords = np.argwhere(volume[0, :, :])
        outlet_coords = np.argwhere(volume[-1, :, :])
        inlet_ids = set(int(node_id[0, y, x]) for y, x in inlet_coords)
        outlet_ids = set(int(node_id[nz - 1, y, x]) for y, x in outlet_coords)
    else:
        raise ValueError("direction must be x, y, or z")

    inlet_ids.discard(-1)
    outlet_ids.discard(-1)

    if not inlet_ids or not outlet_ids:
        return {
            "direction": direction,
            "percolates": False,
            "n_solver_nodes": n_nodes,
            "Q": 0.0,
            "k_relative": 0.0,
            "status": "no inlet or outlet nodes",
            **empty_metrics,
        }

    all_ids = set(range(n_nodes))
    boundary_ids = inlet_ids.union(outlet_ids)
    unknown_ids = sorted(all_ids - boundary_ids)
    unknown_index = {nid: i for i, nid in enumerate(unknown_ids)}

    p_in = 1.0
    p_out = 0.0

    if not unknown_ids:
        # There are no internal nodes. Compute Q directly between faces if edges exist.
        p = np.zeros(n_nodes, dtype=float)
        for nid in inlet_ids:
            p[nid] = p_in
        for nid in outlet_ids:
            p[nid] = p_out
        Q_direct, edge_fluxes, edge_radii = _compute_inlet_Q_and_edge_fluxes(
            volume, dist, node_id, coords, p, inlet_ids, neighbor_specs,
            dx=dx, dy=dy, dz=dz, mu=mu, min_radius=min_radius
        )
        k_rel = mu * Q_direct * L_domain / (A_cross * (p_in - p_out)) if A_cross > 0 and L_domain > 0 else np.nan
        metrics = flux_distribution_metrics(edge_fluxes, edge_radii) if collect_flow_metrics else empty_metrics
        return {
            "direction": direction,
            "percolates": True,
            "n_solver_nodes": n_nodes,
            "Q": float(Q_direct),
            "k_relative": float(k_rel) if np.isfinite(k_rel) else np.nan,
            "status": "ok",
            **metrics,
        }

    rows = []
    cols = []
    data = []
    rhs = np.zeros(len(unknown_ids), dtype=float)

    def add_contribution(i_id, j_id, g):
        row = unknown_index[i_id]
        rows.append(row)
        cols.append(row)
        data.append(g)
        if j_id in unknown_index:
            col = unknown_index[j_id]
            rows.append(row)
            cols.append(col)
            data.append(-g)
        else:
            if j_id in inlet_ids:
                rhs[row] += g * p_in
            elif j_id in outlet_ids:
                rhs[row] += g * p_out

    # Assemble the system using +/- of the positive neighbors.
    for nid in unknown_ids:
        z, y, x = coords[nid]
        for (dz_, dy_, dx_), length in neighbor_specs:
            for sign in (+1, -1):
                zz = z + sign * dz_
                yy = y + sign * dy_
                xx = x + sign * dx_
                if (
                    0 <= zz < nz
                    and 0 <= yy < ny
                    and 0 <= xx < nx
                    and volume[zz, yy, xx]
                ):
                    j_id = int(node_id[zz, yy, xx])
                    ri = max(float(dist[z, y, x]), min_radius)
                    rj = max(float(dist[zz, yy, xx]), min_radius)
                    r_eff = min(ri, rj)
                    g = math.pi * (r_eff ** 4) / (8.0 * mu * length)
                    add_contribution(nid, j_id, g)

    A = sp.csr_matrix((data, (rows, cols)), shape=(len(unknown_ids), len(unknown_ids)))

    try:
        p_unknown = spla.spsolve(A, rhs)
    except Exception as exc:
        return {
            "direction": direction,
            "percolates": True,
            "n_solver_nodes": n_nodes,
            "Q": np.nan,
            "k_relative": np.nan,
            "status": f"solver failure: {exc}",
            **empty_metrics,
        }

    p = np.zeros(n_nodes, dtype=float)
    for nid in inlet_ids:
        p[nid] = p_in
    for nid in outlet_ids:
        p[nid] = p_out
    for nid, idx in unknown_index.items():
        p[nid] = p_unknown[idx]

    Q, edge_fluxes, edge_radii = _compute_inlet_Q_and_edge_fluxes(
        volume, dist, node_id, coords, p, inlet_ids, neighbor_specs,
        dx=dx, dy=dy, dz=dz, mu=mu, min_radius=min_radius
    )

    delta_p = p_in - p_out
    if A_cross <= 0 or L_domain <= 0 or delta_p <= 0:
        k_rel = np.nan
    else:
        k_rel = mu * Q * L_domain / (A_cross * delta_p)

    metrics = flux_distribution_metrics(edge_fluxes, edge_radii) if collect_flow_metrics else empty_metrics

    return {
        "direction": direction,
        "percolates": True,
        "n_solver_nodes": n_nodes,
        "Q": float(Q),
        "k_relative": float(k_rel) if np.isfinite(k_rel) else np.nan,
        "status": "ok",
        **metrics,
    }


def _compute_inlet_Q_and_edge_fluxes(
    volume, dist, node_id, coords, p, inlet_ids, neighbor_specs,
    dx=1.0, dy=1.0, dz=1.0, mu=1.0, min_radius=0.5
):
    """
    Compute inlet Q and fluxes on all unique network edges.
    """
    nz, ny, nx = volume.shape

    # Total flow leaving the inlet: use +/- to capture all neighbors of the inlet node.
    Q = 0.0
    for nid in inlet_ids:
        z, y, x = coords[nid]
        for (dz_, dy_, dx_), length in neighbor_specs:
            for sign in (+1, -1):
                zz = z + sign * dz_
                yy = y + sign * dy_
                xx = x + sign * dx_
                if (
                    0 <= zz < nz
                    and 0 <= yy < ny
                    and 0 <= xx < nx
                    and volume[zz, yy, xx]
                ):
                    j_id = int(node_id[zz, yy, xx])
                    if j_id in inlet_ids:
                        continue
                    ri = max(float(dist[z, y, x]), min_radius)
                    rj = max(float(dist[zz, yy, xx]), min_radius)
                    r_eff = min(ri, rj)
                    g = math.pi * (r_eff ** 4) / (8.0 * mu * length)
                    q = g * (p[nid] - p[j_id])
                    if q > 0:
                        Q += q

    # Absolute flux per unique edge, for entropy/localization.
    edge_fluxes = []
    edge_radii = []
    for nid, (z, y, x) in enumerate(coords):
        for (dz_, dy_, dx_), length in neighbor_specs:
            zz = z + dz_
            yy = y + dy_
            xx = x + dx_
            if (
                0 <= zz < nz
                and 0 <= yy < ny
                and 0 <= xx < nx
                and volume[zz, yy, xx]
            ):
                j_id = int(node_id[zz, yy, xx])
                ri = max(float(dist[z, y, x]), min_radius)
                rj = max(float(dist[zz, yy, xx]), min_radius)
                r_eff = min(ri, rj)
                g = math.pi * (r_eff ** 4) / (8.0 * mu * length)
                edge_fluxes.append(abs(g * (p[nid] - p[j_id])))
                edge_radii.append(r_eff)

    return Q, np.asarray(edge_fluxes, dtype=float), np.asarray(edge_radii, dtype=float)


def permeability_analysis(
    volume,
    labels,
    dist,
    output_folder,
    dx=1.0,
    dy=1.0,
    dz=1.0,
    mu=1.0,
    max_nodes=300000,
):
    rows = []

    for direction in ["x", "y", "z"]:
        sub = extract_percolatesting_subvolume(volume, labels, direction)

        if sub.sum() == 0:
            rows.append({
                "direction": direction,
                "percolates": False,
                "n_solver_nodes": 0,
                "Q": 0.0,
                "k_relative": 0.0,
                "status": "no percolatesting cluster",
                "n_edges_total_solver": 0,
                "n_edges_active_solver": 0,
                "flux_sum_abs": np.nan,
                "flux_entropy": np.nan,
                "flux_participation": np.nan,
                "bottleneck_index_flow": np.nan,
            })
            continue

        result = solve_voxel_resistor_permeability(
            sub,
            dist,
            direction=direction,
            dx=dx,
            dy=dy,
            dz=dz,
            mu=mu,
            max_nodes=max_nodes,
        )
        rows.append(result)

    df = pd.DataFrame(rows)
    df.to_csv(output_folder / "permeability.csv", index=False)
    return df



# ============================================================
# Permeability by multiple methods / new comparative indices
# ============================================================

M2_PER_MD = 9.869233e-16


def m2_to_md(k_m2):
    try:
        if k_m2 is None or not np.isfinite(float(k_m2)):
            return np.nan
        return float(k_m2) / M2_PER_MD
    except Exception:
        return np.nan


def interface_specific_surface_area(volume, dx=1.0, dy=1.0, dz=1.0):
    """
    Solid/pore interfacial area per total volume, S_v [1/unit].

    Count faces between voxels of different phases. For anisotropic voxels:
        interface normal a z -> area dx*dy
        interface normal a y -> area dx*dz
        interface normal a x -> area dy*dz
    """
    volume = volume.astype(bool)
    if volume.size == 0:
        return np.nan

    nz, ny, nx = volume.shape
    bulk_volume = nz * ny * nx * dx * dy * dz
    if bulk_volume <= 0:
        return np.nan

    area = 0.0
    area += np.sum(volume[:-1, :, :] != volume[1:, :, :]) * dx * dy
    area += np.sum(volume[:, :-1, :] != volume[:, 1:, :]) * dx * dz
    area += np.sum(volume[:, :, :-1] != volume[:, :, 1:]) * dy * dz
    return float(area / bulk_volume)


def kozeny_carman_k(phi, specific_surface_area, c_kozeny=5.0, eps=1e-30):
    """
    Kozeny--Carman estimate:
        k = phi^3 / (C * S_v^2 * (1 - phi)^2)

    Returns k in squared physical units if S_v is in 1/unit.
    """
    try:
        phi = float(phi)
        sv = float(specific_surface_area)
    except Exception:
        return np.nan
    if not np.isfinite(phi) or not np.isfinite(sv) or phi <= 0 or phi >= 1 or sv <= 0:
        return np.nan
    return float((phi ** 3) / (float(c_kozeny) * (sv ** 2) * ((1.0 - phi) ** 2 + eps)))


def _tau_or_one(tort, direction):
    tau = tort.get(f"tortuosity_{direction}", np.nan) if tort else np.nan
    try:
        tau = float(tau)
    except Exception:
        tau = np.nan
    if not np.isfinite(tau) or tau <= 0:
        return 1.0
    return tau


def _connected_subvolume_for_direction(volume, labels, direction):
    labs = _percolatesting_labels_for_direction(labels, direction)
    if not labs:
        return np.zeros_like(volume, dtype=bool), []
    return np.isin(labels, labs), labs


def permeability_methods_analysis(
    volume,
    labels,
    dist,
    perc,
    tort,
    perm_df,
    output_folder,
    dx=1.0,
    dy=1.0,
    dz=1.0,
    c_kozeny=5.0,
    selected_methods=None,
    eps=1e-12,
):
    """
    Compute and save permeability estimates using multiple methods.

    Methods:
        resistor_network:
            resistor-network solver already computed in permeability.csv.
        kozeny_carman_total:
            Kozeny--Carman using total porosity and global S_v; returns zero if it does not percolateste.
        kozeny_carman_connected:
            Kozeny--Carman using directionally connected porosity and global S_v.
        capillary_local_radius:
            k = phi_c * mean(r_local^2) / (8*tau_d), using EDT on the percolatesting cluster.
        capillary_throat_median:
            k = phi_c * r_throat_median^2 / (8*tau_d).
        capillary_throat_bottleneck:
            k = phi_c * r_throat_p10^2 / (8*tau_d), conservative bottleneck index.

    Outputs:
        permeability_methods.csv
        permeability_methods_summary.csv
        permeability_method_signature.json
    """
    if selected_methods is None:
        selected_methods = ["all"]
    selected = set(str(m).strip().lower() for m in selected_methods)
    all_methods = {
        "resistor_network",
        "kozeny_carman_total",
        "kozeny_carman_connected",
        "capillary_local_radius",
        "capillary_throat_median",
        "capillary_throat_bottleneck",
    }
    if "all" in selected:
        selected = all_methods

    output_folder = Path(output_folder)
    phi_total = float(volume.mean()) if volume.size else 0.0
    sv_total = interface_specific_surface_area(volume, dx=dx, dy=dy, dz=dz)
    k_kc_total = kozeny_carman_k(phi_total, sv_total, c_kozeny=c_kozeny)

    perm_by_dir = {}
    if perm_df is not None and not perm_df.empty:
        for _, row in perm_df.iterrows():
            perm_by_dir[str(row.get("direction"))] = row.to_dict()

    rows = []
    for direction in ["x", "y", "z"]:
        P_d = bool(perc.get(f"percolates_{direction}_3d", False))
        sub, labs = _connected_subvolume_for_direction(volume, labels, direction)
        phi_c = float(sub.sum() / volume.size) if volume.size else 0.0
        tau = _tau_or_one(tort, direction)

        throat_stats = edge_throat_statistics(sub, dist, dx=dx, dy=dy, dz=dz)
        local_r = dist[sub]
        local_r = local_r[np.isfinite(local_r)] if local_r.size else np.array([], dtype=float)
        mean_r2 = float(np.mean(local_r ** 2)) if local_r.size else np.nan

        # 1) Voxel resistor network.
        if "resistor_network" in selected:
            row0 = perm_by_dir.get(direction, {})
            k = row0.get("k_relative", np.nan)
            try:
                k = float(k)
            except Exception:
                k = np.nan
            rows.append({
                "method": "resistor_network",
                "method_class": "solver_voxel",
                "direction": direction,
                "percolates": P_d,
                "phi_total": phi_total,
                "phi_connected": phi_c,
                "tortuosity": tau,
                "specific_surface_area": sv_total,
                "k_m2": k,
                "k_mD": m2_to_md(k),
                "status": row0.get("status", "no result"),
                "note": "Hydraulic resistor network on pore voxels; preferred physics-based estimator.",
            })

        # 2) Kozeny-Carman total.
        if "kozeny_carman_total" in selected:
            k = k_kc_total if P_d else 0.0
            rows.append({
                "method": "kozeny_carman_total",
                "method_class": "empirical_global",
                "direction": direction,
                "percolates": P_d,
                "phi_total": phi_total,
                "phi_connected": phi_c,
                "tortuosity": tau,
                "specific_surface_area": sv_total,
                "k_m2": k,
                "k_mD": m2_to_md(k),
                "status": "ok" if P_d else "zero: no directional percolation",
                "note": "Kozeny--Carman with total porosity; fast, but it does not resolve flow.",
            })

        # 3) Kozeny-Carman connected.
        if "kozeny_carman_connected" in selected:
            k = kozeny_carman_k(phi_c, sv_total, c_kozeny=c_kozeny) if P_d else 0.0
            rows.append({
                "method": "kozeny_carman_connected",
                "method_class": "empirical_connected",
                "direction": direction,
                "percolates": P_d,
                "phi_total": phi_total,
                "phi_connected": phi_c,
                "tortuosity": tau,
                "specific_surface_area": sv_total,
                "k_m2": k,
                "k_mD": m2_to_md(k),
                "status": "ok" if P_d else "zero: no directional percolation",
                "note": "Kozeny--Carman using only directionally connected porosity.",
            })

        # 4) Capillary by EDT local radius.
        if "capillary_local_radius" in selected:
            k = phi_c * mean_r2 / (8.0 * tau) if P_d and np.isfinite(mean_r2) else 0.0
            rows.append({
                "method": "capillary_local_radius",
                "method_class": "pore_radius_proxy",
                "direction": direction,
                "percolates": P_d,
                "phi_total": phi_total,
                "phi_connected": phi_c,
                "tortuosity": tau,
                "specific_surface_area": sv_total,
                "mean_local_radius2": mean_r2,
                "k_m2": k,
                "k_mD": m2_to_md(k),
                "status": "ok" if P_d else "zero: no directional percolation",
                "note": "Capillary model k=phi_c*<r^2>/(8*tau) using distance to solid.",
            })

        # 5) Capillary by median throat.
        if "capillary_throat_median" in selected:
            r = throat_stats.get("throat_radius_p50", np.nan)
            k = phi_c * (r ** 2) / (8.0 * tau) if P_d and np.isfinite(r) else 0.0
            rows.append({
                "method": "capillary_throat_median",
                "method_class": "throat_proxy",
                "direction": direction,
                "percolates": P_d,
                "phi_total": phi_total,
                "phi_connected": phi_c,
                "tortuosity": tau,
                "specific_surface_area": sv_total,
                "throat_radius_used": r,
                "k_m2": k,
                "k_mD": m2_to_md(k),
                "status": "ok" if P_d else "zero: no directional percolation",
                "note": "Capillary model using the median radius of 6-connected throats.",
            })

        # 6) Capillary by bottleneck throat p10.
        if "capillary_throat_bottleneck" in selected:
            r = throat_stats.get("throat_radius_p10", np.nan)
            k = phi_c * (r ** 2) / (8.0 * tau) if P_d and np.isfinite(r) else 0.0
            rows.append({
                "method": "capillary_throat_bottleneck",
                "method_class": "bottleneck_proxy",
                "direction": direction,
                "percolates": P_d,
                "phi_total": phi_total,
                "phi_connected": phi_c,
                "tortuosity": tau,
                "specific_surface_area": sv_total,
                "throat_radius_used": r,
                "k_m2": k,
                "k_mD": m2_to_md(k),
                "status": "ok" if P_d else "zero: no directional percolation",
                "note": "Conservative estimate using throat p10 as the hydraulic bottleneck.",
            })

    methods_df = pd.DataFrame(rows)
    if not methods_df.empty:
        methods_df.to_csv(output_folder / "permeability_methods.csv", index=False)
    else:
        methods_df = pd.DataFrame(columns=["method", "direction", "k_m2", "k_mD", "status"])
        methods_df.to_csv(output_folder / "permeability_methods.csv", index=False)

    summary_rows = []
    for method, group in methods_df.groupby("method"):
        by_dir = {str(row["direction"]): row for _, row in group.iterrows()}
        kx = float(by_dir.get("x", {}).get("k_m2", np.nan)) if "x" in by_dir else np.nan
        ky = float(by_dir.get("y", {}).get("k_m2", np.nan)) if "y" in by_dir else np.nan
        kz = float(by_dir.get("z", {}).get("k_m2", np.nan)) if "z" in by_dir else np.nan
        # For anisotropy, NaN becomes zero.
        anis = hydraulic_anisotropy_from_k(kx, ky, kz)
        kvals = np.array([kx, ky, kz], dtype=float)
        finite_pos = kvals[np.isfinite(kvals) & (kvals > 0)]
        k_min_pos = float(np.min(finite_pos)) if finite_pos.size else np.nan
        k_max = float(np.nanmax(kvals)) if np.any(np.isfinite(kvals)) else np.nan
        spread = float(k_max / (k_min_pos + eps)) if np.isfinite(k_max) and np.isfinite(k_min_pos) and k_min_pos > 0 else np.nan
        summary_rows.append({
            "method": method,
            "k_x_m2": kx,
            "k_y_m2": ky,
            "k_z_m2": kz,
            "k_x_mD": m2_to_md(kx),
            "k_y_mD": m2_to_md(ky),
            "k_z_mD": m2_to_md(kz),
            "k_max_mD": m2_to_md(k_max),
            "k_min_positive_mD": m2_to_md(k_min_pos),
            "directional_spread_max_over_min_positive": spread,
            **anis,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_folder / "permeability_methods_summary.csv", index=False)

    # Method-agreement indices by direction.
    comparison_rows = []
    for direction, group in methods_df.groupby("direction"):
        positive = group[np.isfinite(group["k_m2"]) & (group["k_m2"] > 0)].copy()
        if positive.empty:
            comparison_rows.append({
                "direction": direction,
                "n_positive_methods": 0,
                "method_spread_max_over_min": np.nan,
                "log10_k_std": np.nan,
                "recommended_k_m2": 0.0,
                "recommended_k_mD": 0.0,
                "recommended_method": "none",
            })
            continue
        k_values = positive["k_m2"].to_numpy(dtype=float)
        spread = float(np.max(k_values) / (np.min(k_values) + eps))
        log_std = float(np.std(np.log10(k_values))) if k_values.size > 1 else 0.0
        # Preference: resistor if valid, otherwise connected KC, otherwise local capillary.
        recommended_order = ["resistor_network", "kozeny_carman_connected", "capillary_local_radius"]
        rec_row = None
        for m in recommended_order:
            cand = positive[positive["method"] == m]
            if not cand.empty:
                rec_row = cand.iloc[0]
                break
        if rec_row is None:
            rec_row = positive.iloc[0]
        comparison_rows.append({
            "direction": direction,
            "n_positive_methods": int(k_values.size),
            "method_spread_max_over_min": spread,
            "log10_k_std": log_std,
            "recommended_k_m2": float(rec_row["k_m2"]),
            "recommended_k_mD": float(rec_row["k_mD"]),
            "recommended_method": str(rec_row["method"]),
        })
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_folder / "permeability_method_consistency.csv", index=False)

    signature = {
        "methods": rows,
        "summary": summary_rows,
        "consistency": comparison_rows,
        "parameters": {
            "c_kozeny": c_kozeny,
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "specific_surface_area": sv_total,
            "phi_total": phi_total,
        },
    }
    save_json(output_folder / "permeability_method_signature.json", signature)
    return methods_df, summary_df, comparison_df, signature


# ============================================================
# Morphohydraulic indices
# ============================================================

def hydraulic_anisotropy_from_k(kx, ky, kz, eps=1e-12):
    """
    A_K index based on the deviatoric norm of the diagonal K tensor.
    Also returns the normalized vector D_K.
    """
    kvals = np.array([
        0.0 if (kx is None or not np.isfinite(kx)) else float(kx),
        0.0 if (ky is None or not np.isfinite(ky)) else float(ky),
        0.0 if (kz is None or not np.isfinite(kz)) else float(kz),
    ], dtype=float)
    K = np.diag(kvals)
    norm_K = np.linalg.norm(K, ord="fro")
    trace = np.trace(K)
    dev = K - (trace / 3.0) * np.eye(3)
    if norm_K <= eps:
        A_K = 0.0
    else:
        A_K = math.sqrt(3.0 / 2.0) * np.linalg.norm(dev, ord="fro") / (norm_K + eps)
    denom = float(np.sum(kvals) + eps)
    D = kvals / denom
    return {
        "hydraulic_anisotropy_AK": float(A_K),
        "dominant_flow_vector_x": float(D[0]),
        "dominant_flow_vector_y": float(D[1]),
        "dominant_flow_vector_z": float(D[2]),
    }


def percolation_gap(volume, direction, dx=1.0, dy=1.0, dz=1.0, connectivity=1,
                    max_radius=None, radius_step=None, eps=1e-12):
    """
    Compute G_d = r_c,d / L_d by isotropic morphological dilation.

    If the sample already percolates, return r_c=0 and G=0.
    If no percolation is found up to max_radius, return NaN and an explanatory status.
    """
    labels0, _ = label_volume(volume, connectivity=connectivity)
    perc0 = percolation_3d(labels0)
    if perc0[f"percolates_{direction}_3d"]:
        return {
            "critical_dilation_radius": 0.0,
            "percolation_gap": 0.0,
            "percolation_gap_status": "already percolates",
        }

    L_domain, _A = _domain_length_and_area(volume.shape, direction, dx=dx, dy=dy, dz=dz)
    if radius_step is None or radius_step <= 0:
        radius_step = min(dx, dy, dz)
    if max_radius is None:
        max_radius = 0.25 * max((volume.shape[2] - 1) * dx, (volume.shape[1] - 1) * dy, (volume.shape[0] - 1) * dz)

    if volume.sum() == 0:
        return {
            "critical_dilation_radius": np.nan,
            "percolation_gap": np.nan,
            "percolation_gap_status": "no pores to dilate",
        }

    # Distance from each solid voxel to the nearest pore. Dilation: dist_to_pore <= r.
    dist_to_pore = ndi.distance_transform_edt(~volume.astype(bool), sampling=(dz, dy, dx))
    radii = np.arange(radius_step, max_radius + 0.5 * radius_step, radius_step)
    for r in radii:
        dilated = volume | (dist_to_pore <= r)
        labels_r, _ = label_volume(dilated, connectivity=connectivity)
        perc_r = percolation_3d(labels_r)
        if perc_r[f"percolates_{direction}_3d"]:
            return {
                "critical_dilation_radius": float(r),
                "percolation_gap": float(r / (L_domain + eps)) if L_domain > 0 else np.nan,
                "percolation_gap_status": "ok",
            }

    return {
        "critical_dilation_radius": np.nan,
        "percolation_gap": np.nan,
        "percolation_gap_status": f"did not percolateste up to radius {max_radius}",
    }


def graph_redundancy_edge_disjoint(volume, direction, max_nodes=50000):
    """
    Estimate R_d^g as the maximum number of edge-disjoint paths using NetworkX.
    For safety, this is optional and limitd to small volumes.

    For large volumes, returns NaN and a not-computed status.
    """
    n_nodes = int(volume.sum())
    if n_nodes == 0:
        return np.nan, "no pores"
    if n_nodes > max_nodes:
        return np.nan, f"not computed: {n_nodes} nodes > limit {max_nodes}"
    try:
        import networkx as nx  # optional dependency
    except Exception:
        return np.nan, "not computed: networkx not instalado"

    nz, ny, nx_ = volume.shape
    node_id = -np.ones(volume.shape, dtype=np.int64)
    coords = np.argwhere(volume)
    node_id[volume] = np.arange(n_nodes)

    G = nx.DiGraph()
    source = "source"
    sink = "sink"

    for nid, (z, y, x) in enumerate(coords):
        G.add_node(int(nid))
        if direction == "x":
            if x == 0:
                G.add_edge(source, int(nid), capacity=1)
            if x == nx_ - 1:
                G.add_edge(int(nid), sink, capacity=1)
        elif direction == "y":
            if y == 0:
                G.add_edge(source, int(nid), capacity=1)
            if y == ny - 1:
                G.add_edge(int(nid), sink, capacity=1)
        elif direction == "z":
            if z == 0:
                G.add_edge(source, int(nid), capacity=1)
            if z == nz - 1:
                G.add_edge(int(nid), sink, capacity=1)

    for nid, (z, y, x) in enumerate(coords):
        for dz_, dy_, dx_ in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            zz = z + dz_
            yy = y + dy_
            xx = x + dx_
            if 0 <= zz < nz and 0 <= yy < ny and 0 <= xx < nx_ and volume[zz, yy, xx]:
                j = int(node_id[zz, yy, xx])
                # Undirected graph represented by two directed edges with capacity 1.
                G.add_edge(int(nid), j, capacity=1)
                G.add_edge(j, int(nid), capacity=1)

    try:
        val = nx.maximum_flow_value(G, source, sink, capacity="capacity")
        return float(val), "ok"
    except Exception as exc:
        return np.nan, f"max-flow failure: {exc}"


def morphohydraulic_indices(
    volume,
    labels,
    dist,
    perc,
    tort,
    perm_df,
    output_folder,
    dx=1.0,
    dy=1.0,
    dz=1.0,
    connectivity=1,
    compute_gap=True,
    gap_max_radius=None,
    gap_radius_step=None,
    compute_graph_redundancy=False,
    graph_max_nodes=50000,
    eps=1e-12,
):
    """
    Compute the directional morphohydraulic signature.

    Main outputs:
        output/morphohydraulic_indices.csv      one row per direction
        output/morphohydraulic_summary.csv      global indices such as A_K and D_K
        output/morphohydraulic_signature.json   complete structure in JSON
    """
    phi = float(volume.mean()) if volume.size > 0 else 0.0
    total_voxels = int(volume.size)

    perm_by_dir = {}
    if perm_df is not None and not perm_df.empty:
        for _, row in perm_df.iterrows():
            perm_by_dir[str(row["direction"])] = row.to_dict()

    kx = float(perm_by_dir.get("x", {}).get("k_relative", 0.0) or 0.0)
    ky = float(perm_by_dir.get("y", {}).get("k_relative", 0.0) or 0.0)
    kz = float(perm_by_dir.get("z", {}).get("k_relative", 0.0) or 0.0)
    if not np.isfinite(kx):
        kx = 0.0
    if not np.isfinite(ky):
        ky = 0.0
    if not np.isfinite(kz):
        kz = 0.0
    anis = hydraulic_anisotropy_from_k(kx, ky, kz, eps=eps)

    rows = []
    for direction in ["x", "y", "z"]:
        labs = _percolatesting_labels_for_direction(labels, direction)
        sub = np.isin(labels, labs) if labs else np.zeros_like(volume, dtype=bool)
        phi_c = float(sub.sum() / total_voxels) if total_voxels > 0 else 0.0
        E_c = float(phi_c / (phi + eps)) if phi > 0 else 0.0
        phi_dead = float(max(phi - phi_c, 0.0))
        dead_fraction = float(max(1.0 - E_c, 0.0))
        P_d = bool(perc[f"percolates_{direction}_3d"])
        tau = tort.get(f"tortuosity_{direction}", np.nan)
        tau_for_formula = tau if (tau is not None and np.isfinite(tau) and tau > 0) else np.nan

        perm_row = perm_by_dir.get(direction, {})
        k_d = perm_row.get("k_relative", 0.0)
        try:
            k_d = float(k_d)
        except Exception:
            k_d = np.nan

        T_d = 0.0
        if P_d and np.isfinite(tau_for_formula):
            T_d = float(E_c / (tau_for_formula + eps))

        throat_stats = edge_throat_statistics(sub, dist, dx=dx, dy=dy, dz=dz)
        L_ref = min(dx, dy, dz)
        mean_r4 = throat_stats["mean_throat_radius4"]
        T_h = 0.0
        if P_d and np.isfinite(tau_for_formula) and np.isfinite(mean_r4) and L_ref > 0:
            T_h = float(phi_c * (mean_r4 / ((L_ref ** 4) + eps)) / (tau_for_formula + eps))

        eta = np.nan
        if np.isfinite(k_d) and phi_c > 0:
            eta = float(k_d / (phi_c + eps))

        if compute_gap:
            gap = percolation_gap(
                volume,
                direction,
                dx=dx,
                dy=dy,
                dz=dz,
                connectivity=connectivity,
                max_radius=gap_max_radius,
                radius_step=gap_radius_step,
            )
        else:
            gap = {
                "critical_dilation_radius": np.nan,
                "percolation_gap": np.nan,
                "percolation_gap_status": "not computed",
            }

        R_cluster = int(len(labs))
        if compute_graph_redundancy and P_d:
            R_graph, R_graph_status = graph_redundancy_edge_disjoint(sub, direction, max_nodes=graph_max_nodes)
        else:
            R_graph, R_graph_status = np.nan, "not computed"

        B_flow = perm_row.get("bottleneck_index_flow", np.nan)
        try:
            B_flow = float(B_flow)
        except Exception:
            B_flow = np.nan
        B_final = B_flow if np.isfinite(B_flow) else throat_stats["bottleneck_index_network"]

        row = {
            "direction": direction,
            "phi_total": phi,
            "percolates": P_d,
            "percolation_indicator": int(P_d),
            "n_percolatesting_clusters": R_cluster,
            "percolatesting_labels": ";".join(str(v) for v in labs),
            "phi_connected": phi_c,
            "connectivity_efficiency": E_c,
            "phi_dead": phi_dead,
            "dead_fraction": dead_fraction,
            "tortuosity": tau,
            "transportability": T_d,
            "hydraulic_transportability": T_h,
            "k_relative": k_d,
            "permeability_yield": eta,
            "critical_dilation_radius": gap["critical_dilation_radius"],
            "percolation_gap": gap["percolation_gap"],
            "percolation_gap_status": gap["percolation_gap_status"],
            "cluster_redundancy": R_cluster,
            "graph_redundancy": R_graph,
            "graph_redundancy_status": R_graph_status,
            "flux_entropy": perm_row.get("flux_entropy", np.nan),
            "flux_participation": perm_row.get("flux_participation", np.nan),
            "n_edges_network": throat_stats["n_edges_network"],
            "n_edges_total_solver": perm_row.get("n_edges_total_solver", np.nan),
            "n_edges_active_solver": perm_row.get("n_edges_active_solver", np.nan),
            "mean_throat_radius": throat_stats["mean_throat_radius"],
            "median_throat_radius": throat_stats["median_throat_radius"],
            "throat_radius_p10": throat_stats["throat_radius_p10"],
            "throat_radius_p50": throat_stats["throat_radius_p50"],
            "mean_throat_radius4": throat_stats["mean_throat_radius4"],
            "bottleneck_index_network": throat_stats["bottleneck_index_network"],
            "bottleneck_index_flow": B_flow,
            "bottleneck_index": B_final,
            "solver_status": perm_row.get("status", "no permeability result"),
        }
        rows.append(row)

    indices_df = pd.DataFrame(rows)
    indices_df.to_csv(output_folder / "morphohydraulic_indices.csv", index=False)

    summary = {
        "phi_total": phi,
        "k_x": kx,
        "k_y": ky,
        "k_z": kz,
        **anis,
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_folder / "morphohydraulic_summary.csv", index=False)

    signature = {
        "summary": summary,
        "directional_indices": rows,
    }
    save_json(output_folder / "morphohydraulic_signature.json", signature)

    return indices_df, summary_df, signature

# ============================================================
# Visualizations
# ============================================================

def plot_slice_porosity(slice_df, output_folder):
    plt.figure()
    plt.plot(slice_df["slice_index"], slice_df["porosity_2d"], marker="o")
    plt.xlabel("Slice")
    plt.ylabel("2D porosity")
    plt.title("Poresity by slice")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_folder / "porosity_by_slice.png", dpi=200)
    plt.close()


def plot_pore_radius_distribution(pore_df, output_folder):
    if pore_df.empty:
        return

    radii = pore_df["radius_eq"].values
    radii = radii[np.isfinite(radii)]
    radii = radii[radii > 0]

    if len(radii) == 0:
        return

    plt.figure()
    plt.hist(radii, bins=50)
    plt.xlabel("Radius equivalent 2D")
    plt.ylabel("Count")
    plt.title("2D equivalent-radius distribution")
    plt.tight_layout()
    plt.savefig(output_folder / "pore_radius_distribution_2d.png", dpi=200)
    plt.close()

    plt.figure()
    plt.hist(np.log10(radii), bins=50)
    plt.xlabel("log10(radius equivalent 2D)")
    plt.ylabel("Count")
    plt.title("Logarithmic 2D radius distribution")
    plt.tight_layout()
    plt.savefig(output_folder / "pore_radius_distribution_2d_log.png", dpi=200)
    plt.close()


def plot_local_radius_distribution(radii, output_folder):
    if radii is None or len(radii) == 0:
        return

    radii = np.asarray(radii)
    radii = radii[np.isfinite(radii)]
    radii = radii[radii > 0]

    if len(radii) == 0:
        return

    plt.figure()
    plt.hist(radii, bins=60)
    plt.xlabel("Local radius by distance to solid")
    plt.ylabel("Pore voxel count")
    plt.title("3D local-radius distribution")
    plt.tight_layout()
    plt.savefig(output_folder / "local_radius_distribution_3d.png", dpi=200)
    plt.close()


def plot_middle_slices(volume, output_folder):
    nz = volume.shape[0]
    indices = sorted(set([0, nz // 2, nz - 1]))

    for idx in indices:
        plt.figure()
        plt.imshow(volume[idx], cmap="gray")
        plt.title(f"Binary mask - slice {idx + 1}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_folder / f"binary_slice_{idx + 1}.png", dpi=200)
        plt.close()


def plot_mip(volume, output_folder):
    """
    Maximum intensity projections.
    """
    projections = {
        "z_projection": volume.max(axis=0),
        "y_projection": volume.max(axis=1),
        "x_projection": volume.max(axis=2),
    }

    for name, img in projections.items():
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.title(name)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_folder / f"{name}.png", dpi=200)
        plt.close()


# ============================================================
# Text report
# ============================================================

def create_markdown_report(
    output_folder,
    files,
    slice_df,
    volume_props,
    perc,
    tort,
    fractal,
    perm_df,
    morph_df=None,
    morph_summary_df=None,
    unit="px",
):
    report_path = output_folder / "report.md"

    lines = []
    lines.append("# Porous-matrix analysis report\n")
    lines.append("## Input\n")
    lines.append(f"- Number of slices: **{len(files)}**")
    lines.append(f"- First slice: `{files[0].name}`")
    lines.append(f"- Last slice: `{files[-1].name}`")
    lines.append(f"- Spatial unit: `{unit}`\n")

    lines.append("## Mean 2D properties\n")
    lines.append(f"- Mean 2D porosity: **{slice_df['porosity_2d'].mean():.6f}**")
    lines.append(f"- Standard deviation of 2D porosity: **{slice_df['porosity_2d'].std():.6f}**")
    lines.append(f"- Mean number of pores per slice: **{slice_df['n_pores_2d'].mean():.2f}**")
    lines.append(f"- Mean 2D equivalent radius: **{slice_df['mean_radius_eq'].mean():.6f} {unit}**")
    lines.append(f"- Median 2D equivalent radius: **{slice_df['median_radius_eq'].median():.6f} {unit}**\n")

    lines.append("## Properties 3D\n")
    lines.append(f"- 3D porosity: **{volume_props['porosity_3d']:.6f}**")
    lines.append(f"- Number of 3D clusters: **{volume_props['n_clusters_3d']}**")
    lines.append(f"- Largest cluster, fraction of pores: **{volume_props['largest_cluster_fraction_of_pores']:.6f}**")
    lines.append(f"- Mean 3D equivalent radius: **{volume_props['mean_equiv_radius_3d']:.6f} {unit}**")
    lines.append(f"- Median 3D equivalent radius: **{volume_props['median_equiv_radius_3d']:.6f} {unit}**\n")

    lines.append("## 3D percolation\n")
    lines.append(f"- Percolates in x: **{perc['percolates_x_3d']}**")
    lines.append(f"- Percolates in y: **{perc['percolates_y_3d']}**")
    lines.append(f"- Percolates in z: **{perc['percolates_z_3d']}**\n")

    lines.append("## Approximate tortuosity\n")
    lines.append(f"- Tortuosity x: **{tort['tortuosity_x']}**")
    lines.append(f"- Tortuosity y: **{tort['tortuosity_y']}**")
    lines.append(f"- Tortuosity z: **{tort['tortuosity_z']}**\n")

    lines.append("## Apollonian/fractal-like fit\n")
    lines.append(
        "- Fit performed on the tail of the cumulative distribution "
        "`N(r >= R) ~ R^(-D)`.\n"
    )
    lines.append(f"- Estimated D: **{fractal['power_law_D_estimate']}**")
    lines.append(f"- Fit R2: **{fractal['power_law_r2']}**")
    lines.append(f"- Points used: **{fractal['power_law_n_points']}**\n")

    lines.append("## Relative permeability\n")
    lines.append(
        "The permeability below comes from a hydraulic resistor network on pore voxels. "
        "It should be interpreted as effective relative permeability, especially if the physical scale "
        "or experimental calibration are unknown.\n"
    )

    lines.append("| Direction | Percolates | Solver nodes | Q | k_relative | Status |")
    lines.append("|---|---:|---:|---:|---:|---|")

    for _, row in perm_df.iterrows():
        lines.append(
            f"| {row['direction']} | {row['percolates']} | {row['n_solver_nodes']} | "
            f"{row['Q']} | {row['k_relative']} | {row['status']} |"
        )

    if morph_summary_df is not None and not morph_summary_df.empty:
        sm = morph_summary_df.iloc[0]
        lines.append("\n## Global morphohydraulic signature\n")
        lines.append(f"- Hydraulic anisotropy A_K: **{sm.get('hydraulic_anisotropy_AK', np.nan)}**")
        lines.append(
            "- Dominant-direction vector D_K: "
            f"**[{sm.get('dominant_flow_vector_x', np.nan)}, "
            f"{sm.get('dominant_flow_vector_y', np.nan)}, "
            f"{sm.get('dominant_flow_vector_z', np.nan)}]**\n"
        )

    if morph_df is not None and not morph_df.empty:
        lines.append("## Directional morphohydraulic indices\n")
        lines.append("| Direction | φ_c,d | E_c,d | φ_dead,d | T_d | T_h,d | η_d | G_d | R_c | H_q,d | Π_q,d | B_d |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in morph_df.iterrows():
            lines.append(
                f"| {row.get('direction')} | {row.get('phi_connected')} | "
                f"{row.get('connectivity_efficiency')} | {row.get('phi_dead')} | "
                f"{row.get('transportability')} | {row.get('hydraulic_transportability')} | "
                f"{row.get('permeability_yield')} | {row.get('percolation_gap')} | "
                f"{row.get('cluster_redundancy')} | {row.get('flux_entropy')} | "
                f"{row.get('flux_participation')} | {row.get('bottleneck_index')} |"
            )

    lines.append("\n## Generated files\n")
    lines.append("- `slice_properties.csv`")
    lines.append("- `volume_properties.csv`")
    lines.append("- `pore_size_distribution.csv`")
    lines.append("- `permeability.csv`")
    lines.append("- `morphohydraulic_indices.csv`")
    lines.append("- `morphohydraulic_summary.csv`")
    lines.append("- `morphohydraulic_signature.json`")
    lines.append("- `permeability_methods.csv`")
    lines.append("- `permeability_methods_summary.csv`")
    lines.append("- `permeability_method_consistency.csv`")
    lines.append("- `permeability_method_signature.json`")
    lines.append("- `porosity_by_slice.png`")
    lines.append("- `pore_radius_distribution_2d.png`")
    lines.append("- `local_radius_distribution_3d.png`")
    lines.append("- projections and binary slices in PNG")
    lines.append("- `debug_crop_slice_*.png`, `debug_clean_slice_*.png`, `debug_grid_slice_*.png` and `debug_mask_slice_*.png` to inspect cropping, grid removal, and segmentation\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze a 3D porous matrix from sequential images in structures/1.png, 2.png, ... Version v6.1: automatic cropping ignores colorbars and computes morphohydraulic indices."
    )

    parser.add_argument("--folder", type=str, default="structures",
                        help="Folder containing 1.png, 2.png, ..., n.png")
    parser.add_argument("--prefix", type=str, default="",
                        help="Optional file prefix. Empty default reads 1.png, 2.png, ...; use f for f1.png")
    parser.add_argument("--ext", type=str, default="",
                        help="Optional extension. Empty default reads png, jpg, jpeg, tif, tiff, bmp, and webp")
    parser.add_argument("--no-strict-sequence", action="store_true",
                        help="Do not require the exact sequence 1, 2, 3, ..., n; only sort numerically")
    parser.add_argument("--output", type=str, default="output",
                        help="Output folder")
    parser.add_argument("--npz-file", type=str, default="",
                        help="Optional: read a binary volume directly from a .npz file instead of folder images")
    parser.add_argument("--npz-key", type=str, default="pores",
                        help="Key inside the .npz file. For the snake generator, normally: pores")
    parser.add_argument("--npz-order", type=str, default="xyz", choices=["xyz", "zyx"],
                        help="Volume order in the .npz file: xyz for pores[x,y,z], zyx for volume[z,y,x]")
    parser.add_argument("--npz-solid-is-true", action="store_true",
                        help="Use if True in the .npz file represents solid, not pore")

    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "blue", "red", "green", "color", "nonblack", "nonwhite", "bright", "dark", "alpha"],
                        help="Pore segmentation mode. For red/blue images, use --mode blue if the pores are blue")

    parser.add_argument("--green-min", type=int, default=80,
                        help="Minimum green-channel value for detecting pores")
    parser.add_argument("--green-margin", type=int, default=40,
                        help="Margin G > R + margin and G > B + margin")
    parser.add_argument("--blue-min", type=int, default=80,
                        help="Minimum blue-channel value for detecting pores in blue mode")
    parser.add_argument("--blue-margin", type=int, default=40,
                        help="Margin B > R + margin and B > G + margin")
    parser.add_argument("--red-min", type=int, default=80,
                        help="Minimum red-channel value for detecting pores in red mode")
    parser.add_argument("--red-margin", type=int, default=40,
                        help="Margin R > G + margin and R > B + margin")
    parser.add_argument("--threshold", type=int, default=128,
                        help="Threshold for bright/dark/alpha modes")
    parser.add_argument("--black-threshold", type=int, default=12,
                        help="Threshold to consider a pixel different from black in nonblack mode")
    parser.add_argument("--white-threshold", type=int, default=245,
                        help="Threshold to consider a pixel different from white in nonwhite mode")
    parser.add_argument("--color-min", type=int, default=25,
                        help="Minimum difference between RGB channels for detecting colored pixels in color/auto mode")
    parser.add_argument("--invert", action="store_true",
                        help="Invert the final mask: pore becomes solid and solid becomes pore")

    parser.add_argument("--crop", type=str, default="auto",
                        choices=["auto", "manual", "none"],
                        help="Crop the central image. auto ignores borders/text; manual uses --roi; none uses the whole image")
    parser.add_argument("--roi", type=str, default="",
                        help="Manual ROI in x1,y1,x2,y2 format. Example: --roi 120,80,980,840")
    parser.add_argument("--crop-padding", type=int, default=0,
                        help="Padding in pixels around the automatic crop")
    parser.add_argument("--crop-bg-threshold", type=int, default=18,
                        help="Sensitivity of automatic cropping relative to the corner background color")
    parser.add_argument("--crop-trim-fraction", type=float, default=0.0,
                        help="Shrink the cropped ROI by a fraction on each side. Example: 0.03 removes 3%% from each border")
    parser.add_argument("--crop-each-slice", action="store_true",
                        help="Compute the crop automatically for each slice. By default, use the ROI of the first slice for all slices")
    parser.add_argument("--keep-grid", action="store_true",
                        help="Keep overlaid black grid/axes. By default, the script tries to remove the grid before segmentation")
    parser.add_argument("--grid-dark-threshold", type=int, default=25,
                        help="Pixels below this gray level may be considered black grid")
    parser.add_argument("--grid-line-fraction", type=float, default=0.35,
                        help="Minimum fraction of dark pixels in a row/column to consider it as grid")
    parser.add_argument("--grid-dilate", type=int, default=1,
                        help="Dilation in pixels of the detected grid mask")
    parser.add_argument("--debug-first-n", type=int, default=3,
                        help="Save the crop and mask of the first N slices for inspection")

    parser.add_argument("--dx", type=float, default=1.0,
                        help="Pixel size in x")
    parser.add_argument("--dy", type=float, default=1.0,
                        help="Pixel size in y")
    parser.add_argument("--dz", type=float, default=1.0,
                        help="Spacing between slices")
    parser.add_argument("--unit", type=str, default="px",
                        help="Spatial unit. Example: px, um, mm")

    parser.add_argument("--downsample", type=int, default=1,
                        help="Spatial downsampling factor. Example: 2 uses every second pixel")
    parser.add_argument("--connectivity", type=int, default=1,
                        choices=[1, 2, 3],
                        help="Connectivity 3D: 1=6, 2=18, 3=26")
    parser.add_argument("--mu", type=float, default=1.0,
                        help="Dynamic viscosity for the relative solver")
    parser.add_argument("--max-nodes", type=int, default=300000,
                        help="Maximum number of pore nodes in the permeability solver")
    parser.add_argument("--skip-permeability", action="store_true",
                        help="Skip permeability calculation by resistor network. Empirical/proxy methods can still be computed")
    parser.add_argument("--permeability-methods", nargs="+", default=["all"],
                        choices=["all", "resistor_network", "kozeny_carman_total", "kozeny_carman_connected", "capillary_local_radius", "capillary_throat_median", "capillary_throat_bottleneck"],
                        help="Permeability methods to save in permeability_methods.csv")
    parser.add_argument("--kozeny-c", type=float, default=5.0,
                        help="Kozeny--Carman constant C. Typical initial value: 5")
    parser.add_argument("--skip-percolation-gap", action="store_true",
                        help="Skip percolation-gap calculation by dilation. Useful if the volume is very large")
    parser.add_argument("--gap-max-radius", type=float, default=None,
                        help="Maximum physical radius tested for the percolation gap. Default: 25%% of the largest domain length")
    parser.add_argument("--gap-radius-step", type=float, default=None,
                        help="Physical dilation step for the percolation gap. Default: the smallest voxel dimension")
    parser.add_argument("--compute-graph-redundancy", action="store_true",
                        help="Compute redundancy by edge-disjoint paths via NetworkX. It can be very expensive; disabled by default")
    parser.add_argument("--graph-max-nodes", type=int, default=50000,
                        help="Maximum number of nodes for attempting graph redundancy")

    args = parser.parse_args()

    output_folder = ensure_output_folder(args.output)

    if args.npz_file:
        print("Reading .npz volume...")
        volume, files, used_rois = load_npz_volume(
            npz_file=args.npz_file,
            key=args.npz_key,
            order=args.npz_order,
            solid_is_true=args.npz_solid_is_true,
            downsample=args.downsample,
        )
    else:
        print("Reading images...")
        volume, files, used_rois = load_volume(
            folder=args.folder,
            prefix=args.prefix,
            ext=args.ext,
            strict_sequence=not args.no_strict_sequence,
            mode=args.mode,
            downsample=args.downsample,
            green_min=args.green_min,
            green_margin=args.green_margin,
            blue_min=args.blue_min,
            blue_margin=args.blue_margin,
            red_min=args.red_min,
            red_margin=args.red_margin,
            threshold=args.threshold,
            black_threshold=args.black_threshold,
            white_threshold=args.white_threshold,
            color_min=args.color_min,
            invert=args.invert,
            crop=args.crop,
            roi=args.roi,
            crop_padding=args.crop_padding,
            crop_bg_threshold=args.crop_bg_threshold,
            crop_trim_fraction=args.crop_trim_fraction,
            crop_each_slice=args.crop_each_slice,
            remove_grid=not args.keep_grid,
            grid_dark_threshold=args.grid_dark_threshold,
            grid_line_fraction=args.grid_line_fraction,
            grid_dilate=args.grid_dilate,
            debug_output_folder=output_folder,
            debug_first_n=args.debug_first_n,
        )

    print(f"Slices read: {len(files)}")
    print(f"First slice: {files[0].name}")
    print(f"Last slice: {files[-1].name}")
    if used_rois:
        print(f"ROI used na first slice x1,y1,x2,y2: {used_rois[0]}")
    print(f"Volume shape z,y,x: {volume.shape}")
    print(f"Initial pore fraction: {volume.mean():.6f}")
    print(f"Minimum pore fraction per slice: {volume.mean(axis=(1, 2)).min():.6f}")
    print(f"Maximum pore fraction per slice: {volume.mean(axis=(1, 2)).max():.6f}")
    if args.mode in ["auto", "nonblack", "color"] and float(volume.mean()) > 0.70:
        print("WARNING: the pore fraction is very high. If your pores are the blue phase and the matrix is red, run with: --mode blue")

    # Adjust scale if x/y downsampling was applied.
    dx = args.dx * args.downsample
    dy = args.dy * args.downsample
    dz = args.dz

    print("Analyzing slices 2D...")
    slice_df, pore_df = analyze_slices(
        volume,
        files,
        output_folder,
        dx=dx,
        dy=dy,
    )

    print("Labeling 3D volume...")
    labels, n_labels = label_volume(volume, connectivity=args.connectivity)

    print("Computing 3D properties...")
    volume_props = volume_cluster_stats(
        volume,
        labels,
        n_labels,
        dx=dx,
        dy=dy,
        dz=dz,
    )

    perc = percolation_3d(labels)

    print("Computing local-radius distribution...")
    radii_local, dist = distance_radius_distribution(
        volume,
        dx=dx,
        dy=dy,
        dz=dz,
    )

    if len(radii_local) > 0:
        radius_stats = {
            "local_radius_mean": float(np.mean(radii_local)),
            "local_radius_median": float(np.median(radii_local)),
            "local_radius_p10": float(np.percentile(radii_local, 10)),
            "local_radius_p25": float(np.percentile(radii_local, 25)),
            "local_radius_p75": float(np.percentile(radii_local, 75)),
            "local_radius_p90": float(np.percentile(radii_local, 90)),
            "local_radius_max": float(np.max(radii_local)),
        }
    else:
        radius_stats = {
            "local_radius_mean": 0.0,
            "local_radius_median": 0.0,
            "local_radius_p10": 0.0,
            "local_radius_p25": 0.0,
            "local_radius_p75": 0.0,
            "local_radius_p90": 0.0,
            "local_radius_max": 0.0,
        }

    print("Fitting approximate power law...")
    fractal = fit_power_law_tail(radii_local, min_percentile=50)

    print("Computing approximate tortuosity...")
    tort = tortuosity_estimates(volume, dx=dx, dy=dy, dz=dz)

    if args.skip_permeability:
        print("Skipping permeability relativa (--skip-permeability)...")
        perm_df = pd.DataFrame([
            {
                "direction": direction,
                "percolates": bool(perc[f"percolates_{direction}_3d"]),
                "n_solver_nodes": 0,
                "Q": np.nan,
                "k_relative": np.nan,
                "status": "not computed due to --skip-permeability",
            }
            for direction in ["x", "y", "z"]
        ])
        perm_df.to_csv(output_folder / "permeability.csv", index=False)
    else:
        print("Computing permeability relativa...")
        perm_df = permeability_analysis(
            volume,
            labels,
            dist,
            output_folder,
            dx=dx,
            dy=dy,
            dz=dz,
            mu=args.mu,
            max_nodes=args.max_nodes,
        )

    print("Computing permeability by multiple methods...")
    perm_methods_df, perm_methods_summary_df, perm_method_consistency_df, perm_method_signature = permeability_methods_analysis(
        volume,
        labels,
        dist,
        perc,
        tort,
        perm_df,
        output_folder,
        dx=dx,
        dy=dy,
        dz=dz,
        c_kozeny=args.kozeny_c,
        selected_methods=args.permeability_methods,
    )

    print("Computing morphohydraulic indices...")
    morph_df, morph_summary_df, morph_signature = morphohydraulic_indices(
        volume,
        labels,
        dist,
        perc,
        tort,
        perm_df,
        output_folder,
        dx=dx,
        dy=dy,
        dz=dz,
        connectivity=args.connectivity,
        compute_gap=not args.skip_percolation_gap,
        gap_max_radius=args.gap_max_radius,
        gap_radius_step=args.gap_radius_step,
        compute_graph_redundancy=args.compute_graph_redundancy,
        graph_max_nodes=args.graph_max_nodes,
    )

    all_volume_props = {}
    all_volume_props.update(volume_props)
    all_volume_props.update(perc)
    all_volume_props.update(radius_stats)
    all_volume_props.update(fractal)
    all_volume_props.update(tort)
    all_volume_props["shape_z_y_x"] = list(volume.shape)
    all_volume_props["n_slices"] = len(files)
    all_volume_props["dx"] = dx
    all_volume_props["dy"] = dy
    all_volume_props["dz"] = dz
    all_volume_props["unit"] = args.unit
    all_volume_props["connectivity"] = args.connectivity
    all_volume_props["segmentation_mode"] = args.mode
    all_volume_props["invert_mask"] = bool(args.invert)
    all_volume_props["crop_mode"] = args.crop
    all_volume_props["crop_roi_first_slice_x1_y1_x2_y2"] = list(used_rois[0]) if used_rois and used_rois[0] is not None else None
    all_volume_props["crop_each_slice"] = bool(args.crop_each_slice)
    all_volume_props["skip_permeability"] = bool(args.skip_permeability)
    all_volume_props["npz_file"] = args.npz_file
    all_volume_props["npz_key"] = args.npz_key if args.npz_file else ""
    all_volume_props["npz_order"] = args.npz_order if args.npz_file else ""
    all_volume_props["permeability_methods"] = ";".join(args.permeability_methods)
    all_volume_props["kozeny_c"] = args.kozeny_c
    all_volume_props["skip_percolation_gap"] = bool(args.skip_percolation_gap)
    all_volume_props["compute_graph_redundancy"] = bool(args.compute_graph_redundancy)
    for col in morph_summary_df.columns:
        all_volume_props[col] = morph_summary_df.iloc[0][col]
    if perm_method_consistency_df is not None and not perm_method_consistency_df.empty:
        for _, row in perm_method_consistency_df.iterrows():
            d = row.get("direction")
            if d in ["x", "y", "z"]:
                all_volume_props[f"recommended_k_{d}_mD"] = row.get("recommended_k_mD", np.nan)
                all_volume_props[f"recommended_k_{d}_method"] = row.get("recommended_method", "")
                all_volume_props[f"k_method_spread_{d}"] = row.get("method_spread_max_over_min", np.nan)

    volume_df = pd.DataFrame([all_volume_props])
    volume_df.to_csv(output_folder / "volume_properties.csv", index=False)
    save_json(output_folder / "volume_properties.json", all_volume_props)

    print("Generating plots...")
    plot_slice_porosity(slice_df, output_folder)
    plot_pore_radius_distribution(pore_df, output_folder)
    plot_local_radius_distribution(radii_local, output_folder)
    plot_middle_slices(volume, output_folder)
    plot_mip(volume, output_folder)

    print("Creating Markdown report...")
    create_markdown_report(
        output_folder,
        files,
        slice_df,
        volume_props,
        perc,
        tort,
        fractal,
        perm_df,
        morph_df=morph_df,
        morph_summary_df=morph_summary_df,
        unit=args.unit,
    )

    print("\nDone.")
    print(f"Results saved in: {output_folder.resolve()}")
    print("\nQuick summary:")
    print(f"  3D porosity: {volume_props['porosity_3d']:.6f}")
    print(f"  Clusters 3D: {volume_props['n_clusters_3d']}")
    print(f"  Largest cluster/pore fraction: {volume_props['largest_cluster_fraction_of_pores']:.6f}")
    print(f"  Percolates x: {perc['percolates_x_3d']}")
    print(f"  Percolates y: {perc['percolates_y_3d']}")
    print(f"  Percolates z: {perc['percolates_z_3d']}")
    print("\nRelative permeability:")
    print(perm_df.to_string(index=False))
    print("\nPermeability by multiple methods:")
    cols_methods = ["method", "direction", "percolates", "k_mD", "status"]
    cols_methods = [c for c in cols_methods if c in perm_methods_df.columns]
    print(perm_methods_df[cols_methods].to_string(index=False))

    print("\nMain morphohydraulic indices:")
    cols_to_show = [
        "direction", "phi_connected", "connectivity_efficiency", "phi_dead",
        "transportability", "hydraulic_transportability", "permeability_yield",
        "percolation_gap", "cluster_redundancy", "flux_entropy",
        "flux_participation", "bottleneck_index"
    ]
    cols_to_show = [c for c in cols_to_show if c in morph_df.columns]
    print(morph_df[cols_to_show].to_string(index=False))


if __name__ == "__main__":
    main()
