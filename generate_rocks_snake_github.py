#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Snake-based 3D synthetic rock generator with target porosity and permeability.

The model generates:
    - binary 3D volume in .npz format
    - clean PNG slices inside structures/ for subsequent analysis
    - visual slices with titles and colorbars inside figures/
    - 0/1 CSV slices inside csv/
    - metrics in JSON/CSV/TXT format

Saved-slice convention:
    0 = pore  = blue
    1 = solid = red

Important note:
    The target permeability is calibrated using the capillary snake model
    based on Hagen-Poiseuille + Darcy. Final validation is recommended using
    a resistor-network analysis script or an LBM/CFD solver.
"""

import os
import csv
import math
import json
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.ndimage import (
    gaussian_filter,
    gaussian_filter1d,
    label,
    generate_binary_structure,
)


# ============================================================
# MAIN PARAMETERS
# ============================================================

PARAMS = {

    # ========================================================
    # ROCK TARGETS
    # ========================================================

    # Target porosity.
    # Example: 0.38 = 38% of the total volume is pore space.
    "TARGET_POROSITY": 0.18,

    # Target permeability in millidarcy.
    # Examples:
    # 100    = low/intermediate
    # 1000   = high
    # 10000  = very high
    "TARGET_K_MD": 200.0,

    # Direction along which the target permeability will be calibrated.
    # It can be "x", "y", or "z".
    "TARGET_K_DIRECTION": "z",

    # Acceptable relative tolerance.
    # 0.20 accepts an error up to 20% relative to the target K.
    "K_REL_TOLERANCE": 0.20,

    # Maximum number of attempts.
    "MAX_ATTEMPTS": 30,


    # ========================================================
    # 3D DOMAIN
    # ========================================================

    # Volume size in voxels.
    # Use 128 for a quick test.
    # Use 192 or 256 for higher visual quality.
    "NX": 160,
    "NY": 160,
    "NZ": 160,

    # Physical resolution.
    # 5e-6 = 5 micrometers per voxel.
    "VOXEL_SIZE": 5e-6,


    # ========================================================
    # GAUSSIAN BACKGROUND TEXTURE
    # ========================================================

    # Gaussian-field scales.
    # Small scales = fine roughness.
    # Large scales = larger pore patches.
    "GAUSSIAN_SCALES": (2, 5, 12, 25),

    # Weight of each scale above.
    "GAUSSIAN_WEIGHTS": (0.40, 0.30, 0.20, 0.10),


    # ========================================================
    # SHORT SNAKES
    # ========================================================

    # Short snakes improve the natural-looking texture.
    "N_SHORT_SNAKES": 100,

    # Mean length of short snakes, in voxels.
    "MEAN_SNAKE_LENGTH": 50,

    # Length variation.
    "SNAKE_LENGTH_STD": 20,


    # ========================================================
    # PERCOLATING SNAKES
    # ========================================================

    # These snakes cross the sample along the target-K direction.
    # They help ensure nonzero permeability in that direction.
    "N_PERCOLATING_SNAKES": 1,

    # Base mean radii of the snakes, in voxels.
    # The code multiplies these radii by RADIUS_SCALE to try to reach target K.
    "BASE_MEAN_RADII": (1.5, 3.0, 6.0),

    # Probability of each radius above.
    "RADIUS_PROBABILITIES": (0.60, 0.30, 0.10),

    # Radius variation along the snake.
    # 0.10 = almost a uniform tube.
    # 0.35 = natural throats and expansions.
    # 0.60 = highly irregular.
    "RADIUS_COEFF_VARIATION": 0.35,

    # Tortuosity.
    # Larger values = more tortuous channels.
    "TORTUOSITY": 1.8,

    # Path smoothing.
    # Larger values = smoother paths.
    "PATH_SMOOTHING": 8,

    # Allowed range for multiplying the radii.
    # Permeability increases approximately with radius^4.
    "RADIUS_SCALE_MIN": 0.25,
    "RADIUS_SCALE_MAX": 3.50,


    # ========================================================
    # OUTPUTS
    # ========================================================

    # Number of XY slices to save.
    "N_XY_SLICES": 160,

    # Output folder.
    "OUTPUT_FOLDER": "target_snake_rock_output",

    # Base seed.
    # Change it to generate another rock with the same targets.
    "SEED": 42,
}


# ============================================================
# CONSTANTS AND CONVERSIONS
# ============================================================

M2_PER_MD = 9.869233e-16


def m2_to_mD(k_m2):
    return k_m2 / M2_PER_MD


def mD_to_m2(k_mD):
    return k_mD * M2_PER_MD


# ============================================================
# GAUSSIAN FIELD
# ============================================================

def normalize_field(field):
    field = field - field.min()
    maximum = field.max()
    if maximum > 0:
        field = field / maximum
    return field


def generate_gaussian_field(shape, scales, weights, rng):
    field = np.zeros(shape, dtype=np.float32)

    for sigma, weight in zip(scales, weights):
        noise = rng.normal(0, 1, shape)
        field += weight * gaussian_filter(noise, sigma=sigma)

    return normalize_field(field)


# ============================================================
# SNAKE GEOMETRY
# ============================================================

def axis_to_index(direction):
    if direction == "x":
        return 0
    if direction == "y":
        return 1
    if direction == "z":
        return 2
    raise ValueError("TARGET_K_DIRECTION must be 'x', 'y', or 'z'.")


def random_3d_direction(rng):
    v = rng.normal(0, 1, 3)
    n = np.linalg.norm(v)
    if n == 0:
        return np.array([1.0, 0.0, 0.0])
    return v / n


def generate_variable_radii(n, mean_radius, coeff_variation, rng, smoothing=5):
    sigma = np.sqrt(np.log(1 + coeff_variation ** 2))
    mu = np.log(mean_radius) - 0.5 * sigma ** 2

    radii = rng.lognormal(mean=mu, sigma=sigma, size=n)
    radii = gaussian_filter1d(radii, sigma=smoothing)
    radii = np.clip(radii, 0.8, None)

    return radii


def generate_percolating_snake(shape, params, rng):
    """
    Generates a snake crossing the sample along the target direction.
    """

    nx, ny, nz = shape
    direction = params["TARGET_K_DIRECTION"]
    axis = axis_to_index(direction)

    dims = np.array([nx, ny, nz], dtype=int)
    n = int(dims[axis])

    coords = np.zeros((n, 3), dtype=float)

    # The main coordinate crosses the entire sample.
    coords[:, axis] = np.arange(n)

    # Transverse axes.
    transverse_axes = [i for i in range(3) if i != axis]

    margin = max(params["BASE_MEAN_RADII"]) * params["RADIUS_SCALE_MAX"] + 4

    for ax in transverse_axes:
        start = rng.uniform(0.25 * dims[ax], 0.75 * dims[ax])
        walk = np.cumsum(rng.normal(0, params["TORTUOSITY"], size=n))

        curve = start + walk
        curve = gaussian_filter1d(curve, sigma=params["PATH_SMOOTHING"])
        curve = np.clip(curve, margin, dims[ax] - margin - 1)
        coords[:, ax] = curve

    mean_radius = rng.choice(
        params["BASE_MEAN_RADII"],
        p=params["RADIUS_PROBABILITIES"],
    )

    radii = generate_variable_radii(
        n=n,
        mean_radius=mean_radius,
        coeff_variation=params["RADIUS_COEFF_VARIATION"],
        rng=rng,
    )

    return {
        "type": "percolating",
        "points": coords,
        "base_radii": radii,
        "base_mean_radius": mean_radius,
    }


def generate_short_snake(shape, params, rng):
    """
    Generates a short snake with a random direction.
    """

    nx, ny, nz = shape
    dims = np.array([nx, ny, nz], dtype=float)

    length = int(
        rng.normal(
            params["MEAN_SNAKE_LENGTH"],
            params["SNAKE_LENGTH_STD"],
        )
    )
    length = max(8, length)

    margin = max(params["BASE_MEAN_RADII"]) * params["RADIUS_SCALE_MAX"] + 4

    p = np.array(
        [
            rng.uniform(margin, nx - margin - 1),
            rng.uniform(margin, ny - margin - 1),
            rng.uniform(margin, nz - margin - 1),
        ],
        dtype=float,
    )

    direction = random_3d_direction(rng)
    points = []

    for _ in range(length):
        noise = rng.normal(0, params["TORTUOSITY"], 3)
        step = direction + 0.25 * noise

        norm = np.linalg.norm(step)
        if norm == 0:
            step = direction
        else:
            step = step / norm

        p = p + step

        for ax in range(3):
            if p[ax] < margin:
                p[ax] = margin
                direction[ax] *= -1
            if p[ax] > dims[ax] - margin - 1:
                p[ax] = dims[ax] - margin - 1
                direction[ax] *= -1

        points.append(p.copy())

    points = np.array(points)

    for ax in range(3):
        points[:, ax] = gaussian_filter1d(
            points[:, ax],
            sigma=params["PATH_SMOOTHING"],
        )

    mean_radius = rng.choice(
        params["BASE_MEAN_RADII"],
        p=params["RADIUS_PROBABILITIES"],
    )

    radii = generate_variable_radii(
        n=len(points),
        mean_radius=mean_radius,
        coeff_variation=params["RADIUS_COEFF_VARIATION"],
        rng=rng,
    )

    return {
        "type": "short",
        "points": points,
        "base_radii": radii,
        "base_mean_radius": mean_radius,
    }


def generate_snake_list(shape, params, rng):
    snakes = []

    for _ in range(params["N_PERCOLATING_SNAKES"]):
        snakes.append(generate_percolating_snake(shape, params, rng))

    for _ in range(params["N_SHORT_SNAKES"]):
        snakes.append(generate_short_snake(shape, params, rng))

    return snakes


# ============================================================
# SNAKE RASTERIZATION
# ============================================================

def add_sphere(volume, center, radius):
    nx, ny, nz = volume.shape

    cx, cy, cz = center
    r = int(np.ceil(radius))

    x1 = max(0, int(cx) - r)
    x2 = min(nx, int(cx) + r + 1)

    y1 = max(0, int(cy) - r)
    y2 = min(ny, int(cy) + r + 1)

    z1 = max(0, int(cz) - r)
    z2 = min(nz, int(cz) + r + 1)

    xs = np.arange(x1, x2)
    ys = np.arange(y1, y2)
    zs = np.arange(z1, z2)

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")

    mask = (
        (xx - cx) ** 2
        + (yy - cy) ** 2
        + (zz - cz) ** 2
    ) <= radius ** 2

    volume[x1:x2, y1:y2, z1:z2][mask] = True


def rasterize_snakes(shape, snakes, radius_scale):
    volume = np.zeros(shape, dtype=bool)

    for snake in snakes:
        points = snake["points"]
        radii = snake["base_radii"] * radius_scale

        for p, r in zip(points, radii):
            add_sphere(volume, p, r)

    return volume


# ============================================================
# PERMEABILITY FROM THE CAPILLARY SNAKE MODEL
# ============================================================

def capillary_snake_permeability(snakes, radius_scale, shape, voxel_size, direction):
    """
    Estimates permeability using only the percolating snakes.

    Model:
        each percolating snake is a tortuous tube;
        conductance follows Hagen-Poiseuille;
        k is obtained from Darcy's law.

    Result:
        k in m².

    Caution:
        this is a geometric estimate, not CFD.
    """

    axis = axis_to_index(direction)

    dims = np.array(shape)
    L = (dims[axis] - 1) * voxel_size

    transverse_axes = [i for i in range(3) if i != axis]
    A = dims[transverse_axes[0]] * voxel_size * dims[transverse_axes[1]] * voxel_size

    conductance_sum = 0.0

    for snake in snakes:
        if snake["type"] != "percolating":
            continue

        points = snake["points"]
        radii_m = snake["base_radii"] * radius_scale * voxel_size

        dif = np.diff(points, axis=0) * voxel_size
        ds = np.sqrt(np.sum(dif ** 2, axis=1))

        # Segment radii.
        r_seg = 0.5 * (radii_m[:-1] + radii_m[1:])
        r_seg = np.clip(r_seg, 0.25 * voxel_size, None)

        integral = np.sum(ds / (r_seg ** 4))

        if integral <= 0:
            continue

        # Geometric conductance without viscosity.
        C = math.pi / (8.0 * integral)
        conductance_sum += C

    if conductance_sum <= 0 or A <= 0 or L <= 0:
        return 0.0

    k = conductance_sum * L / A
    return k


# ============================================================
# EXACT POROSITY MATCHING
# ============================================================

def combine_snakes_with_texture_for_phi(snake_mask, field, target_porosity):
    """
    Creates the final pore space:
        pores = snake_mask OR gaussian_field_pores

    The number of pores selected from the field is chosen to exactly match
    the target porosity whenever possible.
    """

    total = snake_mask.size
    target_voxels = int(round(target_porosity * total))

    snake_voxels = int(snake_mask.sum())

    if snake_voxels > target_voxels:
        return None, {
            "status": "snakes_alone_exceed_target_porosity",
            "snake_phi": snake_voxels / total,
            "phi_final": snake_voxels / total,
        }

    missing = target_voxels - snake_voxels
    pores = snake_mask.copy()

    if missing == 0:
        return pores, {
            "status": "ok",
            "snake_phi": snake_voxels / total,
            "phi_final": pores.mean(),
        }

    candidates = np.flatnonzero(~snake_mask.ravel())

    if missing > len(candidates):
        return None, {
            "status": "not_enough_available_voxels",
            "snake_phi": snake_voxels / total,
            "phi_final": snake_voxels / total,
        }

    field_flat = field.ravel()
    chosen_rel = np.argpartition(field_flat[candidates], missing - 1)[:missing]
    chosen = candidates[chosen_rel]

    pores_flat = pores.ravel()
    pores_flat[chosen] = True
    pores = pores_flat.reshape(snake_mask.shape)

    return pores, {
        "status": "ok",
        "snake_phi": snake_voxels / total,
        "phi_final": pores.mean(),
    }


# ============================================================
# PERCOLATION
# ============================================================

def check_3d_percolation(pores):
    structure = generate_binary_structure(3, 1)
    labels, n = label(pores, structure=structure)

    if n == 0:
        return {
            "n_clusters": 0,
            "largest_cluster_fraction_of_pores": 0.0,
            "percolates_x": False,
            "percolates_y": False,
            "percolates_z": False,
        }

    counts = np.bincount(labels.ravel())
    counts[0] = 0

    largest = counts.max()
    total_pores = pores.sum()

    def axis_percolates(axis):
        if axis == 0:
            a = set(np.unique(labels[0, :, :])) - {0}
            b = set(np.unique(labels[-1, :, :])) - {0}
        elif axis == 1:
            a = set(np.unique(labels[:, 0, :])) - {0}
            b = set(np.unique(labels[:, -1, :])) - {0}
        else:
            a = set(np.unique(labels[:, :, 0])) - {0}
            b = set(np.unique(labels[:, :, -1])) - {0}

        return len(a.intersection(b)) > 0

    return {
        "n_clusters": int(n),
        "largest_cluster_fraction_of_pores": float(largest / total_pores),
        "percolates_x": bool(axis_percolates(0)),
        "percolates_y": bool(axis_percolates(1)),
        "percolates_z": bool(axis_percolates(2)),
    }


# ============================================================
# GENERATION WITH CALIBRATION
# ============================================================

def generate_calibrated_rock(params):
    shape = (params["NX"], params["NY"], params["NZ"])
    rng_global = np.random.default_rng(params["SEED"])

    target_k_m2 = mD_to_m2(params["TARGET_K_MD"])

    best = None
    best_score = np.inf

    for attempt in range(1, params["MAX_ATTEMPTS"] + 1):

        seed = int(rng_global.integers(0, 1_000_000_000))
        rng = np.random.default_rng(seed)

        snakes = generate_snake_list(shape, params, rng)

        # K for radius_scale = 1.
        base_k = capillary_snake_permeability(
            snakes=snakes,
            radius_scale=1.0,
            shape=shape,
            voxel_size=params["VOXEL_SIZE"],
            direction=params["TARGET_K_DIRECTION"],
        )

        if base_k <= 0:
            continue

        # Since k ~ r^4, the radius factor is estimated directly.
        ideal_radius_scale = (target_k_m2 / base_k) ** 0.25

        radius_scale = float(
            np.clip(
                ideal_radius_scale,
                params["RADIUS_SCALE_MIN"],
                params["RADIUS_SCALE_MAX"],
            )
        )

        snake_mask = rasterize_snakes(
            shape=shape,
            snakes=snakes,
            radius_scale=radius_scale,
        )

        field = generate_gaussian_field(
            shape=shape,
            scales=params["GAUSSIAN_SCALES"],
            weights=params["GAUSSIAN_WEIGHTS"],
            rng=rng,
        )

        pores, phi_info = combine_snakes_with_texture_for_phi(
            snake_mask=snake_mask,
            field=field,
            target_porosity=params["TARGET_POROSITY"],
        )

        if pores is None:
            print(
                f"Attempt {attempt:03d} | "
                f"failed: {phi_info['status']} | "
                f"snake_phi={phi_info['snake_phi']:.4f}"
            )
            continue

        phi = pores.mean()

        k_m2 = capillary_snake_permeability(
            snakes=snakes,
            radius_scale=radius_scale,
            shape=shape,
            voxel_size=params["VOXEL_SIZE"],
            direction=params["TARGET_K_DIRECTION"],
        )

        k_mD = m2_to_mD(k_m2)

        k_rel_error = abs(k_mD - params["TARGET_K_MD"]) / params["TARGET_K_MD"]
        phi_rel_error = abs(phi - params["TARGET_POROSITY"]) / params["TARGET_POROSITY"]

        if k_mD > 0:
            log_k_error = abs(np.log10(k_mD / params["TARGET_K_MD"]))
        else:
            log_k_error = np.inf

        score = log_k_error + 3.0 * phi_rel_error
        percolation = check_3d_percolation(pores)

        print(
            f"Attempt {attempt:03d} | "
            f"phi={phi:.5f} | "
            f"k={k_mD:.2f} mD | "
            f"k_error={100 * k_rel_error:.2f}% | "
            f"radius_scale={radius_scale:.4f} | "
            f"percolates x/y/z="
            f"{percolation['percolates_x']}/"
            f"{percolation['percolates_y']}/"
            f"{percolation['percolates_z']}"
        )

        result = {
            "pores": pores,
            "field": field,
            "snake_mask": snake_mask,
            "snakes": snakes,
            "phi": phi,
            "k_m2": k_m2,
            "k_mD": k_mD,
            "k_rel_error": k_rel_error,
            "radius_scale": radius_scale,
            "seed": seed,
            "percolation": percolation,
            "score": score,
        }

        if score < best_score:
            best_score = score
            best = result

        if k_rel_error <= params["K_REL_TOLERANCE"] and phi_rel_error <= 0.01:
            print("Target reached within tolerance.")
            break

    if best is None:
        raise RuntimeError("Could not generate a valid rock. Adjust the parameters.")

    return best


# ============================================================
# OUTPUTS
# ============================================================

def save_binary_rgb_image(data, path):
    """
    Saves an image without axes/colorbar for automatic analysis.

    Convention:
        data = 0 -> pore  -> blue
        data = 1 -> solid -> red
    """

    h, w = data.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Blue pore phase.
    rgb[data == 0] = [0, 0, 255]

    # Red solid phase.
    rgb[data == 1] = [255, 0, 0]

    Image.fromarray(rgb).save(path)


def save_xy_slices(pores, params, output_folder):
    """
    Saves:
        1) structures/1.png, 2.png, ...
           Clean images, without axes, for analysis.
        2) figures/slice_XY_*.png
           Figures with title and colorbar.
        3) csv/slice_XY_*.csv
           0/1 matrices.

    Convention:
        0 = pore
        1 = solid
    """

    nx, ny, nz = pores.shape

    structures_folder = os.path.join(output_folder, "structures")
    figures_folder = os.path.join(output_folder, "figures")
    csv_folder = os.path.join(output_folder, "csv")

    os.makedirs(structures_folder, exist_ok=True)
    os.makedirs(figures_folder, exist_ok=True)
    os.makedirs(csv_folder, exist_ok=True)

    z_indices = np.linspace(0, nz - 1, params["N_XY_SLICES"]).round().astype(int)

    for i, z in enumerate(z_indices, start=1):
        slice_xy = pores[:, :, z]

        # 0 = pore, 1 = solid.
        data = (~slice_xy).astype(np.uint8)
        slice_phi = np.mean(data == 0)

        # Clean image for analysis.
        # Transposed so that x is horizontal and y is vertical.
        save_binary_rgb_image(
            data.T,
            os.path.join(structures_folder, f"{i}.png"),
        )

        # CSV.
        np.savetxt(
            os.path.join(csv_folder, f"{i}.csv"),
            data.T,
            fmt="%d",
            delimiter=",",
        )

        # Visual figure.
        plt.figure(figsize=(7, 7), dpi=140)
        plt.imshow(
            data.T,
            origin="lower",
            cmap="jet",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            aspect="equal",
        )

        plt.title(f"data - XY z={z} | phi={slice_phi:.3f}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True, color="black", linewidth=0.35, alpha=0.35)

        cbar = plt.colorbar(
            orientation="horizontal",
            pad=0.10,
            shrink=0.75,
            extend="both",
        )
        cbar.set_label("data ()")
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

        plt.savefig(
            os.path.join(figures_folder, f"slice_XY_{i:02d}_z_{z:03d}.png"),
            bbox_inches="tight",
        )
        plt.close()


def save_metrics(result, params, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    percolation = result["percolation"]

    metrics = {
        "target_porosity": params["TARGET_POROSITY"],
        "obtained_porosity": result["phi"],
        "target_k_mD": params["TARGET_K_MD"],
        "snake_model_k_mD": result["k_mD"],
        "snake_model_k_m2": result["k_m2"],
        "relative_k_error": result["k_rel_error"],
        "target_k_direction": params["TARGET_K_DIRECTION"],
        "radius_scale": result["radius_scale"],
        "used_seed": result["seed"],
        "n_clusters": percolation["n_clusters"],
        "largest_cluster_fraction_of_pores": percolation["largest_cluster_fraction_of_pores"],
        "percolates_x": percolation["percolates_x"],
        "percolates_y": percolation["percolates_y"],
        "percolates_z": percolation["percolates_z"],
        "voxel_size_m": params["VOXEL_SIZE"],
        "nx": params["NX"],
        "ny": params["NY"],
        "nz": params["NZ"],
    }

    with open(os.path.join(output_folder, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_folder, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["property", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])

    with open(os.path.join(output_folder, "README_RESULT.txt"), "w", encoding="utf-8") as f:
        f.write("SYNTHETIC ROCK CALIBRATED BY THE SNAKE MODEL\n\n")
        f.write("Slice convention:\n")
        f.write("0 = pore  = blue\n")
        f.write("1 = solid = red\n\n")
        f.write(f"Target porosity: {params['TARGET_POROSITY']}\n")
        f.write(f"Obtained porosity: {result['phi']:.6f}\n")
        f.write(f"Target K: {params['TARGET_K_MD']:.6f} mD\n")
        f.write(f"K obtained by the snake model: {result['k_mD']:.6f} mD\n")
        f.write(f"K obtained by the snake model: {result['k_m2']:.6e} m²\n")
        f.write(f"Relative K error: {100 * result['k_rel_error']:.3f}%\n")
        f.write(f"Target direction: {params['TARGET_K_DIRECTION']}\n")
        f.write(f"Radius scale: {result['radius_scale']:.6f}\n\n")
        f.write("3D percolation:\n")
        f.write(f"Percolates x: {percolation['percolates_x']}\n")
        f.write(f"Percolates y: {percolation['percolates_y']}\n")
        f.write(f"Percolates z: {percolation['percolates_z']}\n")
        f.write(f"Number of clusters: {percolation['n_clusters']}\n")
        f.write(
            f"Largest cluster/fraction of pores: "
            f"{percolation['largest_cluster_fraction_of_pores']:.6f}\n\n"
        )
        f.write("Important note:\n")
        f.write(
            "The permeability used during calibration is estimated by the capillary snake model, "
            "based on Hagen-Poiseuille + Darcy. For final validation, run an analysis script "
            "using a resistor network or an LBM/CFD solver on the generated volume.\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    p = PARAMS

    output_folder = p["OUTPUT_FOLDER"]
    os.makedirs(output_folder, exist_ok=True)

    print("==============================================")
    print("SYNTHETIC ROCK GENERATOR WITH TARGET PHI AND K")
    print("==============================================")
    print(f"Target porosity: {p['TARGET_POROSITY']}")
    print(f"Target K: {p['TARGET_K_MD']} mD")
    print(f"Target K direction: {p['TARGET_K_DIRECTION']}")
    print(f"Volume: {p['NX']} x {p['NY']} x {p['NZ']}")
    print(f"Voxel size: {p['VOXEL_SIZE']} m")
    print()

    result = generate_calibrated_rock(p)
    pores = result["pores"]

    print()
    print("==============================================")
    print("BEST RESULT")
    print("==============================================")
    print(f"Obtained porosity: {result['phi']:.6f}")
    print(f"K obtained by snake model: {result['k_mD']:.6f} mD")
    print(f"K obtained by snake model: {result['k_m2']:.6e} m²")
    print(f"Relative K error: {100 * result['k_rel_error']:.3f}%")
    print(f"Radius scale: {result['radius_scale']:.6f}")
    print(f"Used seed: {result['seed']}")

    percolation = result["percolation"]
    print(f"3D clusters: {percolation['n_clusters']}")
    print(
        f"Largest cluster/fraction of pores: "
        f"{percolation['largest_cluster_fraction_of_pores']:.6f}"
    )
    print(f"Percolates x: {percolation['percolates_x']}")
    print(f"Percolates y: {percolation['percolates_y']}")
    print(f"Percolates z: {percolation['percolates_z']}")

    # Save the 3D volume.
    np.savez_compressed(
        os.path.join(output_folder, "calibrated_snake_rock.npz"),
        pores=pores,
        field=result["field"],
        snake_mask=result["snake_mask"],
        voxel_size=p["VOXEL_SIZE"],
        phi=result["phi"],
        snake_model_k_m2=result["k_m2"],
        snake_model_k_mD=result["k_mD"],
        radius_scale=result["radius_scale"],
        used_seed=result["seed"],
    )

    # Save the parameters used.
    with open(os.path.join(output_folder, "used_params.json"), "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2, ensure_ascii=False)

    save_metrics(result, p, output_folder)
    save_xy_slices(pores, p, output_folder)

    print()
    print("Files saved in:")
    print(os.path.abspath(output_folder))
    print()
    print("Clean slices for analysis are in:")
    print(os.path.abspath(os.path.join(output_folder, "structures")))
    print()
    print("To analyze with metrics_calculation.py, run for example:")
    print()
    print(
        f"python3 metrics_calculation.py "
        f"--folder {os.path.join(output_folder, 'structures')} "
        f"--mode blue "
        f"--output {os.path.join(output_folder, 'analysis')} "
        f"--dx {p['VOXEL_SIZE']} --dy {p['VOXEL_SIZE']} --dz {p['VOXEL_SIZE']}"
    )


if __name__ == "__main__":
    main()
