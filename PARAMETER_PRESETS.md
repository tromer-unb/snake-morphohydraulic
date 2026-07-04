# Parameter presets and safe analysis commands

This file provides practical generation presets and safe analysis commands for the Snake--MorphoHydraulic workflow.

The generator currently uses the `PARAMS` dictionary inside:

```text
generate_rocks_snake.py
```

To use one of the presets below, copy the corresponding `PARAMS.update({...})` block into `generate_rocks_snake.py` after the main `PARAMS` dictionary, or manually edit the values inside `PARAMS`.

The analysis commands are designed to avoid common large-system errors in the voxel-resistor permeability solver.

---

## Important note about solver size

The voxel-resistor permeability solver builds a linear system using the connected pore voxels. For large volumes or high connected porosity, the script may report:

```text
system too large for direct solver
```

This does **not** mean that the generated rock is invalid. It only means that the direct solver would require more memory than allowed by the current settings.

Recommended solutions:

```bash
--downsample 2
```

or, for larger/high-porosity rocks:

```bash
--downsample 3
```

You can also increase the solver limit:

```bash
--max-nodes 800000
```

or skip the direct resistor-network permeability calculation:

```bash
--skip-permeability
```

Always provide the physical voxel size during analysis:

```bash
--dx 5e-6 --dy 5e-6 --dz 5e-6 --unit m
```

If the physical voxel size is omitted, the code assumes a voxel size of `1.0`, which can lead to physically unrealistic permeability values.

---

# Preset 01 — Fast smoke test

Purpose: quick test to verify that the generator and analyzer run correctly.

This is the safest first case.

```python
PARAMS.update({
    "TARGET_POROSITY": 0.10,
    "TARGET_K_MD": 20.0,
    "TARGET_K_DIRECTION": "z",

    "NX": 64,
    "NY": 64,
    "NZ": 64,
    "VOXEL_SIZE": 5e-6,

    "N_SHORT_SNAKES": 25,
    "N_PERCOLATING_SNAKES": 1,
    "MEAN_SNAKE_LENGTH": 25,
    "SNAKE_LENGTH_STD": 10,

    "TORTUOSITY": 1.6,
    "PATH_SMOOTHING": 6,

    "OUTPUT_FOLDER": "outputs/01_fast_smoke_test",
    "SEED": 101,
})
```

Analysis:

```bash
python3 metrics_calculation.py \
  --npz-file outputs/01_fast_smoke_test/calibrated_snake_rock.npz \
  --npz-key pores \
  --npz-order xyz \
  --dx 5e-6 --dy 5e-6 --dz 5e-6 \
  --unit m \
  --downsample 1 \
  --max-nodes 300000 \
  --output outputs/01_fast_smoke_test/analysis
```

---

# Preset 02 — Low porosity / low permeability

Purpose: sparse pore space with a narrow preferential transport backbone.

```python
PARAMS.update({
    "TARGET_POROSITY": 0.10,
    "TARGET_K_MD": 20.0,
    "TARGET_K_DIRECTION": "z",

    "NX": 96,
    "NY": 96,
    "NZ": 96,
    "VOXEL_SIZE": 5e-6,

    "N_SHORT_SNAKES": 45,
    "N_PERCOLATING_SNAKES": 1,
    "MEAN_SNAKE_LENGTH": 35,
    "SNAKE_LENGTH_STD": 15,

    "BASE_MEAN_RADII": (1.2, 2.5, 5.0),
    "RADIUS_PROBABILITIES": (0.65, 0.25, 0.10),
    "RADIUS_COEFF_VARIATION": 0.30,

    "TORTUOSITY": 1.8,
    "PATH_SMOOTHING": 8,

    "OUTPUT_FOLDER": "outputs/02_low_phi_low_k",
    "SEED": 102,
})
```

Analysis:

```bash
python3 metrics_calculation.py \
  --npz-file outputs/02_low_phi_low_k/calibrated_snake_rock.npz \
  --npz-key pores \
  --npz-order xyz \
  --dx 5e-6 --dy 5e-6 --dz 5e-6 \
  --unit m \
  --downsample 1 \
  --max-nodes 500000 \
  --output outputs/02_low_phi_low_k/analysis
```

---

# Preset 03 — Low porosity / intermediate permeability

Purpose: low pore volume with a stronger calibrated hydraulic backbone.

```python
PARAMS.update({
    "TARGET_POROSITY": 0.10,
    "TARGET_K_MD": 200.0,
    "TARGET_K_DIRECTION": "z",

    "NX": 96,
    "NY": 96,
    "NZ": 96,
    "VOXEL_SIZE": 5e-6,

    "N_SHORT_SNAKES": 50,
    "N_PERCOLATING_SNAKES": 1,
    "MEAN_SNAKE_LENGTH": 40,
    "SNAKE_LENGTH_STD": 15,

    "BASE_MEAN_RADII": (1.5, 3.0, 6.0),
    "RADIUS_PROBABILITIES": (0.60, 0.30, 0.10),
    "RADIUS_COEFF_VARIATION": 0.35,

    "TORTUOSITY": 1.8,
    "PATH_SMOOTHING": 8,

    "OUTPUT_FOLDER": "outputs/03_low_phi_intermediate_k",
    "SEED": 103,
})
```

Analysis:

```bash
python3 metrics_calculation.py \
  --npz-file outputs/03_low_phi_intermediate_k/calibrated_snake_rock.npz \
  --npz-key pores \
  --npz-order xyz \
  --dx 5e-6 --dy 5e-6 --dz 5e-6 \
  --unit m \
  --downsample 1 \
  --max-nodes 600000 \
  --output outputs/03_low_phi_intermediate_k/analysis
```

---

# Preset 04 — Intermediate reference case

Purpose: balanced reference case for regular use.

This is close to the default manuscript-style case.

```python
PARAMS.update({
    "TARGET_POROSITY": 0.18,
    "TARGET_K_MD": 200.0,
    "TARGET_K_DIRECTION": "z",

    "NX": 160,
    "NY": 160,
    "NZ": 160,
    "VOXEL_SIZE": 5e-6,

    "N_SHORT_SNAKES": 100,
    "N_PERCOLATING_SNAKES": 1,
    "MEAN_SNAKE_LENGTH": 50,
    "SNAKE_LENGTH_STD": 20,

    "BASE_MEAN_RADII": (1.5, 3.0, 6.0),
    "RADIUS_PROBABILITIES": (0.60, 0.30, 0.10),
    "RADIUS_COEFF_VARIATION": 0.35,

    "TORTUOSITY": 1.8,
    "PATH_SMOOTHING": 8,

    "N_XY_SLICES": 160,
    "OUTPUT_FOLDER": "outputs/04_intermediate_reference",
    "SEED": 104,
})
```

Recommended analysis:

```bash
python3 metrics_calculation.py \
  --npz-file outputs/04_intermediate_reference/calibrated_snake_rock.npz \
  --npz-key pores \
  --npz-order xyz \
  --dx 5e-6 --dy 5e-6 --dz 5e-6 \
  --unit m \
  --downsample 2 \
  --max-nodes 800000 \
  --output outputs/04_intermediate_reference/analysis_downsample2
```

Fast morphology-only analysis:

```bash
python3 metrics_calculation.py \
  --npz-file outputs/04_intermediate_reference/calibrated_snake_rock.npz \
  --npz-key pores \
  --npz-order xyz \
  --dx 5e-6 --dy 5e-6 --dz 5e-6 \
  --unit m \
  --skip-permeability \
  --output outputs/04_intermediate_reference/analysis_fast_no_solver
```

---

# Preset 05 — Intermediate porosity / high permeability

Purpose: same porosity range as the reference case, but with a stronger hydraulic backbone.

```python
PARAMS.update({
    "TARGET_POROSITY": 0.18,
    "TARGET_K_MD": 1000.0,
    "TARGET_K_DIRECTION": "z",

    "NX": 160,
    "NY": 160,
    "NZ": 160,
    "VOXEL_SIZE": 5e-6,

    "N_SHORT_SNAKES": 110,
    "N_PERCOLATING_SNAKES": 1,
    "MEAN_SNAKE_LENGTH": 55,
    "SNAKE_LENGTH_STD": 20,

    "BASE_MEAN_RADII": (1.5, 3.0, 6.0),
    "RADIUS_PROBABILITIES": (0.55, 0.30, 0.15),
    "RADIUS_COEFF_VARIATION": 0.35,

    "TORTUOSITY": 1.8,
    "PATH_SMOOTHING": 8,

    "N_XY_SLICES": 160,
    "OUTPUT_FOLDER": "outputs/05_intermediate_phi_high_k",
    "SEED": 105,
})
```

Analysis:

```bash
python3 metrics_calculation.py \
  --npz-file outputs/05_intermediate_phi_high_k/calibrated_snake_rock.npz \
  --npz-key pores \
  --npz-order xyz \
  --dx 5e-6 --dy 5e-6 --dz 5e-6 \
  --unit m \
  --downsample 2 \
  --max-nodes 800000 \
  --output outputs/05_intermediate_phi_high_k/analysis_downsample2
```

---

# Preset 06 — High porosity / high permeability

Purpose: highly connected pore network with strong hydraulic transport.

Because this case may generate a large connected cluster, use `--downsample 3` during analysis.

```python
PARAMS.update({
    "TARGET_POROSITY": 0.28,
    "TARGET_K_MD": 2000.0,
    "TARGET_K_DIRECTION": "z",

    "NX": 160,
    "NY": 160,
    "NZ": 160,
    "VOXEL_SIZE": 5e-6,

    "N_SHORT_SNAKES": 130,
    "N_PERCOLATING_SNAKES": 2,
    "MEAN_SNAKE_LENGTH": 60,
    "SNAKE_LENGTH_STD": 25,

    "BASE_MEAN_RADII": (1.5, 3.0, 6.0),
    "RADIUS_PROBABILITIES": (0.50, 0.35, 0.15),
    "RADIUS_COEFF_VARIATION": 0.40,

    "TORTUOSITY": 1.8,
    "PATH_SMOOTHING": 8,

    "N_XY_SLICES": 160,
    "OUTPUT_FOLDER": "outputs/06_high_phi_high_k",
    "SEED": 106,
})
```

Analysis:

```bash
python3 metrics_calculation.py \
  --npz-file outputs/06_high_phi_high_k/calibrated_snake_rock.npz \
  --npz-key pores \
  --npz-order xyz \
  --dx 5e-6 --dy 5e-6 --dz 5e-6 \
  --unit m \
  --downsample 3 \
  --max-nodes 1000000 \
  --output outputs/06_high_phi_high_k/analysis_downsample3
```

---

# Preset 07 — X-directed permeability

Purpose: generate an anisotropic rock calibrated along the x direction.

```python
PARAMS.update({
    "TARGET_POROSITY": 0.18,
    "TARGET_K_MD": 500.0,
    "TARGET_K_DIRECTION": "x",

    "NX": 128,
    "NY": 128,
    "NZ": 128,
    "VOXEL_SIZE": 5e-6,

    "N_SHORT_SNAKES": 80,
    "N_PERCOLATING_SNAKES": 1,
    "MEAN_SNAKE_LENGTH": 45,
    "SNAKE_LENGTH_STD": 18,

    "BASE_MEAN_RADII": (1.5, 3.0, 6.0),
    "RADIUS_PROBABILITIES": (0.60, 0.30, 0.10),
    "RADIUS_COEFF_VARIATION": 0.35,

    "TORTUOSITY": 2.0,
    "PATH_SMOOTHING": 8,

    "N_XY_SLICES": 128,
    "OUTPUT_FOLDER": "outputs/07_x_directed_k",
    "SEED": 107,
})
```

Analysis:

```bash
python3 metrics_calculation.py \
  --npz-file outputs/07_x_directed_k/calibrated_snake_rock.npz \
  --npz-key pores \
  --npz-order xyz \
  --dx 5e-6 --dy 5e-6 --dz 5e-6 \
  --unit m \
  --downsample 2 \
  --max-nodes 800000 \
  --output outputs/07_x_directed_k/analysis_downsample2
```

---

# Preset 08 — Y-directed permeability

Purpose: generate an anisotropic rock calibrated along the y direction.

```python
PARAMS.update({
    "TARGET_POROSITY": 0.18,
    "TARGET_K_MD": 500.0,
    "TARGET_K_DIRECTION": "y",

    "NX": 128,
    "NY": 128,
    "NZ": 128,
    "VOXEL_SIZE": 5e-6,

    "N_SHORT_SNAKES": 80,
    "N_PERCOLATING_SNAKES": 1,
    "MEAN_SNAKE_LENGTH": 45,
    "SNAKE_LENGTH_STD": 18,

    "BASE_MEAN_RADII": (1.5, 3.0, 6.0),
    "RADIUS_PROBABILITIES": (0.60, 0.30, 0.10),
    "RADIUS_COEFF_VARIATION": 0.35,

    "TORTUOSITY": 2.0,
    "PATH_SMOOTHING": 8,

    "N_XY_SLICES": 128,
    "OUTPUT_FOLDER": "outputs/08_y_directed_k",
    "SEED": 108,
})
```

Analysis:

```bash
python3 metrics_calculation.py \
  --npz-file outputs/08_y_directed_k/calibrated_snake_rock.npz \
  --npz-key pores \
  --npz-order xyz \
  --dx 5e-6 --dy 5e-6 --dz 5e-6 \
  --unit m \
  --downsample 2 \
  --max-nodes 800000 \
  --output outputs/08_y_directed_k/analysis_downsample2
```

---

# Preset 09 — High-resolution publication case

Purpose: visually cleaner digital rocks for figures.

This case is heavier. Use downsampling during analysis.

```python
PARAMS.update({
    "TARGET_POROSITY": 0.18,
    "TARGET_K_MD": 200.0,
    "TARGET_K_DIRECTION": "z",

    "NX": 192,
    "NY": 192,
    "NZ": 192,
    "VOXEL_SIZE": 5e-6,

    "N_SHORT_SNAKES": 140,
    "N_PERCOLATING_SNAKES": 1,
    "MEAN_SNAKE_LENGTH": 65,
    "SNAKE_LENGTH_STD": 25,

    "BASE_MEAN_RADII": (1.5, 3.0, 6.0),
    "RADIUS_PROBABILITIES": (0.60, 0.30, 0.10),
    "RADIUS_COEFF_VARIATION": 0.35,

    "TORTUOSITY": 1.8,
    "PATH_SMOOTHING": 8,

    "N_XY_SLICES": 192,
    "OUTPUT_FOLDER": "outputs/09_high_resolution_publication",
    "SEED": 109,
})
```

Analysis:

```bash
python3 metrics_calculation.py \
  --npz-file outputs/09_high_resolution_publication/calibrated_snake_rock.npz \
  --npz-key pores \
  --npz-order xyz \
  --dx 5e-6 --dy 5e-6 --dz 5e-6 \
  --unit m \
  --downsample 3 \
  --max-nodes 1000000 \
  --output outputs/09_high_resolution_publication/analysis_downsample3
```

---

# Preset 10 — Medium ensemble-safe case

Purpose: safe setting for generating multiple realizations without excessive solver size.

Change only the `SEED` value to create an ensemble.

```python
PARAMS.update({
    "TARGET_POROSITY": 0.16,
    "TARGET_K_MD": 300.0,
    "TARGET_K_DIRECTION": "z",

    "NX": 128,
    "NY": 128,
    "NZ": 128,
    "VOXEL_SIZE": 5e-6,

    "N_SHORT_SNAKES": 70,
    "N_PERCOLATING_SNAKES": 1,
    "MEAN_SNAKE_LENGTH": 45,
    "SNAKE_LENGTH_STD": 18,

    "BASE_MEAN_RADII": (1.5, 3.0, 6.0),
    "RADIUS_PROBABILITIES": (0.60, 0.30, 0.10),
    "RADIUS_COEFF_VARIATION": 0.35,

    "TORTUOSITY": 1.8,
    "PATH_SMOOTHING": 8,

    "N_XY_SLICES": 128,
    "OUTPUT_FOLDER": "outputs/10_ensemble_medium_safe",
    "SEED": 110,
})
```

Analysis:

```bash
python3 metrics_calculation.py \
  --npz-file outputs/10_ensemble_medium_safe/calibrated_snake_rock.npz \
  --npz-key pores \
  --npz-order xyz \
  --dx 5e-6 --dy 5e-6 --dz 5e-6 \
  --unit m \
  --downsample 2 \
  --max-nodes 800000 \
  --output outputs/10_ensemble_medium_safe/analysis_downsample2
```

---

# Preset 11 — Very fast morphology-only analysis

Purpose: analyze large generated rocks or ensembles without solving the expensive resistor-network permeability problem.

Use this with any generated `.npz` file:

```bash
python3 metrics_calculation.py \
  --npz-file outputs/04_intermediate_reference/calibrated_snake_rock.npz \
  --npz-key pores \
  --npz-order xyz \
  --dx 5e-6 --dy 5e-6 --dz 5e-6 \
  --unit m \
  --skip-permeability \
  --output outputs/11_fast_morphology_only
```

This computes morphology, porosity, clusters, pore-radius distribution, percolation, tortuosity, and several morphohydraulic descriptors, but skips the expensive direct resistor-network solve.

---

# Preset 12 — Analyze clean PNG slices instead of `.npz`

Purpose: validate the image-stack workflow.

The `.npz` workflow is preferred for exact numerical analysis, but the slice route is useful for external segmented images.

```bash
python3 metrics_calculation.py \
  --folder outputs/04_intermediate_reference/structures \
  --mode blue \
  --crop none \
  --keep-grid \
  --dx 5e-6 --dy 5e-6 --dz 5e-6 \
  --unit m \
  --downsample 2 \
  --max-nodes 800000 \
  --output outputs/12_analysis_from_slices
```

For the generated images:

```text
blue = pore
red  = solid
```

---

# Recommended command after a large-solver warning

If you see something like:

```text
system too large for direct solver: 660791 nodes
```

run:

```bash
python3 metrics_calculation.py \
  --npz-file target_snake_rock_output/calibrated_snake_rock.npz \
  --npz-key pores \
  --npz-order xyz \
  --dx 5e-6 --dy 5e-6 --dz 5e-6 \
  --unit m \
  --downsample 2 \
  --max-nodes 800000 \
  --output output_downsample2
```

If this is still too slow, use:

```bash
python3 metrics_calculation.py \
  --npz-file target_snake_rock_output/calibrated_snake_rock.npz \
  --npz-key pores \
  --npz-order xyz \
  --dx 5e-6 --dy 5e-6 --dz 5e-6 \
  --unit m \
  --downsample 3 \
  --max-nodes 1000000 \
  --output output_downsample3
```

For a fast diagnostic without direct permeability:

```bash
python3 metrics_calculation.py \
  --npz-file target_snake_rock_output/calibrated_snake_rock.npz \
  --npz-key pores \
  --npz-order xyz \
  --dx 5e-6 --dy 5e-6 --dz 5e-6 \
  --unit m \
  --skip-permeability \
  --output output_fast_no_solver
```

---

# Suggested interpretation of parameter regimes

| Regime | Target porosity | Target permeability | Suggested use |
|---|---:|---:|---|
| Very low pore volume | 0.08--0.12 | 10--50 mD | Sparse networks and bottleneck-dominated transport |
| Low/intermediate | 0.10--0.15 | 50--300 mD | Preferential pathways with limited background porosity |
| Reference case | 0.16--0.20 | 100--500 mD | Balanced synthetic rock generation |
| High transport | 0.18--0.25 | 500--2000 mD | Stronger hydraulic backbones |
| Highly connected | 0.25--0.35 | 1000--5000 mD | Dense pore networks and high connectivity |
| Directional anisotropy | 0.15--0.25 | 100--1000 mD | Calibration along x, y, or z |
| Figure-quality volume | 0.15--0.25 | 100--2000 mD | Use 160³ or 192³ with downsampled analysis |
| Ensemble generation | 0.12--0.22 | 50--1000 mD | Change only `SEED` for repeated realizations |
