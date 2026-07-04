# Snake--MorphoHydraulic

Target-driven 3D digital rock generation and morphohydraulic analysis toolkit for synthetic porous media.

This repository provides a simple, reproducible workflow for generating synthetic three-dimensional digital rocks with prescribed porosity and target permeability, followed by quantitative morphohydraulic characterization.

The workflow is divided into two independent scripts:

- `generate_rocks_snake.py`  
  Generates synthetic 3D digital rocks using a Snake-based pore-generation model.

- `metrics_calculation.py`  
  Computes structural, connectivity, permeability, and morphohydraulic descriptors from either a generated `.npz` volume or a stack of segmented 2D slices.

---

## Main idea

The Snake--MorphoHydraulic workflow separates **digital rock generation** from **independent morphohydraulic validation**.

The generation script creates a binary 3D pore structure using:

- a multiscale stochastic background field;
- tortuous snake-like pore pathways;
- capillary-based permeability calibration;
- exact porosity enforcement;
- export of volumes, slices, figures, metadata, and metrics.

The analysis script then computes:

- porosity and connected porosity;
- dead porosity;
- 3D percolation;
- local pore-radius distribution;
- throat-radius descriptors;
- tortuosity;
- voxel-resistor permeability;
- multiple permeability estimators;
- anisotropy;
- flow localization;
- bottleneck descriptors;
- new morphohydraulic indices.

---

## Repository structure

```text
snake-morphohydraulic/
├── generate_rocks_snake.py
├── metrics_calculation.py
├── PARAMETER_PRESETS.md
├── requirements.txt
├── README.md
├── LICENSE
├── CITATION.cff
├── configs/
│   ├── generation_parameter_presets.py
│   └── analysis_commands.sh
├── examples/
│   └── README.md
└── outputs/
