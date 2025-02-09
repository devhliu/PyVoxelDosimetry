# F-18 Dose Point Kernels

Version: 1.0.0
Created: 2025-02-08 06:04:26
Author: devhliu

## Overview
This folder contains dose point kernel data for F-18, calculated using GATE/GEANT4 Monte Carlo simulations. The kernels include both positron and annihilation photon contributions.

## File Format
Each kernel_[resolution]mm.dat file contains:

### Header (24 bytes):
- dimensions[3]: int32[3] - Size of the kernel array
- voxel_size: float32 - Voxel size in mm
- total_energy: float32 - Total energy per decay
- scaling_factor: float32 - Conversion factor to Gy/(Bq*s)

### Data Section:
- 3D array of float32 values representing dose per decay
- Array dimensions as specified in header
- Values in Gy/(Bq*s)

## Available Resolutions:
1. kernel_1mm.dat (101³ voxels)
2. kernel_2mm.dat (61³ voxels)
3. kernel_3mm.dat (41³ voxels)
4. kernel_4mm.dat (31³ voxels)
5. kernel_5mm.dat (25³ voxels)

## Validation
Kernels have been validated against:
- MIRD Pamphlet No. 25
- Published F-18 FDG PET dosimetry data
- Cross-validated with multiple Monte Carlo codes

## Folder Structure
F18/
├── F18.json                    # Main configuration file
└── validation/
    ├── F18_mird_comparison.csv    # MIRD validation data
    ├── F18_gate_comparison.csv    # GATE MC validation data
    ├── F18_fluka_comparison.csv   # FLUKA MC validation data
    ├── plot_validation.py         # Validation plotting script
    └── F18_validation_comparison.png  # Generated validation plot

## Usage Example:
```python
import numpy as np

def load_f18_kernel(resolution_mm: int) -> np.ndarray:
    filename = f"kernel_{resolution_mm}mm.dat"
    with open(filename, 'rb') as f:
        # Read header
        dims = np.fromfile(f, dtype=np.int32, count=3)
        voxel_size = np.fromfile(f, dtype=np.float32, count=1)[0]
        total_energy = np.fromfile(f, dtype=np.float32, count=1)[0]
        scaling = np.fromfile(f, dtype=np.float32, count=1)[0]
        
        # Read dose kernel data
        kernel = np.fromfile(f, dtype=np.float32)
        kernel = kernel.reshape(dims) * scaling
        
        return kernel

```

