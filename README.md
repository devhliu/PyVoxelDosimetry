# PyVoxelDosimetry

## Internal Dosimetry Package for Nuclear Medicine - Overview
PyVoxelDosimetry is a comprehensive Python package for voxel-level internal dosimetry calculations in nuclear medicine. It supports various radionuclide tracers and provides multiple calculation methods, advanced tissue composition handling, and deep learning-based segmentation.

## Features

### 1. Dosimetry Calculation Methods
- **Local Energy Deposition**: Fast approximation for short-range emissions
- **Dose Kernel Convolution**: Pre-calculated kernels for efficient computation
- **GPU-accelerated Monte Carlo**: Full physics simulation with CUDA support

### 2. Supported Radionuclides
- Positron Emitters: F-18, Ga-68, Cu-64, Zr-89
- Beta Emitters: Y-90, Lu-177, Tb-161
- Beta/Gamma Emitters: I-131
- Alpha Emitters: Ac-225, Pb-212

### 3. Image Processing & Analysis
- Multi-timepoint registration (elastix, ANTs, greedy)
- CT-based tissue composition calculation
- Metal artifact and contrast agent handling
- Time-activity curve fitting and integration

### 4. Deep Learning Segmentation (nnU-Net v2)
- Automated organ segmentation
- Lesion detection and segmentation
- Support for PET/CT and SPECT/CT
  


## Installation

### Prerequisites
- Python ≥ 3.8
- CUDA-capable GPU (recommended)

### Install via pip
```bash
pip install pyvoxeldosimetry
```

### Install from source
```bash
git clone https://github.com/devhliu/pyvoxeldosimetry.git
cd pyvoxeldosimetry
pip install -e .
```

## Quick Start
### Basic Usage
```bash
from pyvoxeldosimetry import MonteCarloCalculator
from pyvoxeldosimetry.tissue import TissueComposition
from pyvoxeldosimetry.segmentation import OrganSegmentation

# Initialize components
tissue_comp = TissueComposition()
organ_seg = OrganSegmentation(model_type="total_body")

# Perform organ segmentation
organ_masks = organ_seg.segment_organs(ct_image)

# Initialize dosimetry calculator
calculator = MonteCarloCalculator(
    radionuclide="Lu177",
    tissue_composition=tissue_comp
)

# Calculate dose rate
dose_rate = calculator.calculate_dose_rate(
    activity_map=spect_data,
    voxel_size=(1, 1, 1)  # mm
)
```

### Time Integration Example
```bash
from pyvoxeldosimetry.time_integration import TimeCurveFitting

# Initialize time curve fitting
fitter = TimeCurveFitting(half_life=6.647 * 24 * 60)  # Lu-177 half-life in minutes

# Fit time-activity curves and calculate accumulated dose
accumulated_dose = fitter.fit_time_activity_curve(
    times=[0, 24, 48, 72],  # hours
    activities=[spect_data_0h, spect_data_24h, spect_data_48h, spect_data_72h]
)
```

### Package Structure
```bash
pyvoxeldosimetry/
├── data/
│   ├── dose_kernels/          # Pre-calculated dose kernels
│   └── pretrained_models/     # nnU-Net models
├── core/                      # Core dosimetry calculations
├── tissue/                    # Tissue composition handling
├── segmentation/             # Deep learning segmentation
├── registration/             # Image registration
└── time_integration/         # Time-activity fitting
```

## Dose Kernel Data
The package includes pre-calculated dose kernels for supported radionuclides at multiple spatial resolutions (1mm, 2mm, 3mm). These are automatically loaded based on the selected radionuclide and voxel size.

## Segmentation Models
Pre-trained nnU-Net v2 models are provided for:

- Total body organ segmentation
- Tumor/lesion segmentation for PET/CT
- Tumor/lesion segmentation for SPECT/CT

## Acknowledgments
- nnU-Net v2 team for the segmentation framework
- Elastix, ANTs, and Greedy teams for registration tools
- The nuclear medicine community for dosimetry standards and data
- 3.5 Sonnet for the initial concept and guidance

## Roadmap
- Support for additional radionuclides
- Web-based visualization interface
- Integration with clinical workflow systems