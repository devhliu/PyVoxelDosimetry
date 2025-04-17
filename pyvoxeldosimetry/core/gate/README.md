# Y90 GATE Monte Carlo Simulation

This directory contains scripts for running Y90 dosimetry calculations using GATE Monte Carlo simulations.

## Overview

The scripts in this directory allow you to:

1. Run Y90 dosimetry simulations using GATE v9.4
2. Calculate dose rates in GBq/h
3. Generate separate beta and gamma dose components
4. Output results in NIfTI (.nii.gz) format

## Requirements

- Docker with GATE 9.4 image (`opengatecollaboration/gate:9.4-docker`)
- Python 3.6+ with the following packages:
  - numpy
  - nibabel
  - pathlib
  - SimpleITK
  - pydicom (for DICOM support)

## Input Files

- Y90 PET image (NIfTI, MHD format, or DICOM series)
- CT image (NIfTI, MHD format, or DICOM series)

## Usage

```bash
python run_y90_simulation.py --pet /path/to/Y90_PET.nii.gz --ct /path/to/CT.nii.gz --output /path/to/output_dir [options]
```

### DICOM Support

The simulation now supports DICOM format for both PET and CT series:

```bash
python run_y90_simulation.py --pet /path/to/PET_DICOM_directory --ct /path/to/CT_DICOM_directory --output /path/to/output_dir [options]
```

When providing DICOM directories, the script automatically:
1. Detects if the input is a DICOM directory
2. Converts PET DICOM series to MHD format with proper activity values (Bq/mL)
3. Converts CT DICOM series to MHD format with proper HU values
4. Proceeds with the simulation using the converted files

#### DICOM Processing Details

The `dicom_utils.py` module handles DICOM processing with the following features:

- **PET DICOM Processing**: Extracts activity concentration values (Bq/mL) from PET DICOM series, applying appropriate rescale factors and vendor-specific quantification factors
- **CT DICOM Processing**: Converts CT values to Hounsfield Units (HU) using rescale slope and intercept from DICOM metadata
- **Automatic Detection**: Identifies DICOM directories and determines modality (PET or CT) from DICOM headers
- **Vendor Support**: Handles vendor-specific DICOM implementations (GE, Siemens, etc.) for proper quantification

### Command Line Arguments

- `--pet`: Path to Y90 PET image (required)
- `--ct`: Path to CT image (required)
- `--output`: Output directory (required)
- `--activity`: Injection Y90 Activity in GBq (default: 4.0)
- `--time`: Scan Time after Injection in hours (default: 2.2)
- `--particles`: Number of particles to simulate (default: 20,000,000)
- `--local`: Use local GATE installation instead of Docker (default: use Docker)

## Output Files

The script generates the following output files:

- `Y90_beta_doserate.nii.gz`: Beta dose rate in GBq/h
- `Y90_gamma_doserate.nii.gz`: Gamma dose rate in GBq/h
- `Y90_total_doserate.nii.gz`: Total dose rate in GBq/h
- `Y90_beta_dose.nii.gz`: Beta absorbed dose
- `Y90_gamma_dose.nii.gz`: Gamma absorbed dose
- `Y90_total_dose.nii.gz`: Total absorbed dose

## Directory Structure

```
gate/
├── mac/
│   └── executor.mac       # GATE macro file for Y90 simulation
├── run_y90_simulation.py  # Python script to run simulation
├── dicom_utils.py         # Utilities for DICOM file handling
├── requirements.txt       # Input/output specifications
└── README.md             # This file
```

## How It Works

1. The script prepares the input data by scaling the PET image to the actual activity based on injection amount and decay time.
2. It creates a directory structure for the GATE simulation with the necessary input files.
3. The GATE simulation is run using Docker (or local installation if specified).
4. The simulation outputs are processed to generate dose rate and absorbed dose files.
5. All results are saved in NIfTI format in the specified output directory.

## GATE Macro File

The `executor.mac` file contains the GATE configuration for the Y90 simulation. It sets up:

- Material properties
- Geometry based on the CT image
- Y90 source definition based on the PET image
- Separate dose actors for beta and gamma components
- Energy spectrum actor for detailed energy deposition analysis

## References

- GATE documentation: https://opengate.readthedocs.io/
- Y90 dosimetry guidelines: https://www.eanm.org/publications/guidelines/