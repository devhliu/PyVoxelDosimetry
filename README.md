# PyVoxelDosimetry

PyVoxelDosimetry is a Python package for performing voxel-based dosimetry calculations, particularly for radionuclide therapies such as Yttrium-90 (Y90). It provides tools for simulating activity distributions, calculating dose distributions, and visualizing results using kernel convolution methods.  
**GATE Monte Carlo simulation support is now provided via the `pyvoxeldosimetry.core.gate` and `pyvoxeldosimetry.core.gate_monte_carlo` modules.**

## Features

- Voxel-based dosimetry calculations for various radionuclides
- Support for kernel convolution methods
- Customizable tissue composition and kernel resolution
- Example scripts for common dosimetry scenarios
- Visualization of activity and dose distributions
- GATE Monte Carlo simulation integration via `core.gate` and `core.gate_monte_carlo`

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/devhliu/PyVoxelDosimetry.git
    cd PyVoxelDosimetry
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

   Typical dependencies include:
   - numpy
   - matplotlib

## Usage

### Example: Single Timepoint Y90 Physical Decay

An example script is provided in `examples/single_timepoint_y90_physical_decay.py`. This script demonstrates how to:

- Create a spherical activity distribution
- Set up a dose calculator for Y90 using kernel convolution methods
- Calculate the dose distribution at a single timepoint
- Visualize the activity and dose distributions

To run the example:

```bash
python examples/single_timepoint_y90_physical_decay.py
```