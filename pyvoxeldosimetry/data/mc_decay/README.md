# Monte Carlo Decay Data

This folder contains detailed decay data for Monte Carlo simulations of various radionuclides.

## Data Structure

Each radionuclide is described in a JSON file with the following structure:

```json
{
    "name": "Radionuclide name",
    "atomic_number": "Z",
    "mass_number": "A",
    "half_life": "value",
    "half_life_unit": "days/hours/minutes",
    "decay_modes": [
        {
            "type": "decay type",
            "branching_ratio": "probability",
            "spectrum": {
                "mean_energy": "MeV",
                "max_energy": "MeV",
                "spectrum_points": {
                    "energy": ["energy points in MeV"],
                    "probability": ["probability density"]
                }
            }
        }
    ],
    "gamma_emissions": [
        {
            "energy": "MeV",
            "intensity": "probability per decay"
        }
    ],
    "physics_parameters": {
        "particle_specific_data": "values"
    }
}
```

# Usage

```bash
from pyvoxeldosimetry.data.mc_decay import load_decay_data

# Load decay data for Lu-177
decay_data = load_decay_data('Lu177')

# Access decay properties
half_life = decay_data['half_life']
beta_spectrum = decay_data['decay_modes'][0]['spectrum']
```
# Updating Data
To add or update radionuclide data:

1. Create a new JSON file with the radionuclide name
2. Follow the data structure format
3. Include all relevant decay modes and emissions
4. Add physics parameters for Monte Carlo simulation

This implementation:

1. **Organizes Decay Data**:
   - Structured JSON format for each radionuclide
   - Includes detailed physics parameters
   - Separates data from implementation

2. **Monte Carlo Features**:
   - GPU-accelerated particle transport
   - Multiple particle types (β, γ, α)
   - Accurate physics modeling
   - Efficient memory management

3. **Physics Implementation**:
   - Beta spectrum sampling
   - Gamma interaction tables
   - Range and attenuation data
   - CUDA kernels for transport

4. **Usage Example**:
```python
from pyvoxeldosimetry import MonteCarloCalculator
from pyvoxeldosimetry.tissue import TissueComposition

# Initialize calculator
mc_calc = MonteCarloCalculator(
    radionuclide="Lu177",
    tissue_composition=tissue_comp,
    n_particles=1000000
)

# Calculate dose rate
dose_rate = mc_calc.calculate_dose_rate(
    activity_map=spect_data,
    voxel_size=(1, 1, 1)
)
```