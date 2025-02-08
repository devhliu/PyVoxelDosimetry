# Dose Kernel Data Structure

Each radionuclide folder contains the following files:
- `kernel_1mm.dat`: Dose point kernel data at 1mm resolution
- `kernel_2mm.dat`: Dose point kernel data at 2mm resolution
- `kernel_3mm.dat`: Dose point kernel data at 3mm resolution
- `metadata.json`: Contains kernel properties and reference information

dose_kernels/
├── [radionuclide]/
    ├── kernel_1mm.dat
    ├── kernel_2mm.dat
    ├── kernel_3mm.dat
    ├── kernel_4mm.dat
    ├── kernel_5mm.dat
    └── metadata.json

Supported radionuclides:
1. F-18 (positron emitter)
2. Ga-68 (positron emitter)
3. Cu-64 (positron emitter/electron capture)
4. Zr-89 (positron emitter)
5. Y-90 (beta emitter)
6. I-131 (beta/gamma emitter)
7. Lu-177 (beta/gamma emitter)
8. Tb-161 (beta emitter)
9. Ac-225 (alpha emitter)
10. Pb-212 (alpha/beta emitter)

Each kernel file contains:
- Energy deposition data
- Range information
- Radiation type specific parameters

Each kernel file contains the dose distribution data in a binary format:

# Example kernel file structure (binary)
```python
struct KernelHeader {
    int32_t dimensions[3];
    float voxel_size;
    float total_energy;
    float scaling_factor;
};

struct KernelData {
    KernelHeader header;
    float data[];  // 3D array of dose values
};
```
Key features of the dose kernels:

1. Resolution options: 1mm to 5mm voxel sizes
2. Symmetry: All kernels are radially symmetric
3. Normalization: Values in Gy/(Bq*s)
4. Medium: Calculated for water (density = 1.0 g/cm³)
5. Physics included:
   - Beta particle transport
   - Positron range (for β+ emitters)
   - Annihilation photons (for β+ emitters)
   - Gamma emissions
   - Bremsstrahlung
   - Secondary electrons
  
Usage example:
```python
import numpy as np

def load_dose_kernel(radionuclide: str, resolution: int) -> np.ndarray:
    """Load dose kernel data for given radionuclide and resolution."""
    kernel_path = f"dose_kernels/{radionuclide}/kernel_{resolution}mm.dat"
    with open(kernel_path, 'rb') as f:
        # Read header
        dimensions = np.fromfile(f, dtype=np.int32, count=3)
        voxel_size = np.fromfile(f, dtype=np.float32, count=1)[0]
        total_energy = np.fromfile(f, dtype=np.float32, count=1)[0]
        scaling_factor = np.fromfile(f, dtype=np.float32, count=1)[0]
        
        # Read kernel data
        kernel_data = np.fromfile(f, dtype=np.float32)
        kernel_data = kernel_data.reshape(dimensions)
        
        return kernel_data * scaling_factor
```
Each radionuclide's kernels are calculated using appropriate physics models and validated against published data when available.