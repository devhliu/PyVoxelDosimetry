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

pyvoxeldosimetry/
└── data/
    └── dose_kernels/
        ├── __init__.py
        ├── base_kernel.py
        ├── kernel_factory.py
        ├── tissue_properties.py
        ├── f18_kernel.py
        ├── F18/
        │   ├── F18.json
        │   └── validation/
        └── cached_kernels/
            ├── F18_water_kernel.npy
            ├── F18_water_xy.png
            └── ...

I'll help you set up the core kernel generator files and structure. Here's the recommended organization:

1. First, create the base kernel generator:

```python
import numpy as np
import json
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class KernelMetadata:
    version: str
    created_at: str
    created_by: str
    method: str

@dataclass
class NuclideInfo:
    name: str
    symbol: str
    half_life: float
    decay_modes: Dict[str, float]

class BaseKernelGenerator(ABC):
    def __init__(self, config_path: Path, tissue_type: str):
        """
        Initialize base kernel generator.
        
        Args:
            config_path: Path to nuclide config JSON
            tissue_type: Type of tissue (water, bone, etc)
        """
        self.config = self._load_config(config_path)
        self.tissue_type = tissue_type
        
    def _load_config(self, config_path: Path) -> Dict:
        with open(config_path, 'r') as f:
            return json.load(f)
    
    @abstractmethod
    def generate_kernel(self, voxel_size: float, grid_size: Tuple[int, int, int]) -> np.ndarray:
        """Generate the dose point kernel."""
        pass
    
    def save_kernel(self, kernel: np.ndarray, output_dir: Path):
        """Save kernel data and visualization."""
        # Save numpy array
        np.save(output_dir / f"{self.config['nuclide']['symbol']}_{self.tissue_type}_kernel.npy", kernel)
        
        # Save visualization
        self._save_visualization(kernel, output_dir)
        
        # Save metadata
        self._save_metadata(kernel, output_dir)
    
    def _save_visualization(self, kernel: np.ndarray, output_dir: Path):
        """Generate and save 2D visualizations of the kernel."""
        import matplotlib.pyplot as plt
        
        center = [s//2 for s in kernel.shape]
        
        # XY plane
        plt.figure(figsize=(8, 8))
        plt.imshow(kernel[:, :, center[2]], cmap='viridis')
        plt.colorbar(label='Dose (Gy/Bq.s)')
        plt.title(f"{self.config['nuclide']['name']} - {self.tissue_type}")
        plt.savefig(output_dir / f"{self.config['nuclide']['symbol']}_{self.tissue_type}_xy.png")
        plt.close()
```

2. Create the radionuclide-specific generator:

```python
from .base_kernel import BaseKernelGenerator
import numpy as np
from pathlib import Path
from typing import Tuple

class F18KernelGenerator(BaseKernelGenerator):
    def __init__(self, tissue_type: str):
        super().__init__(
            Path(__file__).parent / "F18" / "F18.json",
            tissue_type
        )
        
    def generate_kernel(self, voxel_size: float, grid_size: Tuple[int, int, int]) -> np.ndarray:
        """
        Generate F-18 dose point kernel.
        
        Implements positron range and annihilation photon contributions.
        """
        kernel = np.zeros(grid_size)
        center = [s//2 for s in grid_size]
        
        # Positron range contribution
        max_energy = 0.634  # MeV
        beta_range = self._calculate_beta_range(max_energy, self.tissue_type)
        
        # Generate kernel values
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                for z in range(grid_size[2]):
                    r = np.sqrt(((x-center[0])*voxel_size)**2 + 
                              ((y-center[1])*voxel_size)**2 + 
                              ((z-center[2])*voxel_size)**2)
                    if r <= beta_range:
                        kernel[x,y,z] = self._dose_point_value(r)
        
        return kernel * self.config['kernel']['scaling_factor']
    
    def _calculate_beta_range(self, energy: float, tissue: str) -> float:
        """Calculate beta particle range in tissue."""
        # Implement tissue-specific range calculation
        pass
    
    def _dose_point_value(self, distance: float) -> float:
        """Calculate dose point value at given distance."""
        # Implement point dose calculation
        pass
```

3. Create the kernel factory:

```python
from pathlib import Path
from typing import Dict, Type
import numpy as np
from .base_kernel import BaseKernelGenerator
from .f18_kernel import F18KernelGenerator

class KernelFactory:
    """Factory for creating and managing dose kernels."""
    
    _generators: Dict[str, Type[BaseKernelGenerator]] = {
        'F18': F18KernelGenerator,
        # Add other generators as they're implemented
    }
    
    def __init__(self):
        self.cache_dir = Path(__file__).parent / "cached_kernels"
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_kernel(self, 
                  nuclide: str, 
                  tissue_type: str,
                  voxel_size: float = 1.0,
                  grid_size: Tuple[int, int, int] = (101, 101, 101),
                  force_regenerate: bool = False) -> np.ndarray:
        """
        Get or generate a dose kernel.
        
        Args:
            nuclide: Nuclide symbol (e.g., 'F18')
            tissue_type: Type of tissue
            voxel_size: Voxel size in mm
            grid_size: Kernel grid dimensions
            force_regenerate: Force kernel regeneration
        """
        if nuclide not in self._generators:
            raise ValueError(f"Unsupported nuclide: {nuclide}")
            
        cache_path = self.cache_dir / f"{nuclide}_{tissue_type}_kernel.npy"
        
        if not force_regenerate and cache_path.exists():
            return np.load(cache_path)
            
        generator = self._generators[nuclide](tissue_type)
        kernel = generator.generate_kernel(voxel_size, grid_size)
        generator.save_kernel(kernel, self.cache_dir)
        
        return kernel
```

4. Create a tissue properties file:

```python
from dataclasses import dataclass
from typing import Dict

@dataclass
class TissueProperties:
    name: str
    density: float  # g/cm3
    electron_density: float  # e/cm3
    effective_Z: float
    stopping_power_ratio: float  # relative to water
    
TISSUE_DATABASE = {
    'water': TissueProperties(
        name='Water',
        density=1.0,
        electron_density=3.34e23,
        effective_Z=7.42,
        stopping_power_ratio=1.0
    ),
    'bone': TissueProperties(
        name='Cortical Bone',
        density=1.85,
        electron_density=5.91e23,
        effective_Z=13.8,
        stopping_power_ratio=1.15
    ),
    # Add other tissue types
}
```

5. Example usage:

```python
# Example usage script

from pathlib import Path
from pyvoxeldosimetry.data.dose_kernels.kernel_factory import KernelFactory

# Initialize factory
factory = KernelFactory()

# Generate F18 kernel for water
kernel = factory.get_kernel(
    nuclide='F18',
    tissue_type='water',
    voxel_size=1.0,
    grid_size=(101, 101, 101)
)

print(f"Kernel shape: {kernel.shape}")
print(f"Max dose value: {kernel.max():.2e} Gy/Bq.s")
```

# Example usage
from pyvoxeldosimetry.data.dose_kernels.ga68_kernel import Ga68KernelGenerator

# Initialize generator for water
generator = Ga68KernelGenerator(tissue_type='water')

# Generate kernel
kernel = generator.generate_kernel(
    voxel_size=1.0,
    grid_size=(151, 151, 151)
)

# Save kernel with visualizations
output_dir = Path('output/kernels')
generator.save_kernel(kernel, output_dir)

Directory structure should look like:
```
pyvoxeldosimetry/
└── data/
    └── dose_kernels/
        ├── __init__.py
        ├── base_kernel.py
        ├── kernel_factory.py
        ├── tissue_properties.py
        ├── f18_kernel.py
        ├── F18/
        │   ├── F18.json
        │   └── validation/
        └── cached_kernels/
            ├── F18_water_kernel.npy
            ├── F18_water_xy.png
            └── ...
```

This structure provides:
- Clear separation of concerns
- Extensible design for new radionuclides
- Tissue-specific kernel generation
- Caching mechanism
- Visualization capabilities
- Proper configuration management

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

