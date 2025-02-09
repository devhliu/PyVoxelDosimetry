from pathlib import Path
from typing import Dict, Type, Tuple, Optional
import numpy as np
from .base_kernel import BaseKernelGenerator
from .f18_kernel import F18KernelGenerator
from .ga68_kernel import Ga68KernelGenerator
from .y90_kernel import Y90KernelGenerator
from .lu177_kernel import Lu177KernelGenerator
from .tb161_kernel import Tb161KernelGenerator
from .ac225_kernel import Ac225KernelGenerator

class KernelFactory:
    """Factory for creating and managing dose kernels."""
    
    _generators: Dict[str, Type[BaseKernelGenerator]] = {
        'F18': F18KernelGenerator,
        'Ga68': Ga68KernelGenerator,
        'Y90': Y90KernelGenerator,
        'Lu177': Lu177KernelGenerator,
        'Tb161': Tb161KernelGenerator,  # Add Tb161
        'Ac225': Ac225KernelGenerator  # Add Ac225
    }
    
    _default_grid_sizes: Dict[str, Tuple[int, int, int]] = {
        'F18': (101, 101, 101),    # Medium range
        'Ga68': (151, 151, 151),   # Higher energy, larger range
        'Y90': (201, 201, 201),    # High energy beta, largest range
        'Lu177': (81, 81, 81),     # Lower energy, smaller range
        'Tb161': (71, 71, 71),     # Lowest energy, smallest range
        'Ac225': (251, 251, 251)   # Alpha particles + daughters
    }
    
    def __init__(self):
        self.cache_dir = Path(__file__).parent / "cached_kernels"
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_kernel(self, 
                  nuclide: str, 
                  tissue_type: str,
                  voxel_size: float = 1.0,
                  grid_size: Optional[Tuple[int, int, int]] = None,
                  force_regenerate: bool = False) -> np.ndarray:
        """
        Get or generate a dose kernel.
        
        Args:
            nuclide: Nuclide symbol (F18, Ga68, Y90, Lu177)
            tissue_type: Type of tissue (water, lung, soft_tissue, bone, iodine_contrast)
            voxel_size: Voxel size in mm
            grid_size: Kernel grid dimensions (optional, uses default if None)
            force_regenerate: Force kernel regeneration
            
        Returns:
            3D numpy array containing the dose kernel
            
        Raises:
            ValueError: If nuclide is not supported
        """
        if nuclide not in self._generators:
            supported = list(self._generators.keys())
            raise ValueError(f"Unsupported nuclide: {nuclide}. Supported: {supported}")
        
        if grid_size is None:
            grid_size = self._default_grid_sizes.get(nuclide, (101, 101, 101))
            
        cache_path = self.cache_dir / f"{nuclide}_{tissue_type}_kernel.npy"
        
        if not force_regenerate and cache_path.exists():
            return np.load(cache_path)
            
        generator = self._generators[nuclide](tissue_type)
        kernel = generator.generate_kernel(voxel_size, grid_size)
        generator.save_kernel(kernel, self.cache_dir)
        
        return kernel