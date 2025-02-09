from pathlib import Path
import numpy as np
from typing import Tuple, Optional
from scipy.interpolate import interp1d
from .base_kernel import BaseKernelGenerator

class Ga68KernelGenerator(BaseKernelGenerator):
    """
    Ga-68 specific dose kernel generator.
    Handles both positron range and gamma contribution.
    """
    
    def __init__(self, tissue_type: str):
        config_path = Path(__file__).parent / "Ga68" / "Ga68.json"
        super().__init__(config_path, tissue_type)
        self.beta_max_energy = self.config['nuclide']['particle_energies']['beta+_max']
        self.gamma_lines = self.config['nuclide']['particle_energies'].get('gamma_lines', [])
        
    def generate_kernel(self, 
                       voxel_size: float,
                       grid_size: Tuple[int, int, int]) -> np.ndarray:
        """
        Generate Ga-68 dose point kernel.
        
        Args:
            voxel_size: Voxel size in mm
            grid_size: Dimensions of the kernel grid
            
        Returns:
            3D numpy array containing the dose kernel
        """
        kernel = np.zeros(grid_size)
        center = [s//2 for s in grid_size]
        
        # Generate distance matrix
        x, y, z = np.meshgrid(
            np.arange(grid_size[0]) - center[0],
            np.arange(grid_size[1]) - center[1],
            np.arange(grid_size[2]) - center[2],
            indexing='ij'
        )
        r = np.sqrt((x*voxel_size)**2 + (y*voxel_size)**2 + (z*voxel_size)**2)
        
        # Beta contribution (including positron range)
        kernel += self._calculate_beta_contribution(r)
        
        # Annihilation photons
        kernel += self._calculate_annihilation_contribution(r)
        
        # Additional gamma lines
        for gamma in self.gamma_lines:
            kernel += self._calculate_gamma_contribution(r, gamma['energy'], gamma['intensity'])
        
        # Apply tissue-specific scaling
        kernel *= self._get_tissue_scaling_factor()
        
        return kernel
    
    def _calculate_beta_contribution(self, r: np.ndarray) -> np.ndarray:
        """Calculate beta particle dose contribution."""
        # Cole's formula for Ga-68 beta range in tissue
        max_range = 9.0 * self.beta_max_energy**1.5  # mm in water
        tissue_range = max_range * self._get_tissue_scaling_factor()
        
        beta_dose = np.zeros_like(r)
        mask = r <= tissue_range
        
        # Modified point kernel for beta dose
        beta_dose[mask] = (1 - r[mask]/tissue_range)**2 * np.exp(-2*r[mask]/tissue_range)
        return beta_dose
    
    def _calculate_annihilation_contribution(self, r: np.ndarray) -> np.ndarray:
        """Calculate 511 keV annihilation photons contribution."""
        mu = self._get_attenuation_coefficient(0.511)  # cm^-1
        return np.exp(-mu * r/10) / (4*np.pi*r**2) * (r > 0)
    
    def _calculate_gamma_contribution(self, 
                                   r: np.ndarray,
                                   energy: float,
                                   intensity: float) -> np.ndarray:
        """Calculate specific gamma line contribution."""
        mu = self._get_attenuation_coefficient(energy)  # cm^-1
        return intensity * np.exp(-mu * r/10) / (4*np.pi*r**2) * (r > 0)
    
    def _get_tissue_scaling_factor(self) -> float:
        """Get tissue-specific scaling factor."""
        tissue_factors = {
            'water': 1.0,
            'lung': 0.3,
            'soft_tissue': 1.04,
            'bone': 1.6,
            'iodine_contrast': 1.3
        }
        return tissue_factors.get(self.tissue_type, 1.0)
    
    def _get_attenuation_coefficient(self, energy: float) -> float:
        """Get tissue-specific attenuation coefficient."""
        # Simplified lookup table for attenuation coefficients (cm^-1)
        coefficients = {
            'water': 0.096,
            'lung': 0.029,
            'soft_tissue': 0.099,
            'bone': 0.172,
            'iodine_contrast': 0.158
        }
        return coefficients.get(self.tissue_type, 0.096) * (0.511/energy)**3.2
