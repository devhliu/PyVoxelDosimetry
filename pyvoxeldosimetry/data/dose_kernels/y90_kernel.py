from pathlib import Path
import numpy as np
from typing import Tuple, Optional
from scipy.interpolate import interp1d
from .base_kernel import BaseKernelGenerator

class Y90KernelGenerator(BaseKernelGenerator):
    """
    Y-90 specific dose kernel generator.
    Handles beta emission and bremsstrahlung with tissue composition effects.
    """
    
    def __init__(self, tissue_type: str):
        config_path = Path(__file__).parent / "Y90" / "Y90.json"
        super().__init__(config_path, tissue_type)
        self.beta_max_energy = self.config['nuclide']['particle_energies']['beta-_max']
        self.beta_mean_energy = self.config['nuclide']['particle_energies']['beta-_mean']
        self.tissue_properties = self._get_tissue_properties()
        
    def generate_kernel(self, 
                       voxel_size: float,
                       grid_size: Tuple[int, int, int]) -> np.ndarray:
        """
        Generate Y-90 dose point kernel.
        
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
        
        # Beta contribution (high energy)
        kernel += self._calculate_beta_contribution(r)
        
        # Bremsstrahlung contribution (tissue-dependent)
        if self.config['nuclide']['particle_energies'].get('bremsstrahlung'):
            kernel += self._calculate_bremsstrahlung_contribution(r)
        
        # Apply tissue-specific scaling and density correction
        kernel *= self._get_tissue_scaling_factor()
        
        return kernel

    def _get_tissue_properties(self) -> dict:
        """Get tissue-specific properties."""
        properties = {
            'water': {
                'density': 1.0,
                'electron_density': 3.34e23,
                'effective_Z': 7.42,
                'stopping_power_ratio': 1.0
            },
            'lung': {
                'density': 0.26,
                'electron_density': 0.87e23,
                'effective_Z': 7.41,
                'stopping_power_ratio': 1.04
            },
            'soft_tissue': {
                'density': 1.04,
                'electron_density': 3.48e23,
                'effective_Z': 7.46,
                'stopping_power_ratio': 1.04
            },
            'bone': {
                'density': 1.85,
                'electron_density': 5.91e23,
                'effective_Z': 13.8,
                'stopping_power_ratio': 1.15
            },
            'iodine_contrast': {
                'density': 1.30,
                'electron_density': 4.35e23,
                'effective_Z': 53.0,
                'stopping_power_ratio': 1.12
            }
        }
        return properties.get(self.tissue_type, properties['water'])

    def _calculate_beta_contribution(self, r: np.ndarray) -> np.ndarray:
        """
        Calculate beta particle dose contribution with tissue effects.
        Uses Cole's formula modified for tissue composition.
        """
        # Calculate tissue-specific range
        density_factor = self.tissue_properties['density']
        stopping_power_ratio = self.tissue_properties['stopping_power_ratio']
        
        # CSDA range calculation (continuous slowing down approximation)
        max_range = 11.0 * self.beta_max_energy**1.5  # mm in water
        tissue_range = max_range * (1.0/density_factor) * (1.0/stopping_power_ratio)
        
        beta_dose = np.zeros_like(r)
        mask = r <= tissue_range
        
        # Modified point kernel for beta dose with tissue effects
        beta_dose[mask] = (
            (1 - r[mask]/tissue_range)**2 * 
            np.exp(-2*r[mask]/tissue_range) * 
            density_factor * 
            stopping_power_ratio
        )
        
        return beta_dose
        
    def _calculate_bremsstrahlung_contribution(self, r: np.ndarray) -> np.ndarray:
        """
        Calculate bremsstrahlung contribution with tissue composition effects.
        Enhanced model accounting for Z-dependent bremsstrahlung yield.
        """
        effective_Z = self.tissue_properties['effective_Z']
        density = self.tissue_properties['density']
        
        # Bremsstrahlung yield increases with Z^2
        relative_yield = (effective_Z / 7.42)**2  # normalized to water
        
        # Attenuation coefficient (tissue-dependent)
        mu = self._get_attenuation_coefficient()
        
        # Calculate bremsstrahlung dose
        brems = (
            0.015 * relative_yield * density * 
            np.exp(-mu * r/10) / (4*np.pi*r**2) * 
            (r > 0)
        )
        
        return brems
    
    def _get_attenuation_coefficient(self) -> float:
        """Get tissue-specific attenuation coefficient."""
        # Mean energy for bremsstrahlung attenuation (~0.5 MeV)
        base_mu = 0.096  # cm^-1 in water at 0.5 MeV
        return base_mu * (self.tissue_properties['density'] / 1.0)

    def _get_tissue_scaling_factor(self) -> float:
        """Return tissue-specific scaling factor for kernel normalization."""
        # You can customize this logic as needed
        if self.tissue_type == "water":
            return 1.0
        elif self.tissue_type == "bone":
            return 1.15
        elif self.tissue_type == "lung":
            return 1.04
        elif self.tissue_type == "soft_tissue":
            return 1.04
        elif self.tissue_type == "iodine_contrast":
            return 1.12
        else:
            return 1.0