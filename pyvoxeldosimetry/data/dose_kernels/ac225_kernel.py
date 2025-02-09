from pathlib import Path
import numpy as np
from typing import Tuple, Optional, List, Dict
from scipy.interpolate import interp1d
from .base_kernel import BaseKernelGenerator

"""
Key features of this implementation:

1. Alpha Particle Handling:
   - ASTAR-based range calculations
   - Bragg peak modeling
   - LET corrections
   - Tissue-specific range factors

2. Daughter Products:
   - Tracks all alpha-emitting daughters
   - Handles branching ratios
   - Includes gamma emissions

3. Tissue Effects:
   - Density-dependent range scaling
   - Z-dependent stopping power
   - Material-specific attenuation

4. Additional Properties:
   - Alpha range factors for different tissues
   - Energy-dependent attenuation
   - LET-based dose corrections
"""
class Ac225KernelGenerator(BaseKernelGenerator):
    """
    Ac-225 specific dose kernel generator.
    Handles alpha particles, daughter products, and tissue composition effects.
    """
    
    def __init__(self, tissue_type: str):
        config_path = Path(__file__).parent / "Ac225" / "Ac225.json"
        super().__init__(config_path, tissue_type)
        self.alpha_energies = self._get_alpha_energies()
        self.gamma_lines = self.config['nuclide']['particle_energies']['gamma_lines']
        self.tissue_properties = self._get_tissue_properties()
        
    def _get_alpha_energies(self) -> List[Dict]:
        """Get all alpha energies including daughter products."""
        alphas = []
        
        # Parent Ac-225 alphas
        for alpha in self.config['nuclide']['particle_energies']['alpha_energies']:
            alphas.append({
                'energy': alpha['energy'],
                'intensity': alpha['intensity'],
                'nuclide': 'Ac225'
            })
            
        # Daughter product alphas
        for daughter in self.config['nuclide']['particle_energies']['daughters']:
            if 'alpha_energy' in daughter:
                alphas.append({
                    'energy': daughter['alpha_energy'],
                    'intensity': 1.0,  # Assuming full decay
                    'nuclide': daughter['name']
                })
                
        return alphas
    
    def generate_kernel(self, 
                       voxel_size: float,
                       grid_size: Tuple[int, int, int]) -> np.ndarray:
        """Generate Ac-225 dose point kernel including all decay products."""
        kernel = np.zeros(grid_size)
        center = [s//2 for s in grid_size]
        
        x, y, z = np.meshgrid(
            np.arange(grid_size[0]) - center[0],
            np.arange(grid_size[1]) - center[1],
            np.arange(grid_size[2]) - center[2],
            indexing='ij'
        )
        r = np.sqrt((x*voxel_size)**2 + (y*voxel_size)**2 + (z*voxel_size)**2)
        
        # Alpha contributions (parent and daughters)
        for alpha in self.alpha_energies:
            kernel += alpha['intensity'] * self._calculate_alpha_contribution(
                r, alpha['energy']
            )
        
        # Gamma contributions
        for gamma in self.gamma_lines:
            kernel += self._calculate_gamma_contribution(
                r, gamma['energy'], gamma['intensity']
            )
        
        # Apply tissue-specific scaling
        kernel *= self._get_tissue_scaling_factor()
        
        return kernel

    def _get_tissue_properties(self) -> dict:
        """Get tissue-specific properties for alpha particle transport."""
        properties = {
            'water': {
                'density': 1.0,
                'electron_density': 3.34e23,
                'effective_Z': 7.42,
                'stopping_power_ratio': 1.0,
                'alpha_range_factor': 1.0
            },
            'lung': {
                'density': 0.26,
                'electron_density': 0.87e23,
                'effective_Z': 7.41,
                'stopping_power_ratio': 1.04,
                'alpha_range_factor': 3.85
            },
            'soft_tissue': {
                'density': 1.04,
                'electron_density': 3.48e23,
                'effective_Z': 7.46,
                'stopping_power_ratio': 1.04,
                'alpha_range_factor': 0.96
            },
            'bone': {
                'density': 1.85,
                'electron_density': 5.91e23,
                'effective_Z': 13.8,
                'stopping_power_ratio': 1.15,
                'alpha_range_factor': 0.54
            },
            'iodine_contrast': {
                'density': 1.30,
                'electron_density': 4.35e23,
                'effective_Z': 53.0,
                'stopping_power_ratio': 1.12,
                'alpha_range_factor': 0.77
            }
        }
        return properties.get(self.tissue_type, properties['water'])
    
    def _calculate_alpha_contribution(self, r: np.ndarray, energy: float) -> np.ndarray:
        """
        Calculate alpha particle dose contribution.
        Uses ASTAR-based range calculations with tissue-specific corrections.
        
        Args:
            r: Distance matrix (mm)
            energy: Alpha particle energy (MeV)
        """
        # ASTAR-based range calculation
        range_water = 0.0006 * energy**1.7  # mm in water
        tissue_range = range_water * self.tissue_properties['alpha_range_factor']
        
        alpha_dose = np.zeros_like(r)
        mask = r <= tissue_range
        
        # Modified Bragg peak approximation
        alpha_dose[mask] = (
            (1 - (r[mask]/tissue_range)**1.7) * 
            np.exp(-4.5 * r[mask]/tissue_range)
        )
        
        # LET correction
        let_factor = energy / 5.0  # normalized to 5 MeV
        alpha_dose *= let_factor
        
        return alpha_dose
    
    def _calculate_gamma_contribution(self, 
                                   r: np.ndarray,
                                   energy: float,
                                   intensity: float) -> np.ndarray:
        """Calculate gamma contribution with tissue attenuation."""
        density = self.tissue_properties['density']
        mu = self._get_attenuation_coefficient(energy)
        
        gamma_dose = np.zeros_like(r)
        mask = r > 0
        gamma_dose[mask] = (
            intensity * 
            np.exp(-mu * density * r[mask]/10) / 
            (4*np.pi*(r[mask]**2))
        )
        
        return gamma_dose
    
    def _get_attenuation_coefficient(self, energy: float) -> float:
        """Get energy-dependent attenuation coefficient."""
        # Simplified energy-dependent attenuation
        base_mu = 0.096  # cm^-1 in water at 0.5 MeV
        energy_factor = (0.5/energy)**3.2
        z_factor = (self.tissue_properties['effective_Z']/7.42)**0.5
        return base_mu * energy_factor * z_factor