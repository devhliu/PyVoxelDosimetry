from pathlib import Path
import numpy as np
from typing import Tuple, Optional
from scipy.interpolate import interp1d
from .base_kernel import BaseKernelGenerator

class Tb161KernelGenerator(BaseKernelGenerator):
    """
    Tb-161 specific dose kernel generator.
    Handles multiple beta emissions, low-energy gamma lines and tissue composition effects.
    """
    
    def __init__(self, tissue_type: str):
        config_path = Path(__file__).parent / "Tb161" / "Tb161.json"
        super().__init__(config_path, tissue_type)
        self.beta_energies = self.config['nuclide']['particle_energies']['beta-_max']
        self.beta_abundances = self.config['nuclide']['particle_energies']['beta-_abundance']
        self.gamma_lines = self.config['nuclide']['particle_energies']['gamma_lines']
        self.tissue_properties = self._get_tissue_properties()
        
    def _get_tissue_properties(self) -> dict:
        """Get tissue-specific properties."""
        properties = {
            'water': {
                'density': 1.0,
                'electron_density': 3.34e23,
                'effective_Z': 7.42,
                'stopping_power_ratio': 1.0,
                'mu_by_rho': 0.096
            },
            'lung': {
                'density': 0.26,
                'electron_density': 0.87e23,
                'effective_Z': 7.41,
                'stopping_power_ratio': 1.04,
                'mu_by_rho': 0.095
            },
            'soft_tissue': {
                'density': 1.04,
                'electron_density': 3.48e23,
                'effective_Z': 7.46,
                'stopping_power_ratio': 1.04,
                'mu_by_rho': 0.097
            },
            'bone': {
                'density': 1.85,
                'electron_density': 5.91e23,
                'effective_Z': 13.8,
                'stopping_power_ratio': 1.15,
                'mu_by_rho': 0.110
            },
            'iodine_contrast': {
                'density': 1.30,
                'electron_density': 4.35e23,
                'effective_Z': 53.0,
                'stopping_power_ratio': 1.12,
                'mu_by_rho': 0.245
            }
        }
        return properties.get(self.tissue_type, properties['water'])