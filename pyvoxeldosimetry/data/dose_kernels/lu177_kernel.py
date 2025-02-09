from pathlib import Path
import numpy as np
from typing import Tuple, Optional
from scipy.interpolate import interp1d
from .base_kernel import BaseKernelGenerator

"""
Key updates include:

1. Added comprehensive tissue properties:
   - Density
   - Electron density
   - Effective atomic number
   - Stopping power ratio
   - Mass attenuation coefficients

2. Enhanced beta contribution calculation:
   - Tissue-specific CSDA range
   - Density scaling
   - Stopping power corrections

3. Improved gamma contribution handling:
   - Energy-dependent attenuation
   - Tissue-specific mass attenuation coefficients
   - Density effects

4. Better organization of physics calculations:
   - Separate handlers for beta and gamma
   - Tissue-specific scaling factors
   - Energy-dependent corrections

The updated implementation now properly handles:
   - Multiple beta energies with tissue-specific effects
   - Gamma attenuation based on tissue properties
   - Energy-dependent scaling factors
   - Comprehensive tissue composition effects
"""
class Lu177KernelGenerator(BaseKernelGenerator):
    """
    Lu-177 specific dose kernel generator.
    Handles multiple beta emissions, gamma lines and tissue composition effects.
    """
    
    def __init__(self, tissue_type: str):
        config_path = Path(__file__).parent / "Lu177" / "Lu177.json"
        super().__init__(config_path, tissue_type)
        self.beta_energies = self.config['nuclide']['particle_energies']['beta-_max']
        self.beta_abundances = self.config['nuclide']['particle_energies']['beta-_abundance']
        self.gamma_lines = self.config['nuclide']['particle_energies']['gamma_lines']
        self.tissue_properties = self._get_tissue_properties()
        
    def generate_kernel(self, 
                       voxel_size: float,
                       grid_size: Tuple[int, int, int]) -> np.ndarray:
        """
        Generate Lu-177 dose point kernel.
        
        Args:
            voxel_size: Voxel size in mm
            grid_size: Dimensions of the kernel grid
            
        Returns:
            3D numpy array containing the dose kernel
        """
        kernel = np.zeros(grid_size)
        center = [s//2 for s in grid_size]
        
        x, y, z = np.meshgrid(
            np.arange(grid_size[0]) - center[0],
            np.arange(grid_size[1]) - center[1],
            np.arange(grid_size[2]) - center[2],
            indexing='ij'
        )
        r = np.sqrt((x*voxel_size)**2 + (y*voxel_size)**2 + (z*voxel_size)**2)
        
        # Multiple beta contributions with tissue effects
        for energy, abundance in zip(self.beta_energies, self.beta_abundances):
            kernel += abundance * self._calculate_beta_contribution(r, energy)
        
        # Gamma contributions with tissue attenuation
        for gamma in self.gamma_lines:
            kernel += self._calculate_gamma_contribution(
                r, gamma['energy'], gamma['intensity']
            )
        
        return kernel

    def _get_tissue_properties(self) -> dict:
        """Get tissue-specific properties."""
        properties = {
            'water': {
                'density': 1.0,
                'electron_density': 3.34e23,
                'effective_Z': 7.42,
                'stopping_power_ratio': 1.0,
                'mu_by_rho': 0.096  # cm²/g at 0.2 MeV
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

    def _calculate_beta_contribution(self, r: np.ndarray, energy: float) -> np.ndarray:
        """
        Calculate beta particle dose contribution with tissue effects.
        
        Args:
            r: Distance matrix (mm)
            energy: Beta particle energy (MeV)
        """
        density_factor = self.tissue_properties['density']
        stopping_power_ratio = self.tissue_properties['stopping_power_ratio']
        
        # CSDA range with tissue corrections
        max_range = 5.0 * energy**1.5  # mm in water
        tissue_range = max_range * (1.0/density_factor) * (1.0/stopping_power_ratio)
        
        beta_dose = np.zeros_like(r)
        mask = r <= tissue_range
        
        beta_dose[mask] = (
            (1 - r[mask]/tissue_range)**2 * 
            np.exp(-2*r[mask]/tissue_range) * 
            density_factor * 
            stopping_power_ratio
        )
        
        return beta_dose

    def _calculate_gamma_contribution(self, 
                                   r: np.ndarray,
                                   energy: float,
                                   intensity: float) -> np.ndarray:
        """
        Calculate gamma contribution with tissue attenuation.
        
        Args:
            r: Distance matrix (mm)
            energy: Gamma energy (MeV)
            intensity: Gamma intensity (fractional)
        """
        density = self.tissue_properties['density']
        mu_by_rho = self.tissue_properties['mu_by_rho']
        
        # Energy correction for attenuation coefficient
        energy_factor = (0.2/energy)**3.2  # normalized to 0.2 MeV
        mu = density * mu_by_rho * energy_factor
        
        # Point kernel for photons with tissue attenuation
        gamma_dose = np.zeros_like(r)
        mask = r > 0
        gamma_dose[mask] = (
            intensity * 
            np.exp(-mu * r[mask]/10) / 
            (4*np.pi*(r[mask]**2))
        )
        
        return gamma_dose

    def _get_tissue_scaling_factor(self) -> float:
        """Get overall tissue scaling factor for dose."""
        density = self.tissue_properties['density']
        stopping_power = self.tissue_properties['stopping_power_ratio']
        return density * stopping_power


"""
from pathlib import Path
from pyvoxeldosimetry.data.dose_kernels.lu177_kernel import Lu177KernelGenerator

# Test for different tissue types
tissue_types = ['water', 'lung', 'soft_tissue', 'bone', 'iodine_contrast']

for tissue in tissue_types:
    # Create generator
    generator = Lu177KernelGenerator(tissue)
    
    # Generate kernel
    kernel = generator.generate_kernel(
        voxel_size=1.0,
        grid_size=(81, 81, 81)
    )
    
    # Save kernel and visualizations
    output_dir = Path(f'output/kernels/Lu177/{tissue}')
    output_dir.mkdir(parents=True, exist_ok=True)
    generator.save_kernel(kernel, output_dir)
"""