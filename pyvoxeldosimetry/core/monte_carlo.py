"""
GPU-accelerated Monte Carlo dosimetry calculator.
"""
import numpy as np
import cupy as cp
from typing import Dict, Any, Optional, List, Tuple
from .dosimetry_base import DosimetryCalculator
from ..data.mc_decay import load_decay_data

class MonteCarloCalculator(DosimetryCalculator):
    def __init__(self,
                 radionuclide: str,
                 tissue_composition: Any,
                 n_particles: int = 1000000,
                 random_seed: Optional[int] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize Monte Carlo calculator.
        
        Args:
            radionuclide: Radionuclide name
            tissue_composition: Tissue composition object
            n_particles: Number of particles to simulate
            random_seed: Random seed for reproducibility
            config: Additional configuration parameters
        """
        super().__init__(radionuclide, tissue_composition, config)
        self.n_particles = n_particles
        self.decay_data = load_decay_data(radionuclide)
        self.rng = cp.random.RandomState(random_seed)
        self._initialize_physics()
        
    def _initialize_physics(self):
        """Initialize physics parameters and interaction tables."""
        self.interaction_tables = {
            'beta': self._initialize_beta_tables(),
            'gamma': self._initialize_gamma_tables(),
            'alpha': self._initialize_alpha_tables()
        }
        
        # Compile CUDA kernels for different particle types
        self.kernels = {
            'beta': self._compile_beta_kernel(),
            'gamma': self._compile_gamma_kernel(),
            'alpha': self._compile_alpha_kernel()
        }
        
    def calculate_dose_rate(self,
                          activity_map: np.ndarray,
                          voxel_size: Tuple[float, float, float]
                          ) -> np.ndarray:
        """
        Calculate dose rate using GPU-accelerated Monte Carlo.
        
        Args:
            activity_map: 3D array of activity values (Bq)
            voxel_size: Tuple of voxel dimensions (mm)
            
        Returns:
            3D array of dose rate values (Gy/s)
        """
        activity_gpu = cp.asarray(activity_map)
        dose_map_gpu = cp.zeros_like(activity_gpu)
        
        # Initialize particle states
        particle_positions = self._initialize_particle_positions(activity_gpu)
        particle_directions = self._initialize_particle_directions()
        particle_energies = self._sample_initial_energies()
        
        # Simulate each decay mode
        for mode in self.decay_data['decay_modes']:
            if mode['type'] == 'beta-':
                self._simulate_beta(particle_positions, particle_directions,
                                 particle_energies, dose_map_gpu, voxel_size)
            elif mode['type'] in ['gamma', 'x-ray']:
                self._simulate_gamma(particle_positions, particle_directions,
                                  particle_energies, dose_map_gpu, voxel_size)
        
        return cp.asnumpy(dose_map_gpu)
    
    def calculate_absorbed_dose(self,
                              activity_maps: List[np.ndarray],
                              time_points: List[float],
                              voxel_size: Tuple[float, float, float]
                              ) -> np.ndarray:
        """
        Calculate absorbed dose from time series.
        
        Args:
            activity_maps: List of 3D activity arrays
            time_points: List of time points (hours)
            voxel_size: Tuple of voxel dimensions (mm)
            
        Returns:
            3D array of absorbed dose values (Gy)
        """
        dose_map = np.zeros_like(activity_maps[0])
        
        # Calculate dose rate at each time point
        dose_rates = [self.calculate_dose_rate(act_map, voxel_size)
                     for act_map in activity_maps]
        
        # Integrate dose rates over time
        for i in range(len(time_points) - 1):
            dt = (time_points[i+1] - time_points[i]) * 3600  # Convert to seconds
            avg_dose_rate = (dose_rates[i] + dose_rates[i+1]) / 2
            dose_map += avg_dose_rate * dt
            
        return dose_map