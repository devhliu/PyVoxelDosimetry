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
        # Use float32 for large arrays to improve speed/memory
        self.dtype = cp.float32
        
    def _initialize_physics(self):
        """Initialize physics parameters and interaction tables."""
        self.interaction_tables = {
            'beta': self._initialize_beta_tables(),
            'gamma': self._initialize_gamma_tables(),
            'alpha': self._initialize_alpha_tables()
        }
        # Remove explicit kernel compilation, use CuPy fused kernels instead

    def _initialize_particle_positions(self, activity_gpu):
        # Vectorized sampling of initial positions based on activity map
        flat_activity = activity_gpu.ravel()
        probs = flat_activity / cp.sum(flat_activity)
        indices = self.rng.choice(flat_activity.size, size=self.n_particles, p=probs)
        positions = cp.stack(cp.unravel_index(indices, activity_gpu.shape), axis=1)
        return positions.astype(cp.int32)

    def _initialize_particle_directions(self):
        # Vectorized isotropic directions
        phi = self.rng.uniform(0, 2 * cp.pi, self.n_particles, dtype=self.dtype)
        costheta = self.rng.uniform(-1, 1, self.n_particles, dtype=self.dtype)
        sintheta = cp.sqrt(1 - costheta ** 2)
        directions = cp.stack([sintheta * cp.cos(phi), sintheta * cp.sin(phi), costheta], axis=1)
        return directions

    def _sample_initial_energies(self):
        # Vectorized sampling from decay spectrum (placeholder: monoenergetic)
        # For real use, sample from self.decay_data['spectrum'] if available
        energy = cp.full(self.n_particles, self.decay_data.get('mean_energy', 1.0), dtype=self.dtype)
        return energy

    @cp.fuse()
    def _fused_transport_kernel(positions, directions, energies, dose_map, voxel_size, interaction_length):
        # Example: simple straight-line transport with energy deposition
        # This is a placeholder for a real physics model
        # positions: (N, 3), directions: (N, 3), energies: (N,)
        # dose_map: 3D array (on GPU)
        # interaction_length: mean free path (scalar)
        # For each particle, move one step and deposit energy
        new_positions = positions + (directions * interaction_length / cp.array(voxel_size, dtype=cp.float32))
        new_positions = cp.clip(new_positions, 0, cp.array(dose_map.shape) - 1)
        idx = new_positions.astype(cp.int32)
        # Deposit all energy in the new voxel (placeholder)
        cp.atomic.add(dose_map, (idx[:,0], idx[:,1], idx[:,2]), energies)
        # Set energies to zero (absorbed)
        return positions, directions, cp.zeros_like(energies)

    def _simulate_beta(self, positions, directions, energies, dose_map, voxel_size):
        # Example: batch transport loop, fully vectorized
        interaction_length = cp.float32(0.1)  # Placeholder: mean free path in mm
        mask = energies > 0
        while cp.any(mask):
            pos, dir, en = self._fused_transport_kernel(
                positions[mask], directions[mask], energies[mask], dose_map, voxel_size, interaction_length
            )
            positions[mask], directions[mask], energies[mask] = pos, dir, en
            mask = energies > 0

    def _simulate_gamma(self, positions, directions, energies, dose_map, voxel_size):
        # Placeholder: similar to beta, but with different interaction length
        interaction_length = cp.float32(1.0)  # Placeholder: mean free path in mm
        mask = energies > 0
        while cp.any(mask):
            pos, dir, en = self._fused_transport_kernel(
                positions[mask], directions[mask], energies[mask], dose_map, voxel_size, interaction_length
            )
            positions[mask], directions[mask], energies[mask] = pos, dir, en
            mask = energies > 0

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
        activity_gpu = cp.asarray(activity_map, dtype=self.dtype)
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
        
        # Convert deposited energy to dose rate (Gy/s)
        # Placeholder: assume 1 Bq per particle, scale as needed
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
        dose_map = np.zeros_like(activity_maps[0], dtype=np.float32)
        
        # Calculate dose rate at each time point
        dose_rates = [self.calculate_dose_rate(act_map, voxel_size)
                     for act_map in activity_maps]
        
        # Integrate dose rates over time
        for i in range(len(time_points) - 1):
            dt = (time_points[i+1] - time_points[i]) * 3600  # Convert to seconds
            avg_dose_rate = (dose_rates[i] + dose_rates[i+1]) / 2
            dose_map += avg_dose_rate * dt
            
        return dose_map

    # Variance reduction: Russian roulette (example, can be improved)
    def _russian_roulette(self, energies, threshold=0.05):
        survive = self.rng.uniform(0, 1, energies.shape, dtype=self.dtype) < threshold
        energies = cp.where(survive, energies / threshold, 0)
        return energies