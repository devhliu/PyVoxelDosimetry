"""
Dose point kernel convolution calculator.
"""
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from .dosimetry_base import DosimetryCalculator
from ..data.dose_kernels import load_kernel

@dataclass
class DecayProperties:
    half_life: float  # hours
    decay_modes: Dict[str, float]  # mode: branching ratio
    particle_energies: Dict[str, List[float]]  # mode: [energies in MeV]

@dataclass
class KernelConfig:
    name: str
    symbol: str
    atomic_number: int
    mass_number: int
    decay_properties: DecayProperties
    reference: str

class KernelConvolutionCalculator(DosimetryCalculator):
    def __init__(self,
                 radionuclide: str,
                 tissue_name: str,
                 kernel_resolution: float = 1.0,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize calculator."""
        super().__init__(radionuclide, tissue_name, config)
        self.kernel_manager = KernelManager()
        self.kernel_resolution = kernel_resolution
        self.tissue_name = tissue_name
        self._load_dose_kernel()
        
    def _load_dose_kernel(self):
        """Load or generate dose kernel."""
        self.kernel = self.kernel_manager.get_kernel(
            self.radionuclide,
            self.tissue_name,
            self.kernel_resolution,
            size=(64, 64, 64)  # Default size
        )
        
    def calculate_dose_rate(self,
                          activity_map: np.ndarray,
                          voxel_size: Tuple[float, float, float]
                          ) -> np.ndarray:
        """
        Calculate dose rate using kernel convolution.
        
        Args:
            activity_map: 3D array of activity values (Bq)
            voxel_size: Tuple of voxel dimensions (mm)
            
        Returns:
            3D array of dose rate values (Gy/s)
        """
        # Resample activity map if voxel sizes don't match
        if voxel_size != (self.kernel_resolution,)*3:
            activity_map = self._resample_activity(
                activity_map,
                voxel_size,
                (self.kernel_resolution,)*3
            )
            
        # Perform 3D convolution
        dose_rate = np.fft.ifftn(
            np.fft.fftn(activity_map) *
            np.fft.fftn(self.kernel, activity_map.shape)
        ).real
        
        return dose_rate
    
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
    
    def _resample_activity(self,
                          activity_map: np.ndarray,
                          input_voxel_size: Tuple[float, float, float],
                          output_voxel_size: Tuple[float, float, float]
                          ) -> np.ndarray:
        """Resample activity map to match kernel resolution."""
        # Implementation of activity map resampling
        pass