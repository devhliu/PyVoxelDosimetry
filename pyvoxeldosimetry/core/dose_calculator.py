"""
Main dose calculation module for PyVoxelDosimetry.

Created: 2025-02-08 09:45:41
Author: devhliu
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from .dosimetry_base import DosimetryCalculator
from .monte_carlo import MonteCarloCalculator
from .kernel_convolution import KernelConvolutionCalculator
from .gate_monte_carlo import GateMonteCarloCalculator
from .activity_sampler import ActivitySampler
from .image_registration import ImageRegistration

@dataclass
class DoseCalculationResult:
    """Container for dose calculation results."""
    absorbed_dose: np.ndarray
    dose_rate_maps: List[np.ndarray]
    time_points: List[float]
    metadata: Dict[str, Any]

class DoseCalculator:
    """Main class for performing dose calculations."""
    
    def __init__(self,
                 radionuclide: str,
                 method: str = 'kernel',
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize dose calculator.
        
        Args:
            radionuclide: Name of radionuclide
            method: Calculation method ('kernel', 'gpu_monte_carlo', or 'gate_monte_carlo')
            config: Configuration parameters
        """
        self.radionuclide = radionuclide
        self.method = method.lower()
        self.config = config or {}
        
        # Initialize calculator based on method
        if self.method == 'kernel':
            self.calculator = KernelConvolutionCalculator(
                radionuclide=radionuclide,
                tissue_composition=self.config.get('tissue_composition', None),
                kernel_resolution=self.config.get('kernel_resolution', 1.0)
            )
        elif self.method == 'gpu_monte_carlo':
            self.calculator = MonteCarloCalculator(
                radionuclide=radionuclide,
                tissue_composition=self.config.get('tissue_composition', None),
                n_particles=self.config.get('n_particles', 1000000)
            )
        elif self.method == 'gate_monte_carlo':
            self.calculator = GateMonteCarloCalculator(
                radionuclide=radionuclide,
                tissue_composition=self.config.get('tissue_composition', None),
                gate_path=self.config.get('gate_path', None),
                n_particles=self.config.get('n_particles', 1000000),
                config=self.config
            )
        else:
            raise ValueError(f"Unsupported calculation method: {method}")
            
        # Initialize supporting components
        self.activity_sampler = ActivitySampler(
            half_life=self.config.get('half_life', 0.0),
            units=self.config.get('time_units', 'hours')
        )
        
        self.image_registration = ImageRegistration(
            method=self.config.get('registration_method', 'rigid')
        )
        
    def calculate_dose(self,
                      activity_maps: List[np.ndarray],
                      time_points: List[float],
                      voxel_size: Tuple[float, float, float],
                      tissue_densities: Optional[np.ndarray] = None
                      ) -> DoseCalculationResult:
        """
        Calculate absorbed dose from activity maps.
        
        Args:
            activity_maps: List of activity distribution arrays (Bq)
            time_points: List of measurement time points
            voxel_size: Voxel dimensions (mm)
            tissue_densities: Optional tissue density map (g/cm³)
            
        Returns:
            DoseCalculationResult object containing results and metadata
        """
        # Validate inputs
        self._validate_inputs(activity_maps, time_points, voxel_size)
        
        # Register images if needed
        aligned_maps = self._align_activity_maps(activity_maps, voxel_size)
        
        # Calculate dose rates
        dose_rates = []
        for act_map in aligned_maps:
            dose_rate = self.calculator.calculate_dose_rate(
                act_map,
                voxel_size=voxel_size
            )
            if tissue_densities is not None:
                dose_rate *= tissue_densities
            dose_rates.append(dose_rate)
            
        # Calculate absorbed dose
        absorbed_dose = self.calculator.calculate_absorbed_dose(
            aligned_maps,
            time_points,
            voxel_size
        )
        
        # Prepare metadata
        metadata = {
            'radionuclide': self.radionuclide,
            'method': self.method,
            'voxel_size': voxel_size,
            'time_points': time_points,
            'config': self.config,
            'timestamp': '2025-02-08 09:45:41',
            'user': 'devhliu'
        }
        
        return DoseCalculationResult(
            absorbed_dose=absorbed_dose,
            dose_rate_maps=dose_rates,
            time_points=time_points,
            metadata=metadata
        )
        
    def _validate_inputs(self,
                        activity_maps: List[np.ndarray],
                        time_points: List[float],
                        voxel_size: Tuple[float, float, float]):
        """Validate input parameters."""
        if not activity_maps:
            raise ValueError("No activity maps provided")
        if len(activity_maps) != len(time_points):
            raise ValueError("Number of activity maps must match number of time points")
        if len(voxel_size) != 3:
            raise ValueError("Voxel size must be 3D")
            
        shape = activity_maps[0].shape
        if not all(map.shape == shape for map in activity_maps):
            raise ValueError("All activity maps must have the same dimensions")
            
    def _align_activity_maps(self,
                           activity_maps: List[np.ndarray],
                           voxel_size: Tuple[float, float, float]
                           ) -> List[np.ndarray]:
        """Align activity maps if needed."""
        if len(activity_maps) == 1:
            return activity_maps
            
        reference = activity_maps[0]
        aligned_maps = [reference]
        
        for map_i in activity_maps[1:]:
            result = self.image_registration.register(
                fixed_image=reference,
                moving_image=map_i,
                fixed_spacing=voxel_size,
                moving_spacing=voxel_size
            )
            aligned_maps.append(result.transformed_image)
            
        return aligned_maps