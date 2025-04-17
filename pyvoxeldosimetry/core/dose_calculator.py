"""
Main dose calculation module for PyVoxelDosimetry.

Created: 2025-02-08 09:45:41
Author: devhliu
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass

from pyvoxeldosimetry.core.dosimetry_base import DosimetryCalculator
from pyvoxeldosimetry.io.dicom import (
    load_dicom_series, convert_pet_dicom_to_mhd, convert_ct_dicom_to_mhd,
    is_dicom_directory, is_pet_dicom, is_ct_dicom, convert_mhd_to_dicom, detect_input_format
)
from pyvoxeldosimetry.io.nifti import is_nifti_file, convert_mhd_to_nifti
from pyvoxeldosimetry.core.monte_carlo import MonteCarloCalculator
from pyvoxeldosimetry.core.kernel_convolution import KernelConvolutionCalculator
from pyvoxeldosimetry.core.gate_monte_carlo import GateMonteCarloCalculator
from pyvoxeldosimetry.core.activity_sampler import ActivitySampler
from pyvoxeldosimetry.core.image_registration import ImageRegistration

@dataclass
class DoseCalculationResult:
    """Container for dose calculation results."""
    absorbed_dose: Optional[np.ndarray]
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
                tissue_name=self.config.get('tissue_name', 'water'),
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
        
    def calculate_dose(
        self,
        activity_maps: Optional[List[np.ndarray]] = None,
        time_points: Optional[List[float]] = None,
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        tissue_densities: Optional[np.ndarray] = None,
        accumulated_activity: Optional[np.ndarray] = None,
        integration_mode: str = "activity",  # or "dose_rate"
        integration_limit: Optional[float] = None  # e.g., 10 * half_life
    ) -> DoseCalculationResult:
        """
        Calculate absorbed dose or dose rate depending on input.
        """
        # Validate voxel size
        if len(voxel_size) != 3:
            raise ValueError("voxel_size must be a tuple of length 3.")

        # --- Accumulated activity input ---
        if accumulated_activity is not None:
            absorbed_dose = self.calculator.calculate_absorbed_dose_from_accumulated(
                accumulated_activity, voxel_size
            )
            return DoseCalculationResult(
                absorbed_dose=absorbed_dose,
                dose_rate_maps=[],
                time_points=[],
                metadata={"mode": "accumulated_activity"}
            )

        # --- Multiple timepoints ---
        if activity_maps is not None and time_points is not None and len(activity_maps) > 1:
            self._validate_inputs(activity_maps, time_points, voxel_size)
            aligned_maps = self._align_activity_maps(activity_maps, voxel_size)
            if integration_mode == "activity":
                # Integrate activity, then calculate dose
                integrated_activity = self.activity_sampler.integrate_activity(
                    aligned_maps, time_points
                )
                absorbed_dose = self.calculator.calculate_absorbed_dose_from_accumulated(
                    integrated_activity, voxel_size
                )
                return DoseCalculationResult(
                    absorbed_dose=absorbed_dose,
                    dose_rate_maps=[],
                    time_points=time_points,
                    metadata={"mode": "multi_timepoint_activity"}
                )
            elif integration_mode == "dose_rate":
                # Calculate dose rate at each timepoint, then integrate
                dose_rates = [
                    self.calculator.calculate_dose_rate(act, voxel_size)
                    for act in aligned_maps
                ]
                absorbed_dose = self.activity_sampler.integrate_dose_rates(
                    dose_rates, time_points, integration_limit
                )
                return DoseCalculationResult(
                    absorbed_dose=absorbed_dose,
                    dose_rate_maps=dose_rates,
                    time_points=time_points,
                    metadata={"mode": "multi_timepoint_doserate"}
                )
            else:
                raise ValueError(f"Unknown integration_mode: {integration_mode}")

        # --- Single timepoint ---
        if activity_maps is not None and len(activity_maps) == 1:
            self._validate_inputs(activity_maps, time_points or [0], voxel_size)
            aligned_maps = self._align_activity_maps(activity_maps, voxel_size)
            dose_rate = self.calculator.calculate_dose_rate(
                aligned_maps[0], voxel_size
            )
            return DoseCalculationResult(
                absorbed_dose=None,
                dose_rate_maps=[dose_rate],
                time_points=time_points or [],
                metadata={"mode": "single_timepoint"}
            )

        raise ValueError("Invalid input for dose calculation.")
        
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