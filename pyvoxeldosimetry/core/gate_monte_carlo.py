"""
Gate Monte Carlo integration for dosimetry calculations.

This module provides integration with the GATE (Geant4 Application for Tomographic Emission)
Monte Carlo simulation toolkit for accurate dosimetry calculations.
"""

import os
import subprocess
import tempfile
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import SimpleITK as sitk
from pathlib import Path

from pyvoxeldosimetry.io.dicom import (
    load_dicom_series, convert_pet_dicom_to_mhd, convert_ct_dicom_to_mhd,
    is_dicom_directory, is_pet_dicom, is_ct_dicom, convert_mhd_to_dicom, detect_input_format
)
from pyvoxeldosimetry.io.nifti import is_nifti_file, convert_mhd_to_nifti

from pyvoxeldosimetry.core.dosimetry_base import DosimetryCalculator
from pyvoxeldosimetry.core.utils import save_dose_map, load_dose_map

class GateMonteCarloCalculator(DosimetryCalculator):
    """
    Dosimetry calculator using Gate Monte Carlo simulations.
    """
    
    def __init__(self, 
                 radionuclide: str,
                 tissue_composition: Any = None,
                 gate_path: Optional[str] = None,
                 n_particles: int = 1000000,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize Gate Monte Carlo calculator.
        
        Args:
            radionuclide: Name of the radionuclide
            tissue_composition: Object describing tissue properties
            gate_path: Path to Gate executable
            n_particles: Number of particles to simulate
            config: Additional configuration parameters
        """
        super().__init__(radionuclide, tissue_composition, config)
        self.n_particles = n_particles
        self.gate_path = gate_path or self._find_gate_executable()
        
        # Additional Gate-specific configuration
        self.physics_list = self.config.get('physics_list', 'QGSP_BIC_HP_EMZ')
        self.output_dir = self.config.get('output_dir', None)
        self.temp_dir = None
        
    def _find_gate_executable(self) -> str:
        """Find Gate executable in system path."""
        # Try common locations or environment variables
        gate_path = os.environ.get('GATE_HOME', '')
        if gate_path and os.path.exists(os.path.join(gate_path, 'bin', 'Gate')):
            return os.path.join(gate_path, 'bin', 'Gate')
        
        # Check in PATH
        try:
            result = subprocess.run(['which', 'Gate'], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
            
        # Default to 'Gate' and hope it's in the PATH
        return 'Gate'
    
    def _create_gate_macro(self, 
                          activity_map_path: str, 
                          ct_map_path: str, 
                          output_path: str,
                          voxel_size: Tuple[float, float, float]) -> str:
        """
        Create Gate macro file for simulation.
        
        Args:
            activity_map_path: Path to activity map file
            ct_map_path: Path to CT/density map file
            output_path: Path for output files
            voxel_size: Voxel dimensions in mm
            
        Returns:
            Path to created macro file
        """
        macro_content = f"""
/gate/geometry/setMaterialDatabase {self.config.get('material_db', 'GateMaterials.db')}

# World
/gate/world/geometry/setXLength 500 mm
/gate/world/geometry/setYLength 500 mm
/gate/world/geometry/setZLength 500 mm
/gate/world/setMaterial Air

# Patient geometry
/gate/world/daughters/name patient
/gate/world/daughters/insert compressedMatrix
/gate/patient/geometry/setRangeToMaterialFile {ct_map_path}
/gate/patient/geometry/setImage {ct_map_path}
/gate/patient/geometry/setVoxelSize {voxel_size[0]} {voxel_size[1]} {voxel_size[2]} mm

# Physics
/gate/physics/addPhysicsList {self.physics_list}

# Activity source
/gate/source/addSource source voxel
/gate/source/source/reader/insert image
/gate/source/source/imageReader/translator/insert linear
/gate/source/source/imageReader/linearTranslator/setScale 1.0 Bq
/gate/source/source/setImage {activity_map_path}
/gate/source/source/setPosition 0. 0. 0. mm

# Set radionuclide properties
/gate/source/source/setType backtoback
/gate/source/source/gps/particle ion
/gate/source/source/gps/ion {self._get_radionuclide_params()}

# Output
/gate/actor/addActor DoseActor doseActor
/gate/actor/doseActor/save {output_path}
/gate/actor/doseActor/attachTo patient
/gate/actor/doseActor/stepHitType random
/gate/actor/doseActor/setResolution {activity_map_path}
/gate/actor/doseActor/enableDose true
/gate/actor/doseActor/enableSquaredDose true
/gate/actor/doseActor/enableUncertaintyDose true

# Simulation settings
/gate/random/setEngineName MersenneTwister
/gate/random/setEngineSeed auto
/gate/application/setTotalNumberOfPrimaries {self.n_particles}
/gate/application/start
"""
        
        # Write macro to temporary file
        macro_path = os.path.join(self.temp_dir, "simulation.mac")
        with open(macro_path, 'w') as f:
            f.write(macro_content)
            
        return macro_path
    
    def _get_radionuclide_params(self) -> str:
        """Get radionuclide parameters for Gate."""
        # This would be expanded with a proper database of radionuclide parameters
        radionuclide_params = {
            'F18': '9 18 0 0',
            'Ga68': '31 68 0 0',
            'Y90': '39 90 0 0',
            'Lu177': '71 177 0 0',
            'Tb161': '65 161 0 0',
            'Ac225': '89 225 0 0'
        }
        
        return radionuclide_params.get(self.radionuclide, '0 0 0 0')
    
    def _prepare_input_files(self, 
                           activity_map: np.ndarray, 
                           voxel_size: Tuple[float, float, float]) -> Tuple[str, str]:
        """
        Prepare input files for Gate simulation.
        
        Args:
            activity_map: 3D array of activity values
            voxel_size: Voxel dimensions in mm
            
        Returns:
            Tuple of (activity_map_path, ct_map_path)
        """
        # Create temporary directory if needed
        if self.output_dir:
            self.temp_dir = self.output_dir
            os.makedirs(self.temp_dir, exist_ok=True)
        else:
            self.temp_dir = tempfile.mkdtemp(prefix="gate_sim_")
        
        # Save activity map as MHD/RAW
        activity_path = os.path.join(self.temp_dir, "activity.mhd")
        activity_img = sitk.GetImageFromArray(activity_map.astype(np.float32))
        activity_img.SetSpacing([voxel_size[0], voxel_size[1], voxel_size[2]])
        sitk.WriteImage(activity_img, activity_path)
        
        # Create simple CT map (water equivalent for now)
        # In a real implementation, this would use the tissue_composition
        ct_map = np.ones_like(activity_map)
        ct_path = os.path.join(self.temp_dir, "ct.mhd")
        ct_img = sitk.GetImageFromArray(ct_map.astype(np.float32))
        ct_img.SetSpacing([voxel_size[0], voxel_size[1], voxel_size[2]])
        sitk.WriteImage(ct_img, ct_path)
        
        return activity_path, ct_path
    
    def _run_gate_simulation(self, macro_path: str) -> str:
        """
        Run Gate simulation using the provided macro.
        
        Args:
            macro_path: Path to Gate macro file
            
        Returns:
            Path to output dose file
        """
        cmd = [self.gate_path, macro_path]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Gate simulation failed: {e.stderr.decode()}")
        
        # Return path to expected output file
        return os.path.join(self.temp_dir, "dose.mhd")
    
    def _load_dose_results(self, dose_path: str, activity_map_shape: Tuple) -> np.ndarray:
        """
        Load dose results from Gate output.
        
        Args:
            dose_path: Path to dose output file
            activity_map_shape: Shape of original activity map
            
        Returns:
            3D array of dose values
        """
        if not os.path.exists(dose_path):
            raise FileNotFoundError(f"Dose file not found at {dose_path}")
        
        dose_img = sitk.ReadImage(dose_path)
        dose_array = sitk.GetArrayFromImage(dose_img)
        
        # Ensure the shape matches the input
        if dose_array.shape != activity_map_shape:
            raise ValueError(f"Output dose shape {dose_array.shape} doesn't match input shape {activity_map_shape}")
        
        return dose_array
    
    def calculate_dose_rate(self,
                          activity_map: np.ndarray,
                          voxel_size: Tuple[float, float, float]
                          ) -> np.ndarray:
        """
        Calculate dose rate using Gate Monte Carlo simulation.
        
        Args:
            activity_map: 3D array of activity values (Bq)
            voxel_size: Tuple of voxel dimensions (mm)
            
        Returns:
            3D array of dose rate values (Gy/s)
        """
        # Prepare input files
        activity_path, ct_path = self._prepare_input_files(activity_map, voxel_size)
        
        # Create output path
        output_path = os.path.join(self.temp_dir, "dose")
        
        # Create Gate macro
        macro_path = self._create_gate_macro(activity_path, ct_path, output_path, voxel_size)
        
        # Run Gate simulation
        dose_path = self._run_gate_simulation(macro_path)
        
        # Load and return results
        dose_rate = self._load_dose_results(dose_path, activity_map.shape)
        
        # Convert to Gy/s (Gate outputs total dose in Gy)
        # We need to normalize by simulation time and activity
        total_activity = np.sum(activity_map)
        if total_activity > 0:
            dose_rate = dose_rate * total_activity  # Scale by total activity
            
        return dose_rate
    
    def calculate_absorbed_dose(self,
                              activity_maps: List[np.ndarray],
                              time_points: List[float],
                              voxel_size: Tuple[float, float, float]
                              ) -> np.ndarray:
        """
        Calculate absorbed dose from time series of activity maps.
        
        Args:
            activity_maps: List of 3D activity arrays
            time_points: List of time points (hours)
            voxel_size: Tuple of voxel dimensions (mm)
            
        Returns:
            3D array of absorbed dose values (Gy)
        """
        if len(activity_maps) != len(time_points):
            raise ValueError("Number of activity maps must match number of time points")
        
        if len(activity_maps) < 2:
            raise ValueError("At least two time points are required for dose calculation")
        
        # Calculate dose rate at each time point
        dose_rates = []
        for activity_map in activity_maps:
            dose_rate = self.calculate_dose_rate(activity_map, voxel_size)
            dose_rates.append(dose_rate)
        
        # Integrate dose rates over time (trapezoidal rule)
        total_dose = np.zeros_like(dose_rates[0])
        for i in range(len(time_points) - 1):
            dt = (time_points[i+1] - time_points[i]) * 3600  # Convert hours to seconds
            avg_dose_rate = (dose_rates[i] + dose_rates[i+1]) / 2
            total_dose += avg_dose_rate * dt
        
        return total_dose