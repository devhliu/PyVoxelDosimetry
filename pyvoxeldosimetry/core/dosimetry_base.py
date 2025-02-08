"""
Base class for dosimetry calculations.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple

class DosimetryCalculator(ABC):
    """Abstract base class for all dosimetry calculation methods."""
    
    def __init__(self, 
                 radionuclide: str,
                 tissue_composition: Any,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize dosimetry calculator.
        
        Args:
            radionuclide: Name of the radionuclide
            tissue_composition: Object describing tissue properties
            config: Additional configuration parameters
        """
        self.radionuclide = radionuclide
        self.tissue_composition = tissue_composition
        self.config = config or {}
        self._validate_inputs()
        
    def _validate_inputs(self):
        """Validate initialization inputs."""
        if not isinstance(self.radionuclide, str):
            raise TypeError("Radionuclide must be a string")
        
    @abstractmethod
    def calculate_dose_rate(self,
                          activity_map: np.ndarray,
                          voxel_size: Tuple[float, float, float]
                          ) -> np.ndarray:
        """
        Calculate dose rate from activity distribution.
        
        Args:
            activity_map: 3D array of activity values (Bq)
            voxel_size: Tuple of voxel dimensions (mm)
            
        Returns:
            3D array of dose rate values (Gy/s)
        """
        pass
    
    @abstractmethod
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
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Return current configuration."""
        return self.config.copy()