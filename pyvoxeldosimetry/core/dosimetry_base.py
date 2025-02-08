"""
Base class for dosimetry calculations.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional
import nibabel as nib

class DosimetryCalculator(ABC):
    def __init__(self, 
                 radionuclide: str,
                 tissue_composition: Any,
                 config: Dict[str, Any] = None):
        self.radionuclide = radionuclide
        self.tissue_composition = tissue_composition
        self.config = config or {}
        
    @abstractmethod
    def calculate_dose_rate(self, 
                          activity_map: np.ndarray,
                          voxel_size: tuple) -> np.ndarray:
        """Calculate dose rate for given activity distribution."""
        pass
    
    def save_results(self, 
                    dose_map: np.ndarray,
                    output_path: str,
                    reference_nifti: Optional[nib.Nifti1Image] = None):
        """Save dose map as NIfTI file."""
        if reference_nifti is not None:
            nifti_image = nib.Nifti1Image(dose_map, reference_nifti.affine)
        else:
            nifti_image = nib.Nifti1Image(dose_map, np.eye(4))
        nib.save(nifti_image, output_path)