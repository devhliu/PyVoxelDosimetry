"""
Tissue composition calculation from CT images.
"""
import numpy as np
from typing import Dict, Any, Optional
import nibabel as nib
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
import json

@dataclass
class TissueProperties:
    density: float  # g/cm3
    electron_density: float  # e/cm3
    effective_Z: float
    chemical_composition: Dict[str, float]  # element: mass fraction
    hu_value: float
    description: str

class TissueManager:
    def __init__(self):
        self.tissue_db = {
            'water': TissueProperties(
                density=1.0,
                electron_density=3.34e23,
                effective_Z=7.42,
                chemical_composition={'H': 0.111894, 'O': 0.888106},
                hu_value=0,
                description='Water reference material'
            ),
            # Add other tissue types
        }
    
    def load_tissue(self, name: str) -> TissueProperties:
        return self.tissue_db.get(name)

class TissueComposition:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.tissue_types = {
            'air': {'hu_range': (-1000, -900)},
            'lung': {'hu_range': (-900, -500)},
            'soft_tissue': {'hu_range': (-100, 100)},
            'bone': {'hu_range': (300, 3000)},
            'water': {'hu_range': (-10, 10)}
        }
        
    def calculate_composition(self,
                            ct_image: np.ndarray,
                            handle_artifacts: bool = True) -> Dict[str, np.ndarray]:
        """
        Calculate tissue composition maps from CT image.
        
        Args:
            ct_image: CT image array in Hounsfield units
            handle_artifacts: Whether to handle CT artifacts
            
        Returns:
            Dictionary of tissue probability maps
        """
        if handle_artifacts:
            ct_image = self._handle_artifacts(ct_image)
            
        composition_maps = {}
        for tissue, properties in self.tissue_types.items():
            min_hu, max_hu = properties['hu_range']
            mask = (ct_image >= min_hu) & (ct_image <= max_hu)
            composition_maps[tissue] = mask.astype(float)
            
        return composition_maps
    
    def _handle_artifacts(self, ct_image: np.ndarray) -> np.ndarray:
        """
        Handle CT artifacts including metal implants and contrast.
        
        Args:
            ct_image: Original CT image
            
        Returns:
            Processed CT image
        """
        # Metal artifact reduction
        metal_threshold = 2000  # HU
        metal_mask = ct_image > metal_threshold
        
        # Simple metal artifact reduction by smoothing
        corrected_image = ct_image.copy()
        corrected_image[metal_mask] = np.nan
        
        # Fill holes using Gaussian smoothing
        smoothed = gaussian_filter(np.nan_to_num(corrected_image), sigma=1)
        corrected_image[metal_mask] = smoothed[metal_mask]
        
        return corrected_image