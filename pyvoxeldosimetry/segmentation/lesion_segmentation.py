"""
Lesion segmentation using nnU-Net v2
"""
from typing import Dict, List, Optional
import numpy as np
from pathlib import Path
from .nnunet_wrapper import NNUNetWrapper

class LesionSegmentation:
    def __init__(self,
                 modality: str = "pet_ct",
                 use_gpu: bool = True):
        """
        Initialize lesion segmentation
        
        Args:
            modality: Imaging modality ('pet_ct' or 'spect_ct')
            use_gpu: Whether to use GPU for inference
        """
        self.modality = modality
        self.models_path = Path(__file__).parent.parent / "data/pretrained_models/lesion_segmentation"
        
        model_folder = self._get_model_path()
        self.segmenter = NNUNetWrapper(
            model_folder=str(model_folder),
            use_gpu=use_gpu
        )
        
    def _get_model_path(self) -> Path:
        """Get path to the appropriate model based on modality."""
        model_path = self.models_path / self.modality
        if not model_path.exists():
            raise ValueError(f"Model for modality {self.modality} not found")
        return model_path
        
    def segment_lesions(self,
                       functional_image: np.ndarray,
                       ct_image: np.ndarray,
                       spacing: tuple = None,
                       threshold: Optional[float] = None) -> np.ndarray:
        """
        Perform lesion segmentation on functional (PET/SPECT) and CT images
        
        Args:
            functional_image: PET or SPECT image
            ct_image: Corresponding CT image
            spacing: Input image spacing (if None, assumed isotropic 1mm)
            threshold: Optional threshold for probability maps
            
        Returns:
            Binary mask of lesions
        """
        # Prepare input for nnU-Net
        input_images = {
            '0': ct_image,  # CT is typically channel 0
            '1': functional_image  # Functional image is channel 1
        }
        
        # Run segmentation
        segmentation = self.segmenter.predict(
            input_images=input_images,
            spacing=spacing
        )
        
        if threshold is not None:
            segmentation = segmentation > threshold
            
        return segmentation.astype(bool)