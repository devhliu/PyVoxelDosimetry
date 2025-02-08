"""
Organ segmentation using nnU-Net v2
"""
from typing import Dict, List
import numpy as np
from pathlib import Path
from .nnunet_wrapper import NNUNetWrapper

class OrganSegmentation:
    def __init__(self, 
                 model_type: str = "total_body",
                 use_gpu: bool = True):
        """
        Initialize organ segmentation
        
        Args:
            model_type: Type of organ segmentation model to use
            use_gpu: Whether to use GPU for inference
        """
        self.model_type = model_type
        self.models_path = Path(__file__).parent.parent / "data/pretrained_models/organ_segmentation"
        
        model_folder = self._get_model_path()
        self.segmenter = NNUNetWrapper(
            model_folder=str(model_folder),
            use_gpu=use_gpu
        )
        
        self.organ_labels = self._load_organ_labels()
        
    def _get_model_path(self) -> Path:
        """Get path to the appropriate model based on model_type."""
        model_path = self.models_path / self.model_type
        if not model_path.exists():
            raise ValueError(f"Model type {self.model_type} not found")
        return model_path
        
    def _load_organ_labels(self) -> Dict[int, str]:
        """Load organ label definitions."""
        labels_file = self.models_path / self.model_type / "organ_labels.json"
        if labels_file.exists():
            import json
            with open(labels_file, 'r') as f:
                return json.load(f)
        return {}
        
    def segment_organs(self, 
                      ct_image: np.ndarray,
                      spacing: tuple = None) -> Dict[str, np.ndarray]:
        """
        Perform organ segmentation on CT image
        
        Args:
            ct_image: Input CT image
            spacing: Input image spacing (if None, assumed isotropic 1mm)
            
        Returns:
            Dictionary mapping organ names to binary masks
        """
        # Prepare input for nnU-Net
        input_images = {'0': ct_image}  # CT is typically channel 0
        
        # Run segmentation
        segmentation = self.segmenter.predict(
            input_images=input_images,
            spacing=spacing
        )
        
        # Convert to individual organ masks
        organ_masks = {}
        for label_id, organ_name in self.organ_labels.items():
            organ_masks[organ_name] = (segmentation == int(label_id))
            
        return organ_masks