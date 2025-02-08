"""
Wrapper for nnU-Net v2 integration in PyVoxelDosimetry
"""
from typing import Dict, Union, List
import os
import numpy as np
import torch
from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data
from nnunetv2.paths import nnUNet_results, nnUNet_raw

class NNUNetWrapper:
    def __init__(self, 
                 model_folder: str,
                 use_gpu: bool = True,
                 verbose: bool = False):
        """
        Initialize nnU-Net wrapper
        
        Args:
            model_folder: Path to the trained model
            use_gpu: Whether to use GPU for inference
            verbose: Whether to print detailed information
        """
        self.model_folder = model_folder
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.verbose = verbose
        self._verify_model_exists()
        
    def _verify_model_exists(self):
        """Verify that the model exists in the specified folder."""
        model_path = Path(self.model_folder)
        if not model_path.exists():
            raise FileNotFoundError(f"Model folder not found: {self.model_folder}")
            
    def predict(self, 
                input_images: Dict[str, np.ndarray],
                spacing: tuple = None) -> np.ndarray:
        """
        Run inference using nnU-Net
        
        Args:
            input_images: Dictionary of input images (e.g., {'0': ct_image})
            spacing: Input image spacing (if None, assumed isotropic 1mm)
            
        Returns:
            Segmentation mask
        """
        # Prepare temporary folders for nnU-Net
        temp_input_folder = Path(nnUNet_raw) / "temp_inference_input"
        temp_output_folder = Path(nnUNet_raw) / "temp_inference_output"
        
        os.makedirs(temp_input_folder, exist_ok=True)
        os.makedirs(temp_output_folder, exist_ok=True)
        
        try:
            # Save input images in nnU-Net format
            for modality, image in input_images.items():
                np.save(
                    temp_input_folder / f"input_{modality}.npy",
                    image.astype(np.float32)
                )
            
            # Run prediction
            predict_from_raw_data(
                input_folder=str(temp_input_folder),
                output_folder=str(temp_output_folder),
                model_folder=self.model_folder,
                use_gpu=self.use_gpu,
                verbose=self.verbose,
                save_probabilities=False,
                use_sliding_window=True
            )
            
            # Load results
            result_file = next(temp_output_folder.glob("*.npy"))
            segmentation = np.load(result_file)
            
            return segmentation
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_input_folder, ignore_errors=True)
            shutil.rmtree(temp_output_folder, ignore_errors=True)