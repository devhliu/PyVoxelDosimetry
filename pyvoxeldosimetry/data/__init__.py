"""
Data module for PyVoxelDosimetry
"""
from pathlib import Path

# Define paths to data directories
DOSE_KERNELS_PATH = Path(__file__).parent / "dose_kernels"
PRETRAINED_MODELS_PATH = Path(__file__).parent / "pretrained_models"

# Ensure directories exist
DOSE_KERNELS_PATH.mkdir(parents=True, exist_ok=True)
PRETRAINED_MODELS_PATH.mkdir(parents=True, exist_ok=True)

def get_dose_kernel_path(radionuclide: str) -> Path:
    """Get path to dose kernel data for specific radionuclide."""
    kernel_path = DOSE_KERNELS_PATH / radionuclide
    if not kernel_path.exists():
        raise ValueError(f"Dose kernel data not found for {radionuclide}")
    return kernel_path

def get_pretrained_model_path(model_type: str, task: str) -> Path:
    """Get path to pretrained model."""
    model_path = PRETRAINED_MODELS_PATH / task / model_type
    if not model_path.exists():
        raise ValueError(f"Pretrained model not found for {task}/{model_type}")
    return model_path