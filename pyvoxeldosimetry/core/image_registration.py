"""
Image registration and resampling utilities for PyVoxelDosimetry.

Created: 2025-02-08 09:45:41
Author: devhliu
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union
import SimpleITK as sitk
from dataclasses import dataclass

@dataclass
class RegistrationResult:
    """Container for registration results."""
    transformed_image: np.ndarray
    transform_parameters: Dict[str, Any]
    metric_value: float
    success: bool

class ImageRegistration:
    """Image registration class for aligning activity maps and anatomical images."""
    
    def __init__(self,
                 method: str = 'rigid',
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize image registration.
        
        Args:
            method: Registration method ('rigid', 'affine', 'deformable')
            config: Configuration parameters for registration
        """
        self.method = method.lower()
        self.config = config or {}
        self._initialize_registration()
        
    def _initialize_registration(self):
        """Initialize registration algorithm based on method."""
        if self.method == 'rigid':
            self.transform = sitk.Euler3DTransform()
            # self.optimizer = sitk.RegularStepGradientDescentOptimizer()  # REMOVE
            # self.metric = sitk.MattesMutualInformationImageMetric()      # REMOVE
        elif self.method == 'affine':
            self.transform = sitk.AffineTransform(3)
            # self.optimizer = sitk.RegularStepGradientDescentOptimizer()  # REMOVE
            # self.metric = sitk.MattesMutualInformationImageMetric()      # REMOVE
        elif self.method == 'deformable':
            self.transform = sitk.BSplineTransformInitializer(sitk.Image([10,10,10], sitk.sitkFloat32))
            # self.optimizer = sitk.LBFGSBOptimizer()                     # REMOVE
            # self.metric = sitk.MeanSquaresImageMetric()                 # REMOVE
        else:
            raise ValueError(f"Unsupported registration method: {self.method}")
            
    def register(self,
                fixed_image: np.ndarray,
                moving_image: np.ndarray,
                fixed_spacing: Tuple[float, float, float],
                moving_spacing: Tuple[float, float, float]
                ) -> RegistrationResult:
        """
        Register moving image to fixed image.
        
        Args:
            fixed_image: Target image array
            moving_image: Image to be transformed
            fixed_spacing: Voxel spacing of fixed image (mm)
            moving_spacing: Voxel spacing of moving image (mm)
            
        Returns:
            RegistrationResult object containing transformed image and parameters
        """
        # Convert to SimpleITK images
        fixed_sitk = sitk.GetImageFromArray(fixed_image.astype(np.float32))
        fixed_sitk.SetSpacing(fixed_spacing)
        
        moving_sitk = sitk.GetImageFromArray(moving_image.astype(np.float32))
        moving_sitk.SetSpacing(moving_spacing)
        
        # Set up registration
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMeanSquares()
        registration.SetOptimizerAsRegularStepGradientDescent(
            learningRate=self.config.get('learning_rate', 1.0),
            minStep=self.config.get('min_step', 0.01),
            numberOfIterations=self.config.get('max_iterations', 100)
        )
        registration.SetInitialTransform(self.transform)
        
        try:
            # Perform registration
            final_transform = registration.Execute(fixed_sitk, moving_sitk)
            
            # Apply transform
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed_sitk)
            resampler.SetTransform(final_transform)
            resampler.SetInterpolator(sitk.sitkLinear)
            
            transformed_sitk = resampler.Execute(moving_sitk)
            transformed_image = sitk.GetArrayFromImage(transformed_sitk)
            
            return RegistrationResult(
                transformed_image=transformed_image,
                transform_parameters=self._get_transform_parameters(final_transform),
                metric_value=registration.GetMetricValue(),
                success=True
            )
            
        except RuntimeError as e:
            print(f"Registration failed: {str(e)}")
            return RegistrationResult(
                transformed_image=moving_image,
                transform_parameters={},
                metric_value=float('inf'),
                success=False
            )
            
    def _get_transform_parameters(self, transform: sitk.Transform) -> Dict[str, Any]:
        """Extract parameters from transform."""
        params = {
            'type': self.method,
            'parameters': transform.GetParameters(),
            'fixed_parameters': transform.GetFixedParameters()
        }
        return params
    
    def resample_to_reference(self,
                            image: np.ndarray,
                            input_spacing: Tuple[float, float, float],
                            reference_spacing: Tuple[float, float, float],
                            interpolation: str = 'linear'
                            ) -> np.ndarray:
        """
        Resample image to match reference spacing.
        
        Args:
            image: Input image array
            input_spacing: Input voxel spacing (mm)
            reference_spacing: Target voxel spacing (mm)
            interpolation: Interpolation method ('linear', 'nearest', 'cubic')
            
        Returns:
            Resampled image array
        """
        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
        sitk_image.SetSpacing(input_spacing)
        
        # Calculate output size
        output_size = [
            int(round(image.shape[i] * input_spacing[i] / reference_spacing[i]))
            for i in range(3)
        ]
        
        # Set interpolation
        if interpolation == 'linear':
            interp_method = sitk.sitkLinear
        elif interpolation == 'nearest':
            interp_method = sitk.sitkNearestNeighbor
        elif interpolation == 'cubic':
            interp_method = sitk.sitkBSpline
        else:
            raise ValueError(f"Unsupported interpolation method: {interpolation}")
            
        # Perform resampling
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(output_size)
        resampler.SetOutputSpacing(reference_spacing)
        resampler.SetInterpolator(interp_method)
        resampler.SetDefaultPixelValue(0)
        
        resampled_sitk = resampler.Execute(sitk_image)
        return sitk.GetArrayFromImage(resampled_sitk)