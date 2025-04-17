"""
Time-activity curve fitting and dose accumulation.
"""
import numpy as np
from scipy.optimize import curve_fit
from typing import List, Tuple, Optional
import os
import glob
import nibabel as nib
import SimpleITK as sitk
from pyvoxeldosimetry.io.dicom import load_dicom_series, is_dicom_directory
from pyvoxeldosimetry.io.nifti import is_nifti_file

class TimeCurveFitting:
    def __init__(self, half_life: float):
        self.half_life = half_life
        self.decay_constant = np.log(2) / half_life
        
    def fit_time_activity_curve(self,
                              times: List[float],
                              activities: List[np.ndarray],
                              weight_factors: Optional[List[float]] = None
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit time-activity curves for each voxel.
        
        Args:
            times: List of time points
            activities: List of activity maps at each time point
            weight_factors: Optional weighting factors for fitting
            
        Returns:
            Tuple of fitted parameters and accumulated dose
        """
        times = np.array(times)
        activities = np.array(activities)
        
        if weight_factors is None:
            weight_factors = np.ones_like(times)
            
        # Reshape for voxel-wise fitting
        n_times = len(times)
        original_shape = activities[0].shape
        activities_reshaped = activities.reshape(n_times, -1)
        
        # Fit each voxel
        fitted_params = np.zeros((2, activities_reshaped.shape[1]))
        for i in range(activities_reshaped.shape[1]):
            try:
                popt, _ = curve_fit(
                    self._decay_function,
                    times,
                    activities_reshaped[:, i],
                    p0=[activities_reshaped[0, i], self.decay_constant],
                    sigma=1/np.array(weight_factors)
                )
                fitted_params[:, i] = popt
            except:
                fitted_params[:, i] = [0, self.decay_constant]
                
        # Calculate accumulated activity
        accumulated_dose = self._calculate_accumulated_dose(fitted_params)
        
        return (fitted_params.reshape((2, *original_shape)),
                accumulated_dose.reshape(original_shape))
    
    def _decay_function(self,
                       t: float,
                       A0: float,
                       lambda_: float) -> float:
        """Exponential decay function."""
        return A0 * np.exp(-lambda_ * t)
    
    def _calculate_accumulated_dose(self,
                                 fitted_params: np.ndarray,
                                 integration_limit: float = None) -> np.ndarray:
        """Calculate accumulated dose from fitted parameters."""
        if integration_limit is None:
            integration_limit = 100 * self.half_life
            
        A0 = fitted_params[0]
        lambda_ = fitted_params[1]
        
        return A0 / lambda_ * (1 - np.exp(-lambda_ * integration_limit))
    
    @staticmethod
    def load_timepoint_images(image_paths_or_dirs):
        """
        Load multiple time-point PET/SPECT images from DICOM series or NIFTI files.
        Args:
            image_paths_or_dirs (List[str]): List of paths to DICOM directories or NIFTI files for each time point.
        Returns:
            List[np.ndarray]: List of 3D numpy arrays (activity maps) for each time point.
        """
        images = []
        for path in image_paths_or_dirs:
            if os.path.isdir(path) and is_dicom_directory(path):
                # Load DICOM series using centralized function, convert to numpy
                img = load_dicom_series(path)
                arr = sitk.GetArrayFromImage(img)
                # SimpleITK: (z, y, x), convert to (x, y, z)
                arr = np.transpose(arr, (2, 1, 0))
                images.append(arr)
            elif is_nifti_file(path):
                img = nib.load(path)
                arr = np.asarray(img.dataobj)
                images.append(arr)
            else:
                raise ValueError(f"Unsupported image format or path: {path}")
        return images
    
    def fit_time_activity_curve_advanced(self,
                                        times: List[float],
                                        activities: List[np.ndarray],
                                        model: str = "nukfit",
                                        model_name: str = None,
                                        weight_factors: Optional[List[float]] = None,
                                        force_zero_at_t0: bool = False,
                                        integration_limit: float = None,
                                        phys_half_life: float = None,
                                        return_fit_params: bool = False
                                        ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Fit time-activity curves for each voxel using advanced models (NUKFIT/LMFIT) and integrate.
        Args:
            times: List of time points
            activities: List of activity maps at each time point
            model: 'nukfit' or 'lmfit'
            model_name: Specific model name (e.g., 'f2', 'f3', 'mono-exponential', etc.)
            weight_factors: Optional weighting factors for fitting
            force_zero_at_t0: If True, force fit through zero at t=0
            integration_limit: Upper limit for integration (default: 100*half_life)
            phys_half_life: Physical half-life (for models that require it)
            return_fit_params: If True, return fit parameters as dict
        Returns:
            Tuple of (fitted parameters, accumulated dose/activity, fit info dict)
        """
        import warnings
        times = np.array(times)
        activities = np.array(activities)
        if weight_factors is None:
            weight_factors = np.ones_like(times)
        n_times = len(times)
        original_shape = activities[0].shape
        activities_reshaped = activities.reshape(n_times, -1)
        fit_params = {}
        # Model definitions
        def nukfit_f2(t, A1, lambda1):
            lambda_phys = self.decay_constant if phys_half_life is None else np.log(2)/phys_half_life
            return A1 * np.exp(-(lambda1 + lambda_phys) * t)
        def nukfit_f3(t, A1, lambda1, lambda2):
            lambda_phys = self.decay_constant if phys_half_life is None else np.log(2)/phys_half_life
            return A1 * (np.exp(-(lambda1 + lambda_phys) * t) - np.exp(-(lambda2 + lambda_phys) * t))
        def nukfit_f3a(t, A1, alpha, lambda1):
            lambda_phys = self.decay_constant if phys_half_life is None else np.log(2)/phys_half_life
            return A1 * (alpha * np.exp(-(lambda1 + lambda_phys) * t) + (1-alpha) * np.exp(-lambda_phys * t))
        def nukfit_f4a(t, A1, lambda1, A2, lambda2):
            lambda_phys = self.decay_constant if phys_half_life is None else np.log(2)/phys_half_life
            return A1 * np.exp(-(lambda1 + lambda_phys) * t) + A2 * np.exp(-(lambda2 + lambda_phys) * t)
        def nukfit_f4b(t, A1, lambda1, A2, lambda2, lambda_bc):
            lambda_phys = self.decay_constant if phys_half_life is None else np.log(2)/phys_half_life
            return (A1 * np.exp(-(lambda1 + lambda_phys) * t) - A2 * np.exp(-(lambda2 + lambda_phys) * t) - (A1-A2) * np.exp(-(lambda_bc + lambda_phys) * t))
        def nukfit_f4c(t, A1, lambda1, A2, lambda2, lambda_bc):
            lambda_phys = self.decay_constant if phys_half_life is None else np.log(2)/phys_half_life
            return (A1 * np.exp(-(lambda1 + lambda_phys) * t) + A2 * np.exp(-(lambda2 + lambda_phys) * t) - (A1+A2) * np.exp(-(lambda_bc + lambda_phys) * t))
        def lmfit_mono(t, A1, T1):
            return A1 * np.exp(-np.log(2) * t / T1)
        def lmfit_bi(t, A1, T1, T2):
            return A1 * (np.exp(-np.log(2) * t / T1) - np.exp(-np.log(2) * t / T2))
        def lmfit_tri(t, A1, T1, A2, T2, Tphys):
            return A1 * np.exp(-np.log(2) * t / T1) - A2 * np.exp(-np.log(2) * t / T2) - (A1-A2) * np.exp(-np.log(2) * t / Tphys)
        nukfit_models = {
            "f2": (nukfit_f2, [1.0, 0.1]),
            "f3": (nukfit_f3, [1.0, 0.1, 0.01]),
            "f3a": (nukfit_f3a, [1.0, 0.5, 0.1]),
            "f4a": (nukfit_f4a, [1.0, 0.1, 0.5, 0.01]),
            "f4b": (nukfit_f4b, [1.0, 0.1, 0.5, 0.01, 0.693]),
            "f4c": (nukfit_f4c, [1.0, 0.1, 0.5, 0.01, 0.693])
        }
        lmfit_models = {
            "mono-exponential": (lmfit_mono, [1.0, self.half_life]),
            "bi-exponential": (lmfit_bi, [1.0, self.half_life, self.half_life/2]),
            "tri-exponential": (lmfit_tri, [1.0, self.half_life, 0.5, self.half_life/2, self.half_life])
        }
        if model.lower() == "nukfit":
            models = nukfit_models
        elif model.lower() == "lmfit":
            models = lmfit_models
        else:
            raise ValueError(f"Unknown model type: {model}")
        if model_name is None:
            model_name = list(models.keys())[0]
        fit_func, p0 = models[model_name]
        n_params = len(p0)
        fitted_params = np.zeros((n_params, activities_reshaped.shape[1]))
        aucs = np.zeros(activities_reshaped.shape[1])
        fit_info = {"model": model, "model_name": model_name, "params": [], "success": []}
        for i in range(activities_reshaped.shape[1]):
            y = activities_reshaped[:, i]
            if force_zero_at_t0:
                y = y - y[0]
                y[0] = 0
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt, _ = curve_fit(fit_func, times, y, p0=p0, sigma=1/np.array(weight_factors), maxfev=10000)
                fitted_params[:, i] = popt
                fit_info["params"].append(popt.tolist())
                fit_info["success"].append(True)
            except Exception as e:
                fitted_params[:, i] = np.zeros(n_params)
                fit_info["params"].append([0]*n_params)
                fit_info["success"].append(False)
            # Numerical integration for AUC
            aucs[i] = np.trapz(fit_func(times, *fitted_params[:, i]), times)
        aucs = aucs.reshape(original_shape)
        fit_params = fitted_params.reshape((n_params, *original_shape))
        if return_fit_params:
            return fit_params, aucs, fit_info
        return fit_params, aucs