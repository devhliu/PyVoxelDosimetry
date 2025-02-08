"""
Time-activity curve fitting and dose accumulation.
"""
import numpy as np
from scipy.optimize import curve_fit
from typing import List, Tuple, Optional

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