"""
Activity distribution sampling and time integration.
"""
import numpy as np
from typing import List, Tuple, Optional, Union
from scipy.interpolate import interp1d

class ActivitySampler:
    def __init__(self,
                 half_life: float,
                 units: str = "hours"):
        """
        Initialize activity sampler.
        
        Args:
            half_life: Radionuclide half-life
            units: Time units ('hours', 'minutes', 'seconds')
        """
        self.half_life = half_life
        self.units = units
        self._validate_inputs()
        
    def _validate_inputs(self):
        """Validate initialization parameters."""
        if self.half_life <= 0:
            raise ValueError("Half-life must be positive")
        if self.units not in ['hours', 'minutes', 'seconds']:
            raise ValueError("Invalid time units")
            
    def sample_timepoints(self,
                         start_time: float,
                         end_time: float,
                         n_points: int = 10
                         ) -> np.ndarray:
        """
        Generate sampling timepoints.
        
        Args:
            start_time: Start time
            end_time: End time
            n_points: Number of sampling points
            
        Returns:
            Array of timepoints
        """
        return np.linspace(start_time, end_time, n_points)
    
    def integrate_activity(self,
                         activity_maps: List[np.ndarray],
                         time_points: List[float],
                         method: str = 'trapezoid'
                         ) -> np.ndarray:
        """
        Integrate activity over time.
        
        Args:
            activity_maps: List of activity distributions
            time_points: List of measurement times
            method: Integration method
            
        Returns:
            Integrated activity map
        """
        if method == 'trapezoid':
            return self._trapezoid_integration(activity_maps, time_points)
        else:
            raise ValueError(f"Unknown integration method: {method}")
            
    def _trapezoid_integration(self,
                             activity_maps: List[np.ndarray],
                             time_points: List[float]
                             ) -> np.ndarray:
        """Perform trapezoidal integration."""
        result = np.zeros_like(activity_maps[0])
        for i in range(len(time_points) - 1):
            dt = time_points[i+1] - time_points[i]
            avg_activity = (activity_maps[i] + activity_maps[i+1]) / 2
            result += avg_activity * dt
        return result