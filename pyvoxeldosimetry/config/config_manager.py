"""
Configuration manager for PyVoxelDosimetry.

Created: 2025-02-08 10:05:10
Author: devhliu
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

class ConfigManager:
    """Configuration manager for PyVoxelDosimetry package."""
    
    def __init__(self):
        """Initialize configuration manager."""
        self.config_dir = Path(__file__).parent
        self.data_dir = self.config_dir.parent / 'data'
        self._load_default_configs()
        
    def _load_default_configs(self):
        """Load default configurations."""
        with open(self.config_dir / 'default_config.json', 'r') as f:
            self.default_config = json.load(f)
        with open(self.config_dir / 'compute_config.json', 'r') as f:
            self.compute_config = json.load(f)
            
    def get_decay_data(self, nuclide: str) -> Dict[str, Any]:
        """
        Get Monte Carlo decay data for a specific nuclide.
        
        Args:
            nuclide: Radionuclide name (e.g., 'F18', 'Y90', 'Lu177')
            
        Returns:
            Dictionary containing decay data
        """
        decay_path = self.data_dir / 'mc_decay' / f'{nuclide}.json'
        if not decay_path.exists():
            raise ValueError(f"No decay data found for {nuclide}")
            
        with open(decay_path, 'r') as f:
            return json.load(f)
            
    def get_kernel_data(self, nuclide: str) -> Dict[str, Any]:
        """
        Get dose kernel data for a specific nuclide.
        
        Args:
            nuclide: Radionuclide name (e.g., 'F18', 'Y90', 'Lu177')
            
        Returns:
            Dictionary containing kernel data
        """
        kernel_path = self.data_dir / 'dose_kernels' / f'{nuclide}.json'
        if not kernel_path.exists():
            raise ValueError(f"No kernel data found for {nuclide}")
            
        with open(kernel_path, 'r') as f:
            return json.load(f)
            
    def get_compute_config(self, method: str) -> Dict[str, Any]:
        """
        Get computation configuration for specified method.
        
        Args:
            method: Computation method ('monte_carlo' or 'kernel')
            
        Returns:
            Dictionary containing computation parameters
        """
        if method not in ['monte_carlo', 'kernel']:
            raise ValueError(f"Invalid computation method: {method}")
            
        return self.compute_config.get(method, {})
        
    def list_available_nuclides(self, method: str) -> List[str]:
        """
        List available nuclides for specified method.
        
        Args:
            method: Computation method ('monte_carlo' or 'kernel')
            
        Returns:
            List of available nuclide names
        """
        if method == 'monte_carlo':
            data_dir = self.data_dir / 'mc_decay'
        elif method == 'kernel':
            data_dir = self.data_dir / 'dose_kernels'
        else:
            raise ValueError(f"Invalid method: {method}")
            
        return [f.stem for f in data_dir.glob('*.json')]