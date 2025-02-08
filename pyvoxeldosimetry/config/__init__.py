"""
Configuration module for PyVoxelDosimetry.

Created: 2025-02-08 10:05:10
Author: devhliu
"""

from .config_manager import ConfigManager

# Create a singleton instance
config_manager = ConfigManager()

# Export common functions
get_decay_data = config_manager.get_decay_data
get_kernel_data = config_manager.get_kernel_data
get_compute_config = config_manager.get_compute_config