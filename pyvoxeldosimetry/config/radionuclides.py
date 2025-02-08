"""
Configuration for radionuclide properties including decay data and dose kernels.
"""
import json
from pathlib import Path
from typing import Dict, Any

class RadionuclideConfig:
    def __init__(self, config_path: str = None):
        self.config_path = config_path or Path(__file__).parent / "data"
        self.radionuclides = self._load_radionuclide_data()
        
    def _load_radionuclide_data(self) -> Dict[str, Any]:
        """Load radionuclide data from configuration files."""
        radionuclides = {
            "F18": {
                "half_life": 109.77,  # minutes
                "decay_type": "β+",
                "energy": {
                    "mean_beta": 0.250,  # MeV
                    "gamma": [0.511, 0.511]  # MeV
                },
                "branching_ratio": {
                    "β+": 0.967
                }
            },
            # Add other radionuclides similarly
            "Lu177": {
                "half_life": 6.647 * 24 * 60,  # minutes
                "decay_type": "β-",
                "energy": {
                    "mean_beta": 0.149,
                    "gamma": [0.208, 0.113]
                },
                "branching_ratio": {
                    "β-": 1.0
                }
            }
        }
        return radionuclides

    def get_radionuclide(self, name: str) -> Dict[str, Any]:
        """Get properties for a specific radionuclide."""
        return self.radionuclides.get(name)

    def add_radionuclide(self, name: str, properties: Dict[str, Any]) -> None:
        """Add a new radionuclide to the configuration."""
        self.radionuclides[name] = properties