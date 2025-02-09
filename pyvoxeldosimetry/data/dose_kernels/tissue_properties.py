from dataclasses import dataclass
from typing import Dict

@dataclass
class TissueProperties:
    name: str
    density: float  # g/cm3
    electron_density: float  # e/cm3
    effective_Z: float
    stopping_power_ratio: float  # relative to water
    
TISSUE_DATABASE = {
    'water': TissueProperties(
        name='Water',
        density=1.0,
        electron_density=3.34e23,
        effective_Z=7.42,
        stopping_power_ratio=1.0
    ),
    'bone': TissueProperties(
        name='Cortical Bone',
        density=1.85,
        electron_density=5.91e23,
        effective_Z=13.8,
        stopping_power_ratio=1.15
    ),
    # Add other tissue types
}