"""
Monte Carlo decay data initialization
"""
from pathlib import Path
import json
from typing import Dict, Any

MC_DECAY_PATH = Path(__file__).parent

def load_decay_data(radionuclide: str) -> Dict[str, Any]:
    """
    Load decay data for Monte Carlo simulation
    
    Args:
        radionuclide: Name of the radionuclide (e.g., 'F18', 'Lu177')
        
    Returns:
        Dictionary containing decay properties
    """
    decay_file = MC_DECAY_PATH / f"{radionuclide}.json"
    if not decay_file.exists():
        raise ValueError(f"Decay data not found for {radionuclide}")
        
    with open(decay_file, 'r') as f:
        return json.load(f)