import numpy as np
import json
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class KernelMetadata:
    version: str
    created_at: str
    created_by: str
    method: str

@dataclass
class NuclideInfo:
    name: str
    symbol: str
    half_life: float
    decay_modes: Dict[str, float]

class BaseKernelGenerator(ABC):
    def __init__(self, config_path: Path, tissue_type: str):
        """
        Initialize base kernel generator.
        
        Args:
            config_path: Path to nuclide config JSON
            tissue_type: Type of tissue (water, bone, etc)
        """
        self.config = self._load_config(config_path)
        self.tissue_type = tissue_type
        
    def _load_config(self, config_path: Path) -> Dict:
        with open(config_path, 'r') as f:
            return json.load(f)
    
    @abstractmethod
    def generate_kernel(self, voxel_size: float, grid_size: Tuple[int, int, int]) -> np.ndarray:
        """Generate the dose point kernel."""
        pass
    
    def save_kernel(self, kernel: np.ndarray, output_dir: Path):
        """Save kernel data and visualization."""
        # Save numpy array
        np.save(output_dir / f"{self.config['nuclide']['symbol']}_{self.tissue_type}_kernel.npy", kernel)
        
        # Save visualization
        self._save_visualization(kernel, output_dir)
        
        # Save metadata
        self._save_metadata(kernel, output_dir)
    
    def _save_visualization(self, kernel: np.ndarray, output_dir: Path):
        """Generate and save 2D visualizations of the kernel."""
        import matplotlib.pyplot as plt
        
        center = [s//2 for s in kernel.shape]
        
        # XY plane
        plt.figure(figsize=(8, 8))
        plt.imshow(kernel[:, :, center[2]], cmap='viridis')
        plt.colorbar(label='Dose (Gy/Bq.s)')
        plt.title(f"{self.config['nuclide']['name']} - {self.tissue_type}")
        plt.savefig(output_dir / f"{self.config['nuclide']['symbol']}_{self.tissue_type}_xy.png")
        plt.close()
    
    def _save_metadata(self, kernel: np.ndarray, output_dir: Path):
        """Save kernel metadata as JSON."""
        metadata = {
            "nuclide": self.config.get("nuclide", {}),
            "tissue_type": self.tissue_type,
            "kernel_shape": kernel.shape,
            "voxel_size": self.config.get("kernel", {}).get("voxel_size", None),
            "grid_size": self.config.get("kernel", {}).get("grid_size", None),
            "scaling_factor": self.config.get("kernel", {}).get("scaling_factor", None),
            "created_by": self.config.get("metadata", {}).get("created_by", None),
            "created_at": self.config.get("metadata", {}).get("created_at", None),
            "method": self.config.get("metadata", {}).get("method", None),
        }
        meta_path = output_dir / f"{self.config['nuclide']['symbol']}_{self.tissue_type}_kernel_meta.json"
        with open(meta_path, "w") as f:
            import json
            json.dump(metadata, f, indent=2)