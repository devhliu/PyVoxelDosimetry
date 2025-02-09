import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

"""
   - Range up to 20mm (lower energy beta emissions)
   - Multiple beta components reflected in dose rate profile
   - Gamma contribution visible in tail region
"""
def plot_validation_comparisons():
    """Plot validation comparisons for Lu177 kernel."""
    validation_dir = Path(__file__).parent
    
    # Load validation data
    mird_data = pd.read_csv(validation_dir / 'Lu177_mird_comparison.csv')
    gate_data = pd.read_csv(validation_dir / 'Lu177_gate_comparison.csv')
    fluka_data = pd.read_csv(validation_dir / 'Lu177_fluka_comparison.csv')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot dose rate comparisons
    ax1.loglog(mird_data['distance_mm'], mird_data['mird_dose_rate'], 'o-', label='MIRD')
    ax1.loglog(gate_data['distance_mm'], gate_data['gate_dose_rate'], 's-', label='GATE')
    ax1.loglog(fluka_data['distance_mm'], fluka_data['fluka_dose_rate'], '^-', label='FLUKA')
    ax1.loglog(mird_data['distance_mm'], mird_data['calculated_dose_rate'], 'k--', label='PyVoxelDosimetry')
    
    ax1.grid(True)
    ax1.set_xlabel('Distance (mm)')
    ax1.set_ylabel('Dose Rate (Gy/Bq.s)')
    ax1.set_title('Lu-177 Dose Kernel Validation')
    ax1.legend()
    
    # Plot relative differences with multiple beta components
    ax2.semilogx(mird_data['distance_mm'], mird_data['relative_difference'], 'o-', label='vs MIRD')
    ax2.semilogx(gate_data['distance_mm'], gate_data['relative_difference'], 's-', label='vs GATE')
    ax2.semilogx(fluka_data['distance_mm'], fluka_data['relative_difference'], '^-', label='vs FLUKA')
    
    ax2.grid(True)
    ax2.set_xlabel('Distance (mm)')
    ax2.set_ylabel('Relative Difference (%)')
    ax2.set_ylim([-3, 3])
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(validation_dir / 'Lu177_validation_comparison.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    plot_validation_comparisons()