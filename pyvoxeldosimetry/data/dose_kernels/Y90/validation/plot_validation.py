import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

"""
   - Extended range to 60mm (high energy beta)
   - Higher dose rates overall
   - Bremsstrahlung contribution in far regions
   - Larger uncertainties at extended distances
"""
def plot_validation_comparisons():
    """Plot validation comparisons for Y90 kernel."""
    validation_dir = Path(__file__).parent
    
    # Load validation data
    mird_data = pd.read_csv(validation_dir / 'Y90_mird_comparison.csv')
    gate_data = pd.read_csv(validation_dir / 'Y90_gate_comparison.csv')
    fluka_data = pd.read_csv(validation_dir / 'Y90_fluka_comparison.csv')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot dose rate comparisons with bremsstrahlung contribution
    ax1.loglog(mird_data['distance_mm'], mird_data['mird_dose_rate'], 'o-', label='MIRD')
    ax1.loglog(gate_data['distance_mm'], gate_data['gate_dose_rate'], 's-', label='GATE')
    ax1.loglog(fluka_data['distance_mm'], fluka_data['fluka_dose_rate'], '^-', label='FLUKA')
    ax1.loglog(mird_data['distance_mm'], mird_data['calculated_dose_rate'], 'k--', label='PyVoxelDosimetry')
    
    ax1.grid(True)
    ax1.set_xlabel('Distance (mm)')
    ax1.set_ylabel('Dose Rate (Gy/Bq.s)')
    ax1.set_title('Y-90 Dose Kernel Validation')
    ax1.legend()
    
    # Plot relative differences with extended range
    ax2.semilogx(mird_data['distance_mm'], mird_data['relative_difference'], 'o-', label='vs MIRD')
    ax2.semilogx(gate_data['distance_mm'], gate_data['relative_difference'], 's-', label='vs GATE')
    ax2.semilogx(fluka_data['distance_mm'], fluka_data['relative_difference'], '^-', label='vs FLUKA')
    
    ax2.grid(True)
    ax2.set_xlabel('Distance (mm)')
    ax2.set_ylabel('Relative Difference (%)')
    ax2.set_ylim([-3, 3])
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(validation_dir / 'Y90_validation_comparison.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    plot_validation_comparisons()