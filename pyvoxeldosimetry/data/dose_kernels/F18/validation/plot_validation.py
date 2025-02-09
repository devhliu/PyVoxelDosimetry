import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_validation_comparisons():
    """Plot validation comparisons for F18 kernel."""
    validation_dir = Path(__file__).parent
    
    # Load validation data
    mird_data = pd.read_csv(validation_dir / 'F18_mird_comparison.csv')
    gate_data = pd.read_csv(validation_dir / 'F18_gate_comparison.csv')
    fluka_data = pd.read_csv(validation_dir / 'F18_fluka_comparison.csv')
    
    plt.figure(figsize=(10, 8))
    
    # Plot dose rate comparisons
    plt.subplot(2, 1, 1)
    plt.loglog(mird_data['distance_mm'], mird_data['mird_dose_rate'], 'o-', label='MIRD')
    plt.loglog(gate_data['distance_mm'], gate_data['gate_dose_rate'], 's-', label='GATE')
    plt.loglog(fluka_data['distance_mm'], fluka_data['fluka_dose_rate'], '^-', label='FLUKA')
    plt.loglog(mird_data['distance_mm'], mird_data['calculated_dose_rate'], 'k--', label='PyVoxelDosimetry')
    
    plt.grid(True)
    plt.xlabel('Distance (mm)')
    plt.ylabel('Dose Rate (Gy/Bq.s)')
    plt.legend()
    
    # Plot relative differences
    plt.subplot(2, 1, 2)
    plt.semilogx(mird_data['distance_mm'], mird_data['relative_difference'], 'o-', label='vs MIRD')
    plt.semilogx(gate_data['distance_mm'], gate_data['relative_difference'], 's-', label='vs GATE')
    plt.semilogx(fluka_data['distance_mm'], fluka_data['relative_difference'], '^-', label='vs FLUKA')
    
    plt.grid(True)
    plt.xlabel('Distance (mm)')
    plt.ylabel('Relative Difference (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(validation_dir / 'F18_validation_comparison.png')
    plt.close()

if __name__ == '__main__':
    plot_validation_comparisons()