"""
Example of dose rate calculation using Gate Monte Carlo method.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyvoxeldosimetry.core import DoseCalculator
import time
import os

# Create a simple activity distribution (a 3D sphere)
def create_spherical_activity(size=(64, 64, 64), center=(32, 32, 32), radius=10, activity=1000.0):
    """Create a spherical activity distribution."""
    activity_map = np.zeros(size)
    for x in range(size[0]):
        for y in range(size[1]):
            for z in range(size[2]):
                if ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) <= radius**2:
                    activity_map[x, y, z] = activity
    return activity_map

# Create output directory for Gate simulation
output_dir = os.path.join(os.getcwd(), "gate_output")
os.makedirs(output_dir, exist_ok=True)

# Create activity map (smaller for Gate to be faster)
activity_map = create_spherical_activity(size=(20, 20, 20), center=(10, 10, 10), radius=5)
print(f"Total activity: {np.sum(activity_map)} Bq")

# Initialize dose calculator with Gate Monte Carlo method
calculator = DoseCalculator(
    radionuclide="Y90",
    method="gate_monte_carlo",
    config={
        "n_particles": 10000,  # Reduced for example
        "output_dir": output_dir,
        "physics_list": "QGSP_BIC_EMY",
        "tissue_composition": "water"
    }
)

# Calculate dose rate
print("Starting Gate Monte Carlo dose calculation...")
print("This may take several minutes depending on your system...")
start_time = time.time()
voxel_size = (2.0, 2.0, 2.0)  # mm (larger voxels for faster calculation)
dose_rate = calculator.calculator.calculate_dose_rate(
    activity_map=activity_map,
    voxel_size=voxel_size
)
end_time = time.time()
print(f"Calculation completed in {end_time - start_time:.2f} seconds")

# Display results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(activity_map[:, :, 10], cmap='viridis')
plt.colorbar(label='Activity (Bq)')
plt.title('Activity Distribution (Central Slice)')

plt.subplot(1, 2, 2)
plt.imshow(dose_rate[:, :, 10], cmap='hot')
plt.colorbar(label='Dose Rate (Gy/s)')
plt.title('Dose Rate (Central Slice)')

plt.tight_layout()
plt.savefig('gate_monte_carlo_result.png')
plt.show()

# Print some statistics
print(f"Maximum dose rate: {np.max(dose_rate):.2e} Gy/s")
print(f"Mean dose rate in active region: {np.mean(dose_rate[activity_map > 0]):.2e} Gy/s")