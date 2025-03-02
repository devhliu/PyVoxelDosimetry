"""
Example of dose rate calculation using kernel convolution method.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyvoxeldosimetry.core import DoseCalculator

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

# Create activity map
activity_map = create_spherical_activity()
print(f"Total activity: {np.sum(activity_map)} Bq")

# Initialize dose calculator with kernel convolution method
calculator = DoseCalculator(
    radionuclide="F18",
    method="kernel",
    config={
        "kernel_resolution": 1.0,  # mm
        "tissue_composition": "water"
    }
)

# Calculate dose rate
voxel_size = (1.0, 1.0, 1.0)  # mm
dose_rate = calculator.calculator.calculate_dose_rate(
    activity_map=activity_map,
    voxel_size=voxel_size
)

# Display results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(activity_map[:, :, 32], cmap='viridis')
plt.colorbar(label='Activity (Bq)')
plt.title('Activity Distribution (Central Slice)')

plt.subplot(1, 2, 2)
plt.imshow(dose_rate[:, :, 32], cmap='hot')
plt.colorbar(label='Dose Rate (Gy/s)')
plt.title('Dose Rate (Central Slice)')

plt.tight_layout()
plt.savefig('kernel_convolution_result.png')
plt.show()

# Print some statistics
print(f"Maximum dose rate: {np.max(dose_rate):.2e} Gy/s")
print(f"Mean dose rate in active region: {np.mean(dose_rate[activity_map > 0]):.2e} Gy/s")