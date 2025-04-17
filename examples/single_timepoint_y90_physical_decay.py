"""
Example: Single timepoint dosimetry calculation for Y90 using physical decay.
"""
import numpy as np
import matplotlib.pyplot as plt
from pyvoxeldosimetry.core import DoseCalculator

# Create a simple spherical activity distribution
size = (48, 48, 48)
center = (24, 24, 24)
radius = 8
activity = 2e6  # Bq

def create_spherical_activity(size, center, radius, activity):
    arr = np.zeros(size)
    for x in range(size[0]):
        for y in range(size[1]):
            for z in range(size[2]):
                if ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) <= radius**2:
                    arr[x, y, z] = activity
    return arr

activity_map = create_spherical_activity(size, center, radius, activity)
print(f"Total activity: {np.sum(activity_map):.2e} Bq")

# Y90 physical half-life (hours)
y90_half_life = 64.1

# Initialize dose calculator for Y90, kernel method, physical decay
calculator = DoseCalculator(
    radionuclide="Y90",
    method="kernel",
    config={
        "kernel_resolution": 1.0,  # mm
        "tissue_composition": "water",
        "half_life": y90_half_life,
        "time_units": "hours"
    }
)

# Calculate dose rate for single timepoint
voxel_size = (1.0, 1.0, 1.0)  # mm
result = calculator.calculate_dose(
    activity_maps=[activity_map],
    time_points=[2.0],
    voxel_size=voxel_size,
    integration_mode="activity"
)

dose_rate = result.absorbed_dose

# Display results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(activity_map[:, :, center[2]], cmap='viridis')
plt.colorbar(label='Activity (Bq)')
plt.title('Activity Distribution (Central Slice)')
plt.subplot(1, 2, 2)
plt.imshow(dose_rate[:, :, center[2]], cmap='hot')
plt.colorbar(label='Dose Rate (Gy/s)')
plt.title('Dose Rate (Central Slice)')
plt.tight_layout()
plt.savefig('single_timepoint_y90_result.png')
plt.show()

print(f"Maximum dose rate: {np.max(dose_rate):.2e} Gy/s")
print(f"Mean dose rate in active region: {np.mean(dose_rate[activity_map > 0]):.2e} Gy/s")