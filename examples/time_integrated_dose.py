"""
Example of time-integrated dose calculation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyvoxeldosimetry.core import DoseCalculator
import time
import os

# Create a simple activity distribution (a 3D sphere)
def create_spherical_activity(size=(64, 64, 64), center=None, radius=10, activity=1000.0):
    """Create a spherical activity distribution."""
    if center is None:
        center = [s // 2 for s in size]
    
    activity_map = np.zeros(size)
    for x in range(size[0]):
        for y in range(size[1]):
            for z in range(size[2]):
                if ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) <= radius**2:
                    activity_map[x, y, z] = activity
    return activity_map

# Create a series of activity maps with exponential decay
def create_time_series(initial_map, half_life, time_points):
    """Create a time series of activity maps with exponential decay."""
    decay_constant = np.log(2) / half_life
    activity_maps = []
    
    for t in time_points:
        decay_factor = np.exp(-decay_constant * t)
        activity_maps.append(initial_map * decay_factor)
    
    return activity_maps

# Parameters
size = (48, 48, 48)
center = (24, 24, 24)
radius = 10
initial_activity = 1000.0  # Bq
half_life = 109.8  # hours (Lu-177)
time_points = [0, 24, 48, 72, 96, 120]  # hours

# Create output directory
output_dir = os.path.join(os.getcwd(), "time_integrated_results")
os.makedirs(output_dir, exist_ok=True)

# Create initial activity map
initial_map = create_spherical_activity(size=size, center=center, radius=radius, activity=initial_activity)
print(f"Initial total activity: {np.sum(initial_map):.2f} Bq")

# Create time series of activity maps
activity_maps = create_time_series(initial_map, half_life, time_points)

# Print activity at each time point
for i, t in enumerate(time_points):
    total_activity = np.sum(activity_maps[i])
    print(f"Time {t} hours: Total activity = {total_activity:.2f} Bq")

# Initialize dose calculator
calculator = DoseCalculator(
    radionuclide="Lu177",
    method="kernel",  # Using kernel method for speed
    config={
        "kernel_resolution": 1.0,  # mm
        "tissue_composition": "water",
        "half_life": half_life,
        "time_units": "hours"
    }
)

# Calculate dose
print("\nCalculating absorbed dose...")
start_time = time.time()
voxel_size = (1.0, 1.0, 1.0)  # mm

result = calculator.calculate_dose(
    activity_maps=activity_maps,
    time_points=time_points,
    voxel_size=voxel_size
)

end_time = time.time()
print(f"Calculation completed in {end_time - start_time:.2f} seconds")

# Display results
central_slice = size[0] // 2

# Plot activity maps at different time points
plt.figure(figsize=(15, 8))
for i, t in enumerate(time_points):
    if i >= 6:  # Only show up to 6 time points
        break
    plt.subplot(2, 3, i+1)
    plt.imshow(activity_maps[i][:, :, central_slice], cmap='viridis')
    plt.colorbar(label='Activity (Bq)')
    plt.title(f'Activity at t={t}h')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'activity_time_series.png'))
plt.show()

# Plot dose rate maps
plt.figure(figsize=(15, 8))
for i, t in enumerate(time_points):
    if i >= 6:  # Only show up to 6 time points
        break
    plt.subplot(2, 3, i+1)
    plt.imshow(result.dose_rate_maps[i][:, :, central_slice], cmap='hot')
    plt.colorbar(label='Dose Rate (Gy/s)')
    plt.title(f'Dose Rate at t={t}h')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'dose_rate_time_series.png'))
plt.show()

# Plot final absorbed dose
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.imshow(result.absorbed_dose[:, :, central_slice], cmap='hot')
plt.colorbar(label='Absorbed Dose (Gy)')
plt.title('Absorbed Dose (Central Slice)')

# Plot dose profile
plt.subplot(2, 1, 2)
plt.plot(result.absorbed_dose[central_slice, central_slice, :], 'r-')
plt.grid(True)
plt.xlabel('Z Position (voxels)')
plt.ylabel('Absorbed Dose (Gy)')
plt.title('Dose Profile (Central Line)')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'absorbed_dose.png'))
plt.show()

# Print statistics
print("\nDose Statistics:")
print(f"Maximum absorbed dose: {np.max(result.absorbed_dose):.4e} Gy")
print(f"Mean absorbed dose in target: {np.mean(result.absorbed_dose[initial_map > 0]):.4e} Gy")
print(f"Minimum non-zero absorbed dose: {np.min(result.absorbed_dose[result.absorbed_dose > 0]):.4e} Gy")

# Calculate and display cumulative dose-volume histogram (DVH)
plt.figure(figsize=(10, 6))

# Get dose values in target
target_dose = result.absorbed_dose[initial_map > 0].flatten()
target_volume = len(target_dose)

# Sort dose values
sorted_dose = np.sort(target_dose)[::-1]  # High to low
volume_percent = np.arange(1, len(sorted_dose) + 1) / len(sorted_dose) * 100

plt.plot(sorted_dose, volume_percent, 'b-', linewidth=2)
plt.grid(True)
plt.xlabel('Dose (Gy)')
plt.ylabel('Volume (%)')
plt.title('Cumulative Dose-Volume Histogram (DVH)')
plt.savefig(os.path.join(output_dir, 'dvh.png'))
plt.show()

# Save metadata
print("\nMetadata:")
for key, value in result.metadata.items():
    if key != 'config':  # Skip printing full config
        print(f"{key}: {value}")