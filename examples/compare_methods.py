"""
Example comparing all dose rate calculation methods.
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

# Create output directory for results
output_dir = os.path.join(os.getcwd(), "comparison_results")
os.makedirs(output_dir, exist_ok=True)

# Create activity map (small for quick comparison)
size = (32, 32, 32)
activity_map = create_spherical_activity(size=size, radius=8)
print(f"Total activity: {np.sum(activity_map)} Bq")

# Define methods to compare
methods = ["kernel", "gpu_monte_carlo", "gate_monte_carlo"]
results = {}
calculation_times = {}

# Run calculations for each method
voxel_size = (1.0, 1.0, 1.0)  # mm
central_slice = size[0] // 2

for method in methods:
    print(f"\nCalculating dose rate using {method} method...")
    
    # Configure calculator based on method
    config = {
        "tissue_composition": "water"
    }
    
    if method == "gpu_monte_carlo":
        config["n_particles"] = 50000
    elif method == "gate_monte_carlo":
        config["n_particles"] = 5000
        config["output_dir"] = os.path.join(output_dir, "gate_output")
        os.makedirs(config["output_dir"], exist_ok=True)
    
    # Initialize calculator
    calculator = DoseCalculator(
        radionuclide="F18",
        method=method,
        config=config
    )
    
    # Calculate dose rate
    start_time = time.time()
    dose_rate = calculator.calculator.calculate_dose_rate(
        activity_map=activity_map,
        voxel_size=voxel_size
    )
    end_time = time.time()
    
    # Store results
    results[method] = dose_rate
    calculation_times[method] = end_time - start_time
    
    print(f"  Calculation completed in {calculation_times[method]:.2f} seconds")
    print(f"  Maximum dose rate: {np.max(dose_rate):.2e} Gy/s")

# Plot results
plt.figure(figsize=(15, 10))

# Plot activity distribution
plt.subplot(2, 2, 1)
plt.imshow(activity_map[:, :, central_slice], cmap='viridis')
plt.colorbar(label='Activity (Bq)')
plt.title('Activity Distribution (Central Slice)')

# Plot dose rate for each method
for i, method in enumerate(methods):
    plt.subplot(2, 2, i+2)
    plt.imshow(results[method][:, :, central_slice], cmap='hot')
    plt.colorbar(label='Dose Rate (Gy/s)')
    plt.title(f'{method.capitalize()} Method\nTime: {calculation_times[method]:.2f}s')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'method_comparison.png'))
plt.show()

# Create profile comparison
plt.figure(figsize=(10, 6))
for method in methods:
    plt.plot(results[method][central_slice, central_slice, :], 
             label=f"{method.capitalize()} (max={np.max(results[method]):.2e} Gy/s)")

plt.xlabel('Z Position (voxels)')
plt.ylabel('Dose Rate (Gy/s)')
plt.title('Dose Rate Profile Comparison (Central Line)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'profile_comparison.png'))
plt.show()

# Print summary
print("\nSummary of Calculation Methods:")
print("-" * 50)
print(f"{'Method':<20} {'Time (s)':<15} {'Max Dose Rate (Gy/s)':<25}")
print("-" * 50)
for method in methods:
    print(f"{method.capitalize():<20} {calculation_times[method]:<15.2f} {np.max(results[method]):<25.2e}")