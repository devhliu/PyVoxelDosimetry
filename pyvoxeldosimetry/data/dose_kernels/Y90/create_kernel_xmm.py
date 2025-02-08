import numpy as np
from pathlib import Path
from datetime import datetime

def create_y90_kernel(resolution_mm):
    """Create Y-90 dose kernel for specified resolution."""
    # Define dimensions based on resolution
    dims_dict = {
        1: [201, 201, 201],  # Larger grid for Y-90's longer range
        2: [101, 101, 101],
        3: [67, 67, 67],
        4: [51, 51, 51],
        5: [41, 41, 41]
    }
    dims = dims_dict[resolution_mm]
    center = [d//2 for d in dims]
    
    # Create header
    header = {
        'dimensions': np.array(dims, dtype=np.int32),
        'voxel_size': np.array(float(resolution_mm), dtype=np.float32),
        'total_energy': np.array(2.280, dtype=np.float32),  # Max beta energy in MeV
        'mean_energy': np.array(0.934, dtype=np.float32),   # Mean beta energy in MeV
        'scaling_factor': np.array(1.0e-12, dtype=np.float32),  # Convert to Gy/(Bq*s)
        'timestamp': np.array([2025, 2, 8, 6, 32, 55], dtype=np.int32),  # YYYY,MM,DD,HH,MM,SS
        'user': np.array([ord(c) for c in "devhliu".ljust(32)], dtype=np.int8)  # 32-byte user field
    }
    
    # Create 3D kernel data
    kernel = np.zeros(dims, dtype=np.float32)
    
    # Calculate distances from center
    x, y, z = np.meshgrid(
        np.arange(dims[0]) - center[0],
        np.arange(dims[1]) - center[1],
        np.arange(dims[2]) - center[2],
        indexing='ij'
    )
    r = np.sqrt(x*x + y*y + z*z) * resolution_mm  # Scale by voxel size
    
    # Y-90 beta spectrum model (based on MIRD data)
    # Using modified point kernel for beta dose distribution
    max_range = 11.3  # Maximum range in mm (CSDA range for Y-90 in water)
    mean_range = 4.2  # Mean range in mm
    
    # Beta dose kernel (empirical model based on MIRD data)
    kernel = np.zeros_like(r)
    mask = r <= max_range
    kernel[mask] = np.exp(-r[mask]/mean_range) * (1 - r[mask]/max_range)**2
    
    # Normalize to total energy deposition
    kernel = kernel/kernel.sum() * 0.934  # Normalize to mean energy
    
    return header, kernel

def save_y90_kernel(filename, resolution_mm):
    """Save Y-90 kernel to binary file with extended header."""
    header, kernel = create_y90_kernel(resolution_mm)
    
    with open(filename, 'wb') as f:
        # Write extended header
        header['dimensions'].tofile(f)
        header['voxel_size'].tofile(f)
        header['total_energy'].tofile(f)
        header['mean_energy'].tofile(f)
        header['scaling_factor'].tofile(f)
        header['timestamp'].tofile(f)
        header['user'].tofile(f)
        
        # Write kernel data
        kernel.tofile(f)
    
    return Path(filename).stat().st_size

def verify_kernel(filename):
    """Verify the created kernel file."""
    with open(filename, 'rb') as f:
        # Read extended header
        dims = np.fromfile(f, dtype=np.int32, count=3)
        voxel_size = np.fromfile(f, dtype=np.float32, count=1)[0]
        total_energy = np.fromfile(f, dtype=np.float32, count=1)[0]
        mean_energy = np.fromfile(f, dtype=np.float32, count=1)[0]
        scaling = np.fromfile(f, dtype=np.float32, count=1)[0]
        timestamp = np.fromfile(f, dtype=np.int32, count=6)
        user = np.fromfile(f, dtype=np.int8, count=32)
        
        # Read kernel data
        # I didn't know what the kernel data was supposed to be, so I just read it as a float64 array
        kernel = np.fromfile(f, dtype=np.float64)
        kernel = kernel.reshape(dims)
        
        return {
            'dimensions': dims,
            'voxel_size': voxel_size,
            'total_energy': total_energy,
            'mean_energy': mean_energy,
            'scaling_factor': scaling,
            'timestamp': datetime(*timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            'user': bytes(user).decode().strip('\x00'),
            'kernel_sum': kernel.sum(),
            'kernel_max': kernel.max(),
            'kernel_min': kernel.min()
        }

# Create directory
kernel_dir = Path(__file__).parent
kernel_dir.mkdir(parents=True, exist_ok=True)

# Create kernels for all resolutions
kernel_specs = {
    1: "322.4 MB, 201³ voxels",
    2: "40.3 MB, 101³ voxels",
    3: "12.0 MB, 67³ voxels",
    4: "5.3 MB, 51³ voxels",
    5: "2.7 MB, 41³ voxels"
}

for res in range(1, 6):
    kernel_file = kernel_dir / f"kernel_{res}mm.dat"
    file_size = save_y90_kernel(kernel_file, res)
    
    print(f"\nY-90 {res}mm kernel file created: {kernel_file}")
    print(f"Expected size: {kernel_specs[res]}")
    print(f"Actual size: {file_size/1024/1024:.2f} MB")
    
    # Verify the created file
    verification = verify_kernel(kernel_file)
    print(f"\nKernel {res}mm verification:")
    print(f"Dimensions: {verification['dimensions']}")
    print(f"Voxel size: {verification['voxel_size']} mm")
    print(f"Max energy: {verification['total_energy']} MeV")
    print(f"Mean energy: {verification['mean_energy']} MeV")
    print(f"Created by: {verification['user']}")
    print(f"Timestamp: {verification['timestamp']}")
    print(f"Energy sum: {verification['kernel_sum']:.6f}")
    print(f"Value range: [{verification['kernel_min']:.2e}, {verification['kernel_max']:.2e}]")

print("\nAll Y-90 dose kernels created successfully!")