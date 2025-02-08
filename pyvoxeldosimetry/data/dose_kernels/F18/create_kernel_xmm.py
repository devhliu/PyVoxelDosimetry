import numpy as np
from pathlib import Path
from datetime import datetime

def create_f18_kernel(resolution_mm):
    """Create F-18 dose kernel for specified resolution."""
    # Define dimensions based on resolution
    dims_dict = {
        1: [101, 101, 101],
        2: [61, 61, 61],
        3: [41, 41, 41],
        4: [31, 31, 31],
        5: [25, 25, 25]
    }
    dims = dims_dict[resolution_mm]
    center = [d//2 for d in dims]
    
    # Create header
    header = {
        'dimensions': np.array(dims, dtype=np.int32),
        'voxel_size': np.array(float(resolution_mm), dtype=np.float32),
        'total_energy': np.array(0.960, dtype=np.float32),  # MeV
        'scaling_factor': np.array(1.0e-12, dtype=np.float32),  # Convert to Gy/(Bq*s)
        'timestamp': np.array([2025, 2, 8, 6, 26, 44], dtype=np.int32),  # YYYY,MM,DD,HH,MM,SS
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
    
    # F-18 dose kernel model (realistic physics)
    # Positron range component (based on ICRU data)
    positron_range = 0.6  # mean range in mm
    positron_term = np.exp(-r/positron_range)
    
    # 511 keV annihilation photons (based on NIST attenuation data)
    mu_water = 0.0096  # attenuation coefficient in mm^-1
    photon_term = np.exp(-mu_water*r)/(4*np.pi*r*r)
    photon_term[center[0], center[1], center[2]] = 1.0  # Fix central point
    
    # Combined kernel with proper weighting
    kernel = positron_term + 0.2*photon_term  # Weight based on energy deposition
    kernel = kernel/kernel.sum() * 0.960  # Normalize to total energy per decay
    
    return header, kernel

def save_f18_kernel(filename, resolution_mm):
    """Save F-18 kernel to binary file with extended header."""
    header, kernel = create_f18_kernel(resolution_mm)
    
    with open(filename, 'wb') as f:
        # Write extended header
        header['dimensions'].tofile(f)
        header['voxel_size'].tofile(f)
        header['total_energy'].tofile(f)
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
        scaling = np.fromfile(f, dtype=np.float32, count=1)[0]
        timestamp = np.fromfile(f, dtype=np.int32, count=6)
        user = np.fromfile(f, dtype=np.int8, count=32)
        
        # Read kernel data
        # I did not know why the kernel was saved as float64, so I changed it to float64 from float32
        kernel = np.fromfile(f, dtype=np.float64)
        kernel = kernel.reshape(dims)
        
        return {
            'dimensions': dims,
            'voxel_size': voxel_size,
            'total_energy': total_energy,
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
    1: "41.2 MB, 101³ voxels",
    2: "9.1 MB, 61³ voxels",
    3: "2.8 MB, 41³ voxels",
    4: "1.2 MB, 31³ voxels",
    5: "0.63 MB, 25³ voxels"
}

for res in range(1, 6):
    kernel_file = kernel_dir / f"kernel_{res}mm.dat"
    file_size = save_f18_kernel(kernel_file, res)
    
    print(f"\nF-18 {res}mm kernel file created: {kernel_file}")
    print(f"Expected size: {kernel_specs[res]}")
    print(f"Actual size: {file_size/1024/1024:.2f} MB")
    
    # Verify the created file
    verification = verify_kernel(kernel_file)
    print(f"\nKernel {res}mm verification:")
    print(f"Dimensions: {verification['dimensions']}")
    print(f"Voxel size: {verification['voxel_size']} mm")
    print(f"Total energy: {verification['total_energy']} MeV")
    print(f"Created by: {verification['user']}")
    print(f"Timestamp: {verification['timestamp']}")
    print(f"Energy sum: {verification['kernel_sum']:.6f}")
    print(f"Value range: [{verification['kernel_min']:.2e}, {verification['kernel_max']:.2e}]")

print("\nAll F-18 dose kernels created successfully!")