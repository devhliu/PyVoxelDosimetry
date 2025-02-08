def load_y90_kernel(filename):
    """Load Y-90 dose kernel with extended header."""
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
        kernel = np.fromfile(f, dtype=np.float32)
        kernel = kernel.reshape(dims) * scaling
        
        return kernel, {
            'dimensions': dims,
            'voxel_size': voxel_size,
            'total_energy': total_energy,
            'mean_energy': mean_energy,
            'timestamp': datetime(*timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            'user': bytes(user).decode().strip('\x00')
        }

# Example usage:
resolution = 1  # or 2,3,4,5
kernel, info = load_y90_kernel(f"pyvoxeldosimetry/data/dose_kernels/Y90/kernel_{resolution}mm.dat")