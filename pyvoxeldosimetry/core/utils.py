"""
Utility functions for PyVoxelDosimetry.

Created: 2025-02-08 09:50:56
Author: devhliu
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
import json
import nibabel as nib
from datetime import datetime
from scipy.interpolate import interp1d

def load_kernel(filename: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load dose point kernel from file.
    
    Args:
        filename: Path to kernel file
        
    Returns:
        Tuple containing:
        - 3D numpy array containing kernel data
        - Dictionary with kernel metadata
    """
    with open(filename, 'rb') as f:
        # Read header
        dims = np.fromfile(f, dtype=np.int32, count=3)
        voxel_size = np.fromfile(f, dtype=np.float32, count=1)[0]
        total_energy = np.fromfile(f, dtype=np.float32, count=1)[0]
        scaling = np.fromfile(f, dtype=np.float32, count=1)[0]
        timestamp = np.fromfile(f, dtype=np.int32, count=6)
        user = np.fromfile(f, dtype=np.int8, count=32)
        
        # Read kernel data
        kernel = np.fromfile(f, dtype=np.float32)
        kernel = kernel.reshape(dims) * scaling
        
        metadata = {
            'dimensions': dims,
            'voxel_size': voxel_size,
            'total_energy': total_energy,
            'scaling_factor': scaling,
            'creation_date': datetime(*timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            'created_by': bytes(user).decode().strip('\x00')
        }
        
        return kernel, metadata

def save_dose_map(filename: Union[str, Path],
                 dose_map: np.ndarray,
                 voxel_size: Tuple[float, float, float],
                 metadata: Dict[str, Any],
                 affine: Optional[np.ndarray] = None):
    """
    Save dose map to NIFTI format.
    
    Args:
        filename: Output filename (.nii or .nii.gz)
        dose_map: 3D dose array
        voxel_size: Voxel dimensions in mm
        metadata: Dictionary of metadata
        affine: 4x4 affine transformation matrix (optional)
    """
    # Ensure the filename has the correct extension
    filename = str(filename)
    if not filename.endswith(('.nii', '.nii.gz')):
        filename += '.nii.gz'
        
    # Create default affine if none provided
    if affine is None:
        # Create diagonal matrix with voxel sizes
        affine = np.diag(list(voxel_size) + [1.0])
        # Center the volume
        center_offset = np.array(dose_map.shape) * np.array(voxel_size) / -2.0
        affine[:3, 3] = center_offset
        
    # Update metadata with creation info
    metadata.update({
        'creation_date': '2025-02-08 09:50:56',
        'created_by': 'devhliu',
        'voxel_size': voxel_size,
        'dimensions': dose_map.shape,
        'affine_matrix': affine.tolist()
    })
        
    # Create NIFTI image
    nifti_img = nib.Nifti1Image(dose_map.astype(np.float32), affine)
    
    # Add metadata to header
    header = nifti_img.header
    header.set_data_dtype(np.float32)
    header['pixdim'][1:4] = voxel_size
    
    # Add description with creation info
    creation_info = f"Created by {metadata['created_by']} at {metadata['creation_date']}"
    header['descrip'] = creation_info
    
    # Store additional metadata in extension
    metadata_json = json.dumps(metadata)
    extension = nib.nifti1.Nifti1Extension(
        code=44,  # User-defined extension
        content=metadata_json.encode('utf-8')
    )
    nifti_img.header.extensions.append(extension)
    
    # Save the file
    nib.save(nifti_img, filename)

def load_dose_map(filename: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load dose map from NIFTI file.
    
    Args:
        filename: Input filename (.nii or .nii.gz)
        
    Returns:
        Tuple containing:
        - 3D dose array
        - Dictionary with metadata
    """
    # Load NIFTI file
    nifti_img = nib.load(filename)
    
    # Get dose map data
    dose_map = np.asarray(nifti_img.dataobj)
    
    # Extract metadata from header
    header = nifti_img.header
    affine = nifti_img.affine
    
    metadata = {
        'voxel_size': tuple(header['pixdim'][1:4].tolist()),
        'dimensions': dose_map.shape,
        'affine_matrix': affine.tolist(),
        'data_type': str(header.get_data_dtype()),
        'description': str(header['descrip'])
    }
    
    # Extract additional metadata from extension
    for extension in nifti_img.header.extensions:
        if extension.code == 44:  # User-defined extension
            try:
                ext_metadata = json.loads(extension.get_content().decode('utf-8'))
                metadata.update(ext_metadata)
            except:
                pass
                
    return dose_map, metadata

def interpolate_timepoints(time_points: List[float],
                         values: List[np.ndarray],
                         new_times: List[float],
                         method: str = 'linear') -> List[np.ndarray]:
    """
    Interpolate 3D arrays across time points.
    
    Args:
        time_points: Original time points
        values: List of 3D arrays corresponding to time points
        new_times: Time points to interpolate to
        method: Interpolation method ('linear', 'cubic', 'previous')
        
    Returns:
        List of interpolated 3D arrays
    """
    if len(time_points) != len(values):
        raise ValueError("Number of time points must match number of values")
    
    original_shape = values[0].shape
    n_voxels = np.prod(original_shape)
    
    # Reshape arrays for interpolation
    values_2d = np.array([v.reshape(-1) for v in values])
    
    # Create interpolator
    if method == 'previous':
        interpolator = interp1d(time_points, values_2d, axis=0, kind='previous',
                              bounds_error=False, fill_value='extrapolate')
    else:
        interpolator = interp1d(time_points, values_2d, axis=0, kind=method,
                              bounds_error=False, fill_value='extrapolate')
    
    # Perform interpolation
    interpolated_values = interpolator(new_times)
    
    # Reshape back to 3D
    return [v.reshape(original_shape) for v in interpolated_values]

def resample_dose_map(dose_map: np.ndarray,
                     input_spacing: Tuple[float, float, float],
                     output_spacing: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample dose map to new voxel spacing using NIFTI-based resampling.
    
    Args:
        dose_map: Input dose map
        input_spacing: Current voxel spacing (mm)
        output_spacing: Desired voxel spacing (mm)
        
    Returns:
        Tuple containing:
        - Resampled dose map
        - New affine matrix
    """
    # Create input affine matrix
    input_affine = np.diag(list(input_spacing) + [1.0])
    
    # Create input NIFTI image
    input_img = nib.Nifti1Image(dose_map, input_affine)
    
    # Calculate output shape
    output_shape = np.ceil(np.array(dose_map.shape) * 
                          np.array(input_spacing) / 
                          np.array(output_spacing)).astype(int)
    
    # Create output affine matrix
    output_affine = np.diag(list(output_spacing) + [1.0])
    
    # Perform resampling
    resampled_img = nib.processing.conform(
        input_img,
        out_shape=output_shape,
        voxel_size=output_spacing,
        order=3  # 3rd order spline interpolation
    )
    
    return np.asarray(resampled_img.dataobj), resampled_img.affine

def calculate_dvh(dose_map: np.ndarray,
                 roi_mask: np.ndarray,
                 bins: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate dose-volume histogram.
    
    Args:
        dose_map: 3D dose array
        roi_mask: Binary mask defining region of interest
        bins: Number of histogram bins
        
    Returns:
        Tuple of (dose_bins, volume_fraction)
    """
    if dose_map.shape != roi_mask.shape:
        raise ValueError("Dose map and ROI mask must have same dimensions")
        
    # Extract doses in ROI
    roi_doses = dose_map[roi_mask > 0]
    
    if len(roi_doses) == 0:
        raise ValueError("ROI mask is empty")
        
    # Calculate histogram
    hist, edges = np.histogram(roi_doses, bins=bins)
    
    # Convert to cumulative
    cum_dvh = 1.0 - np.cumsum(hist) / len(roi_doses)
    
    return edges[1:], cum_dvh

def verify_file_integrity(filename: Union[str, Path]) -> bool:
    """
    Verify integrity of saved dose map or kernel file.
    
    Args:
        filename: File to verify
        
    Returns:
        True if file is valid, False otherwise
    """
    try:
        file_ext = Path(filename).suffix.lower()
        
        if file_ext in ['.nii', '.gz']:  # NIFTI files
            img = nib.load(filename)
            img.header['pixdim']  # Check header accessibility
            return True
            
        elif file_ext == '.dat':  # Kernel file
            with open(filename, 'rb') as f:
                # Check header can be read
                dims = np.fromfile(f, dtype=np.int32, count=3)
                if len(dims) != 3:
                    return False
                    
                # Check file size matches expected size
                expected_size = (np.prod(dims) + 11) * 4  # Data + header
                actual_size = Path(filename).stat().st_size
                if expected_size != actual_size:
                    return False
                    
        else:
            return False
            
        return True
        
    except Exception:
        return False

def get_file_metadata(filename: Union[str, Path]) -> Dict[str, Any]:
    """
    Get metadata from dose map or kernel file.
    
    Args:
        filename: File to read
        
    Returns:
        Dictionary of metadata
    """
    file_ext = Path(filename).suffix.lower()
    
    if file_ext in ['.nii', '.gz']:  # NIFTI files
        _, metadata = load_dose_map(filename)
        
    elif file_ext == '.dat':  # Kernel file
        _, metadata = load_kernel(filename)
        
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
        
    return metadata