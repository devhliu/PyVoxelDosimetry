#!/usr/bin/env python
"""
Dosimetry Simulation Runner

This script prepares input data, runs a GATE Monte Carlo simulation for radionuclide dosimetry,
and processes the output files to generate doserate and dosimetry results.

Requirements:
- PET input file in Bq/mL units (supported formats: MHD/RAW or DICOM series)
- CT input file in HU units (supported formats: MHD/RAW or DICOM series)
- Docker with GATE 9.4 image

Outputs:
- PET Doserate in GBq/h
- PET Dosimetry and Deposited Energy files for beta and gamma
- All output files in mhd format
"""

import os
import sys
import argparse
import subprocess
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import shutil
import time
import glob
import pydicom

from pyvoxeldosimetry.io.dicom import (
    convert_pet_dicom_to_mhd,
    convert_ct_dicom_to_mhd,
    is_dicom_directory,
    is_pet_dicom,
    is_ct_dicom,
    convert_mhd_to_dicom,
    detect_input_format
)
from pyvoxeldosimetry.io.nifti import (
    is_nifti_file,
    convert_mhd_to_nifti
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run GATE Monte Carlo simulation for radionuclide dosimetry')
    parser.add_argument('--pet', type=str, required=True, 
                      help='Path to PET image (mhd format or DICOM directory, in Bq/mL)')
    parser.add_argument('--ct', type=str, required=True, 
                      help='Path to CT image (mhd format or DICOM directory, in HU units)')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--radionuclide', type=str, help='Radionuclide name (e.g., Y90, Lu177, I131). Overrides DICOM tags if specified.')
    parser.add_argument('--half_life', type=float, help='Half-life of the radionuclide in hours. Overrides DICOM tags if specified.')
    parser.add_argument('--activity', type=float, default=4.0, help='Injection Activity in GBq')
    parser.add_argument('--time', type=float, default=2.2, help='Scan Time after Injection in hours')
    parser.add_argument('--particles', type=int, default=20000000, help='Number of particles to simulate')
    parser.add_argument('--cpu', type=int, default=8, help='Number of CPU cores for parallel processing')
    parser.add_argument('--local', dest='use_docker', action='store_false', default=True, 
                      help='Use local GATE installation instead of Docker (default: use Docker)')
    return parser.parse_args()


def resample_ct_to_pet(ct_path, pet_path, output_path):
    """Resample CT to match PET dimensions and voxel size."""
    print(f"Resampling CT to match PET dimensions")
    
    # Load images
    pet_img = sitk.ReadImage(pet_path)
    ct_img = sitk.ReadImage(ct_path)
    
    # Create resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(pet_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000)  # Air HU value for CT
    
    # Resample CT to match PET
    resampled_ct = resampler.Execute(ct_img)
    
    # Save resampled CT
    sitk.WriteImage(resampled_ct, output_path)
    print(f"CT resampled to match PET dimensions and saved to {output_path}")
    
    return resampled_ct


def prepare_data_directory(pet_path, ct_path, output_dir):
    """Prepare data directory structure for simulation."""
    # Create directories
    data_dir = os.path.join(output_dir, 'data')
    output_result_dir = os.path.join(output_dir, 'output')
    mac_dir = os.path.join(output_dir, 'mac')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_result_dir, exist_ok=True)
    os.makedirs(mac_dir, exist_ok=True)
    
    # Copy PET file to data directory
    pet_dest_path = os.path.join(data_dir, 'PET.mhd')
    pet_raw_file = pet_path.replace('.mhd', '.raw')
    pet_dest_raw = os.path.join(data_dir, 'PET.raw')
    
    # Copy MHD file if it exists
    shutil.copy(pet_path, pet_dest_path)
    # Copy RAW file if it exists
    if os.path.exists(pet_raw_file):
        shutil.copy(pet_raw_file, pet_dest_raw)
    print(f"PET file copied to {pet_dest_path}")
    
    # Resample CT to match PET dimensions and voxel size
    ct_dest_path = os.path.join(data_dir, 'CT.mhd')
    resample_ct_to_pet(ct_path, pet_path, ct_dest_path)
    
    # Copy macro file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    shutil.copy(os.path.join(script_dir, 'mac', 'executor.mac'), 
                os.path.join(mac_dir, 'executor.mac'))
    
    # Copy HU to material conversion file
    hu_material_path = os.path.join(script_dir, 'data', 'HU_to_material.txt')
    shutil.copy(hu_material_path, os.path.join(data_dir, 'HU_to_material.txt'))
    print(f"HU to material conversion table copied to {os.path.join(data_dir, 'HU_to_material.txt')}")
    
    # Copy GateMaterials.db file
    gate_materials_path = os.path.join(script_dir, 'data', 'GateMaterials.db')
    shutil.copy(gate_materials_path, os.path.join(data_dir, 'GateMaterials.db'))
    print(f"Gate materials database copied to {os.path.join(data_dir, 'GateMaterials.db')}")
    
    # Copy Materials.xml file
    gate_materials_xml_path = os.path.join(script_dir, 'data', 'Materials.xml')
    shutil.copy(gate_materials_xml_path, os.path.join(data_dir, 'Materials.xml'))
    print(f"Gate materials xml database copied to {os.path.join(data_dir, 'Materials.xml')}")
    
    return data_dir, output_result_dir, mac_dir


def get_radionuclide_info(pet_path, radionuclide=None, half_life=None):
    """Extract radionuclide information from DICOM tags or use provided values."""
    # Default values (will be overridden if information is available)
    radionuclide_name = "Y90"  # Default radionuclide
    radionuclide_half_life = 64.1  # Default half-life in hours for Y90
    atomic_number = 39  # Default atomic number for Y90
    mass_number = 90  # Default mass number for Y90
    
    # Check if the input is a DICOM directory
    if is_dicom_directory(pet_path):
        # Get the first DICOM file
        dicom_files = glob.glob(os.path.join(pet_path, "*.dcm"))
        if not dicom_files:
            dicom_files = glob.glob(os.path.join(pet_path, "*"))
        
        if dicom_files:
            try:
                # Read the DICOM file
                dcm = pydicom.dcmread(dicom_files[0])
                
                # Try to extract radionuclide information from RadiopharmaceuticalInformationSequence
                if hasattr(dcm, 'RadiopharmaceuticalInformationSequence'):
                    info = dcm.RadiopharmaceuticalInformationSequence[0]
                    
                    # Extract radionuclide name
                    if hasattr(info, 'RadionuclideCodeSequence') and len(info.RadionuclideCodeSequence) > 0:
                        code_seq = info.RadionuclideCodeSequence[0]
                        if hasattr(code_seq, 'CodeMeaning'):
                            radionuclide_name = code_seq.CodeMeaning
                    
                    # Extract half-life
                    if hasattr(info, 'RadionuclideHalfLife'):
                        # Convert from seconds to hours
                        radionuclide_half_life = float(info.RadionuclideHalfLife) / 3600.0
                    
                    # Map common radionuclide names to atomic/mass numbers
                    radionuclide_map = {
                        "Y-90": (39, 90),
                        "Y90": (39, 90),
                        "Lu-177": (71, 177),
                        "Lu177": (71, 177),
                        "I-131": (53, 131),
                        "I131": (53, 131),
                        "Tc-99m": (43, 99),
                        "Tc99m": (43, 99),
                        "F-18": (9, 18),
                        "F18": (9, 18)
                    }
                    
                    # Try to match the radionuclide name to get atomic/mass numbers
                    for key, value in radionuclide_map.items():
                        if key.lower() in radionuclide_name.lower():
                            atomic_number, mass_number = value
                            break
            except Exception as e:
                print(f"Warning: Could not extract radionuclide information from DICOM: {e}")
    
    # Override with command line arguments if provided
    if radionuclide:
        radionuclide_name = radionuclide
        # Map the radionuclide name to atomic/mass numbers
        radionuclide_map = {
            "Y90": (39, 90, 64.1),
            "Y-90": (39, 90, 64.1),
            "Lu177": (71, 177, 161.5),
            "Lu-177": (71, 177, 161.5),
            "I131": (53, 131, 192.5),
            "I-131": (53, 131, 192.5),
            "Tc99m": (43, 99, 6.01),
            "Tc-99m": (43, 99, 6.01),
            "F18": (9, 18, 1.83),
            "F-18": (9, 18, 1.83)
        }
        
        if radionuclide_name in radionuclide_map:
            atomic_number, mass_number, default_half_life = radionuclide_map[radionuclide_name]
            if not half_life:  # Only use default if not explicitly provided
                radionuclide_half_life = default_half_life
        else:
            print(f"Warning: Unknown radionuclide '{radionuclide_name}'. Using default values.")
    
    # Override half-life if explicitly provided
    if half_life:
        radionuclide_half_life = half_life
    
    print(f"Using radionuclide: {radionuclide_name} (Z={atomic_number}, A={mass_number})")
    print(f"Half-life: {radionuclide_half_life} hours")
    
    return radionuclide_name, radionuclide_half_life, atomic_number, mass_number


def scale_pet_activity(pet_path, output_path, injection_activity_gbq, scan_time_h, half_life):
    """Scale PET image from Bq/mL to absolute activity in Bq based on injection and decay."""
    # Load PET image
    pet_img = sitk.ReadImage(pet_path)
    pet_data = sitk.GetArrayFromImage(pet_img)
    
    # Get voxel volume in mL (mm³ to mL conversion: 1000 mm³ = 1 mL)
    spacing = pet_img.GetSpacing()
    voxel_volume_ml = spacing[0] * spacing[1] * spacing[2] / 1000.0
    
    # Convert from Bq/mL to Bq by multiplying by voxel volume
    pet_data_bq = pet_data * voxel_volume_ml
    
    # Calculate decay factor
    decay_factor = 2 ** (-scan_time_h / half_life)
    
    # Calculate remaining activity in Bq (GBq to Bq conversion)
    remaining_activity_bq = injection_activity_gbq * decay_factor * 1e9
    
    # Normalize PET data to sum to 1.0 (representing activity distribution)
    pet_sum = np.sum(pet_data_bq)
    if pet_sum > 0:
        pet_data_normalized = pet_data_bq / pet_sum
    else:
        raise ValueError("PET image contains no positive values")
    
    # Scale to actual activity in Bq
    pet_data_activity = pet_data_normalized * remaining_activity_bq
    
    # Create new SimpleITK image with activity data
    activity_img = sitk.GetImageFromArray(pet_data_activity)
    activity_img.CopyInformation(pet_img)
    
    # Save as MHD file
    sitk.WriteImage(activity_img, output_path)
    
    return remaining_activity_bq


def run_gate_simulation(output_dir, mac_dir, n_particles, pet_path, atomic_number, mass_number, use_docker=True, n_cpu=4):
    """Run GATE simulation using Docker or local installation."""
    start_time = time.time()
    
    # Get PET image dimensions and voxel size using SimpleITK
    pet_img = sitk.ReadImage(pet_path)
    pet_shape = pet_img.GetSize()
    pet_voxel_size = pet_img.GetSpacing()
    
    print(f"PET image dimensions: {pet_shape}")
    print(f"PET voxel size: {pet_voxel_size} mm")
    print(f"Using {n_cpu} CPU cores for parallel processing")
    
    # Replace placeholders in the macro file
    macro_path = os.path.join(mac_dir, 'executor.mac')
    with open(macro_path, 'r') as f:
        macro_content = f.read()
    
    # Replace number of particles
    macro_content = macro_content.replace('[nb]', str(n_particles))
    
    # Replace number of threads for parallel processing
    macro_content = macro_content.replace('setNumberOfThreads n-cores', f'setNumberOfThreads {n_cpu}')
    
    # Replace PET voxel size - ensure proper formatting with spaces between values
    macro_content = macro_content.replace('[PET_VOXEL_SIZE_X]', str(pet_voxel_size[0]))
    macro_content = macro_content.replace('[PET_VOXEL_SIZE_Y]', str(pet_voxel_size[1]))
    macro_content = macro_content.replace('[PET_VOXEL_SIZE_Z]', str(pet_voxel_size[2]))
    
    # Replace PET matrix size
    macro_content = macro_content.replace('[PET_MATRIX_SIZE_X]', str(pet_shape[0]))
    macro_content = macro_content.replace('[PET_MATRIX_SIZE_Y]', str(pet_shape[1]))
    macro_content = macro_content.replace('[PET_MATRIX_SIZE_Z]', str(pet_shape[2]))
    
    # Calculate world size (PET dimensions plus 32 voxels margin in each direction)
    world_size_x = (pet_shape[0] + 32) * pet_voxel_size[0]
    world_size_y = (pet_shape[1] + 32) * pet_voxel_size[1]
    world_size_z = (pet_shape[2] + 32) * pet_voxel_size[2]
    
    # Replace world size placeholders
    macro_content = macro_content.replace('[WORLD_SIZE_X]', str(world_size_x))
    macro_content = macro_content.replace('[WORLD_SIZE_Y]', str(world_size_y))
    macro_content = macro_content.replace('[WORLD_SIZE_Z]', str(world_size_z))
    
    # Replace radionuclide information
    macro_content = macro_content.replace('/gate/source/y90source/gps/ion 39 90 0 0', 
                                        f'/gate/source/y90source/gps/ion {atomic_number} {mass_number} 0 0')
    
    # Update source path to use generic PET filename
    macro_content = macro_content.replace('/gate/source/y90source/setImage /APP/data/Y90_PET.mhd', 
                                          '/gate/source/y90source/setImage /APP/data/PET.mhd')
    
    # Write the updated macro file
    with open(macro_path, 'w') as f:
        f.write(macro_content)
    
    if use_docker:
        # Run using Docker
        # Create a script to run the GATE simulation
        run_script_path = os.path.join(output_dir, 'run_gate.sh')
        with open(run_script_path, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('# Copy Materials.xml to the current directory to prevent I/O warning\n')
            f.write('cp /APP/data/Materials.xml .\n')
            f.write('# Set the macro path for GATE\n')
            f.write('export GC_DOT_GATE_DIR=/APP\n')
            f.write('# Run the GATE simulation with the full path\n')
            f.write('Gate /APP/mac/executor.mac\n')
        
        # Make the script executable
        os.chmod(run_script_path, 0o755)
        
        cmd = [
            'docker', 'run', '-ti', '--rm',
            '--cpus', f'{n_cpu}',
            '-v', f'{output_dir}:/APP',
            'opengatecollaboration/gate:9.4-docker',
            '/bin/bash', '/APP/run_gate.sh'
        ]
    else:
        # Run using local GATE installation
        cmd = ['Gate', macro_path]
    
    print(f"Running GATE simulation with {n_particles} particles...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Simulation completed in {time.time() - start_time:.2f} seconds")
    except subprocess.CalledProcessError as e:
        print(f"Error running GATE simulation: {e}")
        sys.exit(1)


def convert_to_dose_rate(dose_path, output_path, total_activity_bq):
    """Convert GATE dose output to dose rate in Gy/s and GBq/h."""
    # Load dose file using SimpleITK
    dose_img = sitk.ReadImage(dose_path)
    dose_data = sitk.GetArrayFromImage(dose_img)
    
    # Convert to dose rate (Gy/s)
    # GATE simulates total dose for all particles, so we need to normalize
    # by the number of decays simulated and multiply by activity
    dose_rate_gy_s = dose_data * (total_activity_bq / 1e9)  # Convert to GBq for scaling
    
    # Convert to GBq/h (1 Gy/s = 3600 GBq/h)
    dose_rate_gbq_h = dose_rate_gy_s * 3600
    
    # Create new SimpleITK image with dose rate data
    rate_img = sitk.GetImageFromArray(dose_rate_gbq_h)
    rate_img.CopyInformation(dose_img)
    
    # Save as MHD file
    sitk.WriteImage(rate_img, output_path)
    
    return dose_rate_gbq_h


def process_simulation_results(output_result_dir, total_activity_bq, radionuclide_name, half_life, output_format='mhd', reference_pet_path=None, reference_ct_path=None):
    """Process GATE simulation results to generate dose rate files."""
    # Create a directory for final output files
    final_output_dir = os.path.join(os.path.dirname(output_result_dir), 'final_output')
    os.makedirs(final_output_dir, exist_ok=True)
    
    # Beta dose
    beta_dose_path = os.path.join(output_result_dir, 'beta_dose-Dose.mhd')
    beta_dose_rate_path = os.path.join(output_result_dir, 'beta_doserate.mhd')
    beta_dose_rate = convert_to_dose_rate(beta_dose_path, beta_dose_rate_path, total_activity_bq)
    print(f"Beta dose rate calculated")
    
    # Gamma dose
    gamma_dose_path = os.path.join(output_result_dir, 'gamma_dose-Dose.mhd')
    gamma_dose_rate_path = os.path.join(output_result_dir, 'gamma_doserate.mhd')
    gamma_dose_rate = convert_to_dose_rate(gamma_dose_path, gamma_dose_rate_path, total_activity_bq)
    print(f"Gamma dose rate calculated")
    
    # Total dose (beta + gamma)
    total_dose_rate = beta_dose_rate + gamma_dose_rate
    total_dose_rate_path = os.path.join(output_result_dir, 'total_doserate.mhd')
    
    # Create total dose rate image
    total_rate_img = sitk.GetImageFromArray(total_dose_rate)
    # Copy image information from beta dose rate image
    beta_rate_img = sitk.ReadImage(beta_dose_rate_path)
    total_rate_img.CopyInformation(beta_rate_img)
    
    # Save total dose rate
    sitk.WriteImage(total_rate_img, total_dose_rate_path)
    
    # Calculate absorbed dose (Gy) for a standard treatment time (one half-life of the radionuclide)
    treatment_time_h = half_life  # One half-life of the radionuclide
    
    # Beta absorbed dose
    beta_absorbed_dose = beta_dose_rate * treatment_time_h / 3600  # Convert from GBq/h to Gy
    beta_absorbed_dose_path = os.path.join(output_result_dir, f'{radionuclide_name}_beta_dose.mhd')
    beta_absorbed_img = sitk.GetImageFromArray(beta_absorbed_dose)
    beta_absorbed_img.CopyInformation(beta_rate_img)
    sitk.WriteImage(beta_absorbed_img, beta_absorbed_dose_path)
    
    # Gamma absorbed dose
    gamma_absorbed_dose = gamma_dose_rate * treatment_time_h / 3600  # Convert from GBq/h to Gy
    gamma_absorbed_dose_path = os.path.join(output_result_dir, f'{radionuclide_name}_gamma_dose.mhd')
    gamma_absorbed_img = sitk.GetImageFromArray(gamma_absorbed_dose)
    gamma_absorbed_img.CopyInformation(beta_rate_img)
    sitk.WriteImage(gamma_absorbed_img, gamma_absorbed_dose_path)
    
    # Total absorbed dose
    total_absorbed_dose = beta_absorbed_dose + gamma_absorbed_dose
    total_absorbed_dose_path = os.path.join(output_result_dir, f'{radionuclide_name}_total_dose.mhd')
    total_absorbed_img = sitk.GetImageFromArray(total_absorbed_dose)
    total_absorbed_img.CopyInformation(beta_rate_img)
    sitk.WriteImage(total_absorbed_img, total_absorbed_dose_path)
    
    # List of result files to convert
    result_files = [
        ('beta_doserate.mhd', f'{radionuclide_name}_beta_doserate'),
        ('gamma_doserate.mhd', f'{radionuclide_name}_gamma_doserate'),
        ('total_doserate.mhd', f'{radionuclide_name}_total_doserate'),
        (f'{radionuclide_name}_beta_dose.mhd', f'{radionuclide_name}_beta_dose'),
        (f'{radionuclide_name}_gamma_dose.mhd', f'{radionuclide_name}_gamma_dose'),
        (f'{radionuclide_name}_total_dose.mhd', f'{radionuclide_name}_total_dose')
    ]
    
    # Convert results to the appropriate output format
    if output_format == 'dicom' and reference_pet_path:
        # Use PET DICOM as reference for dose outputs
        for mhd_file, base_name in result_files:
            mhd_path = os.path.join(output_result_dir, mhd_file)
            dicom_output_dir = os.path.join(final_output_dir, base_name)
            convert_mhd_to_dicom(
                mhd_path, 
                dicom_output_dir, 
                reference_pet_path,
                series_description=base_name
            )
            print(f"{base_name} saved as DICOM series to {dicom_output_dir}")
    
    elif output_format == 'nifti':
        # Convert to NIfTI format
        for mhd_file, base_name in result_files:
            mhd_path = os.path.join(output_result_dir, mhd_file)
            nifti_path = os.path.join(final_output_dir, f"{base_name}.nii.gz")
            convert_mhd_to_nifti(mhd_path, nifti_path)
            print(f"{base_name} saved as NIfTI to {nifti_path}")
    
    else:  # Default to MHD format
        # Copy MHD files to final output directory
        for mhd_file, base_name in result_files:
            mhd_path = os.path.join(output_result_dir, mhd_file)
            raw_path = mhd_path.replace('.mhd', '.raw')
            
            # Copy MHD file
            shutil.copy(mhd_path, os.path.join(final_output_dir, mhd_file))
            # Copy RAW file if it exists
            if os.path.exists(raw_path):
                shutil.copy(raw_path, os.path.join(final_output_dir, raw_path.split('/')[-1]))
            
            print(f"{base_name} saved as MHD to {os.path.join(final_output_dir, mhd_file)}")
    
    return final_output_dir


def detect_input_format(file_path):
    """Detect the format of an input file or directory.
    
    Args:
        file_path (str): Path to the input file or directory
        
    Returns:
        str: Format of the input ('dicom', 'nifti', 'mhd', or 'unknown')
    """
    # Check if it's a directory (potential DICOM)
    if os.path.isdir(file_path):
        if is_dicom_directory(file_path):
            return 'dicom'
        else:
            return 'unknown'
    
    # Check file extension for other formats
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == '.mhd':
        return 'mhd'
    elif file_ext == '.nii' or (len(os.path.splitext(os.path.splitext(file_path)[0])[1]) > 0 and 
                               os.path.splitext(os.path.splitext(file_path)[0])[1].lower() == '.nii' and 
                               file_ext == '.gz'):
        return 'nifti'
    else:
        return 'unknown'


def process_input_files(pet_path, ct_path, temp_dir, force_update=False):
    """Process input files, converting to MHD if necessary.
    
    Args:
        pet_path (str): Path to PET image (MHD, NIfTI file, or DICOM directory)
        ct_path (str): Path to CT image (MHD, NIfTI file, or DICOM directory)
        temp_dir (str): Temporary directory to store converted files
        force_update (bool): Force update of converted files even if they already exist
        
    Returns:
        tuple: (processed_pet_path, processed_ct_path, pet_format, ct_format)
    """
    os.makedirs(temp_dir, exist_ok=True)
    processed_pet_path = pet_path
    processed_ct_path = ct_path
    
    # Detect input formats
    pet_format = detect_input_format(pet_path)
    ct_format = detect_input_format(ct_path)
    
    print(f"Detected PET format: {pet_format}")
    print(f"Detected CT format: {ct_format}")
    
    # Process PET file
    if pet_format == 'dicom':
        processed_pet_path = os.path.join(temp_dir, 'PET.mhd')
        # Check if converted file already exists and force_update is False
        if not force_update and os.path.exists(processed_pet_path) and os.path.exists(processed_pet_path.replace('.mhd', '.raw')):
            print(f"Using existing converted PET file: {processed_pet_path}")
        else:
            if not is_pet_dicom(pet_path):
                print(f"Warning: DICOM directory {pet_path} does not appear to contain PET images")
            convert_pet_dicom_to_mhd(pet_path, processed_pet_path)
            print(f"PET DICOM converted to MHD and saved to {processed_pet_path}")
    elif pet_format == 'nifti':
        # Convert NIfTI to MHD using SimpleITK
        processed_pet_path = os.path.join(temp_dir, 'PET.mhd')
        if not force_update and os.path.exists(processed_pet_path) and os.path.exists(processed_pet_path.replace('.mhd', '.raw')):
            print(f"Using existing converted PET file: {processed_pet_path}")
        else:
            pet_img = sitk.ReadImage(pet_path)
            sitk.WriteImage(pet_img, processed_pet_path)
            print(f"PET NIfTI file converted to MHD and saved to {processed_pet_path}")
    
    # Process CT file
    if ct_format == 'dicom':
        processed_ct_path = os.path.join(temp_dir, 'CT.mhd')
        if not force_update and os.path.exists(processed_ct_path) and os.path.exists(processed_ct_path.replace('.mhd', '.raw')):
            print(f"Using existing converted CT file: {processed_ct_path}")
        else:
            if not is_ct_dicom(ct_path):
                print(f"Warning: DICOM directory {ct_path} does not appear to contain CT images")
            convert_ct_dicom_to_mhd(ct_path, processed_ct_path)
            print(f"CT DICOM converted to MHD and saved to {processed_ct_path}")
    elif ct_format == 'nifti':
        # Convert NIfTI to MHD using SimpleITK
        processed_ct_path = os.path.join(temp_dir, 'CT.mhd')
        if not force_update and os.path.exists(processed_ct_path) and os.path.exists(processed_ct_path.replace('.mhd', '.raw')):
            print(f"Using existing converted CT file: {processed_ct_path}")
        else:
            ct_img = sitk.ReadImage(ct_path)
            sitk.WriteImage(ct_img, processed_ct_path)
            print(f"CT NIfTI file converted to MHD and saved to {processed_ct_path}")
    
    return processed_pet_path, processed_ct_path, pet_format, ct_format


def main():
    """Main function to run the dosimetry simulation."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Create temporary directory for processing
    temp_dir = os.path.join(args.output, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process input files (convert to MHD if necessary)
    processed_pet_path, processed_ct_path, pet_format, ct_format = process_input_files(
        args.pet, args.ct, temp_dir, getattr(args, 'force_update', False)
    )
    
    # Get radionuclide information from DICOM or command line arguments
    radionuclide_name, half_life, atomic_number, mass_number = get_radionuclide_info(
        args.pet if pet_format == 'dicom' else None,
        args.radionuclide,
        args.half_life
    )
    
    # Determine output format based on input format
    # If both inputs are the same format, use that format for output
    # Otherwise, prioritize DICOM > NIfTI > MHD
    if pet_format == ct_format:
        output_format = pet_format
    elif 'dicom' in [pet_format, ct_format]:
        output_format = 'dicom'
    elif 'nifti' in [pet_format, ct_format]:
        output_format = 'nifti'
    else:
        output_format = 'mhd'
    
    print(f"Using output format: {output_format}")
    
    # Prepare data directory
    data_dir, output_result_dir, mac_dir = prepare_data_directory(
        processed_pet_path, processed_ct_path, args.output
    )
    
    # Scale PET activity based on injection and decay
    pet_path = os.path.join(data_dir, 'PET.mhd')
    scaled_pet_path = os.path.join(data_dir, 'PET_scaled.mhd')
    total_activity_bq = scale_pet_activity(
        pet_path, scaled_pet_path, args.activity, args.time, half_life
    )
    
    # Replace original PET with scaled PET
    shutil.copy(scaled_pet_path, pet_path)
    print(f"PET activity scaled to {total_activity_bq/1e9:.4f} GBq")
    
    # Run GATE simulation
    run_gate_simulation(
        args.output, mac_dir, args.particles, pet_path, atomic_number, mass_number, args.use_docker, args.cpu
    )
    
    # Process simulation results with format preservation
    reference_pet_path = args.pet if pet_format == 'dicom' else None
    reference_ct_path = args.ct if ct_format == 'dicom' else None
    
    process_simulation_results(
        output_result_dir, 
        total_activity_bq,
        radionuclide_name,
        half_life,
        output_format,
        reference_pet_path,
        reference_ct_path
    )
    
    print("\nDosimetry simulation completed successfully!")
    print(f"Results saved to {args.output} in {output_format.upper()} format")


if __name__ == "__main__":
    main()

