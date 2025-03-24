#!/usr/bin/env python
"""
Y90 Dosimetry Simulation Runner

This script prepares input data, runs a GATE Monte Carlo simulation for Y90 dosimetry,
and processes the output files to generate doserate and dosimetry results.

Requirements:
- Y90_PET.mhd (.raw) input file in Bq/mL units
- CT.mhd (.raw) input file in HU units
- Docker with GATE 9.4 image

Outputs:
- Y90 PET Doserate in GBq/h
- Y90 PET Dosimetry and Deposited Energy files for beta and gamma
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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Y90 GATE Monte Carlo simulation')
    parser.add_argument('--pet', type=str, required=True, help='Path to Y90 PET image (mhd format, in Bq/mL)')
    parser.add_argument('--ct', type=str, required=True, help='Path to CT image (mhd format, in HU units)')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--activity', type=float, default=4.0, help='Injection Y90 Activity in GBq')
    parser.add_argument('--time', type=float, default=2.2, help='Scan Time after Injection in hours')
    parser.add_argument('--particles', type=int, default=20000000, help='Number of particles to simulate')
    parser.add_argument('--cpu', type=int, default=4, help='Number of CPU cores for parallel processing')
    parser.add_argument('--local', dest='use_docker', action='store_false', default=True, help='Use local GATE installation instead of Docker (default: use Docker)')
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
    pet_dest_path = os.path.join(data_dir, 'Y90_PET.mhd')
    pet_raw_file = pet_path.replace('.mhd', '.raw')
    pet_dest_raw = os.path.join(data_dir, 'Y90_PET.raw')
    
    # Copy MHD file
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
    
    return data_dir, output_result_dir, mac_dir


def scale_pet_activity(pet_path, output_path, injection_activity_gbq, scan_time_h):
    """Scale PET image from Bq/mL to absolute activity in Bq based on injection and decay."""
    # Load PET image
    pet_img = sitk.ReadImage(pet_path)
    pet_data = sitk.GetArrayFromImage(pet_img)
    
    # Get voxel volume in mL (mm³ to mL conversion: 1000 mm³ = 1 mL)
    spacing = pet_img.GetSpacing()
    voxel_volume_ml = spacing[0] * spacing[1] * spacing[2] / 1000.0
    
    # Convert from Bq/mL to Bq by multiplying by voxel volume
    pet_data_bq = pet_data * voxel_volume_ml
    
    # Y90 half-life in hours
    half_life = 64.1
    
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


def run_gate_simulation(output_dir, mac_dir, n_particles, pet_path, use_docker=True, n_cpu=4):
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
    
    # Replace PET voxel size
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
    
    with open(macro_path, 'w') as f:
        f.write(macro_content)
    
    if use_docker:
        # Run using Docker
        cmd = [
            'docker', 'run', '-ti', '--rm',
            '--cpus', f'{n_cpu}',
            '-v', f'{output_dir}:/APP',
            'opengatecollaboration/gate:9.4-docker',
            f'/APP/mac/executor.mac'
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


def process_simulation_results(output_result_dir, total_activity_bq):
    """Process GATE simulation results to generate dose rate files."""
    # Beta dose
    beta_dose_path = os.path.join(output_result_dir, 'beta_dose-Dose.mhd')
    beta_dose_rate_path = os.path.join(output_result_dir, 'Y90_beta_doserate.mhd')
    beta_dose_rate = convert_to_dose_rate(beta_dose_path, beta_dose_rate_path, total_activity_bq)
    print(f"Beta dose rate saved to {beta_dose_rate_path}")
    
    # Gamma dose
    gamma_dose_path = os.path.join(output_result_dir, 'gamma_dose-Dose.mhd')
    gamma_dose_rate_path = os.path.join(output_result_dir, 'Y90_gamma_doserate.mhd')
    gamma_dose_rate = convert_to_dose_rate(gamma_dose_path, gamma_dose_rate_path, total_activity_bq)
    print(f"Gamma dose rate saved to {gamma_dose_rate_path}")
    
    # Total dose (beta + gamma)
    total_dose_rate = beta_dose_rate + gamma_dose_rate
    total_dose_rate_path = os.path.join(output_result_dir, 'Y90_total_doserate.mhd')
    
    # Create total dose rate image
    total_rate_img = sitk.GetImageFromArray(total_dose_rate)
    # Copy image information from beta dose rate image
    beta_rate_img = sitk.ReadImage(beta_dose_rate_path)
    total_rate_img.CopyInformation(beta_rate_img)
    
    # Save total dose rate
    sitk.WriteImage(total_rate_img, total_dose_rate_path)
    print(f"Total dose rate saved to {total_dose_rate_path}")
    
    # Calculate absorbed dose (Gy) for a standard treatment time (e.g., 64.1 hours = 1 half-life)
    treatment_time_h = 64.1  # One half-life of Y90
    
    # Beta absorbed dose
    beta_absorbed_dose = beta_dose_rate * treatment_time_h / 3600  # Convert from GBq/h to Gy
    beta_absorbed_dose_path = os.path.join(output_result_dir, 'Y90_beta_dose.mhd')
    beta_absorbed_img = sitk.GetImageFromArray(beta_absorbed_dose)
    beta_absorbed_img.CopyInformation(beta_rate_img)
    sitk.WriteImage(beta_absorbed_img, beta_absorbed_dose_path)
    print(f"Beta absorbed dose saved to {beta_absorbed_dose_path}")
    
    # Gamma absorbed dose
    gamma_absorbed_dose = gamma_dose_rate * treatment_time_h / 3600  # Convert from GBq/h to Gy
    gamma_absorbed_dose_path = os.path.join(output_result_dir, 'Y90_gamma_dose.mhd')
    gamma_absorbed_img = sitk.GetImageFromArray(gamma_absorbed_dose)
    gamma_absorbed_img.CopyInformation(beta_rate_img)
    sitk.WriteImage(gamma_absorbed_img, gamma_absorbed_dose_path)
    print(f"Gamma absorbed dose saved to {gamma_absorbed_dose_path}")
    
    # Total absorbed dose
    total_absorbed_dose = beta_absorbed_dose + gamma_absorbed_dose
    total_absorbed_dose_path = os.path.join(output_result_dir, 'Y90_total_dose.mhd')
    total_absorbed_img = sitk.GetImageFromArray(total_absorbed_dose)
    total_absorbed_img.CopyInformation(beta_rate_img)
    sitk.WriteImage(total_absorbed_img, total_absorbed_dose_path)
    print(f"Total absorbed dose saved to {total_absorbed_dose_path}")


def main():
    """Main function to run the Y90 dosimetry simulation."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Prepare data directory
    data_dir, output_result_dir, mac_dir = prepare_data_directory(
        args.pet, args.ct, args.output
    )
    
    # Scale PET activity based on injection and decay
    pet_path = os.path.join(data_dir, 'Y90_PET.mhd')
    scaled_pet_path = os.path.join(data_dir, 'Y90_PET_scaled.mhd')
    total_activity_bq = scale_pet_activity(
        pet_path, scaled_pet_path, args.activity, args.time
    )
    
    # Replace original PET with scaled PET
    shutil.copy(scaled_pet_path, pet_path)
    print(f"PET activity scaled to {total_activity_bq/1e9:.4f} GBq")
    
    # Run GATE simulation
    run_gate_simulation(
        args.output, mac_dir, args.particles, pet_path, args.use_docker, args.cpu
    )
    
    # Process simulation results
    process_simulation_results(output_result_dir, total_activity_bq)
    
    print("\nY90 dosimetry simulation completed successfully!")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

