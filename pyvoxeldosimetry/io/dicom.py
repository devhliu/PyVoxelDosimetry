import os
import numpy as np
import SimpleITK as sitk
import pydicom
from pathlib import Path
import glob
import nibabel as nib
import shutil

def load_dicom_series(dicom_dir):
    print(f"Loading DICOM series from {dicom_dir}")
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    if not dicom_names:
        raise ValueError(f"No DICOM series found in {dicom_dir}")
    reader.SetFileNames(dicom_names)
    return reader.Execute()

def convert_pet_dicom_to_mhd(dicom_dir, output_path):
    pet_img = load_dicom_series(dicom_dir)
    dicom_files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    if not dicom_files:
        dicom_files = glob.glob(os.path.join(dicom_dir, "*"))
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")
    dcm = pydicom.dcmread(dicom_files[0])
    rescale_slope = 1.0
    rescale_intercept = 0.0
    if hasattr(dcm, 'RescaleSlope'):
        rescale_slope = float(dcm.RescaleSlope)
    if hasattr(dcm, 'RescaleIntercept'):
        rescale_intercept = float(dcm.RescaleIntercept)
    activity_concentration_factor = 1.0
    try:
        if '0x7053' in dcm:
            activity_concentration_factor = float(dcm[0x7053, 0x1000].value)
    except:
        pass
    try:
        if hasattr(dcm, 'RadiopharmaceuticalInformationSequence'):
            info = dcm.RadiopharmaceuticalInformationSequence[0]
            if hasattr(info, 'RadionuclideTotalDose'):
                pass
    except:
        pass
    pet_array = sitk.GetArrayFromImage(pet_img)
    pet_array = pet_array * rescale_slope + rescale_intercept
    pet_array = pet_array * activity_concentration_factor
    pet_array = np.maximum(pet_array, 0)
    converted_img = sitk.GetImageFromArray(pet_array)
    converted_img.CopyInformation(pet_img)
    sitk.WriteImage(converted_img, output_path)
    print(f"PET DICOM series converted to MHD and saved to {output_path}")
    return output_path

def convert_ct_dicom_to_mhd(dicom_dir, output_path):
    ct_img = load_dicom_series(dicom_dir)
    dicom_files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    if not dicom_files:
        dicom_files = glob.glob(os.path.join(dicom_dir, "*"))
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")
    dcm = pydicom.dcmread(dicom_files[0])
    rescale_slope = 1.0
    rescale_intercept = 0.0
    if hasattr(dcm, 'RescaleSlope'):
        rescale_slope = float(dcm.RescaleSlope)
    if hasattr(dcm, 'RescaleIntercept'):
        rescale_intercept = float(dcm.RescaleIntercept)
    ct_array = sitk.GetArrayFromImage(ct_img)
    ct_array = ct_array * rescale_slope + rescale_intercept
    converted_img = sitk.GetImageFromArray(ct_array)
    converted_img.CopyInformation(ct_img)
    sitk.WriteImage(converted_img, output_path)
    print(f"CT DICOM series converted to MHD and saved to {output_path}")
    return output_path

def is_dicom_directory(path):
    if not os.path.isdir(path):
        return False
    dcm_files = glob.glob(os.path.join(path, "*.dcm"))
    if dcm_files:
        return True
    files = os.listdir(path)
    if not files:
        return False
    try:
        pydicom.dcmread(os.path.join(path, files[0]))
        return True
    except:
        return False

def is_pet_dicom(dicom_dir):
    dicom_files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    if not dicom_files:
        dicom_files = glob.glob(os.path.join(dicom_dir, "*"))
    if not dicom_files:
        return False
    try:
        dcm = pydicom.dcmread(dicom_files[0])
        modality = getattr(dcm, 'Modality', '')
        return modality == 'PT'
    except:
        return False

def is_ct_dicom(dicom_dir):
    dicom_files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    if not dicom_files:
        dicom_files = glob.glob(os.path.join(dicom_dir, "*"))
    if not dicom_files:
        return False
    try:
        dcm = pydicom.dcmread(dicom_files[0])
        modality = getattr(dcm, 'Modality', '')
        return modality == 'CT'
    except:
        return False

def is_nifti_file(file_path):
    if not os.path.isfile(file_path):
        return False
    extensions = ('.nii', '.nii.gz')
    if not any(file_path.endswith(ext) for ext in extensions):
        return False
    try:
        nib.load(file_path)
        return True
    except:
        return False

def convert_mhd_to_nifti(mhd_path, output_path):
    img = sitk.ReadImage(mhd_path)
    data = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()
    affine = np.eye(4)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]
    affine[0, 3] = origin[0]
    affine[1, 3] = origin[1]
    affine[2, 3] = origin[2]
    if direction != (1, 0, 0, 0, 1, 0, 0, 0, 1):
        dir_matrix = np.array(direction).reshape(3, 3)
        affine[:3, :3] = np.dot(affine[:3, :3], dir_matrix)
    nifti_img = nib.Nifti1Image(np.transpose(data, (2, 1, 0)), affine)
    nib.save(nifti_img, output_path)
    print(f"MHD file converted to NIfTI and saved to {output_path}")
    return output_path

def convert_mhd_to_dicom(mhd_path, output_dir, reference_dicom_dir, series_description=None):
    os.makedirs(output_dir, exist_ok=True)
    mhd_img = sitk.ReadImage(mhd_path)
    mhd_data = sitk.GetArrayFromImage(mhd_img)
    dicom_files = glob.glob(os.path.join(reference_dicom_dir, "*.dcm"))
    if not dicom_files:
        dicom_files = glob.glob(os.path.join(reference_dicom_dir, "*"))
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {reference_dicom_dir}")
    dicom_files.sort(key=lambda x: pydicom.dcmread(x, stop_before_pixels=True).InstanceNumber)
    reference_dcm = pydicom.dcmread(dicom_files[0])
    new_series_uid = pydicom.uid.generate_uid()
    for i, dicom_file in enumerate(dicom_files):
        if i >= mhd_data.shape[0]:
            break
        dcm = pydicom.dcmread(dicom_file)
        slice_data = mhd_data[i, :, :]
        rescale_slope = 1.0
        rescale_intercept = 0.0
        if hasattr(dcm, 'RescaleSlope'):
            rescale_slope = float(dcm.RescaleSlope)
        if hasattr(dcm, 'RescaleIntercept'):
            rescale_intercept = float(dcm.RescaleIntercept)
        if rescale_slope != 0:
            slice_data = (slice_data - rescale_intercept) / rescale_slope
        if dcm.BitsAllocated == 16:
            if dcm.PixelRepresentation == 0:
                slice_data = np.clip(slice_data, 0, 65535).astype(np.uint16)
            else:
                slice_data = np.clip(slice_data, -32768, 32767).astype(np.int16)
        elif dcm.BitsAllocated == 8:
            slice_data = np.clip(slice_data, 0, 255).astype(np.uint8)
        else:
            slice_data = np.clip(slice_data, 0, 65535).astype(np.uint16)
        dcm.PixelData = slice_data.tobytes()
        dcm.SeriesInstanceUID = new_series_uid
        if series_description:
            dcm.SeriesDescription = series_description
        dcm.InstanceNumber = i + 1
        dcm.SOPInstanceUID = pydicom.uid.generate_uid()
        output_file = os.path.join(output_dir, f"slice_{i:04d}.dcm")
        dcm.save_as(output_file)
    print(f"MHD file converted to DICOM series and saved to {output_dir}")
    return output_dir

def detect_input_format(file_path):
    from pyvoxeldosimetry.io.nifti import is_nifti_file
    if is_dicom_directory(file_path):
        return 'dicom'
    elif is_nifti_file(file_path):
        return 'nifti'
    elif file_path.endswith('.mhd'):
        return 'mhd'
    else:
        return 'mhd'