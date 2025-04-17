import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk

def is_nifti_file(file_path):
    if not os.path.isfile(file_path):
        return False
    extensions = ('.nii', '.nii.gz')
    if not any(file_path.endswith(ext) for ext in extensions):
        return False
    try:
        nib.load(file_path)
        return True
    except Exception:
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