import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk

def load_nifti(fs, path):
    local = f"/tmp/{os.path.basename(path)}"
    try:
        fs.get(path, local)
        img = nib.load(local)

        data = img.get_fdata(dtype=np.float32)
        affine = img.affine
        spacing = img.header.get_zooms()[:3]
        return data, affine, spacing, img.header
    finally:
        if os.path.exists(local): os.remove(local)

def numpy_to_sitk(data, spacing):
    img = sitk.GetImageFromArray(data.transpose(2, 1, 0))
    img.SetSpacing([float(s) for s in spacing])
    return img
