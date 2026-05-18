import numpy as np
from nibabel.processing import resample_from_to


def resample_mask_to_mri(mask_img, mri_img):
    resampled_img = resample_from_to(mask_img, mri_img, order=0)
    resampled_data = resampled_img.get_fdata()
    return resampled_img, resampled_data


def get_relevant_slices(mask_data):
    slices = [i for i in range(mask_data.shape[2]) if np.sum(mask_data[:, :, i]) > 0]
    if len(slices) == 0:
        return None
    return {
        "all": slices,
        "first": slices[0],
        "middle": slices[len(slices) // 2],
        "last": slices[-1]
    }