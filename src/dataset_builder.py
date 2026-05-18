import os
import numpy as np
from scipy.ndimage import zoom

from src.loader import load_nifti
from src.preprocessing import resample_mask_to_mri


def normalize_slice(slice_data):
    slice_data = slice_data.astype(np.float32)
    mean = np.mean(slice_data)
    std = np.std(slice_data)

    if std == 0:
        return np.zeros_like(slice_data, dtype=np.float32)

    return (slice_data - mean) / (std + 1e-6)


def resize_slice(slice_data, target_size=(256, 256), is_mask=False):
    h, w = slice_data.shape
    target_h, target_w = target_size

    zoom_factors = (target_h / h, target_w / w)

    if is_mask:
        return zoom(slice_data, zoom_factors, order=0).astype(np.float32)
    else:
        return zoom(slice_data, zoom_factors, order=1).astype(np.float32)


def get_center_from_mask(mask_slice):
    coords = np.argwhere(mask_slice > 0)

    if len(coords) == 0:
        return None

    center_y = int(np.mean(coords[:, 0]))
    center_x = int(np.mean(coords[:, 1]))
    return center_y, center_x


def crop_fixed_roi(slice_data, center, crop_size=160):
    h, w = slice_data.shape
    cy, cx = center
    half = crop_size // 2

    y_min = cy - half
    y_max = cy + half
    x_min = cx - half
    x_max = cx + half

    # padding si el crop se sale
    pad_top = max(0, -y_min)
    pad_bottom = max(0, y_max - h)
    pad_left = max(0, -x_min)
    pad_right = max(0, x_max - w)

    y_min = max(0, y_min)
    y_max = min(h, y_max)
    x_min = max(0, x_min)
    x_max = min(w, x_max)

    cropped = slice_data[y_min:y_max, x_min:x_max]

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        cropped = np.pad(
            cropped,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0
        )

    return cropped


def build_2d_segmentation_dataset(
    dataset_root,
    sequence="T2",
    rater="r3",
    max_patients=None,
    target_size=(256, 256),
    use_roi=False,
    roi_crop_size=160,
    negative_keep_prob=0.2
):
    X = []
    y = []
    metadata = []

    patient_folders = sorted([
        f for f in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, f))
    ])

    if max_patients is not None:
        patient_folders = patient_folders[:max_patients]

    for patient_id in patient_folders:
        base_folder = os.path.join(dataset_root, patient_id)

        mri_path = os.path.join(base_folder, f"{patient_id}_{sequence}.nii.gz")
        em_path = os.path.join(base_folder, f"{patient_id}_em_{rater}.nii.gz")
        ut_path = os.path.join(base_folder, f"{patient_id}_ut_{rater}.nii.gz")

        if use_roi:
            if not (os.path.exists(mri_path) and os.path.exists(em_path) and os.path.exists(ut_path)):
                continue
        else:
            if not (os.path.exists(mri_path) and os.path.exists(em_path)):
                continue

        try:
            mri_img, mri_data = load_nifti(mri_path)
            em_img, _ = load_nifti(em_path)
            _, em_data = resample_mask_to_mri(em_img, mri_img)

            if use_roi:
                ut_img, _ = load_nifti(ut_path)
                _, ut_data = resample_mask_to_mri(ut_img, mri_img)
            else:
                ut_data = None

            for slice_idx in range(mri_data.shape[2]):
                mri_slice = mri_data[:, :, slice_idx]
                em_slice = em_data[:, :, slice_idx]

                if np.sum(em_slice) == 0:
                    continue

                if use_roi:
                    ut_slice = ut_data[:, :, slice_idx]
                    center = get_center_from_mask(ut_slice)

                    if center is None:
                        continue

                    mri_slice = crop_fixed_roi(mri_slice, center, crop_size=roi_crop_size)
                    em_slice = crop_fixed_roi(em_slice, center, crop_size=roi_crop_size)

                mri_slice = normalize_slice(mri_slice)
                em_slice = (em_slice > 0).astype(np.float32)

                mri_slice = resize_slice(mri_slice, target_size=target_size, is_mask=False)
                em_slice = resize_slice(em_slice, target_size=target_size, is_mask=True)

                X.append(mri_slice)
                y.append(em_slice)

                metadata.append({
                    "patient_id": patient_id,
                    "slice_idx": slice_idx,
                    "original_shape": mri_data[:, :, slice_idx].shape,
                    "resized_shape": target_size,
                    "is_positive": int(np.sum(em_slice) > 0),
                    "used_roi": use_roi
                })

        except Exception:
            continue

    X = np.stack(X).astype(np.float32)
    y = np.stack(y).astype(np.float32)

    return X, y, metadata