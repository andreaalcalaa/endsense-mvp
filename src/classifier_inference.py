import os
import numpy as np
import torch

from src.loader import load_nifti
from src.preprocessing import resample_mask_to_mri
from src.classification_dataset_builder import (
    normalize_slice,
    get_center_from_mask,
    crop_fixed_roi,
    resize_slice,
)
from src.classifier_model import SimpleCNNClassifier


def run_patient_classification(
    base_folder,
    patient_id,
    sequence="T2",
    rater="r3",
    target_size=(128, 128),
    roi_crop_size=160,
    threshold=0.55,
    model_path="cnn_endometrioma_classifier_best.pth",
):
    mri_path = os.path.join(base_folder, f"{patient_id}_{sequence}.nii.gz")
    ut_path = os.path.join(base_folder, f"{patient_id}_ut_{rater}.nii.gz")
    em_path = os.path.join(base_folder, f"{patient_id}_em_{rater}.nii.gz")

    if not (os.path.exists(mri_path) and os.path.exists(ut_path)):
        raise FileNotFoundError("No se encontraron MRI o máscara de útero para este paciente.")

    mri_img, mri_data = load_nifti(mri_path)
    ut_img, _ = load_nifti(ut_path)
    _, ut_data = resample_mask_to_mri(ut_img, mri_img)

    has_ground_truth = os.path.exists(em_path)
    em_data = None
    if has_ground_truth:
        em_img, _ = load_nifti(em_path)
        _, em_data = resample_mask_to_mri(em_img, mri_img)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNNClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    slice_results = []

    for slice_idx in range(mri_data.shape[2]):
        mri_slice = mri_data[:, :, slice_idx]
        ut_slice = ut_data[:, :, slice_idx]

        center = get_center_from_mask(ut_slice)
        if center is None:
            continue

        roi = crop_fixed_roi(mri_slice, center, crop_size=roi_crop_size)
        roi = normalize_slice(roi)
        roi = resize_slice(roi, target_size=target_size)

        image = torch.tensor(roi, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            logit = model(image)
            prob = torch.sigmoid(logit).item()

        pred_label = int(prob > threshold)

        true_label = None
        if has_ground_truth and em_data is not None:
            em_slice = em_data[:, :, slice_idx]
            true_label = int(np.sum(em_slice) > 0)

        slice_results.append({
            "slice_idx": slice_idx,
            "probability": prob,
            "pred_label": pred_label,
            "true_label": true_label
        })

    if len(slice_results) == 0:
        raise ValueError("No se pudieron construir ROIs válidos para este paciente.")

    max_prob = max(r["probability"] for r in slice_results)
    positive_slices = sum(r["pred_label"] for r in slice_results)

    patient_pred = int(max_prob > threshold)

    return {
        "patient_id": patient_id,
        "sequence": sequence,
        "threshold": threshold,
        "max_probability": max_prob,
        "positive_slices": positive_slices,
        "patient_prediction": patient_pred,
        "slice_results": slice_results
    }