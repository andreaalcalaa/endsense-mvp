import numpy as np


def extract_features(mask):
    slices = [i for i in range(mask.shape[2]) if np.sum(mask[:, :, i]) > 0]

    if len(slices) == 0:
        return {
            "num_slices": 0,
            "total_area": 0.0,
            "max_area": 0.0,
            "volume": 0.0
        }

    areas = [np.sum(mask[:, :, i]) for i in slices]

    return {
        "num_slices": len(slices),
        "total_area": float(sum(areas)),
        "max_area": float(max(areas)),
        "volume": float(sum(areas))
    }