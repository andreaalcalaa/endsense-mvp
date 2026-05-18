from src.dataset_builder import build_2d_segmentation_dataset

dataset_root = "C:/Users/andya/Downloads/UT-EndoMRI/UT-EndoMRI/D1_MHS"

X, y, metadata = build_2d_segmentation_dataset(
    dataset_root=dataset_root,
    sequence="T2",
    rater="r3",
    max_patients=5
)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Ejemplo metadata:", metadata[:5])

positive_slices = sum(m["is_positive"] for m in metadata)
print("Slices positivos:", positive_slices)
print("Slices negativos:", len(metadata) - positive_slices)