import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

from src.dataset_builder import build_2d_segmentation_dataset
from src.torch_dataset import MRISegmentationDataset
from src.unet_model import UNetSmall

# =========================
# CONFIGURACIÓN
# =========================
dataset_root = "C:/Users/andya/Downloads/UT-EndoMRI/UT-EndoMRI/D1_MHS"
sequence = "T2"
rater = "r3"
max_patients = 30
target_size = (256, 256)

thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

# =========================
# FUNCIÓN DE POSTPROCESADO
# =========================
def keep_largest_component(binary_mask):
    labeled_mask, num_features = label(binary_mask)

    if num_features == 0:
        return binary_mask

    largest_component = 0
    largest_size = 0

    for component_id in range(1, num_features + 1):
        component_size = np.sum(labeled_mask == component_id)
        if component_size > largest_size:
            largest_size = component_size
            largest_component = component_id

    cleaned_mask = (labeled_mask == largest_component).astype(np.float32)
    return cleaned_mask

# =========================
# CARGAR DATASET
# =========================
X, y, metadata = build_2d_segmentation_dataset(
    dataset_root=dataset_root,
    sequence=sequence,
    rater=rater,
    max_patients=max_patients,
    target_size=target_size,
    use_roi=True,
    roi_crop_size=160,
    negative_keep_prob=0.2
)

dataset = MRISegmentationDataset(X, y)

# =========================
# SPLIT POR PACIENTE
# =========================
patient_ids = sorted(list(set(m["patient_id"] for m in metadata)))
np.random.seed(42)
np.random.shuffle(patient_ids)

split_idx = int(0.8 * len(patient_ids))
train_patients = set(patient_ids[:split_idx])
val_patients = set(patient_ids[split_idx:])

val_indices = [i for i, m in enumerate(metadata) if m["patient_id"] in val_patients]

# =========================
# CARGAR MODELO
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetSmall().to(device)
model.load_state_dict(torch.load("unet_small_endometrioma.pth", map_location=device))
model.eval()

# =========================
# BUSCAR UN SLICE POSITIVO EN VALIDACIÓN
# =========================
positive_idx = None
for idx in val_indices:
    if metadata[idx]["is_positive"] == 1:
        positive_idx = idx
        break

if positive_idx is None:
    print("No se encontró slice positivo en validación.")
else:
    image = torch.tensor(X[positive_idx], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    true_mask = y[positive_idx]

    with torch.no_grad():
        pred_mask = model(image)
        pred_prob = torch.sigmoid(pred_mask).cpu().numpy()[0, 0]

    # =========================
    # VISUALIZACIÓN
    # =========================
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # MRI
    axes[0, 0].imshow(X[positive_idx], cmap="gray")
    axes[0, 0].set_title("MRI slice")
    axes[0, 0].axis("off")

    # máscara real
    axes[0, 1].imshow(X[positive_idx], cmap="gray")
    masked_true = np.ma.masked_where(true_mask == 0, true_mask)
    axes[0, 1].imshow(masked_true, cmap="spring", alpha=0.6)
    axes[0, 1].set_title("Máscara real")
    axes[0, 1].axis("off")

    # mapa de probabilidad crudo
    axes[0, 2].imshow(pred_prob, cmap="hot")
    axes[0, 2].set_title("Mapa de probabilidad")
    axes[0, 2].axis("off")

    # MRI + mapa de probabilidad
    axes[0, 3].imshow(X[positive_idx], cmap="gray")
    prob_overlay = np.ma.masked_where(pred_prob < 0.05, pred_prob)
    axes[0, 3].imshow(prob_overlay, cmap="hot", alpha=0.5)
    axes[0, 3].set_title("MRI + probabilidad")
    axes[0, 3].axis("off")

    # thresholds
    flat_axes = axes[1, :].flatten()
    for i, thr in enumerate(thresholds[:4]):
        pred_binary = (pred_prob > thr).astype(np.float32)
        pred_clean = keep_largest_component(pred_binary)

        flat_axes[i].imshow(X[positive_idx], cmap="gray")
        masked_clean = np.ma.masked_where(pred_clean == 0, pred_clean)
        flat_axes[i].imshow(masked_clean, cmap="autumn", alpha=0.6)
        flat_axes[i].set_title(f"Thr={thr}")
        flat_axes[i].axis("off")

    plt.tight_layout()
    plt.show()

    # segunda figura para thresholds restantes
    if len(thresholds) > 4:
        fig2, axes2 = plt.subplots(1, len(thresholds[4:]), figsize=(10, 4))
        if len(thresholds[4:]) == 1:
            axes2 = [axes2]

        for ax, thr in zip(axes2, thresholds[4:]):
            pred_binary = (pred_prob > thr).astype(np.float32)
            pred_clean = keep_largest_component(pred_binary)

            ax.imshow(X[positive_idx], cmap="gray")
            masked_clean = np.ma.masked_where(pred_clean == 0, pred_clean)
            ax.imshow(masked_clean, cmap="autumn", alpha=0.6)
            ax.set_title(f"Thr={thr}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    print("Metadata del slice:", metadata[positive_idx])
    print("Pixeles positivos en máscara real:", int(np.sum(true_mask)))

    for thr in thresholds:
        pred_binary = (pred_prob > thr).astype(np.float32)
        pred_clean = keep_largest_component(pred_binary)
        print(f"Threshold {thr:.2f} -> pixeles positivos limpios: {int(np.sum(pred_clean))}")