import numpy as np
import torch
import matplotlib.pyplot as plt

from src.classification_dataset_builder import build_2d_classification_dataset
from src.torch_classification_dataset import MRIClassificationDataset
from src.classifier_model import SimpleCNNClassifier

# =========================
# CONFIGURACIÓN
# =========================
dataset_root = "C:/Users/andya/Downloads/UT-EndoMRI/UT-EndoMRI/D1_MHS"
sequence = "T2"
rater = "r3"
max_patients = 30
target_size = (128, 128)
roi_crop_size = 160

# =========================
# CARGAR DATASET
# =========================
X, y, metadata = build_2d_classification_dataset(
    dataset_root=dataset_root,
    sequence=sequence,
    rater=rater,
    max_patients=max_patients,
    target_size=target_size,
    roi_crop_size=roi_crop_size,
    negative_keep_prob=0.3
)

print("X shape:", X.shape)
print("y shape:", y.shape)

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

model = SimpleCNNClassifier().to(device)
model.load_state_dict(torch.load("cnn_endometrioma_classifier.pth", map_location=device))
model.eval()

# =========================
# ELEGIR UN EJEMPLO DE VALIDACIÓN
# =========================
positive_indices = [i for i in val_indices if y[i] == 1]
negative_indices = [i for i in val_indices if y[i] == 0]

print("Positivos en validación:", len(positive_indices))
print("Negativos en validación:", len(negative_indices))

# Cambia esto a negative_indices[0] si quieres ver un negativo
sample_idx = positive_indices[0] if len(positive_indices) > 0 else val_indices[0]

image = torch.tensor(X[sample_idx], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
true_label = int(y[sample_idx])

with torch.no_grad():
    logit = model(image)
    prob = torch.sigmoid(logit).item()

pred_label = int(prob > 0.55)

# =========================
# VISUALIZACIÓN
# =========================
plt.figure(figsize=(6, 6))
plt.imshow(X[sample_idx], cmap="gray")
plt.title(
    f"Predicción: {pred_label} | Prob: {prob:.4f} | Real: {true_label}\n"
    f"Paciente: {metadata[sample_idx]['patient_id']} | Slice: {metadata[sample_idx]['slice_idx']}"
)
plt.axis("off")
plt.show()

print("Metadata:", metadata[sample_idx])
print("True label:", true_label)
print("Predicted probability:", prob)
print("Predicted label:", pred_label)