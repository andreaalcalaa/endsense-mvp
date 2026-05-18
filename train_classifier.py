import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

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
batch_size = 8
epochs = 5
lr = 1e-3

# =========================
# DATASET
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
print("Positivos:", int(np.sum(y)))
print("Negativos:", int(len(y) - np.sum(y)))

# =========================
# SPLIT POR PACIENTE
# =========================
patient_ids = sorted(list(set(m["patient_id"] for m in metadata)))
np.random.seed(42)
np.random.shuffle(patient_ids)

split_idx = int(0.8 * len(patient_ids))
train_patients = set(patient_ids[:split_idx])
val_patients = set(patient_ids[split_idx:])

train_indices = [i for i, m in enumerate(metadata) if m["patient_id"] in train_patients]
val_indices = [i for i, m in enumerate(metadata) if m["patient_id"] in val_patients]

print("Train patients:", train_patients)
print("Val patients:", val_patients)
print("Train samples:", len(train_indices))
print("Val samples:", len(val_indices))

dataset = MRIClassificationDataset(X, y)
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# =========================
# MODELO
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = SimpleCNNClassifier().to(device)

num_positive = np.sum(y)
num_negative = len(y) - num_positive
pos_weight_value = num_negative / max(num_positive, 1)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([pos_weight_value], device=device)
)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

# =========================
# ENTRENAMIENTO
# =========================
for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    val_loss /= len(val_loader)

    acc = accuracy_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Acc: {acc:.4f} | Recall: {rec:.4f} | Precision: {prec:.4f} | F1: {f1:.4f}"
    )

torch.save(model.state_dict(), "cnn_endometrioma_classifier.pth")
print("Modelo guardado como: cnn_endometrioma_classifier.pth")