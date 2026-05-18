import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.dataset_builder import build_2d_segmentation_dataset
from src.torch_dataset import MRISegmentationDataset
from src.unet_model import UNetSmall

# =========================
# CONFIGURACIÓN
# =========================
dataset_root = "C:/Users/andya/Downloads/UT-EndoMRI/UT-EndoMRI/D1_MHS"
sequence = "T2"
rater = "r3"
max_patients=30
target_size = (256, 256)
batch_size = 4
epochs = 40
lr = 1e-3

# =========================
# CARGA DEL DATASET
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

print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================
# SPLIT POR PACIENTE
# =========================
patient_ids = sorted(list(set(m["patient_id"] for m in metadata)))
print("Pacientes encontrados:", patient_ids)

np.random.seed(42)
np.random.shuffle(patient_ids)

split_idx = int(0.8 * len(patient_ids))
train_patients = set(patient_ids[:split_idx])
val_patients = set(patient_ids[split_idx:])

train_indices = [i for i, m in enumerate(metadata) if m["patient_id"] in train_patients]
val_indices = [i for i, m in enumerate(metadata) if m["patient_id"] in val_patients]

print("Train patients:", train_patients)
print("Val patients:", val_patients)
print("Train slices:", len(train_indices))
print("Val slices:", len(val_indices))

# =========================
# DATASET / DATALOADER
# =========================
dataset = MRISegmentationDataset(X, y)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# =========================
# MODELO
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = UNetSmall().to(device)

# Peso para clase positiva
num_positive = sum(m["is_positive"] for m in metadata)
num_negative = len(metadata) - num_positive
pos_weight_value = num_negative / max(num_positive, 1)

def dice_loss(preds, targets, eps=1e-6):
    preds = torch.sigmoid(preds)

    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def combined_loss(preds, targets):
    bce = nn.BCEWithLogitsLoss()(preds, targets)
    d = dice_loss(preds, targets)
    return 0.3 * bce + 0.7 * d
criterion = combined_loss
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# =========================
# FUNCIÓN DICE
# =========================
def dice_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()

# =========================
# ENTRENAMIENTO
# =========================
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_dice = 0.0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_dice += dice_score(outputs.detach(), masks)

    train_loss /= len(train_loader)
    train_dice /= len(train_loader)

    model.eval()
    val_loss = 0.0
    val_dice = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item()
            val_dice += dice_score(outputs, masks)

    val_loss /= len(val_loader)
    val_dice /= len(val_loader)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"
    )

# =========================
# GUARDAR MODELO
# =========================
torch.save(model.state_dict(), "unet_small_endometrioma.pth")
print("Modelo guardado como: unet_small_endometrioma.pth")