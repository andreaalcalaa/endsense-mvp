import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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

max_patients = 80
target_size = (128, 128)
roi_crop_size = 160

batch_size = 8
epochs = 6
lr = 1e-4

threshold = 0.5
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)


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
    negative_keep_prob=1.0
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

train_indices = [
    i for i, m in enumerate(metadata)
    if m["patient_id"] in train_patients
]

val_indices = [
    i for i, m in enumerate(metadata)
    if m["patient_id"] in val_patients
]

print("Train patients:", train_patients)
print("Val patients:", val_patients)
print("Train samples:", len(train_indices))
print("Val samples:", len(val_indices))

dataset = MRIClassificationDataset(X, y)
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)


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

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=1e-4
)


# =========================
# HISTORIAL
# =========================
history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": [],
    "val_recall": [],
    "val_precision": [],
    "val_f1": []
}


# =========================
# ENTRENAMIENTO
# =========================
best_val_f1 = -1.0
best_epoch = 0
best_model_path = "cnn_endometrioma_classifier_best.pth"
for epoch in range(epochs):
    model.train()

    train_loss = 0.0
    train_labels = []
    train_preds = []

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        probs = torch.sigmoid(outputs)
        preds = (probs > threshold).float()

        train_labels.extend(labels.detach().cpu().numpy())
        train_preds.extend(preds.detach().cpu().numpy())

    train_loss /= len(train_loader)

    train_acc = accuracy_score(
        train_labels,
        train_preds
    )

    model.eval()

    val_loss = 0.0
    val_labels = []
    val_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()

            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())

    val_loss /= len(val_loader)

    val_acc = accuracy_score(
        val_labels,
        val_preds
    )

    val_rec = recall_score(
        val_labels,
        val_preds,
        zero_division=0
    )

    val_prec = precision_score(
        val_labels,
        val_preds,
        zero_division=0
    )

    val_f1 = f1_score(
        val_labels,
        val_preds,
        zero_division=0
    )
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch + 1

    torch.save(
        model.state_dict(),
        best_model_path
    )

    print(
        f"Nuevo mejor modelo guardado "
        f"en epoch {best_epoch} | "
        f"Val F1: {best_val_f1:.4f}"
    )

    history["epoch"].append(epoch + 1)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_accuracy"].append(train_acc)
    history["val_accuracy"].append(val_acc)
    history["val_recall"].append(val_rec)
    history["val_precision"].append(val_prec)
    history["val_f1"].append(val_f1)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Recall: {val_rec:.4f} | "
        f"Precision: {val_prec:.4f} | "
        f"F1: {val_f1:.4f}"
    )


# =========================
# GUARDAR MODELO
# =========================
last_model_path = "cnn_endometrioma_classifier_last.pth"

torch.save(
    model.state_dict(),
    last_model_path
)

print(f"Último modelo guardado como: {last_model_path}")

print(
    f"Mejor modelo guardado como: "
    f"{best_model_path} | "
    f"Epoch: {best_epoch} | "
    f"Val F1: {best_val_f1:.4f}"
)




# =========================
# GUARDAR HISTORIAL
# =========================
history_csv = os.path.join(output_dir, "training_history.csv")

with open(history_csv, "w", encoding="utf-8") as f:
    f.write(
        "epoch,train_loss,val_loss,train_accuracy,val_accuracy,"
        "val_recall,val_precision,val_f1\n"
    )

    for i in range(len(history["epoch"])):
        f.write(
            f"{history['epoch'][i]},"
            f"{history['train_loss'][i]:.6f},"
            f"{history['val_loss'][i]:.6f},"
            f"{history['train_accuracy'][i]:.6f},"
            f"{history['val_accuracy'][i]:.6f},"
            f"{history['val_recall'][i]:.6f},"
            f"{history['val_precision'][i]:.6f},"
            f"{history['val_f1'][i]:.6f}\n"
        )

print(f"Historial guardado en: {history_csv}")


# =========================
# GRÁFICA DE LOSS
# =========================
plt.figure(figsize=(8, 5))
plt.plot(history["epoch"], history["train_loss"], marker="o", label="Training loss")
plt.plot(history["epoch"], history["val_loss"], marker="o", label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

loss_path = os.path.join(output_dir, "loss_curve.png")
plt.savefig(loss_path, dpi=300)
plt.show()

print(f"Gráfica de loss guardada en: {loss_path}")


# =========================
# GRÁFICA DE ACCURACY
# =========================
plt.figure(figsize=(8, 5))
plt.plot(history["epoch"], history["train_accuracy"], marker="o", label="Training accuracy")
plt.plot(history["epoch"], history["val_accuracy"], marker="o", label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

accuracy_path = os.path.join(output_dir, "accuracy_curve.png")
plt.savefig(accuracy_path, dpi=300)
plt.show()

print(f"Gráfica de accuracy guardada en: {accuracy_path}")