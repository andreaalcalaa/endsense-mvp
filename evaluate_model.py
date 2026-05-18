import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.classifier_inference import run_patient_classification

# =========================
# CONFIG
# =========================

DATASET_ROOT = r"C:/Users/andya/Downloads/UT-EndoMRI/UT-EndoMRI/D1_MHS"

SEQUENCE = "T2"
RATER = "r3"

THRESHOLD = 0.75

MODEL_PATH = "cnn_endometrioma_classifier.pth"

# =========================
# LOAD PATIENTS
# =========================

patients = sorted([
    f for f in os.listdir(DATASET_ROOT)
    if os.path.isdir(os.path.join(DATASET_ROOT, f))
])

print(f"Pacientes encontrados: {len(patients)}")

# =========================
# EVALUATION STORAGE
# =========================

results = []

# =========================
# LOOP
# =========================

for patient_id in patients:

    print(f"\nProcesando {patient_id}...")

    base_folder = os.path.join(DATASET_ROOT, patient_id)

    mri_path = os.path.join(
        base_folder,
        f"{patient_id}_{SEQUENCE}.nii.gz"
    )

    ut_path = os.path.join(
        base_folder,
        f"{patient_id}_ut_{RATER}.nii.gz"
    )

    em_path = os.path.join(
        base_folder,
        f"{patient_id}_em_{RATER}.nii.gz"
    )

    # =========================
    # REQUIRED FILES
    # =========================

    if not os.path.exists(mri_path):
        print("MRI faltante")
        continue

    if not os.path.exists(ut_path):
        print("UT faltante")
        continue

    # =========================
    # GROUND TRUTH
    # =========================

    real_label = 1 if os.path.exists(em_path) else 0

    # =========================
    # RUN MODEL
    # =========================

    try:

        result = run_patient_classification(
            base_folder=base_folder,
            patient_id=patient_id,
            sequence=SEQUENCE,
            rater=RATER,
            target_size=(128, 128),
            roi_crop_size=160,
            threshold=THRESHOLD,
            model_path=MODEL_PATH,
        )

        pred_label = result["patient_prediction"]

        max_prob = result["max_probability"]

        results.append({
            "patient_id": patient_id,
            "real": real_label,
            "pred": pred_label,
            "probability": max_prob
        })

        print(
            f"GT={real_label} | "
            f"PRED={pred_label} | "
            f"PROB={max_prob:.2f}"
        )

    except Exception as e:
        print(f"ERROR: {e}")

# =========================
# DATAFRAME
# =========================

df = pd.DataFrame(results)

print("\n=========================")
print("RESULTADOS")
print("=========================\n")

print(df.head())

# =========================
# METRICS
# =========================

y_true = df["real"]
y_pred = df["pred"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\n=========================")
print("MÉTRICAS")
print("=========================\n")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# =========================
# CLASSIFICATION REPORT
# =========================

print("\n=========================")
print("CLASSIFICATION REPORT")
print("=========================\n")

print(classification_report(y_true, y_pred))

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negativo", "Positivo"],
    yticklabels=["Negativo", "Positivo"]
)

plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")

plt.tight_layout()

plt.show()

# =========================
# SAVE CSV
# =========================

csv_name = f"evaluation_results_{SEQUENCE}.csv"

df.to_csv(csv_name, index=False)

print(f"\nCSV guardado: {csv_name}")