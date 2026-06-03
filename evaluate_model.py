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
    classification_report,
    roc_auc_score,
    roc_curve
)

from src.classifier_inference import run_patient_classification


# =========================
# CONFIG
# =========================

DATASET_ROOT = r"C:/Users/andya/Downloads/UT-EndoMRI/UT-EndoMRI/D1_MHS"

SEQUENCE = "T2"
RATER = "r3"

THRESHOLD = 0.50

MODEL_PATH = "cnn_endometrioma_classifier_best.pth"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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

    if not os.path.exists(mri_path):
        print("MRI faltante")
        continue

    if not os.path.exists(ut_path):
        print("UT faltante")
        continue

    real_label = 1 if os.path.exists(em_path) else 0

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

if df.empty:
    raise ValueError("No se generaron resultados. Revisa rutas o archivos faltantes.")

print("\n=========================")
print("RESULTADOS")
print("=========================\n")

print(df.head())


# =========================
# METRICS
# =========================

y_true = df["real"]
y_pred = df["pred"]
y_prob = df["probability"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = recall

try:
    auc = roc_auc_score(y_true, y_prob)
except ValueError:
    auc = None

print("\n=========================")
print("MÉTRICAS")
print("=========================\n")

print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"Sensitivity : {sensitivity:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1 Score    : {f1:.4f}")

if auc is not None:
    print(f"AUC         : {auc:.4f}")
else:
    print("AUC         : No calculable")


# =========================
# CLASSIFICATION REPORT
# =========================

print("\n=========================")
print("CLASSIFICATION REPORT")
print("=========================\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Negativo", "Positivo"],
        zero_division=0
    )
)


# =========================
# SAVE METRICS CSV
# =========================

metrics_df = pd.DataFrame([{
    "accuracy": accuracy,
    "precision": precision,
    "recall_sensitivity": sensitivity,
    "specificity": specificity,
    "f1_score": f1,
    "auc": auc if auc is not None else "NA",
    "true_negative": tn,
    "false_positive": fp,
    "false_negative": fn,
    "true_positive": tp,
    "threshold": THRESHOLD,
    "model_path": MODEL_PATH
}])

metrics_path = os.path.join(
    OUTPUT_DIR,
    f"evaluation_metrics_{SEQUENCE}.csv"
)

metrics_df.to_csv(metrics_path, index=False)

print(f"\nMétricas guardadas: {metrics_path}")


# =========================
# CONFUSION MATRIX
# =========================

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

cm_path = os.path.join(
    OUTPUT_DIR,
    f"confusion_matrix_{SEQUENCE}.png"
)

plt.savefig(cm_path, dpi=300)
plt.show()

print(f"Matriz de confusión guardada: {cm_path}")


# =========================
# ROC CURVE
# =========================

if auc is not None:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    plt.figure(figsize=(6, 5))

    plt.plot(
        fpr,
        tpr,
        marker="o",
        label=f"AUC = {auc:.3f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Clasificador aleatorio"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    roc_path = os.path.join(
        OUTPUT_DIR,
        f"roc_curve_{SEQUENCE}.png"
    )

    plt.savefig(roc_path, dpi=300)
    plt.show()

    print(f"Curva ROC guardada: {roc_path}")


# =========================
# SAVE FULL RESULTS CSV
# =========================

csv_path = os.path.join(
    OUTPUT_DIR,
    f"evaluation_results_{SEQUENCE}.csv"
)

df.to_csv(csv_path, index=False)

print(f"CSV guardado: {csv_path}")