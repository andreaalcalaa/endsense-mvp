import os
import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

from src.loader import load_nifti
from src.preprocessing import resample_mask_to_mri, get_relevant_slices
from src.features import extract_features
from src.visualization import build_triptych_figure
from src.classifier_inference import run_patient_classification


st.set_page_config(
    page_title="Endsense",
    layout="wide",
    initial_sidebar_state="expanded"
)


def html_block(html: str):
    cleaned = "".join(line.strip() for line in html.strip().splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


html_block("""
<style>
.main { background-color: #0b1020; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }
h1, h2, h3 { color: #f4f7fb; }

.card {
    background: linear-gradient(180deg, #131a2e 0%, #101729 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 24px 28px;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.22);
}

.hero {
    background: linear-gradient(135deg, #111827 0%, #0f172a 55%, #172554 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 28px;
    padding: 34px 38px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

.hero-kicker {
    color:#93c5fd;
    font-size:0.82rem;
    font-weight:800;
    letter-spacing:2px;
    text-transform:uppercase;
    margin-bottom:12px;
}

.hero-title {
    font-size:3.4rem;
    font-weight:850;
    color:#f8fafc;
    letter-spacing:-1.5px;
    line-height:1;
    margin-bottom:14px;
}

.hero-subtitle {
    color:#cbd5e1;
    font-size:1.15rem;
    line-height:1.6;
    max-width:760px;
}

.card-title {
    color: #94a3b8;
    font-size: 1rem;
    margin-bottom: 0.4rem;
    font-weight: 600;
}

.card-value {
    color: #ffffff;
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.05;
}

.section-title {
    color: #ffffff;
    font-size: 1.45rem;
    font-weight: 700;
    margin-top: 1.6rem;
    margin-bottom: 1rem;
}

.risk-low {
    background-color: rgba(34,197,94,0.15);
    color: #86efac;
    padding: 14px 18px;
    border-radius: 14px;
    border: 1px solid rgba(34,197,94,0.3);
    font-weight: 700;
    font-size: 1rem;
}

.risk-mid {
    background-color: rgba(250,204,21,0.14);
    color: #fde68a;
    padding: 14px 18px;
    border-radius: 14px;
    border: 1px solid rgba(250,204,21,0.3);
    font-weight: 700;
    font-size: 1rem;
}

.risk-high {
    background-color: rgba(239,68,68,0.14);
    color: #fca5a5;
    padding: 14px 18px;
    border-radius: 14px;
    border: 1px solid rgba(239,68,68,0.3);
    font-weight: 700;
    font-size: 1rem;
}

.viz-label {
    text-align: center;
    color: #cbd5e1;
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 8px;
}

.viz-helper {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: -2px;
    margin-bottom: 14px;
}

.empty-panel {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 320px;
    background: #111827;
    color: #94a3b8;
    border: 1px dashed rgba(255,255,255,0.15);
    border-radius: 14px;
    font-size: 1rem;
    text-align: center;
    padding: 18px;
}

.loading-box {
    background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 20px 24px;
    margin-top: 18px;
    margin-bottom: 18px;
    color: #cbd5e1;
    font-size: 1.05rem;
    font-weight: 600;
}

div.stButton > button:first-child {
    height: 3.2rem;
    border-radius: 14px;
    font-size: 1.05rem;
    font-weight: 700;
}
</style>
""")


# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.markdown("## Estudios pendientes")

    dataset_root = "data/demo/D1_MHS"
    studies_path = "data/demo/studies.csv"

    if not os.path.exists(studies_path):
        st.warning("No se encontró el listado de estudios.")
        st.stop()

    studies_df = pd.read_csv(studies_path)
    study_options = studies_df["study_id"].tolist()

    def format_study_label(study_id):
        row = studies_df[studies_df["study_id"] == study_id].iloc[0]
        priority = row["priority"]
        status = row["status"]

        if priority == "Alta":
            icon = "🔴"
        elif priority == "Media":
            icon = "🟡"
        else:
            icon = "🟢"

        return f"{icon} {study_id} · {status}"

    selected_study_id = st.selectbox(
        "Lista de revisión",
        study_options,
        format_func=format_study_label,
        key="study_selectbox"
    )

    selected_study = studies_df[
        studies_df["study_id"] == selected_study_id
    ].iloc[0]

    patient_id = selected_study["folder_id"]
    study_id = selected_study["study_id"]
    upload_date = selected_study["upload_date"]
    study_status = selected_study["status"]
    study_priority = selected_study["priority"]

    sequence = st.selectbox(
        "Secuencia MRI",
        ["T2", "T1FS"],
        key="sequence_selectbox"
    )

    rater = "r3"
    threshold = 0.55


# =========================
# HEADER
# =========================

html_block("""
<div class="hero">
    <div class="hero-kicker">Clinical Review Dashboard</div>
    <div class="hero-title">Endsense</div>
    <div class="hero-subtitle">
        Plataforma de apoyo para priorización y revisión clínica de estudios MRI pélvicos.
    </div>
</div>
""")


# =========================
# STUDY OVERVIEW
# =========================

overview_col1, overview_col2 = st.columns([1, 2])

with overview_col1:
    html_block(f"""
    <div class="card">
        <div class="card-title">Estudios pendientes</div>
        <div class="card-value">{len(studies_df)}</div>
    </div>
    """)

with overview_col2:
    html_block(f"""
    <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:18px;">
            <div>
                <div class="card-title">ESTUDIO ACTIVO</div>
                <div class="card-value">{study_id}</div>
            </div>
            <div style="background:#1e293b; padding:12px 18px; border-radius:14px; color:#cbd5e1; font-weight:700; font-size:0.95rem;">
                {study_status} · Prioridad {study_priority}
            </div>
        </div>
        <div style="color:#94a3b8; font-size:1rem; margin-top:8px;">
            MRI pélvico · Secuencia {sequence} · Cargado {upload_date}
        </div>
    </div>
    """)


analyze_button = st.button(
    "Generar análisis clínico",
    use_container_width=True
)


# =========================
# ANALYSIS
# =========================

if analyze_button:
    try:
        base_folder = os.path.join(dataset_root, patient_id)

        mri_path = os.path.join(base_folder, f"{patient_id}_{sequence}.nii.gz")
        ut_path = os.path.join(base_folder, f"{patient_id}_ut_{rater}.nii.gz")
        em_path = os.path.join(base_folder, f"{patient_id}_em_{rater}.nii.gz")

        has_mri = os.path.exists(mri_path)
        has_ut = os.path.exists(ut_path)
        has_em = os.path.exists(em_path)

        missing_required = []

        if not has_mri:
            missing_required.append(f"Imagen MRI no encontrada: {mri_path}")
        if not has_ut:
            missing_required.append(f"Referencia anatómica no encontrada: {ut_path}")

        if missing_required:
            st.error("No se puede procesar este estudio porque faltan archivos requeridos.")
            for msg in missing_required:
                st.write(msg)

        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.markdown(
                "<div class='loading-box'>Inicializando análisis clínico...</div>",
                unsafe_allow_html=True
            )
            progress_bar.progress(10)
            time.sleep(0.2)

            mri_img, mri_data = load_nifti(mri_path)

            status_text.markdown(
                "<div class='loading-box'>Cargando secuencias MRI...</div>",
                unsafe_allow_html=True
            )
            progress_bar.progress(25)
            time.sleep(0.2)

            ut_img, _ = load_nifti(ut_path)
            _, ut_data = resample_mask_to_mri(ut_img, mri_img)

            ut_slices = get_relevant_slices(ut_data)
            ut_slice_list = ut_slices["all"] if ut_slices else []

            status_text.markdown(
                "<div class='loading-box'>Evaluando región anatómica relevante...</div>",
                unsafe_allow_html=True
            )
            progress_bar.progress(45)
            time.sleep(0.2)

            em_data = None
            em_slices = None
            em_slice_list = []

            if has_em:
                em_img, _ = load_nifti(em_path)
                _, em_data = resample_mask_to_mri(em_img, mri_img)
                em_slices = get_relevant_slices(em_data)
                em_slice_list = em_slices["all"] if em_slices else []

            common_slices = (
                sorted(set(ut_slice_list).intersection(set(em_slice_list)))
                if has_em else []
            )

            status_text.markdown(
                "<div class='loading-box'>Procesando hallazgos relevantes...</div>",
                unsafe_allow_html=True
            )
            progress_bar.progress(70)
            time.sleep(0.2)

            result = run_patient_classification(
                base_folder=base_folder,
                patient_id=patient_id,
                sequence=sequence,
                rater=rater,
                target_size=(128, 128),
                roi_crop_size=160,
                threshold=threshold,
                model_path="cnn_endometrioma_classifier.pth",
            )

            status_text.markdown(
                "<div class='loading-box'>Generando reporte clínico...</div>",
                unsafe_allow_html=True
            )
            progress_bar.progress(95)
            time.sleep(0.2)

            progress_bar.progress(100)
            status_text.success("Análisis completado")

            html_block("<div class='section-title'>Resultado del análisis</div>")

            c1, c2, c3 = st.columns(3)

            with c1:
                html_block(f"""
                <div class="card">
                    <div class="card-title">Probabilidad estimada</div>
                    <div class="card-value">{result['max_probability']:.2f}</div>
                </div>
                """)

            with c2:
                html_block(f"""
                <div class="card">
                    <div class="card-title">Cortes relevantes</div>
                    <div class="card-value">{result['positive_slices']}</div>
                </div>
                """)

            with c3:
                if not has_em:
                    pred_text = (
                        "Sospecha de lesión"
                        if result["patient_prediction"] == 1
                        else "Sin evidencia de lesión"
                    )
                else:
                    pred_text = (
                        "Hallazgos compatibles"
                        if result["patient_prediction"] == 1
                        else "Sin hallazgos relevantes"
                    )

                html_block(f"""
                <div class="card">
                    <div class="card-title">Interpretación</div>
                    <div class="card-value">{pred_text}</div>
                </div>
                """)

            prob = result["max_probability"]

            if prob < 0.45:
                html_block("<div class='risk-low'>Baja probabilidad clínica</div>")
            elif prob < 0.60:
                html_block("<div class='risk-mid'>Probabilidad intermedia — requiere revisión</div>")
            else:
                html_block("<div class='risk-high'>Alta probabilidad de hallazgo relevante</div>")

            st.caption(
                "Este análisis es una herramienta de apoyo y no sustituye la interpretación clínica profesional."
            )

            html_block("<div class='section-title'>Visualización MRI</div>")
            html_block("<div class='viz-helper'>Corte representativo del estudio enfocado en la región de interés clínica.</div>")

            label_col1, label_col2, label_col3 = st.columns(3)

            with label_col1:
                html_block("<div class='viz-label'>Imagen MRI</div>")

            with label_col2:
                html_block("<div class='viz-label'>Región anatómica relevante</div>")

            with label_col3:
                html_block("<div class='viz-label'>Zona de análisis</div>")

            if has_em and em_data is not None:
                fig = build_triptych_figure(
                    mri_data, ut_data, em_data, ut_slices, em_slices
                )
                st.pyplot(fig, use_container_width=True)
            else:
                preview_slice = (
                    ut_slices["middle"] if ut_slices else (mri_data.shape[2] // 2)
                )

                v1, v2, v3 = st.columns(3)

                with v1:
                    fig1, ax1 = plt.subplots(figsize=(4, 4), facecolor="#0b1020")
                    ax1.set_facecolor("#0b1020")
                    ax1.imshow(mri_data[:, :, preview_slice].T, cmap="gray", origin="lower")
                    ax1.axis("off")
                    st.pyplot(fig1, use_container_width=True)

                with v2:
                    fig2, ax2 = plt.subplots(figsize=(4, 4), facecolor="#0b1020")
                    ax2.set_facecolor("#0b1020")
                    ax2.imshow(mri_data[:, :, preview_slice].T, cmap="gray", origin="lower")
                    masked = np.ma.masked_where(
                        ut_data[:, :, preview_slice].T == 0,
                        ut_data[:, :, preview_slice].T
                    )
                    ax2.imshow(masked, cmap="spring", alpha=0.55, origin="lower")
                    ax2.axis("off")
                    st.pyplot(fig2, use_container_width=True)

                with v3:
                    html_block("<div class='empty-panel'>No se identifican regiones con características compatibles con endometrioma</div>")

            html_block("<div class='section-title'>Cortes con mayor probabilidad</div>")

            top_slices = sorted(
                result["slice_results"],
                key=lambda x: x["probability"],
                reverse=True
            )[:3]

            top_cols = st.columns(3)

            for i, slice_info in enumerate(top_slices):
                slice_idx = int(slice_info["slice_idx"])
                prob_slice = slice_info["probability"]

                with top_cols[i]:
                    fig, ax = plt.subplots(figsize=(4, 4), facecolor="#0b1020")
                    ax.set_facecolor("#0b1020")
                    ax.imshow(mri_data[:, :, slice_idx].T, cmap="gray", origin="lower")
                    ax.axis("off")
                    st.pyplot(fig, use_container_width=True)

                    html_block(f"""
                    <div class="card">
                        <div class="card-title">Corte {slice_idx}</div>
                        <div class="card-value">{prob_slice:.2f}</div>
                    </div>
                    """)

            with st.expander("Ver detalle técnico"):
                summary_data = {
                    "study_id": study_id,
                    "internal_folder_id": patient_id,
                    "sequence": sequence,
                    "anatomical_region_slices": len(ut_slice_list),
                    "analysis_region_slices": len(em_slice_list),
                    "common_slices": len(common_slices),
                    "has_anatomical_reference": len(ut_slice_list) > 0,
                    "has_target_region": has_em and len(em_slice_list) > 0,
                    "model_threshold": threshold,
                    "max_probability": result["max_probability"],
                    "positive_slices": result["positive_slices"],
                    "model_prediction": result["patient_prediction"]
                }

                summary_df = pd.DataFrame([summary_data])
                st.dataframe(summary_df, use_container_width=True)

                if has_em and em_data is not None:
                    em_features = extract_features(em_data)
                else:
                    em_features = {
                        "num_slices": 0,
                        "total_area": 0.0,
                        "max_area": 0.0,
                        "volume": 0.0
                    }

                ut_features = extract_features(ut_data)

                features = {
                    **{f"target_{k}": v for k, v in em_features.items()},
                    **{f"anatomical_{k}": v for k, v in ut_features.items()},
                    "target_anatomical_ratio": em_features["volume"] / (
                        ut_features["volume"] + 1e-5
                    )
                }

                features_df = pd.DataFrame([features])
                results_df = pd.DataFrame(result["slice_results"])

                st.markdown("### Variables extraídas")
                st.dataframe(features_df, use_container_width=True)

                st.markdown("### Detalle por corte")
                st.dataframe(results_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error en análisis: {e}")