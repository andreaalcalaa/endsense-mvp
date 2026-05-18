import os
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

st.markdown("""
<style>
    .main { background-color: #0b1020; }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 { color: #f4f7fb; }

    .subtitle {
        color: #b8c0d4;
        font-size: 1rem;
        margin-top: -0.5rem;
        margin-bottom: 1.5rem;
    }

    .card {
        background: linear-gradient(180deg, #131a2e 0%, #101729 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.18);
    }

    .card-title {
        color: #c9d2e3;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }

    .card-value {
        color: #ffffff;
        font-size: 1.6rem;
        font-weight: 700;
    }

    .section-title {
        color: #ffffff;
        font-size: 1.15rem;
        font-weight: 600;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }

    .risk-low {
        background-color: rgba(34,197,94,0.15);
        color: #86efac;
        padding: 10px 14px;
        border-radius: 12px;
        border: 1px solid rgba(34,197,94,0.3);
        font-weight: 600;
    }

    .risk-mid {
        background-color: rgba(250,204,21,0.14);
        color: #fde68a;
        padding: 10px 14px;
        border-radius: 12px;
        border: 1px solid rgba(250,204,21,0.3);
        font-weight: 600;
    }

    .risk-high {
        background-color: rgba(239,68,68,0.14);
        color: #fca5a5;
        padding: 10px 14px;
        border-radius: 12px;
        border: 1px solid rgba(239,68,68,0.3);
        font-weight: 600;
    }

    .viz-card {
        background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px 18px 8px 18px;
        margin-top: 8px;
        margin-bottom: 16px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.20);
    }

    .viz-label {
        text-align: center;
        color: #cbd5e1;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 8px;
    }

    .viz-helper {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: -2px;
        margin-bottom: 12px;
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
        font-size: 0.95rem;
        text-align: center;
        padding: 16px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# Endsense")
st.markdown(
    "<div class='subtitle'>Sistema de apoyo para el análisis de MRI pélvico y detección de endometrioma</div>",
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("## Parámetros de análisis")

    dataset_root = st.text_input(
    "Ruta de la cohorte",
    "data/demo/D1_MHS"
    )

    sequence = st.selectbox(
        "Secuencia MRI",
        ["T2", "T1FS"],
        key="sequence_selectbox"
    )

    rater = "r3"

    threshold = st.slider(
        "Umbral de decisión",
        min_value=0.40,
        max_value=0.70,
        value=0.55,
        step=0.01,
        key="threshold_slider"
    )

    patient_options = []
    if os.path.exists(dataset_root):
        patient_options = sorted([
            f for f in os.listdir(dataset_root)
            if os.path.isdir(os.path.join(dataset_root, f))
        ])

    if not patient_options:
        st.warning("No se encontraron estudios en la ruta indicada.")
        st.stop()

    patient_id = st.selectbox(
        "Estudio",
        patient_options,
        key="patient_selectbox"
    )

    patient_options = []
    if os.path.exists(dataset_root):
        patient_options = sorted([
            f for f in os.listdir(dataset_root)
            if os.path.isdir(os.path.join(dataset_root, f))
        ])

    if not patient_options:
        st.warning("No se encontraron estudios en la ruta indicada.")
        st.stop()



st.markdown("<div class='section-title'>Análisis integral del estudio</div>", unsafe_allow_html=True)

if st.button("Analizar estudio completo"):
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
            mri_img, mri_data = load_nifti(mri_path)
            ut_img, _ = load_nifti(ut_path)
            _, ut_data = resample_mask_to_mri(ut_img, mri_img)

            ut_slices = get_relevant_slices(ut_data)
            ut_slice_list = ut_slices["all"] if ut_slices else []

            em_data = None
            em_slices = None
            em_slice_list = []

            if has_em:
                em_img, _ = load_nifti(em_path)
                _, em_data = resample_mask_to_mri(em_img, mri_img)
                em_slices = get_relevant_slices(em_data)
                em_slice_list = em_slices["all"] if em_slices else []

            common_slices = sorted(set(ut_slice_list).intersection(set(em_slice_list))) if has_em else []

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

            tab1, tab2, tab3 = st.tabs(["Resumen del estudio", "Visualización MRI", "Detalle técnico"])

            with tab1:
                st.markdown("<div class='section-title'>Resultado del análisis</div>", unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown(
                        f"""
                        <div class="card">
                            <div class="card-title">Probabilidad máxima</div>
                            <div class="card-value">{result['max_probability']:.2f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with c2:
                    st.markdown(
                        f"""
                        <div class="card">
                            <div class="card-title">Cortes con hallazgos</div>
                            <div class="card-value">{result['positive_slices']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with c3:
                    if not has_em:
                        pred_text = "Sospecha de lesión" if result["patient_prediction"] == 1 else "Sin evidencia de lesión"
                    else:
                        pred_text = "Hallazgos compatibles" if result["patient_prediction"] == 1 else "Sin hallazgos relevantes"

                    st.markdown(
                        f"""
                        <div class="card">
                            <div class="card-title">Interpretación del estudio</div>
                            <div class="card-value">{pred_text}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                prob = result["max_probability"]
                if prob < 0.45:
                    st.markdown("<div class='risk-low'>Baja probabilidad clínica</div>", unsafe_allow_html=True)
                elif prob < 0.60:
                    st.markdown("<div class='risk-mid'>Probabilidad intermedia — requiere revisión</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='risk-high'>Alta probabilidad de hallazgo relevante</div>", unsafe_allow_html=True)

                if has_em and len(em_slice_list) > 0:
                    st.success("Se identifican regiones compatibles con endometrioma en el estudio.")
                else:
                    st.info("No se identifican hallazgos compatibles con endometrioma en este estudio con la información disponible.")

                st.caption("Este análisis es una herramienta de apoyo y no sustituye la interpretación clínica profesional.")

            with tab2:
                st.markdown("<div class='section-title'>Visualización del estudio</div>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='viz-helper'>Visualización de un corte representativo del estudio enfocado en la región de interés clínica.</div>",
                    unsafe_allow_html=True
                )

                label_col1, label_col2, label_col3 = st.columns(3)

                with label_col1:
                    st.markdown("<div class='viz-label'>Imagen MRI</div>", unsafe_allow_html=True)

                with label_col2:
                    st.markdown("<div class='viz-label'>Región anatómica relevante</div>", unsafe_allow_html=True)

                with label_col3:
                    st.markdown("<div class='viz-label'>Zona de análisis</div>", unsafe_allow_html=True)

                if has_em and em_data is not None:
                    st.markdown("<div class='viz-card'>", unsafe_allow_html=True)
                    fig = build_triptych_figure(mri_data, ut_data, em_data, ut_slices, em_slices)
                    st.pyplot(fig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    preview_slice = ut_slices["middle"] if ut_slices else (mri_data.shape[2] // 2)

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        fig1, ax1 = plt.subplots(figsize=(4, 4), facecolor="#0b1020")
                        ax1.set_facecolor("#0b1020")
                        ax1.imshow(mri_data[:, :, preview_slice].T, cmap="gray", origin="lower")
                        ax1.axis("off")
                        st.pyplot(fig1, use_container_width=True)

                    with c2:
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

                    with c3:
                        st.markdown(
                            "<div class='empty-panel'>No se identifican regiones con características compatibles con endometrioma</div>",
                            unsafe_allow_html=True
                        )

            with tab3:
                st.markdown("<div class='section-title'>Resumen técnico del estudio</div>", unsafe_allow_html=True)

                summary_data = {
                    "study_id": patient_id,
                    "sequence": sequence,
                    "evaluator": rater,
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
                    "target_anatomical_ratio": em_features["volume"] / (ut_features["volume"] + 1e-5)
                }

                features_df = pd.DataFrame([features])

                st.markdown("<div class='section-title'>Variables extraídas</div>", unsafe_allow_html=True)
                st.dataframe(features_df, use_container_width=True)

                st.markdown("<div class='section-title'>Cortes con mayor probabilidad</div>", unsafe_allow_html=True)

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

                        st.markdown(
                            f"""
                            <div class="card">
                                <div class="card-title">Corte {slice_idx}</div>
                                <div class="card-value">{prob_slice:.2f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                st.markdown("<div class='section-title'>Detalle por corte</div>", unsafe_allow_html=True)
                results_df = pd.DataFrame(result["slice_results"])
                st.dataframe(results_df, use_container_width=True)

                summary_csv = summary_df.to_csv(index=False).encode("utf-8")
                features_csv = features_df.to_csv(index=False).encode("utf-8")
                slice_csv = results_df.to_csv(index=False).encode("utf-8")

                d1, d2, d3 = st.columns(3)

                with d1:
                    st.download_button(
                        label="Descargar resumen CSV",
                        data=summary_csv,
                        file_name=f"{patient_id}_{sequence}_summary.csv",
                        mime="text/csv"
                    )

                with d2:
                    st.download_button(
                        label="Descargar variables CSV",
                        data=features_csv,
                        file_name=f"{patient_id}_{sequence}_features.csv",
                        mime="text/csv"
                    )

                with d3:
                    st.download_button(
                        label="Descargar cortes CSV",
                        data=slice_csv,
                        file_name=f"{patient_id}_{sequence}_slice_predictions.csv",
                        mime="text/csv"
                    )

    except Exception as e:
        st.error(f"Error en análisis: {e}")