import os
import zipfile
import tempfile
import shutil
import time
import numpy as np
import streamlit as st
import pandas as pd

from src.loader import load_nifti
from src.preprocessing import resample_mask_to_mri, get_relevant_slices
from src.features import extract_features
from src.classifier_inference import run_patient_classification


st.set_page_config(
    page_title="Endsense",
    layout="wide",
    initial_sidebar_state="expanded"
)


def html_block(html: str):
    cleaned = "".join(line.strip() for line in html.strip().splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


def clinical_level(prob):
    if prob < 0.45:
        return "Baja sospecha clínica", "risk-low"
    elif prob < 0.60:
        return "Sospecha intermedia", "risk-mid"
    else:
        return "Alta sospecha clínica", "risk-high"


def normalize_slice(slice_img):
    img = slice_img.astype(np.float32)
    img = np.nan_to_num(img)
    img = img - np.min(img)
    max_val = np.max(img)
    if max_val > 0:
        img = img / max_val
    return (img * 255).astype(np.uint8)


def orient_slice(volume, slice_idx):
    return np.flipud(volume[:, :, slice_idx].T)


def make_overlay(mri_slice, mask_slice, color=(255, 99, 71), alpha=0.50):
    base = normalize_slice(mri_slice)
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    mask = mask_slice > 0
    color_arr = np.array(color, dtype=np.float32)
    rgb[mask] = (1 - alpha) * rgb[mask] + alpha * color_arr
    return np.clip(rgb, 0, 255).astype(np.uint8)


def find_uploaded_study_folder(extract_dir, sequence):
    nii_files = []

    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith(".nii.gz"):
                nii_files.append((root, file))

    if not nii_files:
        return None, None, []

    sequence_candidates = [
        (root, file)
        for root, file in nii_files
        if file.endswith(f"_{sequence}.nii.gz")
    ]

    selected_root, selected_file = (
        sequence_candidates[0] if sequence_candidates else nii_files[0]
    )

    patient_id = selected_file.split("_")[0]
    return selected_root, patient_id, nii_files
def find_flexible_file(base_folder, patient_id, keywords, sequence=None):
    matches = []

    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if not file.lower().endswith(".nii.gz"):
                continue

            lower = file.lower()

            if patient_id.lower() not in lower:
                continue

            if sequence is not None and sequence.lower() not in lower:
                continue

            for keyword in keywords:
                if keyword.lower() in lower:
                    matches.append(os.path.join(root, file))

    return matches[0] if matches else None


def ensure_expected_file(found_path, expected_path):
    if found_path is None:
        return None

    if os.path.abspath(found_path) != os.path.abspath(expected_path):
        shutil.copyfile(found_path, expected_path)

    return expected_path


html_block("""
<style>
.main { background-color: #0b1020; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }

.card {
    background: linear-gradient(180deg, #131a2e 0%, #101729 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 22px 24px;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.22);
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    backdrop-filter: blur(10px);
}

.card:hover {
    transform: translateY(-3px);
    border-color: rgba(147,197,253,0.22);
    box-shadow: 0 14px 32px rgba(0,0,0,0.32);
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
    font-weight: 800;
    font-size: 1.05rem;
}

.risk-mid {
    background-color: rgba(250,204,21,0.14);
    color: #fde68a;
    padding: 14px 18px;
    border-radius: 14px;
    border: 1px solid rgba(250,204,21,0.3);
    font-weight: 800;
    font-size: 1.05rem;
}

.risk-high {
    background-color: rgba(239,68,68,0.14);
    color: #fca5a5;
    padding: 14px 18px;
    border-radius: 14px;
    border: 1px solid rgba(239,68,68,0.3);
    font-weight: 800;
    font-size: 1.05rem;
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
    height: 3.3rem;
    border-radius: 16px;
    font-size: 1.02rem;
    font-weight: 700;
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    color: white;
    border: none;
    transition: transform 0.15s ease, box-shadow 0.15s ease, opacity 0.15s ease;
    box-shadow: 0 8px 22px rgba(37,99,235,0.35);
}

div.stButton > button:first-child:hover {
    transform: translateY(-2px);
    opacity: 0.96;
    box-shadow: 0 12px 28px rgba(37,99,235,0.42);
}

@media (max-width: 768px) {
    .hero { padding: 24px 22px; border-radius: 22px; }
    .hero-title { font-size: 2.4rem; }
    .hero-subtitle { font-size: 1rem; }
    .card { padding: 18px 20px; border-radius: 18px; }
    .card-value { font-size: 1.8rem; }
    .section-title { font-size: 1.25rem; }
}

[data-testid="stImage"] img {
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 10px 24px rgba(0,0,0,0.24);
}
</style>
""")


# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.markdown("## Worklist")

    input_mode = st.radio(
        "Modo de entrada",
        ["Demo", "Upload ZIP"],
        horizontal=False,
        key="input_mode_radio"
    )

    rater = "r3"
    threshold = 0.55
    can_analyze = True

    selected_base_folder = None
    patient_id = None
    study_id = None
    upload_date = None
    study_status = None
    study_priority = None
    sequence = None
    studies_count = 0

    if input_mode == "Demo":
        dataset_root = "data/demo/D1_MHS"
        studies_path = "data/demo/studies.csv"

        if not os.path.exists(studies_path):
            st.warning("No se encontró el listado de estudios.")
            st.stop()

        studies_df = pd.read_csv(studies_path)
        studies_count = len(studies_df)
        study_options = studies_df["study_id"].tolist()

        def format_study_label(study_id_value):
            row = studies_df[studies_df["study_id"] == study_id_value].iloc[0]
            priority = row["priority"]
            status = row["status"]

            if priority == "Alta":
                icon = "🔴"
            elif priority == "Media":
                icon = "🟡"
            else:
                icon = "🟢"

            return f"{icon} {study_id_value} · {status}"

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

        selected_base_folder = os.path.join(dataset_root, patient_id)

    else:
        uploaded_zip = st.file_uploader(
            "Subir estudio (.zip)",
            type=["zip"],
            key="uploaded_study_zip"
        )

        study_id = "Uploaded Study"
        upload_date = "Ahora"
        study_status = "Nuevo"
        study_priority = "Pendiente"
        studies_count = 1

        sequence = st.selectbox(
            "Secuencia MRI",
            ["T2", "T1FS"],
            key="upload_sequence_selectbox"
        )

        if uploaded_zip is None:
            can_analyze = False
            st.info("Sube un archivo ZIP para habilitar el análisis.")
        else:
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, uploaded_zip.name)

            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.read())

            extract_dir = os.path.join(temp_dir, "study")

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            selected_base_folder, patient_id, nii_files = find_uploaded_study_folder(
                extract_dir,
                sequence
            )

            if patient_id is None or selected_base_folder is None:
                can_analyze = False
                st.error("No se encontraron archivos .nii.gz dentro del ZIP.")
            else:
                study_id = f"UPLOAD-{patient_id}"
                st.success(f"Estudio detectado: {patient_id}")


# Reset analysis if study, mode, sequence or folder changes
current_case_key = f"{input_mode}_{patient_id}_{sequence}_{selected_base_folder}"

if st.session_state.get("selected_case_key") != current_case_key:
    keys_to_clear = [
        "analysis_ready",
        "mri_data",
        "ut_data",
        "em_data",
        "result",
        "ut_slices",
        "em_slices",
        "common_slices",
        "ut_slice_list",
        "em_slice_list",
        "has_em",
        "has_ut",
        "has_mri",
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state["selected_case_key"] = current_case_key


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
        <div class="card-title">Estudios disponibles</div>
        <div class="card-value">{studies_count}</div>
    </div>
    """)

with overview_col2:
    html_block(f"""
    <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:18px;">
            <div>
                <div class="card-title">ESTUDIO ACTIVO</div>
                <div class="card-value">{study_id if study_id else "Sin estudio"}</div>
            </div>
            <div style="background:#1e293b; padding:12px 18px; border-radius:14px; color:#cbd5e1; font-weight:700; font-size:0.95rem;">
                {study_status if study_status else "Pendiente"} · Prioridad {study_priority if study_priority else "N/A"}
            </div>
        </div>
        <div style="color:#94a3b8; font-size:1rem; margin-top:8px;">
            MRI pélvico · Secuencia {sequence if sequence else "N/A"} · Cargado {upload_date if upload_date else "N/A"}
        </div>
    </div>
    """)


analyze_button = st.button(
    "Generar análisis clínico",
    use_container_width=True
)


# =========================
# RUN ANALYSIS
# =========================

if analyze_button and can_analyze:
    try:
        base_folder = selected_base_folder

        if base_folder is None or patient_id is None:
            st.error("No hay un estudio válido para analizar.")
            st.stop()

        expected_mri_path = os.path.join(base_folder, f"{patient_id}_{sequence}.nii.gz")
        expected_ut_path = os.path.join(base_folder, f"{patient_id}_ut_{rater}.nii.gz")
        expected_em_path = os.path.join(base_folder, f"{patient_id}_em_{rater}.nii.gz")

        mri_found_path = find_flexible_file(
            base_folder,
            patient_id,
            [f"_{sequence.lower()}", sequence.lower()],
            sequence=sequence
        )

        ut_found_path = find_flexible_file(
            base_folder,
            patient_id,
            ["_ut_", "uterus", "ut"]
        )

        em_found_path = find_flexible_file(
            base_folder,
            patient_id,
            ["_em_", "endometrioma", "em"]
        )

        mri_path = ensure_expected_file(mri_found_path, expected_mri_path)
        ut_path = ensure_expected_file(ut_found_path, expected_ut_path)
        em_path = ensure_expected_file(em_found_path, expected_em_path)

        has_mri = mri_path is not None and os.path.exists(mri_path)
        has_ut = ut_path is not None and os.path.exists(ut_path)
        has_em = em_path is not None and os.path.exists(em_path)

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
                "<div class='loading-box'>Evaluando área de referencia...</div>",
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
                "<div class='loading-box'>Procesando hallazgos de interés...</div>",
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
                "<div class='loading-box'>Generando lectura asistida...</div>",
                unsafe_allow_html=True
            )
            progress_bar.progress(95)
            time.sleep(0.2)

            progress_bar.progress(100)
            status_text.success("Análisis completado")

            st.session_state["analysis_ready"] = True
            st.session_state["mri_data"] = mri_data
            st.session_state["ut_data"] = ut_data
            st.session_state["em_data"] = em_data
            st.session_state["result"] = result
            st.session_state["ut_slices"] = ut_slices
            st.session_state["em_slices"] = em_slices
            st.session_state["common_slices"] = common_slices
            st.session_state["ut_slice_list"] = ut_slice_list
            st.session_state["em_slice_list"] = em_slice_list
            st.session_state["has_em"] = has_em
            st.session_state["has_ut"] = has_ut
            st.session_state["has_mri"] = has_mri
            st.session_state["analysis_patient_id"] = patient_id
            st.session_state["analysis_base_folder"] = base_folder

    except Exception as e:
        st.error(f"Error en análisis: {e}")


# =========================
# DISPLAY ANALYSIS
# =========================

if st.session_state.get("analysis_ready"):
    mri_data = st.session_state["mri_data"]
    ut_data = st.session_state["ut_data"]
    em_data = st.session_state["em_data"]
    result = st.session_state["result"]
    ut_slices = st.session_state["ut_slices"]
    em_slices = st.session_state["em_slices"]
    common_slices = st.session_state["common_slices"]
    ut_slice_list = st.session_state["ut_slice_list"]
    em_slice_list = st.session_state["em_slice_list"]
    has_em = st.session_state["has_em"]

    prob = result["max_probability"]
    level_text, risk_class = clinical_level(prob)

    html_block("<div class='section-title'>Resultado del análisis</div>")

    c1, c2, c3 = st.columns(3)

    with c1:
        html_block(f"""
        <div class="card">
            <div class="card-title">Sospecha clínica</div>
            <div class="card-value">{level_text}</div>
        </div>
        """)

    with c2:
        html_block(f"""
        <div class="card">
            <div class="card-title">Probabilidad del hallazgo</div>
            <div class="card-value">{prob * 100:.0f}%</div>
        </div>
        """)

    with c3:
        if not has_em:
            pred_text = (
                "Revisión recomendada"
                if result["patient_prediction"] == 1
                else "Sin hallazgos prioritarios"
            )
        else:
            pred_text = (
                "Hallazgos compatibles"
                if result["patient_prediction"] == 1
                else "Sin hallazgos relevantes"
            )

        html_block(f"""
        <div class="card">
            <div class="card-title">Lectura asistida</div>
            <div class="card-value">{pred_text}</div>
        </div>
        """)

    html_block(f"<div class='{risk_class}'>{level_text}</div>")

    st.caption(
        "Este análisis es una herramienta de apoyo y no sustituye la interpretación clínica profesional."
    )

    html_block("<div class='section-title'>Visualización MRI</div>")
    html_block("<div class='viz-helper'>Corte representativo del estudio enfocado en la región de interés clínica.</div>")

    if has_em and em_data is not None:
        selected_slice = (
            common_slices[len(common_slices) // 2]
            if common_slices
            else em_slices["middle"]
        )

        mri_view = normalize_slice(orient_slice(mri_data, selected_slice))
        ut_view = make_overlay(
            orient_slice(mri_data, selected_slice),
            orient_slice(ut_data, selected_slice),
            color=(34, 197, 94),
            alpha=0.45
        )
        em_view = make_overlay(
            orient_slice(mri_data, selected_slice),
            orient_slice(em_data, selected_slice),
            color=(239, 68, 68),
            alpha=0.55
        )

        v1, v2, v3 = st.columns(3)

        with v1:
            html_block("<div class='viz-label'>Imagen MRI</div>")
            st.image(
                mri_view,
                caption=f"Corte {selected_slice}",
                use_container_width=True,
                clamp=True
            )

        with v2:
            html_block("<div class='viz-label'>Área de referencia</div>")
            st.image(
                ut_view,
                caption=f"Corte {selected_slice}",
                use_container_width=True,
                clamp=True
            )

        with v3:
            html_block("<div class='viz-label'>Hallazgo evaluado</div>")
            st.image(
                em_view,
                caption=f"Corte {selected_slice}",
                use_container_width=True,
                clamp=True
            )

    else:
        preview_slice = (
            ut_slices["middle"] if ut_slices else (mri_data.shape[2] // 2)
        )

        mri_view = normalize_slice(orient_slice(mri_data, preview_slice))
        ut_view = make_overlay(
            orient_slice(mri_data, preview_slice),
            orient_slice(ut_data, preview_slice),
            color=(34, 197, 94),
            alpha=0.45
        )

        v1, v2, v3 = st.columns(3)

        with v1:
            html_block("<div class='viz-label'>Imagen MRI</div>")
            st.image(
                mri_view,
                caption=f"Corte {preview_slice}",
                use_container_width=True,
                clamp=True
            )

        with v2:
            html_block("<div class='viz-label'>Área de referencia</div>")
            st.image(
                ut_view,
                caption=f"Corte {preview_slice}",
                use_container_width=True,
                clamp=True
            )

        with v3:
            html_block("<div class='viz-label'>Hallazgo evaluado</div>")
            html_block("<div class='empty-panel'>No se identifican regiones con características compatibles con endometrioma</div>")

    html_block("<div class='section-title'>Cortes prioritarios para revisión</div>")

    top_slices = sorted(
        result["slice_results"],
        key=lambda x: x["probability"],
        reverse=True
    )[:3]

    if top_slices:
        main_slice = int(top_slices[0]["slice_idx"])
        main_prob = top_slices[0]["probability"]

        main_img = normalize_slice(orient_slice(mri_data, main_slice))

        html_block(f"""
        <div class="card">
            <div class="card-title">Corte principal sugerido</div>
            <div class="card-value">Corte {main_slice} · {main_prob * 100:.0f}%</div>
        </div>
        """)

        st.image(
            main_img,
            caption=f"Corte prioritario {main_slice} · Probabilidad del hallazgo {main_prob * 100:.0f}%",
            use_container_width=True,
            clamp=True
        )

        html_block("<div class='section-title'>Otros cortes de interés</div>")

        top_cols = st.columns(3)

        for i, slice_info in enumerate(top_slices):
            slice_idx = int(slice_info["slice_idx"])
            prob_slice = slice_info["probability"]
            img = normalize_slice(orient_slice(mri_data, slice_idx))

            with top_cols[i]:
                html_block(f"<div class='viz-label'>Corte {slice_idx}</div>")
                st.image(
                    img,
                    caption=f"{prob_slice * 100:.0f}% probabilidad",
                    use_container_width=True,
                    clamp=True
                )

                html_block(f"""
                <div class="card">
                    <div class="card-title">Probabilidad del hallazgo</div>
                    <div class="card-value">{prob_slice * 100:.0f}%</div>
                </div>
                """)

    html_block("<div class='section-title'>Explorador de cortes</div>")
    html_block("<div class='viz-helper'>Explora manualmente el estudio MRI y revisa cualquier corte axial.</div>")

    total_slices = mri_data.shape[2]

    default_slice = (
        int(top_slices[0]["slice_idx"])
        if top_slices
        else total_slices // 2
    )

    selected_slice = st.slider(
        "Seleccionar corte axial",
        min_value=0,
        max_value=total_slices - 1,
        value=default_slice,
        step=1,
        key="slice_explorer_slider"
    )

    view_mode = st.radio(
        "Vista",
        ["MRI limpio", "Área de referencia", "Hallazgo evaluado"],
        horizontal=True,
        key="slice_view_mode"
    )

    if view_mode == "MRI limpio":
        explorer_img = normalize_slice(
            orient_slice(mri_data, selected_slice)
        )

    elif view_mode == "Área de referencia":
        explorer_img = make_overlay(
            orient_slice(mri_data, selected_slice),
            orient_slice(ut_data, selected_slice),
            color=(34, 197, 94),
            alpha=0.45
        )

    else:
        if has_em and em_data is not None:
            explorer_img = make_overlay(
                orient_slice(mri_data, selected_slice),
                orient_slice(em_data, selected_slice),
                color=(239, 68, 68),
                alpha=0.55
            )
        else:
            explorer_img = normalize_slice(
                orient_slice(mri_data, selected_slice)
            )

    st.image(
        explorer_img,
        caption=f"{view_mode} · Corte {selected_slice}",
        use_container_width=True,
        clamp=True
    )

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