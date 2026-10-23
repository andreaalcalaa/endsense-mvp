# EndSense

**Exploring CNN-Based Endometrioma Identification in Pelvic MRI**

Academic capstone project · Data Science Specialization · Python · PyTorch · CNN · Streamlit

> [!IMPORTANT]
> **Research prototype only — not a medical device.** EndSense explores whether deep learning can flag MRI images with potential endometrioma-related findings for closer expert review. It does **not** provide a diagnosis, has **not** been clinically validated, and must not be used for screening, triage, treatment, or individual medical decisions.

## Overview

EndSense is an academic proof of concept developed by a two-person team. Its goal was to investigate the use of convolutional neural networks (CNNs) to classify the **presence or absence of endometrioma-related findings** in pelvic magnetic resonance imaging (MRI) slices. The proposed use case is to **draw attention to potentially relevant findings**, not to replace a radiologist or gynecologist.

The project follows the **CRISP-DM** framework and includes MRI data preparation, model training, experimental validation, and a Streamlit-based interface for exploring model outputs.

## Dataset

The project used **UT-EndoMRI**, a publicly hosted research dataset of pelvic MRI studies and associated segmentation masks.

- **Source:** [UT-EndoMRI on Zenodo](https://zenodo.org/records/13749613)
- **DOI:** [10.5281/zenodo.13749613](https://doi.org/10.5281/zenodo.13749613)
- **Input format:** NIfTI medical imaging volumes (`.nii.gz`)
- **Use in this project:** Research and educational experimentation with endometrioma-related MRI findings

The source dataset and the actual subset used for model training are **not necessarily identical**. This repository does not claim that all records or imaging sequences in the source dataset were used. Consult the Zenodo record for dataset authorship, citation requirements, and current reuse terms. Public availability does not automatically permit redistribution; **do not commit source MRI scans or masks** without confirming authorization.

## Methodology

### 1. Data understanding and exploration

The project report documents exploratory analysis of positive and negative cases, relevant MRI slices, image dimensions, availability of masks, and variation across MRI sequences. The team also inspected original images, segmentation masks, and overlays. Class imbalance and study heterogeneity were identified as challenges.

### 2. Data preparation

The documented preparation workflow included:

- Reading volumetric MRI studies in NIfTI format.
- Converting 3D volumes into 2D slices.
- Resizing and intensity normalization.
- Filtering relevant slices.
- Generating binary classification labels.
- Preparing data for model training and validation.

**Validation split:** Training and validation data were separated **by patient**, so slices from a given patient were not intentionally assigned to both sets. This reduces the risk of leakage from closely related slices, but does not eliminate every source of bias or leakage.

### 3. CNN classification

A CNN was implemented and trained to classify the **presence or absence of endometrioma-related findings** in MRI slices. The codebase uses Python and PyTorch. The academic work monitored model performance using classification metrics and selected the best validation model based on F1-score.

### 4. U-Net exploration

The team also explored **U-Net** as a potential architecture for medical image segmentation. The technical report describes this as an **exploratory/conceptual direction**, distinct from the reported final CNN classification results. The repository includes U-Net-related experimental code; no clinically validated segmentation performance is claimed here.

### 5. Streamlit interface

A Streamlit interface was developed to make the experimental model workflow accessible through a visual application. Any predictions or visualizations are **research outputs only**, not clinically calibrated probabilities or diagnoses.

## Final reported validation results

The team's technical report records the following **final preliminary validation metrics** for its best CNN model:

| Metric | Reported value |
| --- | ---: |
| Validation accuracy | **~64%** |
| Best validation F1-score | **~0.45** |

The evaluation workflow also considered precision, recall (sensitivity), and a confusion matrix. These additional metric values are **not specified in the technical report**, so they are not invented here.

**Interpretation:** These figures document the outcome of an academic experiment, **not** reliable detection performance in clinical practice. The F1-score, class imbalance, limited sample size, and absence of independent clinical validation prevent claims of diagnostic usefulness. The patient-level split improves the methodology but does not establish generalizability to new institutions or imaging protocols.

## Repository structure

The repository has included the following components (filenames and setup should be checked against the current GitHub version):

| File / directory | Purpose |
| --- | --- |
| `train_classifier.py` | CNN training workflow |
| `evaluate_model.py` | Experimental classification evaluation |
| `train_unet.py` | U-Net-related segmentation experiment |
| `app.py` | Streamlit application |
| `src/` | Supporting code for data/model/inference workflows |
| `requirements.txt` | Python dependencies |

## Local development setup

> [!NOTE]
> The project currently requires local configuration of dataset paths and model artifacts. The commands below describe environment setup; they are **not** a verified, one-command reproduction of the reported results.

```bash
git clone https://github.com/andreaalcalaa/endsense-mvp.git
cd endsense-mvp
python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

After configuring compatible input data, local paths, and the required trained model artifacts, launch the interface:

```bash
streamlit run app.py
```

**Before attempting training or inference:** Review the scripts for machine-specific paths, expected MRI sequences and masks, model weight filenames, and preprocessing assumptions. Do not interpret any app output as medical advice.

## Team and contributions

**Raquel Mireles and Andrea Alcalá** — Two-person academic capstone team.

| Area | Main responsibility |
| --- | --- |
| Python implementation, CNN training, and user interface | **Andrea Alcalá** |
| Research, documentation, and data preprocessing | **Raquel Mireles** |

The project was collaborative. This table describes the principal responsibilities shared by the team; it does not assign exclusive authorship of every individual script or experiment.

## Limitations and responsible use

- **No clinical validation:** Results must not guide patient care.
- **Limited evaluation:** Reported validation accuracy and F1-score are insufficient to establish safe performance.
- **Class imbalance and data heterogeneity:** Both were identified during exploratory analysis.
- **Generalization:** Patient-level validation does not substitute for testing on independent external cohorts.
- **Dataset rights:** Follow the Zenodo dataset's attribution and reuse requirements; avoid uploading MRI data or masks without permission.
- **Reproducibility:** Environment versions, dataset subset, preprocessing choices, split manifest, and model artifacts should be documented before presenting the project as fully reproducible.

## Future work

- [ ] Publish a de-identified **patient-level split summary** and class distribution, if permitted.
- [ ] Add the complete confusion matrix and per-class precision, recall, and F1-score.
- [ ] Compare against a majority-class or other simple baseline.
- [ ] Test generalization on an independent dataset, subject to appropriate permissions.
- [ ] Replace hard-coded local file paths with configuration parameters.
- [ ] Add approved interface screenshots and a concise architecture diagram.
- [ ] Clearly separate exploratory U-Net experiments from the final CNN results.
- [ ] Verify dataset citation and any code/model redistribution permissions before adding a license.

## References

- **UT-EndoMRI dataset:** https://doi.org/10.5281/zenodo.13749613
- **Project technical report:** *Reporte Técnico – EndSense*, Raquel Mireles and Andrea Alcalá (academic project document; not automatically included in this repository).

---

*EndSense is a research and educational prototype. It is not intended for medical diagnosis, screening, triage, or treatment.*
