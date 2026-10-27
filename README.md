# AI-Based Seizure Pattern Detection from EEG Signals

An educational research prototype that preprocesses EEG segments, extracts signal features, trains a binary classifier, and evaluates seizure/non-seizure classification with sensitivity, specificity, precision, recall, F1-score, confusion matrix, and false-alarm/missed-detection counts.

> **Safety and data note:** The bundled EEG-like sample data is synthetic and is only for checking the software workflow. It is not patient EEG, does not demonstrate clinical accuracy, and must never be used for diagnosis or treatment. Use a properly licensed, documented public EEG dataset for meaningful experiments.

## Features
- Band-pass filtering and optional notch filtering
- Time-domain and frequency-domain feature extraction
- Random Forest classification (scikit-learn)
- Evaluation metrics including sensitivity, specificity, false alarms, and missed detections
- Signal/feature visualisation and Streamlit demo
- Subject/group-aware split option when subject IDs are available

## Project structure
- `app.py` — Streamlit demo
- `src/generate_sample_data.py` — creates synthetic example segments
- `src/eeg_pipeline.py` — filtering and feature extraction
- `src/train.py` — train/test and save model
- `src/evaluate.py` — independent dataset evaluation
- `data/sample/eeg_segments.csv` — synthetic feature-ready segment dataset
- `docs/PROJECT_REPORT.md` — editable report draft
- `docs/EEG_Seizure_Detection_Presentation.pptx` — editable PPT starter
- `notebooks/EEG_Exploration.ipynb` — exploratory notebook starter

## Setup (Windows PowerShell)
Open this folder in VS Code and run:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If `py` is not recognised, install Python from the official Python website and enable **Add Python to PATH** during installation.

## Run
```powershell
python src/generate_sample_data.py
python src/train.py
python src/evaluate.py
streamlit run app.py
```

## Dataset format
The included CSV is a feature-table example. For a real dataset, either provide one row per EEG segment with pre-extracted features, or adapt the loader to read the dataset's raw EDF/CSV format and use `src/eeg_pipeline.py`.

Feature-table columns expected by the trainer:
- Feature columns: `mean`, `std`, `variance`, `rms`, `line_length`, `zero_crossings`, `delta_power`, `theta_power`, `alpha_power`, `beta_power`, `gamma_power`, `dominant_frequency`
- Target: `label` (`non_seizure` or `seizure`)
- Optional grouping column: `subject_id` (recommended for subject-independent evaluation)

## Important evaluation guidance
- Do not randomly split overlapping segments from the same patient across train and test.
- Prefer patient-/subject-independent splits (`subject_id`) or the dataset's official split.
- Report sensitivity, specificity, precision, recall, F1, confusion matrix, false alarms, and missed detections; accuracy alone is insufficient.
- Document dataset source, license, channels, sampling rate, segment duration, label definition, preprocessing, and split method.
- Do not claim results on real EEG if you only used synthetic sample data.

This is a research/education prototype, not a clinical diagnostic tool.
