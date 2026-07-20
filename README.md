# NephroAI — Explainable CKD Detection Across Two Clinical Populations

## Problem
Most existing Chronic Kidney Disease ML studies are trained and evaluated on a
single dataset, and few provide per-patient explanations for their predictions.
This project builds a CKD classifier trained on a **unified, harmonized dataset
combining two independently-collected clinical sources** (UCI + Enam Medical
College, Bangladesh), and adds SHAP-based explainability so every prediction
comes with a clear, feature-level reason.

## Data Sources
- [UCI CKD Dataset](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) — 400 patients, 24 features, raw continuous values
- [Risk Factor Prediction of CKD, Enam Medical College](https://archive.ics.uci.edu/dataset/857/risk+factor+prediction+of+chronic+kidney+disease) — 200 patients, pre-binned/discretized features

These two datasets used incompatible representations (raw numbers vs.
pre-made ranges, different encodings for the same yes/no fields). A key part
of this project was reverse-engineering the second dataset's exact binning
scheme and applying it to the first, producing one genuinely unified 599-patient
table — not just a naive concatenation.

## Methodology
1. Data cleaning (missing-value detection, malformed-row handling)
2. Cross-dataset harmonization (see `notebooks/` for the full reasoning)
3. Leakage-safe train/test split → imputation → ordinal encoding
4. Model comparison: Logistic Regression, Random Forest, XGBoost (5-fold
   stratified cross-validation)
5. SHAP explainability (global + per-patient)
6. Streamlit deployment for live predictions

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.983 | 0.992 | 0.981 | 0.987 |
| **Random Forest** | **0.985** | 0.992 | 0.984 | **0.988** |
| XGBoost | 0.978 | 0.982 | 0.984 | 0.983 |

Top predictive features (SHAP): hemoglobin, packed cell volume, albumin —
consistent with prior single-dataset studies, despite this model training on
a combined cross-population dataset.

## How to Run

Install dependencies:
```bash
pip install -r requirements.txt
```

**Explore the full pipeline (data cleaning → training → SHAP):**
```bash
jupyter notebook notebooks/NephroAI_pipeline.ipynb
```

**Run the live prediction app:**
```bash
streamlit run app/streamlit_app.py
```
(Requires the four `.pkl` files in `models/` — already included in this repo.)

## Project Structure
```
NephroAI/
├── data/          raw datasets (UCI + Enam Medical College)
├── notebooks/     full pipeline: cleaning, merging, training, SHAP
├── models/        saved model + preprocessing objects (.pkl)
└── app/           Streamlit prediction app
```

## Limitations
- Continuous precision was sacrificed for both datasets to enable a genuinely
  unified schema (see notebook for full reasoning)
- One numeric feature (`su`) had internally inconsistent binning in the
  source dataset itself — flagged rather than silently corrected
- Research prototype only — not a validated clinical diagnostic tool

## Tech Stack
Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit
