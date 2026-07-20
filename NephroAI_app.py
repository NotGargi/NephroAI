"""
NephroAI - CKD Prediction App
------------------------------
Takes a patient's raw clinical values, runs them through the SAME preprocessing
pipeline used during training (binning, encoding), then predicts CKD risk and
explains the prediction with a SHAP waterfall plot.

Run with: streamlit run streamlit_app.py
Needs the 4 .pkl files (model, imputer, encoder, feature metadata) sitting in
a models/ folder next to this file.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ======================================================================
# Load the trained model + preprocessing pieces (cached so this only
# happens once, not on every single click)
# ======================================================================
@st.cache_resource
def load_pipeline():
    model = joblib.load('Models/ckd_random_forest_model.pkl')
    imputer = joblib.load('Models/ckd_imputer.pkl')
    encoder = joblib.load('Models/ckd_ordinal_encoder.pkl')
    metadata = joblib.load('Models/ckd_feature_metadata.pkl')
    return model, imputer, encoder, metadata

model, imputer, encoder, metadata = load_pipeline()
feature_order = metadata['feature_order']
binary_cols = metadata['binary_cols']
ordinal_cols = metadata['ordinal_cols']


# ======================================================================
# The exact same binning rules we reverse-engineered from Enam's dataset
# during training - reused here so a new patient's raw numbers get sorted
# into the identical categories the model was actually trained on.
# ======================================================================
bin_rules = {
    'bgr':  (112.0, 448.0, 42.0), 'bu':   (48.1, 352.9, 38.1),
    'sod':  (118.0, 158.0, 5.0),  'sc':   (3.65, 28.85, 3.15),
    'pot':  (7.31, 42.59, 4.41),  'hemo': (6.1, 16.5, 1.3),
    'pcv':  (17.9, 49.1, 3.9),    'rbcc': (2.69, 7.41, 0.59),
    'wbcc': (4980.0, 24020.0, 2380.0),
}

def format_number(x):
    x = round(x, 4)
    return str(int(x)) if x == int(x) else str(x)

def sort_into_bin(value, lo, hi, step):
    """Same logic as training - open-ended extremes so any raw value lands somewhere."""
    edges = list(np.arange(lo, hi + step / 2, step))
    edges = [-np.inf] + edges + [np.inf]
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            if edges[i] == -np.inf:
                return f"< {format_number(lo)}"
            elif edges[i + 1] == np.inf:
                return f"≥ {format_number(hi)}"
            else:
                return f"{format_number(edges[i])} - {format_number(edges[i+1])}"
    return f"≥ {format_number(hi)}"   # fallback for anything above the last edge

age_edges = [-np.inf, 12, 20, 27, 35, 43, 51, 59, 66, 74, np.inf]
age_labels = ['< 12', '12 - 20', '20 - 27', '27 - 35', '35 - 43',
              '43 - 51', '51 - 59', '59 - 66', '66 - 74', '≥ 74']

def bin_age(value):
    for i in range(len(age_edges) - 1):
        if age_edges[i] <= value < age_edges[i + 1]:
            return age_labels[i]
    return age_labels[-1]

sg_lookup = {
    1.005: '< 1.007', 1.010: '1.009 - 1.011', 1.015: '1.015 - 1.017',
    1.020: '1.019 - 1.021', 1.025: '≥ 1.023'
}

def group_albumin_sugar_score(value):
    value = int(value)
    if value <= 0: return '< 0'
    elif value == 1: return '1 - 1'
    elif value == 2: return '2 - 2'
    elif value == 3: return '3 - 3'
    else: return '≥ 4'

def bp_is_high(value):
    return 0 if value < 90 else 1

def bp_severity_level(value):
    if value < 90: return 0
    elif value < 100: return 1
    else: return 2


# ======================================================================
# Page setup
# ======================================================================
st.set_page_config(page_title="NephroAI - CKD Risk Prediction", layout="wide")
st.title("NephroAI - Chronic Kidney Disease Risk Prediction")
st.caption("Research prototype trained on a combined UCI + Enam Medical College dataset.")

tab_predict, tab_model_info = st.tabs(["Patient Prediction", "Model Info"])

with tab_predict:
    st.subheader("Enter Patient Values")
    vitals_tab, blood_tab, urine_tab, history_tab = st.tabs(
        ["Vitals", "Blood Test Values", "Urine Test Values", "Medical History"]
    )

    with vitals_tab:
        age_raw = st.number_input("Age (years)", min_value=0, max_value=120, value=45)
        bp_raw = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=180, value=80)

    with blood_tab:
        col1, col2 = st.columns(2)
        with col1:
            bgr_raw = st.number_input("Blood Glucose Random (mg/dL)", min_value=0.0, value=120.0)
            bu_raw = st.number_input("Blood Urea (mg/dL)", min_value=0.0, value=50.0)
            sc_raw = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, value=1.2)
            sod_raw = st.number_input("Sodium (mEq/L)", min_value=0.0, value=138.0)
        with col2:
            pot_raw = st.number_input("Potassium (mEq/L)", min_value=0.0, value=4.5)
            hemo_raw = st.number_input("Hemoglobin (g/dL)", min_value=0.0, value=13.5)
            pcv_raw = st.number_input("Packed Cell Volume (%)", min_value=0.0, value=40.0)
            rbcc_raw = st.number_input("Red Blood Cell Count (millions/cmm)", min_value=0.0, value=5.0)
            wbcc_raw = st.number_input("White Blood Cell Count (cells/cumm)", min_value=0.0, value=8000.0)

    with urine_tab:
        sg_raw = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025], index=2)
        al_raw = st.selectbox("Albumin (0-5)", [0, 1, 2, 3, 4, 5], index=0)
        su_raw = st.selectbox("Sugar (0-5)", [0, 1, 2, 3, 4, 5], index=0)
        rbc_raw = st.radio("Red Blood Cells", ["normal", "abnormal"])
        pc_raw = st.radio("Pus Cell", ["normal", "abnormal"])
        pcc_raw = st.radio("Pus Cell Clumps", ["notpresent", "present"])
        ba_raw = st.radio("Bacteria", ["notpresent", "present"])

    with history_tab:
        col3, col4 = st.columns(2)
        with col3:
            htn_raw = st.radio("Hypertension", ["no", "yes"])
            dm_raw = st.radio("Diabetes Mellitus", ["no", "yes"])
            cad_raw = st.radio("Coronary Artery Disease", ["no", "yes"])
        with col4:
            appet_raw = st.radio("Appetite", ["good", "poor"])
            pe_raw = st.radio("Pedal Edema", ["no", "yes"])
            ane_raw = st.radio("Anemia", ["no", "yes"])

    predict_clicked = st.button("Predict CKD Risk", type="primary")

    if predict_clicked:
        # ---- Build the raw patient row, applying the SAME transforms as training ----
        row = {
            'bp (Diastolic)': bp_is_high(bp_raw),
            'bp limit': bp_severity_level(bp_raw),
            'sg': sg_lookup[sg_raw],
            'al': group_albumin_sugar_score(al_raw),
            'rbc': 1 if rbc_raw == 'abnormal' else 0,
            'su': group_albumin_sugar_score(su_raw),
            'pc': 1 if pc_raw == 'abnormal' else 0,
            'pcc': 1 if pcc_raw == 'present' else 0,
            'ba': 1 if ba_raw == 'present' else 0,
            'bgr': sort_into_bin(bgr_raw, *bin_rules['bgr']),
            'bu': sort_into_bin(bu_raw, *bin_rules['bu']),
            'sod': sort_into_bin(sod_raw, *bin_rules['sod']),
            'sc': sort_into_bin(sc_raw, *bin_rules['sc']),
            'pot': sort_into_bin(pot_raw, *bin_rules['pot']),
            'hemo': sort_into_bin(hemo_raw, *bin_rules['hemo']),
            'pcv': sort_into_bin(pcv_raw, *bin_rules['pcv']),
            'rbcc': sort_into_bin(rbcc_raw, *bin_rules['rbcc']),
            'wbcc': sort_into_bin(wbcc_raw, *bin_rules['wbcc']),
            'htn': 1 if htn_raw == 'yes' else 0,
            'dm': 1 if dm_raw == 'yes' else 0,
            'cad': 1 if cad_raw == 'yes' else 0,
            'appet': 1 if appet_raw == 'poor' else 0,
            'pe': 1 if pe_raw == 'yes' else 0,
            'ane': 1 if ane_raw == 'yes' else 0,
            'age': bin_age(age_raw),
        }

        patient_df = pd.DataFrame([row])[feature_order]

        # ordinal-encode the range/bin columns using the SAME fitted encoder from training
        patient_df[ordinal_cols] = encoder.transform(patient_df[ordinal_cols])
        # binary columns are already 0/1, just make sure they're proper numbers
        for col in binary_cols + ['bp limit']:
            patient_df[col] = patient_df[col].astype(int)

        # ---- Predict ----
        prediction = model.predict(patient_df)[0]
        probability = model.predict_proba(patient_df)[0][1]   # probability of "ckd"

        st.divider()
        if prediction == 1:
            st.error(f"**Prediction: Likely CKD** (model confidence: {probability:.1%})")
        else:
            st.success(f"**Prediction: Likely Not CKD** (model confidence: {1-probability:.1%})")

        # ---- SHAP explanation for this one patient ----
        st.subheader("Why this prediction? (SHAP explanation)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(patient_df)

        fig, ax = plt.subplots()
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0, :, 1],
                base_values=explainer.expected_value[1],
                data=patient_df.iloc[0],
                feature_names=patient_df.columns.tolist()
            ),
            show=False
        )
        st.pyplot(fig)

        st.caption(
            "For research/educational purposes only. Not a diagnostic tool. "
            "Always consult a qualified medical professional."
        )

with tab_model_info:
    st.subheader("Model Performance (5-fold cross-validation)")
    st.table(pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Score': [0.9849, 0.9921, 0.9840, 0.9879]
    }))
    st.markdown(
        "**Model**: Random Forest (200 trees)  \n"
        "**Training data**: 599 patients, combined from the UCI CKD dataset (399) "
        "and the Enam Medical College Risk Factor dataset (200)  \n"
        "**Note**: both datasets were harmonized into a single shared binned/ordinal "
        "feature representation before training - see project README for full methodology."
    )
