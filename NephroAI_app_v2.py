"""
NephroAI - CKD Prediction Portal
------------------------------
Takes a patient's raw clinical values, runs them through the SAME preprocessing
pipeline used during training (binning, encoding), then predicts CKD risk and
explains the prediction with an interactive SHAP-based breakdown.

Run with: streamlit run NephroAI_app_v2.py
Needs the 4 .pkl files (model, imputer, encoder, feature metadata) sitting in
a Models/ folder next to this file, and assets/kidney_banner.png for the side panel.
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go
import base64
from pathlib import Path

# ======================================================================
# Page setup - must be the first Streamlit call
# ======================================================================
st.set_page_config(
    page_title="NephroAI | CKD Risk Portal",
    page_icon="\U0001FA78",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# Marine Blue theme
# ======================================================================
MARINE = {
    "mist":   "#CFDBDC",  # lightest - backgrounds / subtle fills
    "seafoam":"#F3B79B",  # light accent (was seafoam, now light orange)
    "teal":   "#E8674F",  # secondary accent (was teal, now orange)
    "ocean":  "#367CA3",  # primary accent / interactive
    "abyss":  "#113147",  # darkest - headers / text / high-risk
}

CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600&family=Inter:wght@400;500;600&family=Anton&display=swap');

    html, body, [class*="css"]  {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background: linear-gradient(180deg, #F7FAFB 0%, #FFFFFF 100%);
    }}

    /* ---- Hero header ---- */
    .hero-title {{
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 2.4rem;
        color: {MARINE['abyss']};
        margin-bottom: 0.15rem;
        letter-spacing: -0.01em;
    }}
    .hero-sub {{
        color: #4169E1;
        font-size: 1.02rem;
        font-weight: 500;
        margin-bottom: 1.4rem;
    }}

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {MARINE['abyss']} 0%, #0B2436 100%);
    }}
    section[data-testid="stSidebar"] * {{
        color: #EAF3F5 !important;
    }}
    section[data-testid="stSidebar"] .block-container,
    section[data-testid="stSidebarUserContent"],
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
        padding-top: 0rem !important;
    }}
    .sidebar-brand-row {{
        display: flex;
        align-items: center;
        gap: 4px;
        margin-bottom: 20px;
        margin-top: -36px;
        position: relative;
        z-index: 2;
    }}
    .sidebar-logo {{
        flex-shrink: 0;
    }}
    .sidebar-brand {{
        font-family: 'Fraunces', serif;
        font-weight: 800;
        font-size: 2rem;
        color: #FFFFFF !important;
        letter-spacing: -0.02em;
        line-height: 1;
    }}
    section[data-testid="stSidebar"] .sidebar-tagline {{
        font-family: 'Anton', sans-serif;
        font-style: normal;
        font-weight: 400;
        font-size: 1.35rem;
        letter-spacing: 0.01em;
        color: #F1AD94 !important;
        margin: 4px 0 14px 0;
        white-space: nowrap;
    }}
    .sidebar-tag {{
        display: inline-block;
        background: rgba(146,204,213,0.15);
        border: 1px solid rgba(166,208,211,0.35);
        color: {MARINE['teal']} !important;
        border-radius: 999px;
        padding: 3px 12px;
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }}
    .sidebar-divider {{
        border: none;
        border-top: 1px solid rgba(166,208,211,0.25);
        margin: 6px 0;
    }}
    section[data-testid="stSidebar"] .sidebar-fact {{
        font-size: 1rem;
        line-height: 1.5;
        margin: 0 0 4px 0;
        color: {MARINE['mist']} !important;
    }}

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        border-bottom: 1px solid {MARINE['mist']};
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 42px;
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        color: #5C7A87;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: transparent !important;
        color: {MARINE['abyss']} !important;
        font-weight: 600;
    }}

    /* ---- Wizard step indicator (replaces st.tabs for the assessment form) ---- */
    .wizard-tabs {{
        display: flex;
        gap: 24px;
        font-family: 'Inter', sans-serif;
    }}
    .wizard-tab {{
        font-size: 1rem;
        color: #5C7A87;
        padding-bottom: 10px;
    }}
    .wizard-tab.active {{
        color: {MARINE['abyss']};
        font-weight: 600;
        border-bottom: 2px solid #E8674F;
    }}
    hr.wizard-divider {{
        margin-top: -1px;
        border: none;
        border-top: 1px solid {MARINE['mist']};
    }}

    /* ---- Section labels ---- */
    .section-label {{
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #4169E1;
        margin: 4px 0 10px 0;
    }}

    /* ---- Submit button ---- */
    div.stFormSubmitButton > button {{
        background: #407AF7;
        color: #FFFFFF;
        font-weight: 600;
        font-size: 1.1rem;
        border: none;
        border-radius: 999px;
        padding: 0.9rem 2.5rem;
        width: auto;
        box-shadow: 0 6px 16px rgba(64,122,247,0.30);
        transition: transform 0.15s ease;
    }}
    div.stFormSubmitButton > button:focus:not(:active) {{
        background: #407AF7 !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 6px 16px rgba(64,122,247,0.30);
    }}
    div.stFormSubmitButton > button p {{
        color: inherit !important;
    }}
    
    /* Generic buttons except Back */
    div.stFormSubmitButton:not(.st-key-back_btn_mid):not(.st-key-back_btn_last) > button:hover {{
    background: #2F5FD1 !important;
    color: #FFFFFF !important;
    border: none !important;
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(64,122,247,0.4);
    }}

    /* ---- Result cards ---- */
    .result-card {{
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
    }}
    .risk-high {{
        background: linear-gradient(135deg, #16394F 0%, {MARINE['abyss']} 100%);
        color: #F2F8F9;
    }}
    .risk-mod {{
        background: linear-gradient(135deg, {MARINE['ocean']} 0%, #2C6588 100%);
        color: #F2F8F9;
    }}
    .risk-low {{
        background: linear-gradient(135deg, {MARINE['seafoam']} 0%, {MARINE['teal']} 100%);
        color: {MARINE['abyss']};
    }}
    .risk-label {{
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.85;
        margin-bottom: 2px;
    }}
    .risk-value {{
        font-family: 'Inter', sans-serif;
        font-variant-numeric: tabular-nums;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        line-height: 1.1;
    }}

    .factor-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 14px;
        border-radius: 10px;
        background: #F7FAFB;
        border: 1px solid {MARINE['mist']};
        margin-bottom: 8px;
        font-size: 0.92rem;
    }}
    .factor-tag {{
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 999px;
        letter-spacing: 0.03em;
    }}
    .tag-up {{ background: {MARINE['abyss']}; color: white; }}
    .tag-down {{ background: {MARINE['teal']}; color: {MARINE['abyss']}; }}

    .stCaption, .stMarkdown p {{
        color: #45606B;
    }}

    /* ---- Force all input labels & radio option text to be visible on the light background ----
       Wrapped in :where() so this carries ZERO specificity - it still beats Streamlit's own
       unstyled defaults (thanks to !important), but can never outrank the sidebar's own text
       color rule below, which needs to stay in charge inside the sidebar. */
    :where(
        .stApp [data-testid="stWidgetLabel"] p,
        .stApp [data-testid="stWidgetLabel"] label,
        .stApp [data-testid="stWidgetLabel"] div,
        .stApp label,
        .stApp .stRadio p,
        .stApp .stSelectbox p,
        .stApp .stNumberInput p,
        .stApp [data-testid="stMarkdownContainer"] p
    ) {{
        color: {MARINE['abyss']} !important;
        opacity: 1 !important;
        font-weight: 500;
    }}
    :where(.stApp .stRadio [role="radiogroup"] label) {{
        color: {MARINE['abyss']} !important;
    }}
    /* Sidebar text always wins over the rule above, regardless of source order */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] span {{
        color: #EAF3F5 !important;
        opacity: 1 !important;
    }}
    html {{
        color-scheme: light;
    }}
    /* Hide the fullscreen-expand icon on the sidebar banner image (covers multiple Streamlit versions) */
    [data-testid="StyledFullScreenButton"],
    button[title="View fullscreen"],
    section[data-testid="stSidebar"] [data-testid="stImage"] button,
    section[data-testid="stSidebar"] [data-testid="stElementToolbar"],
    section[data-testid="stSidebar"] div[data-testid="stImage"] > div > button {{
        display: none !important;
        visibility: hidden !important;
    }}
    /* ---- Model Info / Back to Assessment toggle button ---- */
    .st-key-model_info_btn button {{
        background: linear-gradient(135deg, #F0705A 0%, #E8674F 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(232,103,79,0.30);
        transition: transform 0.15s ease;
    }}
    .st-key-model_info_btn button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(232,103,79,0.4);
        color: #FFFFFF !important;
    }}
    .st-key-model_info_btn button p {{
        color: #FFFFFF !important;
    }}

             /* ---- Back button (Vitals/Blood/Urine/History wizard) ---- */
    .stElementContainer.st-key-back_btn_mid .stFormSubmitButton > button,
    .stElementContainer.st-key-back_btn_last .stFormSubmitButton > button {{
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        background-image: none !important;

        color: #000000 !important;
        border: 1px solid #000000 !important;
        border-radius: 999px !important;
        font-weight: 600 !important;

        box-shadow: none !important;
        outline: none !important;
        transition: all 0.15s ease !important;
    }}

    .stElementContainer.st-key-back_btn_mid .stFormSubmitButton > button:hover,
    .stElementContainer.st-key-back_btn_last .stFormSubmitButton > button:hover,
    .stElementContainer.st-key-back_btn_mid .stFormSubmitButton > button:focus,
    .stElementContainer.st-key-back_btn_last .stFormSubmitButton > button:focus,
    .stElementContainer.st-key-back_btn_mid .stFormSubmitButton > button:active,
    .stElementContainer.st-key-back_btn_last .stFormSubmitButton > button:active {{
        background: #ECECEC !important;
        background-color: #ECECEC !important;
        background-image: none !important;

        color: #000000 !important;
        border: 1px solid #000000 !important;

        box-shadow: none !important;
        outline: none !important;
        transform: translateY(-1px) !important;
    }}

    .stElementContainer.st-key-back_btn_mid .stFormSubmitButton > button *,
    .stElementContainer.st-key-back_btn_last .stFormSubmitButton > button * {{
        color: #000000 !important;
        fill: #000000 !important;
    }}

    /* ---- Pull Next / Assess CKD Risk right up against Back ---- */
    .st-key-next_btn_wrap_mid,
    .st-key-next_btn_wrap_last {{
        margin-left: -1.6rem !important;
    }}
    
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ======================================================================
# FIX: initialize every wizard-widget key in session_state up front, with
# the same defaults each widget already used. This guarantees the keys
# exist no matter which wizard step is currently rendered, so the final
# "Assess CKD Risk" step can never hit a KeyError when it reads values
# from steps that aren't the currently-rendered one.
# ======================================================================
DEFAULTS = {
    "age_input": 45, "bp_input": 80,
    "bgr_input": 120.0, "bu_input": 50.0, "sc_input": 1.2, "sod_input": 138.0,
    "pot_input": 4.5, "hemo_input": 13.5, "pcv_input": 40.0,
    "rbcc_input": 5.0, "wbcc_input": 8000.0,
    "sg_input": 1.015, "al_input": 0, "su_input": 0,
    "rbc_input": "normal", "pc_input": "normal",
    "pcc_input": "notpresent", "ba_input": "notpresent",
    "htn_input": "no", "dm_input": "no", "cad_input": "no",
    "appet_input": "good", "pe_input": "no", "ane_input": "no",
}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

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
                return f"\u2265 {format_number(hi)}"
            else:
                return f"{format_number(edges[i])} - {format_number(edges[i+1])}"
    return f"\u2265 {format_number(hi)}"

age_edges = [-np.inf, 12, 20, 27, 35, 43, 51, 59, 66, 74, np.inf]
age_labels = ['< 12', '12 - 20', '20 - 27', '27 - 35', '35 - 43',
              '43 - 51', '51 - 59', '59 - 66', '66 - 74', '\u2265 74']

def bin_age(value):
    for i in range(len(age_edges) - 1):
        if age_edges[i] <= value < age_edges[i + 1]:
            return age_labels[i]
    return age_labels[-1]

sg_lookup = {
    1.005: '< 1.007', 1.010: '1.009 - 1.011', 1.015: '1.015 - 1.017',
    1.020: '1.019 - 1.021', 1.025: '\u2265 1.023'
}

def group_albumin_sugar_score(value):
    value = int(value)
    if value <= 0: return '< 0'
    elif value == 1: return '1 - 1'
    elif value == 2: return '2 - 2'
    elif value == 3: return '3 - 3'
    else: return '\u2265 4'

def bp_is_high(value):
    return 0 if value < 90 else 1

def bp_severity_level(value):
    if value < 90: return 0
    elif value < 100: return 1
    else: return 2

# Human-friendly labels for every feature, used in the results panel
FRIENDLY_NAMES = {
    'age': 'Age', 'bp (Diastolic)': 'Diastolic Blood Pressure', 'bp limit': 'BP Severity',
    'sg': 'Urine Specific Gravity', 'al': 'Albumin Level', 'su': 'Sugar Level',
    'rbc': 'Red Blood Cells (urine)', 'pc': 'Pus Cells (urine)', 'pcc': 'Pus Cell Clumps',
    'ba': 'Bacteria (urine)', 'bgr': 'Blood Glucose (Random)', 'bu': 'Blood Urea',
    'sod': 'Sodium', 'sc': 'Serum Creatinine', 'pot': 'Potassium', 'hemo': 'Hemoglobin',
    'pcv': 'Packed Cell Volume', 'rbcc': 'Red Blood Cell Count', 'wbcc': 'White Blood Cell Count',
    'htn': 'Hypertension', 'dm': 'Diabetes Mellitus', 'cad': 'Coronary Artery Disease',
    'appet': 'Appetite', 'pe': 'Pedal Edema', 'ane': 'Anemia',
}

# ======================================================================
# Sidebar - branding panel
# ======================================================================
with st.sidebar:
    logo_path = Path(__file__).parent / "assets" / "nephro_logo.png"
    logo_b64 = ""
    if logo_path.exists():
        logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()

    logo_html = (
        f'<img src="data:image/png;base64,{logo_b64}" style="width:52px;height:52px;object-fit:contain;">'
        if logo_b64 else ""
    )

    st.markdown(
        f'''
        <div class="sidebar-brand-row">
            <div class="sidebar-logo">{logo_html}</div>
            <div class="sidebar-brand">NephroAI</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    banner_path = Path(__file__).parent / "assets" / "kidney_banner.png"
    if banner_path.exists():
        st.image(str(banner_path), use_container_width=True)

    st.markdown(
        '<p class="sidebar-fact" style="margin-top:12px;">A clinical decision-support '
        'portal that estimates chronic kidney disease risk from routine blood and urine '
        'values, and explains the reasoning behind every prediction.</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown(
        '<p class="sidebar-fact"><b>Model:</b> Random Forest (200 trees)<br>'
        '<b>Trained on:</b> 599 patients<br>'
        '<b>Sources:</b> UCI CKD + Enam Medical College</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown(
        '<p class="sidebar-fact"><b>Disclaimer:</b> For research and educational '
        'use only. Not a diagnostic device. Always confirm with a qualified '
        'nephrologist.</p>',
        unsafe_allow_html=True,
    )

# ======================================================================
# Header - title + a smart top-right button that toggles Model Info
# ======================================================================
if "show_model_info" not in st.session_state:
    st.session_state.show_model_info = False

header_col, button_col = st.columns([5, 1])
with header_col:
    st.markdown('<div class="hero-title">Chronic Kidney Disease Risk Portal</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Enter a patient\'s clinical values below to generate a risk '
        'assessment with a full explanation of the contributing factors.</div>',
        unsafe_allow_html=True,
    )
with button_col:
    st.write("")
    btn_label = "\U00002190 Back to Assessment" if st.session_state.show_model_info else "Model Info"
    if st.button(btn_label, use_container_width=True, key="model_info_btn"):
        st.session_state.show_model_info = not st.session_state.show_model_info
        st.rerun()

# ======================================================================
# Patient Assessment - step-by-step wizard (Vitals -> Blood -> Urine -> History)
# "Next" advances through steps; only the final step's button runs the assessment.
# ======================================================================
if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 0

STEP_NAMES = ["Vitals", "Blood Test Values", "Urine Test Values", "Medical History"]

if not st.session_state.show_model_info:

    # ---- Visual step indicator (replaces the old clickable st.tabs) ----
    tabs_html = "".join(
        f'<span class="wizard-tab{" active" if i == st.session_state.wizard_step else ""}">{name}</span>'
        for i, name in enumerate(STEP_NAMES)
    )
    st.markdown(
        f'<div class="wizard-tabs">{tabs_html}</div><hr class="wizard-divider">',
        unsafe_allow_html=True,
    )

    with st.form("patient_form"):
        step = st.session_state.wizard_step

        if step == 0:
            st.markdown('<div class="section-label">Basic Vitals</div>', unsafe_allow_html=True)
            st.number_input("Age (years)", min_value=0, max_value=120,
                             value=st.session_state.get("age_input", 45), key="age_input")
            st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=180,
                             value=st.session_state.get("bp_input", 80), key="bp_input")

        elif step == 1:
            st.markdown('<div class="section-label">Blood Test Values</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.number_input("Blood Glucose Random (mg/dL)", min_value=0.0,
                                 value=st.session_state.get("bgr_input", 120.0), key="bgr_input")
                st.number_input("Blood Urea (mg/dL)", min_value=0.0,
                                 value=st.session_state.get("bu_input", 50.0), key="bu_input")
                st.number_input("Serum Creatinine (mg/dL)", min_value=0.0,
                                 value=st.session_state.get("sc_input", 1.2), key="sc_input")
                st.number_input("Sodium (mEq/L)", min_value=0.0,
                                 value=st.session_state.get("sod_input", 138.0), key="sod_input")
            with col2:
                st.number_input("Potassium (mEq/L)", min_value=0.0,
                                 value=st.session_state.get("pot_input", 4.5), key="pot_input")
                st.number_input("Hemoglobin (g/dL)", min_value=0.0,
                                 value=st.session_state.get("hemo_input", 13.5), key="hemo_input")
                st.number_input("Packed Cell Volume (%)", min_value=0.0,
                                 value=st.session_state.get("pcv_input", 40.0), key="pcv_input")
                st.number_input("Red Blood Cell Count (millions/cmm)", min_value=0.0,
                                 value=st.session_state.get("rbcc_input", 5.0), key="rbcc_input")
                st.number_input("White Blood Cell Count (cells/cumm)", min_value=0.0,
                                 value=st.session_state.get("wbcc_input", 8000.0), key="wbcc_input")

        elif step == 2:
            st.markdown('<div class="section-label">Urine Test Values</div>', unsafe_allow_html=True)
            sg_options = [1.005, 1.010, 1.015, 1.020, 1.025]
            st.selectbox("Specific Gravity", sg_options,
                         index=sg_options.index(st.session_state.get("sg_input", 1.015)), key="sg_input")
            su_al_options = [0, 1, 2, 3, 4, 5]
            st.selectbox("Albumin (0-5)", su_al_options,
                         index=su_al_options.index(st.session_state.get("al_input", 0)), key="al_input")
            st.selectbox("Sugar (0-5)", su_al_options,
                         index=su_al_options.index(st.session_state.get("su_input", 0)), key="su_input")
            st.radio("Red Blood Cells", ["normal", "abnormal"], horizontal=True, key="rbc_input")
            st.radio("Pus Cell", ["normal", "abnormal"], horizontal=True, key="pc_input")
            st.radio("Pus Cell Clumps", ["notpresent", "present"], horizontal=True, key="pcc_input")
            st.radio("Bacteria", ["notpresent", "present"], horizontal=True, key="ba_input")

        else:  # step == 3, last step
            st.markdown('<div class="section-label">Medical History</div>', unsafe_allow_html=True)
            col3, col4 = st.columns(2)
            with col3:
                st.radio("Hypertension", ["no", "yes"], horizontal=True, key="htn_input")
                st.radio("Diabetes Mellitus", ["no", "yes"], horizontal=True, key="dm_input")
                st.radio("Coronary Artery Disease", ["no", "yes"], horizontal=True, key="cad_input")
            with col4:
                st.radio("Appetite", ["good", "poor"], horizontal=True, key="appet_input")
                st.radio("Pedal Edema", ["no", "yes"], horizontal=True, key="pe_input")
                st.radio("Anemia", ["no", "yes"], horizontal=True, key="ane_input")

        st.write("")
        if step == 0:
            next_clicked = st.form_submit_button("Next")
            back_clicked = False
            predict_clicked = False
        elif step < 3:
            back_col, next_col, _spacer = st.columns([1, 1, 4], gap="small")
            with back_col:
                back_clicked = st.form_submit_button("\U00002190 Back", key="back_btn_mid")
            with next_col:
                with st.container(key="next_btn_wrap_mid"):
                    next_clicked = st.form_submit_button("Next")
            predict_clicked = False
        else:
            back_col, next_col, _spacer = st.columns([1, 1.6, 3], gap="small")
            with back_col:
                back_clicked = st.form_submit_button("\U00002190 Back", key="back_btn_last")
            with next_col:
                with st.container(key="next_btn_wrap_last"):
                    predict_clicked = st.form_submit_button("Assess CKD Risk")
            next_clicked = False

    if back_clicked:
        st.session_state.wizard_step -= 1
        st.rerun()

    if next_clicked:
        st.session_state.wizard_step += 1
        st.rerun()

    if predict_clicked:
        # ---- Build the raw patient row, applying the SAME transforms as training ----
        # Values are pulled from session_state (via each widget's key) so answers from
        # earlier steps are still available once we've moved on to later steps.
        # Every key is guaranteed to exist thanks to the DEFAULTS block near the top
        # of this file, so this can never KeyError even if a step was skipped.
        ss = st.session_state
        age_v   = ss["age_input"]
        bp_v    = ss["bp_input"]
        bgr_v   = ss["bgr_input"]
        bu_v    = ss["bu_input"]
        sc_v    = ss["sc_input"]
        sod_v   = ss["sod_input"]
        pot_v   = ss["pot_input"]
        hemo_v  = ss["hemo_input"]
        pcv_v   = ss["pcv_input"]
        rbcc_v  = ss["rbcc_input"]
        wbcc_v  = ss["wbcc_input"]
        sg_v    = ss["sg_input"]
        al_v    = ss["al_input"]
        su_v    = ss["su_input"]
        rbc_v   = ss["rbc_input"]
        pc_v    = ss["pc_input"]
        pcc_v   = ss["pcc_input"]
        ba_v    = ss["ba_input"]
        htn_v   = ss["htn_input"]
        dm_v    = ss["dm_input"]
        cad_v   = ss["cad_input"]
        appet_v = ss["appet_input"]
        pe_v    = ss["pe_input"]
        ane_v   = ss["ane_input"]

        row = {
            'bp (Diastolic)': bp_is_high(bp_v),
            'bp limit': bp_severity_level(bp_v),
            'sg': sg_lookup[sg_v],
            'al': group_albumin_sugar_score(al_v),
            'rbc': 1 if rbc_v == 'abnormal' else 0,
            'su': group_albumin_sugar_score(su_v),
            'pc': 1 if pc_v == 'abnormal' else 0,
            'pcc': 1 if pcc_v == 'present' else 0,
            'ba': 1 if ba_v == 'present' else 0,
            'bgr': sort_into_bin(bgr_v, *bin_rules['bgr']),
            'bu': sort_into_bin(bu_v, *bin_rules['bu']),
            'sod': sort_into_bin(sod_v, *bin_rules['sod']),
            'sc': sort_into_bin(sc_v, *bin_rules['sc']),
            'pot': sort_into_bin(pot_v, *bin_rules['pot']),
            'hemo': sort_into_bin(hemo_v, *bin_rules['hemo']),
            'pcv': sort_into_bin(pcv_v, *bin_rules['pcv']),
            'rbcc': sort_into_bin(rbcc_v, *bin_rules['rbcc']),
            'wbcc': sort_into_bin(wbcc_v, *bin_rules['wbcc']),
            'htn': 1 if htn_v == 'yes' else 0,
            'dm': 1 if dm_v == 'yes' else 0,
            'cad': 1 if cad_v == 'yes' else 0,
            'appet': 1 if appet_v == 'poor' else 0,
            'pe': 1 if pe_v == 'yes' else 0,
            'ane': 1 if ane_v == 'yes' else 0,
            'age': bin_age(age_v),
        }
        patient_df = pd.DataFrame([row])[feature_order]
        patient_df[ordinal_cols] = encoder.transform(patient_df[ordinal_cols])
        for col in binary_cols + ['bp limit']:
            patient_df[col] = patient_df[col].astype(int)

        # ---- Predict ----
        prediction = model.predict(patient_df)[0]
        probability = model.predict_proba(patient_df)[0][1]  # probability of "ckd"

        if probability >= 0.66:
            risk_tier, risk_class = "High Risk", "risk-high"
        elif probability >= 0.33:
            risk_tier, risk_class = "Moderate Risk", "risk-mod"
        else:
            risk_tier, risk_class = "Low Risk", "risk-low"

        st.divider()
        st.markdown('<div class="section-label">Assessment Result</div>', unsafe_allow_html=True)

        result_col, gauge_col = st.columns([1, 1.2])

        with result_col:
            st.markdown(
                f'<div class="result-card {risk_class}">'
                f'<div class="risk-label">Predicted Outcome</div>'
                f'<div class="risk-value">{"Likely CKD" if prediction == 1 else "Likely Not CKD"}</div>'
                f'<div style="margin-top:10px; font-size:0.95rem;">Model confidence: '
                f'<b>{probability:.1%}</b> probability of CKD &middot; classified as '
                f'<b>{risk_tier}</b></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "This is a statistical estimate from a machine learning model, not a "
                "medical diagnosis. Always confirm with a qualified nephrologist."
            )

        with gauge_col:
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={'suffix': "%", 'font': {'color': MARINE['abyss'], 'size': 40}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': MARINE['abyss']},
                    'bar': {'color': MARINE['abyss']},
                    'bgcolor': "white",
                    'steps': [
                        {'range': [0, 33], 'color': MARINE['seafoam']},
                        {'range': [33, 66], 'color': MARINE['teal']},
                        {'range': [66, 100], 'color': MARINE['ocean']},
                    ],
                },
                title={'text': "CKD Probability", 'font': {'color': MARINE['ocean'], 'size': 16}},
            ))
            gauge_fig.update_layout(
                height=230, margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", font={'family': "Inter"},
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

        # ---- SHAP explanation for this one patient ----
        st.markdown('<div class="section-label" style="margin-top:1.2rem;">Why This Prediction?</div>', unsafe_allow_html=True)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(patient_df)
        patient_shap = shap_values[0, :, 1]

        contrib_df = pd.DataFrame({
            'feature': patient_df.columns,
            'friendly': [FRIENDLY_NAMES.get(c, c) for c in patient_df.columns],
            'value': patient_df.iloc[0].values,
            'shap': patient_shap,
        }).sort_values('shap', key=abs, ascending=False)

        top_factors = contrib_df.head(5)

        text_col, chart_col = st.columns([1, 1.3])

        with text_col:
            st.markdown("**Top contributing factors**")
            for _, r in top_factors.iterrows():
                direction = "Raises risk" if r['shap'] > 0 else "Lowers risk"
                tag_class = "tag-up" if r['shap'] > 0 else "tag-down"
                st.markdown(
                    f'<div class="factor-row"><span>{r["friendly"]}</span>'
                    f'<span class="factor-tag {tag_class}">{direction}</span></div>',
                    unsafe_allow_html=True,
                )

        with chart_col:
            plot_df = contrib_df.reindex(contrib_df['shap'].abs().sort_values(ascending=False).index).head(10)
            plot_df = plot_df.sort_values('shap')
            colors = [MARINE['abyss'] if v > 0 else MARINE['teal'] for v in plot_df['shap']]

            bar_fig = go.Figure(go.Bar(
                x=plot_df['shap'],
                y=plot_df['friendly'],
                orientation='h',
                marker_color=colors,
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            ))
            bar_fig.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Impact on CKD prediction",
                font={'family': "Inter", 'color': MARINE['abyss']},
                xaxis=dict(zeroline=True, zerolinecolor=MARINE['mist'], gridcolor=MARINE['mist']),
            )
            st.plotly_chart(bar_fig, use_container_width=True)
            st.caption(
                f"Dark navy bars push toward CKD, orange bars push away from it. "
                f"Baseline model expectation: {explainer.expected_value[1]:.1%}."
            )

        st.caption(
            "For research/educational purposes only. Not a diagnostic tool. "
            "Always consult a qualified medical professional."
        )

# ======================================================================
# Model Info panel
# ======================================================================
else:
    st.markdown(f'<div class="section-label" style="color:{MARINE["abyss"]};">5-Fold Cross-Validation Performance</div>', unsafe_allow_html=True)

    # ---- STEP 1 CHANGE: circular metric cards with a counting-up animation ----
    metrics = {'Accuracy': 0.9849, 'Precision': 0.9921, 'Recall': 0.9840, 'F1 Score': 0.9879}

    circles_html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;700&display=swap');
        html, body {{
            margin: 0;
            padding: 0;
            overflow: visible;
        }}
        .metric-row {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 16px;
            font-family: 'Inter', sans-serif;
            padding: 10px 4px 0 4px;
        }}
        .metric-circle {{
            flex: 1 1 0;
            width: 100%;
            max-width: 210px;
            aspect-ratio: 1 / 1;
            border-radius: 50%;
            background: linear-gradient(135deg, #5B8CF9 0%, #407AF7 100%);
            border: none;
            color: #FFFFFF;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            box-shadow: 0 6px 16px rgba(17,49,71,0.12);
            box-sizing: border-box;
            margin: 0 auto;
        }}
        .metric-label {{
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            opacity: 0.9;
            margin-bottom: 8px;
        }}
        .metric-value {{
            font-size: 2.6rem;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
        }}
    </style>

    <div class="metric-row">
        {"".join(
            f'<div class="metric-circle"><div class="metric-label">{label}</div>'
            f'<div class="metric-value" data-target="{value*100:.2f}">0%</div></div>'
            for label, value in metrics.items()
        )}
    </div>

    <script>
        const els = document.querySelectorAll('.metric-value');
        els.forEach(el => {{
            const target = parseFloat(el.getAttribute('data-target'));
            const duration = 1200;
            const start = performance.now();
            function tick(now) {{
                const progress = Math.min((now - start) / duration, 1);
                const eased = 1 - Math.pow(1 - progress, 3);
                const current = (eased * target).toFixed(2);
                el.textContent = current + '%';
                if (progress < 1) requestAnimationFrame(tick);
                else el.textContent = target.toFixed(2) + '%';
            }}
            requestAnimationFrame(tick);
        }});
    </script>
    """

    components.html(circles_html, height=240)
    # ---- END STEP 1 CHANGE ----

    st.markdown('<div class="section-label" style="margin-top:1rem;">Methodology</div>', unsafe_allow_html=True)
    st.markdown(
        "**Model**: Random Forest (200 trees)  \n"
        "**Training data**: 599 patients, combined from the UCI CKD dataset (399) "
        "and the Enam Medical College Risk Factor dataset (200)  \n"
        "**Note**: both datasets were harmonized into a single shared binned/ordinal "
        "feature representation before training - see project README for full methodology."
    )
