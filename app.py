import io
import json
import joblib
import pandas as pd
import streamlit as st

from utils import clean_telco, segment_customer

st.set_page_config(
    page_title="Customer Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #0a0c10 !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

section[data-testid="stSidebar"] { display: none !important; }

[data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

.app-shell {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem 2rem 4rem;
}

/* ── Header ── */
.app-header {
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 1.5rem; margin-bottom: 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.header-brand { display: flex; align-items: center; gap: 0.75rem; }
.brand-icon {
    width: 42px; height: 42px; border-radius: 10px;
    background: linear-gradient(135deg, #4f46e5, #2563eb);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; box-shadow: 0 0 24px rgba(79,70,229,0.25);
}
.brand-name { font-size: 1.1rem; font-weight: 600; color: #f1f5f9; letter-spacing: -0.01em; }
.brand-sub {
    font-size: 0.65rem; color: #475569; text-transform: uppercase;
    letter-spacing: 0.1em; font-weight: 500; margin-top: 1px;
}
.header-meta { display: flex; align-items: center; gap: 1rem; font-size: 0.8rem; }
.status-dot { display: inline-flex; align-items: center; gap: 6px; color: #64748b; }
.dot-green { width: 7px; height: 7px; border-radius: 50%; background: #10b981; box-shadow: 0 0 6px #10b981; }
.divider-v { width: 1px; height: 16px; background: rgba(255,255,255,0.08); }
.sys-id { font-family: monospace; font-size: 0.7rem; color: #475569; }

/* ── Tabs ── */
[data-testid="stTabs"] { margin-top: 0; }
[data-testid="stTabs"] > div:first-child {
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
    gap: 0 !important;
    padding: 0 !important;
    margin-bottom: 1.5rem !important;
}
button[data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: #475569 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 0.75rem 1.25rem !important;
    margin: 0 !important;
    letter-spacing: 0.01em !important;
    border-radius: 0 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #f1f5f9 !important;
    border-bottom: 2px solid #3b82f6 !important;
}
button[data-baseweb="tab"]:hover { color: #94a3b8 !important; }
[data-testid="stTabPanel"] { padding: 0 !important; }

/* ── Form card (via :has selector) ── */
[data-testid="stForm"] { background: transparent !important; border: none !important; padding: 0 !important; }
div[data-testid="stVerticalBlock"]:has([data-testid="stForm"]) {
    background: #11141a !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-top: 2px solid rgba(59,130,246,0.5) !important;
    border-radius: 14px !important;
    padding: 1rem 1.25rem 1.5rem !important;
    box-shadow: 0 4px 32px rgba(0,0,0,0.35) !important;
}

/* ── Widget labels ── */
.stNumberInput label, .stSelectbox label, div[data-testid="stWidgetLabel"] > p {
    font-size: 0.72rem !important; font-weight: 500 !important;
    color: #64748b !important; text-transform: uppercase !important;
    letter-spacing: 0.06em !important; margin-bottom: 4px !important;
}

/* ── Inputs ── */
.stNumberInput input, [data-baseweb="input"] input {
    background-color: #0a0c10 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important; color: #f1f5f9 !important;
    font-family: 'SF Mono', 'Fira Code', monospace !important;
    font-size: 0.875rem !important;
}
.stNumberInput input:focus, [data-baseweb="input"] input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}

/* ── Selects ── */
[data-baseweb="select"] > div:first-child {
    background-color: #0a0c10 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important; color: #f1f5f9 !important;
}
[data-baseweb="select"] svg { color: #475569 !important; }
[data-baseweb="menu"] { background-color: #1a1f2e !important; border: 1px solid rgba(255,255,255,0.1) !important; }
[data-baseweb="menu"] li { color: #cbd5e1 !important; }
[data-baseweb="menu"] li:hover { background-color: rgba(59,130,246,0.15) !important; }

/* ── Buttons ── */
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #2563eb, #4f46e5) !important;
    color: #fff !important; border: none !important; border-radius: 9px !important;
    font-weight: 600 !important; font-size: 0.875rem !important;
    padding: 0.65rem 1.5rem !important; letter-spacing: 0.02em !important;
    transition: all 0.2s !important; width: 100% !important; margin-top: 0.5rem !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    box-shadow: 0 0 24px rgba(79,70,229,0.35) !important; transform: translateY(-1px) !important;
}
[data-testid="stFormSubmitButton"] > button:active { transform: translateY(0) !important; }

[data-testid="stButton"] > button {
    background: rgba(255,255,255,0.04) !important; color: #64748b !important;
    border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 7px !important;
    font-weight: 500 !important; font-size: 0.75rem !important;
    padding: 0.35rem 0.85rem !important; transition: all 0.15s !important;
}
[data-testid="stButton"] > button:hover {
    background: rgba(255,255,255,0.08) !important; color: #94a3b8 !important;
    border-color: rgba(255,255,255,0.15) !important;
}

/* ── Download button override ── */
[data-testid="stDownloadButton"] > button {
    background: rgba(59,130,246,0.08) !important; color: #60a5fa !important;
    border: 1px solid rgba(59,130,246,0.25) !important; border-radius: 8px !important;
    font-weight: 500 !important; font-size: 0.8rem !important;
    padding: 0.5rem 1.1rem !important; transition: all 0.15s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(59,130,246,0.15) !important; border-color: rgba(59,130,246,0.4) !important;
}

/* ── File uploader (hidden — not used) ── */

/* ── Results: single customer ── */
.results-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }
.metric-card {
    background: #11141a; border: 1px solid; border-radius: 14px;
    padding: 1.25rem; position: relative; overflow: hidden;
}
.metric-glow {
    position: absolute; top: -20px; right: -20px;
    width: 80px; height: 80px; border-radius: 50%;
    filter: blur(24px); pointer-events: none;
}
.metric-header {
    display: flex; align-items: center;
    justify-content: space-between; margin-bottom: 1rem;
    position: relative; z-index: 1;
}
.metric-title { display: flex; align-items: center; gap: 6px; font-size: 0.78rem; font-weight: 500; color: #94a3b8; }
.metric-icon { font-size: 0.9rem; }
.badge {
    font-size: 0.65rem; font-family: monospace; font-weight: 700;
    padding: 2px 8px; border-radius: 4px; border: 1px solid; letter-spacing: 0.05em;
}
.metric-value {
    font-size: 3rem; font-weight: 300; color: #f1f5f9;
    font-family: 'SF Mono', 'Fira Code', monospace;
    letter-spacing: -0.02em; line-height: 1;
    margin-bottom: 1rem; position: relative; z-index: 1;
}
.metric-bar-track {
    width: 100%; height: 4px; background: rgba(255,255,255,0.05);
    border-radius: 2px; overflow: hidden; margin-bottom: 0.5rem;
}
.metric-bar-fill { height: 100%; border-radius: 2px; }
.metric-sub { font-size: 0.7rem; color: #475569; margin: 0; }

.assessment-card { background: #11141a; border: 1px solid rgba(255,255,255,0.05); border-radius: 14px; padding: 1.5rem; }
.assessment-section-title {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.7rem; font-weight: 600; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1.25rem;
}
.segment-row {
    display: flex; align-items: flex-start; justify-content: space-between;
    background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1.25rem; gap: 1rem;
}
.sub-label { font-size: 0.7rem; color: #475569; margin-bottom: 4px; }
.segment-name { font-size: 1.05rem; font-weight: 600; color: #f1f5f9; display: flex; align-items: center; gap: 8px; margin: 0; }
.pulse-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #f59e0b; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(.85)} }
.reco-text {
    font-size: 0.875rem; color: #94a3b8; line-height: 1.6;
    padding: 1rem; border: 1px solid rgba(255,255,255,0.05);
    border-radius: 8px; background: rgba(59,130,246,0.04); margin: 0;
}

.empty-state {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    min-height: 420px; background: #11141a; border: 1px dashed rgba(255,255,255,0.07);
    border-radius: 14px; text-align: center; padding: 2rem; color: #334155;
}
.empty-icon { font-size: 2.5rem; margin-bottom: 1rem; opacity: 0.3; }
.empty-state h3 { font-size: 1rem; color: #475569; font-weight: 500; margin-bottom: 0.5rem; }
.empty-state p { font-size: 0.825rem; color: #334155; max-width: 280px; line-height: 1.6; margin: 0; }

/* ── Batch results table ── */
.batch-card {
    background: #11141a; border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px; overflow: hidden; margin-bottom: 1rem;
}
.batch-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 1.25rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.batch-title { font-size: 0.7rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; }
.batch-count { font-size: 0.7rem; color: #475569; font-family: monospace; }
.batch-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.batch-table thead tr { border-bottom: 1px solid rgba(255,255,255,0.06); }
.batch-table thead th {
    padding: 0.6rem 1rem; text-align: left;
    font-size: 0.65rem; font-weight: 600; color: #475569;
    text-transform: uppercase; letter-spacing: 0.08em;
}
.batch-table tbody tr { border-bottom: 1px solid rgba(255,255,255,0.03); transition: background 0.1s; }
.batch-table tbody tr:last-child { border-bottom: none; }
.batch-table tbody tr:hover { background: rgba(255,255,255,0.02); }
.batch-table td { padding: 0.75rem 1rem; color: #cbd5e1; vertical-align: middle; }
.batch-table td.mono { font-family: monospace; }
.risk-pill {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 0.65rem; font-weight: 700; font-family: monospace;
    padding: 2px 8px; border-radius: 4px; border: 1px solid; letter-spacing: 0.05em;
}
.summary-bar {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;
    margin-bottom: 1rem;
}
.summary-tile {
    background: #11141a; border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px; padding: 1rem 1.25rem;
    display: flex; align-items: center; justify-content: space-between;
}
.summary-tile-label { font-size: 0.7rem; color: #475569; text-transform: uppercase; letter-spacing: 0.08em; }
.summary-tile-value { font-size: 1.6rem; font-weight: 300; font-family: monospace; color: #f1f5f9; }
.sample-intro {
    background: #11141a; border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px; padding: 2.5rem 2rem; text-align: center; margin-bottom: 1.5rem;
}
.sample-intro h3 { font-size: 1rem; color: #f1f5f9; font-weight: 600; margin-bottom: 0.5rem; }
.sample-intro p { font-size: 0.85rem; color: #64748b; max-width: 420px; margin: 0 auto 1.5rem; line-height: 1.6; }
.customer-chips {
    display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin-bottom: 1.5rem;
}
.chip {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.72rem; color: #64748b;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    churn_model = joblib.load("models/churn_model.joblib")
    value_model = joblib.load("models/value_model.joblib")
    with open("models/metadata.json", "r") as f:
        meta = json.load(f)
    return churn_model, value_model, meta


churn_model, value_model, meta = load_artifacts()
CHURN_THR = float(meta["churn_threshold"])
VALUE_THR = float(meta["value_threshold"])

EXAMPLES = {
    "loyal": dict(tenure=48, monthly=65.0, contract="Two year", paperless="No",
                  internet="DSL", payment="Credit card (automatic)", senior=0, partner="Yes"),
    "risky": dict(tenure=3, monthly=95.0, contract="Month-to-month", paperless="Yes",
                  internet="Fiber optic", payment="Electronic check", senior=1, partner="No"),
}

SAMPLE_BATCH = [
    {"customerID": "C-001", "name": "Alice M.",    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",  "Dependents": "Yes", "tenure": 52, "PhoneService": "Yes", "MultipleLines": "Yes",              "InternetService": "DSL",         "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",  "StreamingMovies": "No",  "Contract": "Two year",       "PaperlessBilling": "No",  "PaymentMethod": "Credit card (automatic)",    "MonthlyCharges": 72.5,  "TotalCharges": 3770.0},
    {"customerID": "C-002", "name": "Brian K.",    "gender": "Male",   "SeniorCitizen": 0, "Partner": "No",   "Dependents": "No",  "tenure": 4,  "PhoneService": "Yes", "MultipleLines": "No",               "InternetService": "Fiber optic", "OnlineSecurity": "No",  "OnlineBackup": "No",  "DeviceProtection": "No",  "TechSupport": "No",  "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",  "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",           "MonthlyCharges": 99.0,  "TotalCharges": 396.0},
    {"customerID": "C-003", "name": "Carol T.",    "gender": "Female", "SeniorCitizen": 1, "Partner": "No",   "Dependents": "No",  "tenure": 2,  "PhoneService": "Yes", "MultipleLines": "No",               "InternetService": "Fiber optic", "OnlineSecurity": "No",  "OnlineBackup": "No",  "DeviceProtection": "No",  "TechSupport": "No",  "StreamingTV": "No",  "StreamingMovies": "No",  "Contract": "Month-to-month",  "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",           "MonthlyCharges": 79.5,  "TotalCharges": 159.0},
    {"customerID": "C-004", "name": "David R.",    "gender": "Male",   "SeniorCitizen": 0, "Partner": "Yes",  "Dependents": "Yes", "tenure": 36, "PhoneService": "Yes", "MultipleLines": "Yes",              "InternetService": "Fiber optic", "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "One year",        "PaperlessBilling": "Yes", "PaymentMethod": "Credit card (automatic)",    "MonthlyCharges": 110.0, "TotalCharges": 3960.0},
    {"customerID": "C-005", "name": "Eva L.",      "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",  "Dependents": "No",  "tenure": 60, "PhoneService": "No",  "MultipleLines": "No phone service", "InternetService": "DSL",         "OnlineSecurity": "Yes", "OnlineBackup": "No",  "DeviceProtection": "Yes", "TechSupport": "No",  "StreamingTV": "No",  "StreamingMovies": "No",  "Contract": "Two year",       "PaperlessBilling": "No",  "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 45.0,  "TotalCharges": 2700.0},
    {"customerID": "C-006", "name": "Frank N.",    "gender": "Male",   "SeniorCitizen": 1, "Partner": "No",   "Dependents": "No",  "tenure": 8,  "PhoneService": "Yes", "MultipleLines": "No",               "InternetService": "Fiber optic", "OnlineSecurity": "No",  "OnlineBackup": "No",  "DeviceProtection": "No",  "TechSupport": "No",  "StreamingTV": "No",  "StreamingMovies": "No",  "Contract": "Month-to-month",  "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",           "MonthlyCharges": 75.0,  "TotalCharges": 600.0},
    {"customerID": "C-007", "name": "Grace P.",    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",  "Dependents": "Yes", "tenure": 44, "PhoneService": "Yes", "MultipleLines": "Yes",              "InternetService": "DSL",         "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "No",  "TechSupport": "Yes", "StreamingTV": "Yes", "StreamingMovies": "No",  "Contract": "One year",        "PaperlessBilling": "No",  "PaymentMethod": "Mailed check",               "MonthlyCharges": 68.0,  "TotalCharges": 2992.0},
    {"customerID": "C-008", "name": "Henry S.",    "gender": "Male",   "SeniorCitizen": 0, "Partner": "No",   "Dependents": "No",  "tenure": 1,  "PhoneService": "Yes", "MultipleLines": "No",               "InternetService": "Fiber optic", "OnlineSecurity": "No",  "OnlineBackup": "No",  "DeviceProtection": "No",  "TechSupport": "No",  "StreamingTV": "No",  "StreamingMovies": "No",  "Contract": "Month-to-month",  "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",           "MonthlyCharges": 70.35, "TotalCharges": 70.35},
    {"customerID": "C-009", "name": "Isla W.",     "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",  "Dependents": "Yes", "tenure": 72, "PhoneService": "Yes", "MultipleLines": "Yes",              "InternetService": "Fiber optic", "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Two year",        "PaperlessBilling": "Yes", "PaymentMethod": "Credit card (automatic)",    "MonthlyCharges": 115.0, "TotalCharges": 8280.0},
    {"customerID": "C-010", "name": "James D.",    "gender": "Male",   "SeniorCitizen": 0, "Partner": "No",   "Dependents": "No",  "tenure": 15, "PhoneService": "Yes", "MultipleLines": "No",               "InternetService": "No",          "OnlineSecurity": "No internet service", "OnlineBackup": "No internet service", "DeviceProtection": "No internet service", "TechSupport": "No internet service", "StreamingTV": "No internet service", "StreamingMovies": "No internet service", "Contract": "Month-to-month", "PaperlessBilling": "No", "PaymentMethod": "Mailed check", "MonthlyCharges": 20.05, "TotalCharges": 300.75},
]

if "ex" not in st.session_state:
    st.session_state.ex = EXAMPLES["loyal"]
if "batch_run" not in st.session_state:
    st.session_state.batch_run = False


def risk_attrs(prob, threshold):
    if prob >= 0.7:
        return "HIGH",   "#f43f5e"
    elif prob >= threshold:
        return "MEDIUM", "#f59e0b"
    else:
        return "LOW",    "#10b981"


def run_batch(rows):
    names = [r.pop("name", r.get("customerID", f"Row {i+1}")) for i, r in enumerate(rows)]
    ids   = [r.pop("customerID", f"C-{i+1:03d}") for i, r in enumerate(rows)]
    df = pd.DataFrame(rows)
    df_clean = clean_telco(df)
    df_value = df_clean.drop(columns=["TotalCharges"], errors="ignore")

    churn_probs = churn_model.predict_proba(df_clean)[:, 1]
    value_preds = value_model.predict(df_value)

    results = []
    for i, (cid, name, cp, vp) in enumerate(zip(ids, names, churn_probs, value_preds)):
        seg, reco = segment_customer(float(cp), float(vp), CHURN_THR, VALUE_THR)
        label, _ = risk_attrs(float(cp), CHURN_THR)
        results.append({
            "customerID": cid, "name": name,
            "churn_prob": float(cp), "risk_label": label,
            "ltv": float(vp), "segment": seg, "recommendation": reco,
        })
    return results


st.markdown('<div class="app-shell">', unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
  <div class="header-brand">
    <div class="brand-icon">⚡</div>
    <div>
      <div class="brand-name">Customer Intelligence</div>
      <div class="brand-sub">Predictive Analysis System</div>
    </div>
  </div>
  <div class="header-meta">
    <div class="status-dot"><span class="dot-green"></span><span>Model Active v2.4</span></div>
    <div class="divider-v"></div>
    <span class="sys-id">SYS_ID: 9481-A</span>
  </div>
</div>
""", unsafe_allow_html=True)

tab_single, tab_batch = st.tabs(["⚡  Single Customer", "⊞  Batch Scoring"])

# ─────────────────────────────────────────────
# TAB 1 — Single Customer
# ─────────────────────────────────────────────
with tab_single:
    bc1, bc2, _ = st.columns([2, 2, 8])
    with bc1:
        if st.button("Loyal Profile"):
            st.session_state.ex = EXAMPLES["loyal"]
            st.rerun()
    with bc2:
        if st.button("At-Risk Profile"):
            st.session_state.ex = EXAMPLES["risky"]
            st.rerun()

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    col_form, col_results = st.columns([5, 7], gap="large")

    with col_form:
        st.markdown('<div class="card-section-title" style="font-size:0.7rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.75rem">👤 &nbsp; Target Profile</div>', unsafe_allow_html=True)
        ex = st.session_state.ex
        with st.form("predict_form"):
            ci1, ci2 = st.columns(2)
            with ci1:
                tenure = st.number_input("Tenure (Months)", 0, 72, int(ex["tenure"]))
            with ci2:
                monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, float(ex["monthly"]))
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"],
                                    index=["Month-to-month", "One year", "Two year"].index(ex["contract"]))
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"],
                                    index=["DSL", "Fiber optic", "No"].index(ex["internet"]))
            payment = st.selectbox("Payment Method",
                                   ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                                   index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(ex["payment"]))
            cs1, cs2 = st.columns(2)
            with cs1:
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"],
                                         index=["Yes", "No"].index(ex["paperless"]))
            with cs2:
                senior = st.selectbox("Senior Citizen", [0, 1],
                                      index=[0, 1].index(int(ex["senior"])),
                                      format_func=lambda x: "Yes" if x else "No")
            partner = st.selectbox("Has Partner", ["Yes", "No"],
                                   index=["Yes", "No"].index(ex["partner"]))
            submitted = st.form_submit_button("⚡  Run Prediction Model", use_container_width=True)

    with col_results:
        if submitted:
            row = {
                "gender": "Female", "SeniorCitizen": int(senior), "Partner": partner,
                "Dependents": "No", "tenure": int(tenure), "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": internet,
                "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
                "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": contract, "PaperlessBilling": paperless,
                "PaymentMethod": payment, "MonthlyCharges": float(monthly),
                "TotalCharges": float(monthly) * float(tenure),
            }
            input_df = clean_telco(pd.DataFrame([row]))
            input_df_value = input_df.drop(columns=["TotalCharges"], errors="ignore")
            churn_prob = float(churn_model.predict_proba(input_df)[:, 1][0])
            value_pred = float(value_model.predict(input_df_value)[0])
            segment, reco = segment_customer(churn_prob, value_pred, CHURN_THR, VALUE_THR)

            churn_pct = int(churn_prob * 100)
            r_label, r_col = risk_attrs(churn_prob, CHURN_THR)
            r_bg, r_bdr = r_col + "1a", r_col + "33"

            if "High Value" in segment and "At Risk" in segment:
                p_label, p_col = "CRITICAL", "#f59e0b"
            elif "At Risk" in segment:
                p_label, p_col = "HIGH",     "#f43f5e"
            else:
                p_label, p_col = "STABLE",   "#10b981"
            p_bg, p_bdr = p_col + "1a", p_col + "33"

            st.markdown(f"""
<div class="results-grid">
  <div class="metric-card" style="border-color:{r_bdr}">
    <div class="metric-glow" style="background:{r_col}20"></div>
    <div class="metric-header">
      <div class="metric-title"><span class="metric-icon">⚠</span> Churn Risk</div>
      <span class="badge" style="color:{r_col};background:{r_bg};border-color:{r_bdr}">{r_label}</span>
    </div>
    <div class="metric-value">{churn_pct}<span style="font-size:1.4rem;color:{r_col}">%</span></div>
    <div class="metric-bar-track">
      <div class="metric-bar-fill" style="width:{min(churn_pct,100)}%;background:linear-gradient(90deg,#dc2626,#f59e0b)"></div>
    </div>
    <p class="metric-sub">Probability of leaving within 30 days</p>
  </div>
  <div class="metric-card" style="border-color:rgba(16,185,129,0.2)">
    <div class="metric-glow" style="background:rgba(16,185,129,0.1)"></div>
    <div class="metric-header">
      <div class="metric-title"><span class="metric-icon">◈</span> Predicted LTV</div>
      <span class="badge" style="color:#10b981;background:rgba(16,185,129,0.1);border-color:rgba(16,185,129,0.2)">VALUE</span>
    </div>
    <div class="metric-value">{value_pred:,.0f}<span style="font-size:1.1rem;color:#10b981;margin-left:3px">pts</span></div>
    <div class="metric-bar-track">
      <div class="metric-bar-fill" style="width:70%;background:linear-gradient(90deg,#059669,#2dd4bf)"></div>
    </div>
    <p class="metric-sub">Estimated lifetime value score</p>
  </div>
</div>
<div class="assessment-card">
  <div class="assessment-section-title">🛡 &nbsp; Strategic Assessment</div>
  <div class="segment-row">
    <div>
      <div class="sub-label">Business Segment</div>
      <div class="segment-name">{segment} <span class="pulse-dot"></span></div>
    </div>
    <div style="text-align:right">
      <div class="sub-label">Intervention Priority</div>
      <span class="badge" style="color:{p_col};background:{p_bg};border-color:{p_bdr};font-size:0.68rem;padding:3px 10px">{p_label}</span>
    </div>
  </div>
  <p class="reco-text">{reco}</p>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="empty-state">
  <div class="empty-icon">⚡</div>
  <h3>Awaiting Parameters</h3>
  <p>Adjust the customer profile on the left and run the prediction model to generate insights.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TAB 2 — Batch Scoring
# ─────────────────────────────────────────────
with tab_batch:
    if not st.session_state.batch_run:
        customer_names = [r["name"] for r in SAMPLE_BATCH]
        chips_html = "".join(f'<span class="chip">{n}</span>' for n in customer_names)
        st.markdown(f"""
<div class="sample-intro">
  <h3>Sample Batch — 10 Customers</h3>
  <p>A pre-loaded set of 10 diverse customers spanning different contract types,
     tenure lengths, and service configurations. Run the model to score all of them at once.</p>
  <div class="customer-chips">{chips_html}</div>
</div>
""", unsafe_allow_html=True)
        _, btn_col, _ = st.columns([3, 4, 3])
        with btn_col:
            if st.button("⊞  Run Batch Predictions on Sample", use_container_width=True):
                st.session_state.batch_run = True
                st.rerun()
    else:
        results = run_batch([r.copy() for r in SAMPLE_BATCH])

        n_high   = sum(1 for r in results if r["risk_label"] == "HIGH")
        n_medium = sum(1 for r in results if r["risk_label"] == "MEDIUM")
        n_low    = sum(1 for r in results if r["risk_label"] == "LOW")

        st.markdown(f"""
<div class="summary-bar">
  <div class="summary-tile">
    <div>
      <div class="summary-tile-label">High Risk</div>
      <div class="summary-tile-value" style="color:#f43f5e">{n_high}</div>
    </div>
    <span style="font-size:1.4rem;opacity:0.3">⚠</span>
  </div>
  <div class="summary-tile">
    <div>
      <div class="summary-tile-label">Medium Risk</div>
      <div class="summary-tile-value" style="color:#f59e0b">{n_medium}</div>
    </div>
    <span style="font-size:1.4rem;opacity:0.3">◑</span>
  </div>
  <div class="summary-tile">
    <div>
      <div class="summary-tile-label">Low Risk</div>
      <div class="summary-tile-value" style="color:#10b981">{n_low}</div>
    </div>
    <span style="font-size:1.4rem;opacity:0.3">✓</span>
  </div>
</div>
""", unsafe_allow_html=True)

        color_map = {"HIGH": "#f43f5e", "MEDIUM": "#f59e0b", "LOW": "#10b981"}

        rows_html = ""
        for r in results:
            col   = color_map[r["risk_label"]]
            bg    = col + "1a"
            bdr   = col + "33"
            pct   = int(r["churn_prob"] * 100)
            bar_w = min(pct, 100)
            rows_html += f"""
<tr>
  <td class="mono" style="color:#64748b;font-size:0.72rem">{r['customerID']}</td>
  <td style="color:#f1f5f9;font-weight:500">{r['name']}</td>
  <td>
    <div style="display:flex;align-items:center;gap:8px">
      <div style="width:60px;height:3px;background:rgba(255,255,255,0.05);border-radius:2px;overflow:hidden">
        <div style="width:{bar_w}%;height:100%;background:{col};border-radius:2px"></div>
      </div>
      <span class="mono" style="color:{col};font-size:0.8rem">{pct}%</span>
    </div>
  </td>
  <td><span class="risk-pill" style="color:{col};background:{bg};border-color:{bdr}">{r['risk_label']}</span></td>
  <td class="mono" style="color:#94a3b8">{r['ltv']:,.0f}</td>
  <td style="color:#94a3b8;font-size:0.75rem">{r['segment']}</td>
</tr>"""

        st.markdown(f"""
<div class="batch-card">
  <div class="batch-header">
    <span class="batch-title">⊞ &nbsp; Prediction Results</span>
    <span class="batch-count">{len(results)} customers scored</span>
  </div>
  <table class="batch-table">
    <thead>
      <tr>
        <th>ID</th><th>Customer</th><th>Churn Prob</th>
        <th>Risk</th><th>LTV Score</th><th>Segment</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)

        dl_col, reset_col, _ = st.columns([3, 2, 7])
        with dl_col:
            csv_df = pd.DataFrame([{
                "Customer ID":     r["customerID"],
                "Name":            r["name"],
                "Churn Prob (%)":  round(r["churn_prob"] * 100, 1),
                "Risk Level":      r["risk_label"],
                "LTV Score":       round(r["ltv"], 0),
                "Segment":         r["segment"],
                "Recommendation":  r["recommendation"],
            } for r in results])
            st.download_button(
                "↓  Download Results CSV",
                data=csv_df.to_csv(index=False).encode(),
                file_name="batch_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with reset_col:
            if st.button("← Reset", use_container_width=True):
                st.session_state.batch_run = False
                st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
