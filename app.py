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

[data-testid="stMainBlockContainer"] {
    padding: 0 !important;
    max-width: 100% !important;
}

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

.app-shell {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem 2rem 4rem;
}

/* ── Header ── */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.header-brand { display: flex; align-items: center; gap: 0.75rem; }
.brand-icon {
    width: 42px; height: 42px; border-radius: 10px;
    background: linear-gradient(135deg, #4f46e5, #2563eb);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
    box-shadow: 0 0 24px rgba(79,70,229,0.25);
}
.brand-name {
    font-size: 1.1rem; font-weight: 600; color: #f1f5f9; letter-spacing: -0.01em;
}
.brand-sub {
    font-size: 0.65rem; color: #475569; text-transform: uppercase;
    letter-spacing: 0.1em; font-weight: 500; margin-top: 1px;
}
.header-meta { display: flex; align-items: center; gap: 1rem; font-size: 0.8rem; }
.status-dot {
    display: inline-flex; align-items: center; gap: 6px; color: #64748b;
}
.dot-green {
    width: 7px; height: 7px; border-radius: 50%; background: #10b981;
    box-shadow: 0 0 6px #10b981;
}
.divider-v { width: 1px; height: 16px; background: rgba(255,255,255,0.08); }
.sys-id { font-family: monospace; font-size: 0.7rem; color: #475569; }

/* ── Column card surfaces ── */
/* Only target the main 5:7 layout row — identified by adjacent .form-layout-marker */
.form-layout-marker + div [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child > [data-testid="stVerticalBlock"],
div:has(> .form-layout-marker) ~ div [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child > [data-testid="stVerticalBlock"] {
    background: #11141a;
    border: 1px solid rgba(255,255,255,0.05);
    border-top: 2px solid rgba(59,130,246,0.5);
    border-radius: 14px;
    padding: 1.25rem 1.25rem 1.5rem;
    box-shadow: 0 4px 32px rgba(0,0,0,0.35);
}

/* Simpler reliable approach: give form column a card feel via the form container */
[data-testid="stForm"] {
    background: transparent !important;
}
div[data-testid="stVerticalBlock"]:has([data-testid="stForm"]) {
    background: #11141a !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-top: 2px solid rgba(59,130,246,0.5) !important;
    border-radius: 14px !important;
    padding: 1rem 1.25rem 1.5rem !important;
    box-shadow: 0 4px 32px rgba(0,0,0,0.35) !important;
}

/* ── Card ── */
.card {
    background: #11141a;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 32px rgba(0,0,0,0.4);
}
.card-accent-bar {
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, rgba(59,130,246,0.6), transparent);
}
.card-section-title {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 1.25rem; color: #64748b;
}

/* ── Native widget overrides ── */
div[data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

.stNumberInput label, .stSelectbox label, div[data-testid="stWidgetLabel"] > p {
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    margin-bottom: 4px !important;
}

.stNumberInput input, [data-baseweb="input"] input {
    background-color: #0a0c10 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
    font-family: 'SF Mono', 'Fira Code', monospace !important;
    font-size: 0.875rem !important;
}
.stNumberInput input:focus, [data-baseweb="input"] input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}

[data-baseweb="select"] > div:first-child {
    background-color: #0a0c10 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
}
[data-baseweb="select"] svg { color: #475569 !important; }
[data-baseweb="menu"] { background-color: #1a1f2e !important; border: 1px solid rgba(255,255,255,0.1) !important; }
[data-baseweb="menu"] li { color: #cbd5e1 !important; }
[data-baseweb="menu"] li:hover { background-color: rgba(59,130,246,0.15) !important; }

[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #2563eb, #4f46e5) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 0.65rem 1.5rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
    width: 100% !important;
    margin-top: 0.5rem !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    box-shadow: 0 0 24px rgba(79,70,229,0.35) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stFormSubmitButton"] > button:active { transform: translateY(0) !important; }

[data-testid="stButton"] > button {
    background: rgba(255,255,255,0.04) !important;
    color: #64748b !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 7px !important;
    font-weight: 500 !important;
    font-size: 0.75rem !important;
    padding: 0.35rem 0.85rem !important;
    transition: all 0.15s !important;
}
[data-testid="stButton"] > button:hover {
    background: rgba(255,255,255,0.08) !important;
    color: #94a3b8 !important;
    border-color: rgba(255,255,255,0.15) !important;
}

/* ── Results ── */
.results-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-bottom: 1rem;
}
.metric-card {
    background: #11141a;
    border: 1px solid;
    border-radius: 14px;
    padding: 1.25rem;
    position: relative;
    overflow: hidden;
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
.metric-title {
    display: flex; align-items: center; gap: 6px;
    font-size: 0.78rem; font-weight: 500; color: #94a3b8;
}
.metric-icon { font-size: 0.9rem; }
.badge {
    font-size: 0.65rem; font-family: monospace; font-weight: 700;
    padding: 2px 8px; border-radius: 4px; border: 1px solid;
    letter-spacing: 0.05em;
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

.assessment-card {
    background: #11141a;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 1.5rem;
}
.assessment-section-title {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.7rem; font-weight: 600; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 1.25rem;
}
.segment-row {
    display: flex; align-items: flex-start; justify-content: space-between;
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.25rem;
    gap: 1rem;
}
.sub-label { font-size: 0.7rem; color: #475569; margin-bottom: 4px; }
.segment-name {
    font-size: 1.05rem; font-weight: 600; color: #f1f5f9;
    display: flex; align-items: center; gap: 8px;
    margin: 0;
}
.pulse-dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; background: #f59e0b;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.85); }
}
.reco-text {
    font-size: 0.875rem; color: #94a3b8; line-height: 1.6;
    padding: 1rem;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 8px;
    background: rgba(59,130,246,0.04);
    margin: 0;
}

.empty-state {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: 420px;
    background: #11141a;
    border: 1px dashed rgba(255,255,255,0.07);
    border-radius: 14px;
    text-align: center;
    padding: 2rem;
    color: #334155;
}
.empty-icon { font-size: 2.5rem; margin-bottom: 1rem; opacity: 0.3; }
.empty-state h3 { font-size: 1rem; color: #475569; font-weight: 500; margin-bottom: 0.5rem; }
.empty-state p { font-size: 0.825rem; color: #334155; max-width: 280px; line-height: 1.6; margin: 0; }

/* Quick-load buttons */
.load-btns {
    display: flex; gap: 8px; margin-bottom: 1.25rem;
}
.load-btn {
    flex: 1; padding: 6px 12px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 7px;
    font-size: 0.72rem; font-weight: 500;
    color: #64748b; cursor: pointer;
    text-align: center;
    transition: all 0.15s;
}
.load-btn:hover { background: rgba(255,255,255,0.08); color: #94a3b8; }
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

if "ex" not in st.session_state:
    st.session_state.ex = EXAMPLES["loyal"]

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
    <div class="status-dot">
      <span class="dot-green"></span>
      <span>Model Active v2.4</span>
    </div>
    <div class="divider-v"></div>
    <span class="sys-id">SYS_ID: 9481-A</span>
  </div>
</div>
""", unsafe_allow_html=True)

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
    st.markdown('<div class="card"><div class="card-accent-bar"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-section-title">👤 &nbsp; Target Profile</div>', unsafe_allow_html=True)

    ex = st.session_state.ex

    with st.form("predict_form"):
        ci1, ci2 = st.columns(2)
        with ci1:
            tenure = st.number_input("Tenure (Months)", 0, 72, int(ex["tenure"]))
        with ci2:
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, float(ex["monthly"]))

        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"],
            index=["Month-to-month", "One year", "Two year"].index(ex["contract"]),
        )
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
            index=["DSL", "Fiber optic", "No"].index(ex["internet"]),
        )
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(ex["payment"]),
        )

        cs1, cs2 = st.columns(2)
        with cs1:
            paperless = st.selectbox(
                "Paperless Billing", ["Yes", "No"],
                index=["Yes", "No"].index(ex["paperless"]),
            )
        with cs2:
            senior_opts = [0, 1]
            senior = st.selectbox(
                "Senior Citizen", senior_opts,
                index=senior_opts.index(int(ex["senior"])),
                format_func=lambda x: "Yes" if x else "No",
            )

        partner = st.selectbox(
            "Has Partner", ["Yes", "No"],
            index=["Yes", "No"].index(ex["partner"]),
        )

        submitted = st.form_submit_button("⚡  Run Prediction Model", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with col_results:
    if submitted:
        row = {
            "gender": "Female", "SeniorCitizen": int(senior),
            "Partner": partner, "Dependents": "No", "tenure": int(tenure),
            "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": internet, "OnlineSecurity": "No",
            "OnlineBackup": "No", "DeviceProtection": "No",
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

        if churn_prob >= 0.7:
            r_label, r_col = "HIGH",   "#f43f5e"
        elif churn_prob >= CHURN_THR:
            r_label, r_col = "MEDIUM", "#f59e0b"
        else:
            r_label, r_col = "LOW",    "#10b981"

        r_bg  = r_col + "1a"
        r_bdr = r_col + "33"

        if "High Value" in segment and "At Risk" in segment:
            p_label, p_col = "CRITICAL", "#f59e0b"
        elif "At Risk" in segment:
            p_label, p_col = "HIGH",     "#f43f5e"
        else:
            p_label, p_col = "STABLE",   "#10b981"

        p_bg  = p_col + "1a"
        p_bdr = p_col + "33"

        bar_w = min(churn_pct, 100)

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
      <div class="metric-bar-fill" style="width:{bar_w}%;background:linear-gradient(90deg,#dc2626,#f59e0b)"></div>
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

st.markdown("</div>", unsafe_allow_html=True)
