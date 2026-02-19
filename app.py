import json
import joblib
import pandas as pd
import streamlit as st

from utils import clean_telco, segment_customer

st.set_page_config(
    page_title="Customer Intelligence System",
    page_icon="📊",
    layout="wide"
)

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

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("📊 Customer Intelligence")
    st.caption("Churn Risk + Value Scoring + Segmentation")
    st.divider()
    st.write(f"**Churn threshold:** {CHURN_THR:.2f}")
    st.write(f"**Value threshold (median):** {VALUE_THR:,.0f}")

# ---------------- Main ----------------
st.title("Customer Intelligence System")

st.subheader("Enter Customer Details")

# Example customers
example_loyal = {
    "tenure": 48, "MonthlyCharges": 65.0, "Contract": "Two year",
    "PaperlessBilling": "No", "InternetService": "DSL",
    "PaymentMethod": "Credit card (automatic)", "SeniorCitizen": 0, "Partner": "Yes"
}

example_risky = {
    "tenure": 3, "MonthlyCharges": 95.0, "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check", "SeniorCitizen": 1, "Partner": "No"
}

if "form_data" not in st.session_state:
    st.session_state.form_data = example_loyal

colA, colB = st.columns(2)

with colA:
    if st.button("Load Loyal Example"):
        st.session_state.form_data = example_loyal
    if st.button("Load At-Risk Example"):
        st.session_state.form_data = example_risky

with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.number_input("Tenure (months)", 0, 72, int(st.session_state.form_data["tenure"]))
        monthly = st.number_input("Monthly Charges", 0.0, 200.0, float(st.session_state.form_data["MonthlyCharges"]))
        senior = st.selectbox("Senior Citizen", [0, 1],
                              index=[0,1].index(int(st.session_state.form_data["SeniorCitizen"])))

    with col2:
        contract = st.selectbox("Contract",
                                ["Month-to-month", "One year", "Two year"],
                                index=["Month-to-month", "One year", "Two year"]
                                .index(st.session_state.form_data["Contract"]))
        internet = st.selectbox("Internet Service",
                                ["DSL", "Fiber optic", "No"],
                                index=["DSL", "Fiber optic", "No"]
                                .index(st.session_state.form_data["InternetService"]))
        partner = st.selectbox("Partner", ["Yes", "No"],
                               index=["Yes","No"]
                               .index(st.session_state.form_data["Partner"]))

    with col3:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"],
                                 index=["Yes","No"]
                                 .index(st.session_state.form_data["PaperlessBilling"]))
        payment = st.selectbox("Payment Method",
                               ["Electronic check", "Mailed check",
                                "Bank transfer (automatic)", "Credit card (automatic)"],
                               index=["Electronic check", "Mailed check",
                                      "Bank transfer (automatic)", "Credit card (automatic)"]
                               .index(st.session_state.form_data["PaymentMethod"]))

    submitted = st.form_submit_button("Predict")

if submitted:
    row = {
        "gender": "Female",
        "SeniorCitizen": int(senior),
        "Partner": partner,
        "Dependents": "No",
        "tenure": int(tenure),
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": internet,
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly),
        "TotalCharges": float(monthly) * float(tenure),
    }

    input_df = pd.DataFrame([row])
    input_df = clean_telco(input_df)
    input_df_value = input_df.drop(columns=["TotalCharges"], errors="ignore")

    churn_prob = float(churn_model.predict_proba(input_df)[:, 1][0])
    value_pred = float(value_model.predict(input_df_value)[0])

    segment, reco = segment_customer(churn_prob, value_pred, CHURN_THR, VALUE_THR)

    st.divider()
    st.subheader("Prediction Result")

    m1, m2, m3 = st.columns(3)
    m1.metric("Churn Probability", f"{churn_prob:.1%}")
    m2.metric("Predicted Value", f"{value_pred:,.0f}")
    m3.metric("Segment", segment)

    st.info(reco)
