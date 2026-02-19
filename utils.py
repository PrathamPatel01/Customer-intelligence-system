import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


CHURN_THRESHOLD_DEFAULT = 0.4  # from your tuning


def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # drop ID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # TotalCharges fix
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Churn to 0/1 if still Yes/No
    if "Churn" in df.columns and df["Churn"].dtype == "object":
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def make_value_target(df: pd.DataFrame) -> pd.Series:
    # CLV proxy / value score
    return df["MonthlyCharges"] * df["tenure"]


def get_feature_groups(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # treat SeniorCitizen as categorical
    if "SeniorCitizen" in num_cols:
        num_cols.remove("SeniorCitizen")
        cat_cols.append("SeniorCitizen")

    return num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    numeric_tf = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_tf = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols),
        ]
    )


def segment_customer(churn_prob: float, value: float, churn_threshold: float, value_threshold: float) -> tuple[str, str]:
    """Returns (segment_label, recommendation)"""
    high_risk = churn_prob >= churn_threshold
    high_value = value >= value_threshold

    if high_value and high_risk:
        return ("High Value - At Risk", "Priority retention: personalized offer + proactive support outreach.")
    if high_value and not high_risk:
        return ("High Value - Loyal", "Maintain loyalty: perks, rewards, and prevent service issues.")
    if (not high_value) and high_risk:
        return ("Low Value - At Risk", "Low-cost retention: automated discount or plan optimization.")
    return ("Low Value - Stable", "No urgent action: nurture with light engagement and upsell opportunities.")
