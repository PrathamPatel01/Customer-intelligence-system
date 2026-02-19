import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error

from utils import (
    clean_telco,
    make_value_target,
    get_feature_groups,
    build_preprocessor,
    CHURN_THRESHOLD_DEFAULT,
)

DATA_PATH = "data/data.csv"
OUT_DIR = "models"

def main():
    df = pd.read_csv(DATA_PATH)
    df = clean_telco(df)

    # ---------- CHURN MODEL ----------
    y = df["Churn"]
    X = df.drop(columns=["Churn"], errors="ignore")

    num_cols, cat_cols = get_feature_groups(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(num_cols, cat_cols)

    churn_pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=2000))
    ])

    churn_pipe.fit(X_train, y_train)
    proba = churn_pipe.predict_proba(X_test)[:, 1]

    churn_threshold = CHURN_THRESHOLD_DEFAULT
    pred = (proba >= churn_threshold).astype(int)

    print("\n=== CHURN MODEL ===")
    print("ROC-AUC:", roc_auc_score(y_test, proba))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print("Report:\n", classification_report(y_test, pred))

    # ---------- VALUE MODEL ----------
    df["Value"] = make_value_target(df)
    yv = df["Value"]

    # Leakage note: drop TotalCharges because it's almost the same as Value proxy
    Xv = df.drop(columns=["Churn", "Value", "TotalCharges"], errors="ignore")

    Xv_train, Xv_test, yv_train, yv_test = train_test_split(
        Xv, yv, test_size=0.2, random_state=42
    )

    num_cols_v, cat_cols_v = get_feature_groups(Xv)

    preprocessor_v = build_preprocessor(num_cols_v, cat_cols_v)

    value_pipe = Pipeline(steps=[
        ("preprocess", preprocessor_v),
        ("model", LinearRegression())
    ])

    value_pipe.fit(Xv_train, yv_train)
    value_pred = value_pipe.predict(Xv_test)

    mae = mean_absolute_error(yv_test, value_pred)
    rmse = np.sqrt(mean_squared_error(yv_test, value_pred))

    print("\n=== VALUE MODEL ===")
    print("MAE:", mae)
    print("RMSE:", rmse)

    # ---------- THRESHOLDS / METADATA ----------
    value_threshold = float(df["Value"].median())

    metadata = {
        "churn_threshold": churn_threshold,
        "value_threshold": value_threshold,
        "features_churn": {"num_cols": num_cols, "cat_cols": cat_cols},
        "features_value": {"num_cols": num_cols_v, "cat_cols": cat_cols_v},
        "data_path": DATA_PATH,
    }

    # ---------- SAVE ----------
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    joblib.dump(churn_pipe, f"{OUT_DIR}/churn_model.joblib")
    joblib.dump(value_pipe, f"{OUT_DIR}/value_model.joblib")

    with open(f"{OUT_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved:")
    print(f"- {OUT_DIR}/churn_model.joblib")
    print(f"- {OUT_DIR}/value_model.joblib")
    print(f"- {OUT_DIR}/metadata.json")


if __name__ == "__main__":
    main()
