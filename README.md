# 📊 Customer Intelligence System

**Churn Prediction + Customer Value Scoring + Business Segmentation**

An end-to-end Machine Learning and Data Analytics project that predicts customer churn, estimates customer value, and assigns actionable business segments. 

---

## 🚀 Project Overview

Businesses often struggle to identify:

- Which customers are likely to churn  
- Which customers are most valuable  
- Which customers require immediate retention efforts  

This system integrates predictive modeling with business segmentation to provide data-driven customer intelligence.

---

## 🧠 What This Project Does

For any given customer, the system:

1. Predicts **Churn Probability**
2. Estimates **Customer Value (proxy)**
3. Classifies the customer into a **business segment**
4. Provides a recommended action

---

## 🏗️ System Architecture

### 1️⃣ Churn Prediction Model
- Model: Logistic Regression
- Preprocessing:
  - OneHot Encoding for categorical features
  - Standard Scaling for numeric features
- Evaluation:
  - ROC-AUC ≈ 0.84
- Threshold tuning:
  - Default 0.5 → Adjusted to 0.4
  - Improves recall for churn detection

Output:
- Probability that a customer will churn

---

### 2️⃣ Customer Value Model
- Model: Linear Regression
- Target (Proxy):


- `TotalCharges` excluded to prevent data leakage

Output:
- Predicted customer value score

---

### 3️⃣ Customer Segmentation

Customers are classified using:

- Churn Threshold = 0.4
- Value Threshold = Median value

Segments:

| Segment | Description | Business Action |
|----------|------------|----------------|
| High Value – At Risk | Important customers likely to churn | Priority retention |
| High Value – Loyal | Valuable and stable | Loyalty maintenance |
| Low Value – At Risk | Risky but lower value | Low-cost retention |
| Low Value – Stable | Low risk & low value | Monitor |

---

## 📊 Example Output

For a test customer:

- Churn Probability: 67%
- Predicted Value: 2,450
- Segment: **High Value – At Risk**
- Recommendation: Priority retention strategy

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

---
