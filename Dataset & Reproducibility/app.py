
import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import shap
import matplotlib.pyplot as plt

model = xgb.XGBClassifier()
model.load_model("xgb_heart_model.json")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_columns.pkl")
cont_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

st.title("CardioAI — Heart Disease Risk Predictor")
st.markdown("Enter patient values and click **Predict**.")

with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        age      = st.number_input("Age (20–80)",          min_value=20, max_value=80,  value=63)
        sex      = st.selectbox("Sex (0=Female, 1=Male)",  [0, 1], index=1)
        trestbps = st.number_input("Resting BP (80–200)",  min_value=80, max_value=200, value=145)
        chol     = st.number_input("Cholesterol (100–600)",min_value=100,max_value=600, value=233)
        fbs      = st.selectbox("Fasting BS >120 (0/1)",   [0, 1], index=1)
        thalach  = st.number_input("Max HR (70–210)",       min_value=70, max_value=210, value=150)
        exang    = st.selectbox("Exercise Angina (0/1)",    [0, 1], index=0)
    with col2:
        oldpeak  = st.number_input("ST Depression (0–6)",  min_value=0.0, max_value=6.0, value=2.3)
        cp       = st.selectbox("Chest Pain Type (0–3)",   [0,1,2,3], index=0)
        restecg  = st.selectbox("Resting ECG (0/1/2)",     [0,1,2], index=2)
        slope    = st.selectbox("ST Slope (0/1/2)",         [0,1,2], index=2)
        ca       = st.number_input("Vessels (0–3)",         min_value=0, max_value=3, value=0)
        thal     = st.selectbox("Thal (1/2/3)",             [1,2,3], index=2)
    submitted = st.form_submit_button("Predict")

if submitted:
    row = {col: 0.0 for col in feature_cols}
    for k, v in zip(["age","sex","trestbps","chol","fbs","thalach","exang","oldpeak","ca"],
                    [age, sex, trestbps, chol, fbs, thalach, exang, oldpeak, ca]):
        row[k] = float(v)
    for col_name, val in [(f"cp_{cp}", 1), (f"restecg_{restecg}", 1),
                          (f"slope_{slope}", 1), (f"thal_{thal}", 1)]:
        if col_name in row:
            row[col_name] = 1.0

    X_in = pd.DataFrame([row])[feature_cols]
    X_in[cont_cols] = scaler.transform(X_in[cont_cols])
    prob = model.predict_proba(X_in)[0][1]
    pred = int(prob > 0.5)

    if pred == 1:
        st.error(f"Disease Present - Confidence: {prob*100:.1f}%")
    else:
        st.success(f"No Disease Detected - Confidence: {(1-prob)*100:.1f}%")

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_in)
    shap_df = pd.Series(sv[0], index=feature_cols).abs().sort_values(ascending=False).head(3)
    st.markdown("**Top 3 Driving Features:**")
    fig, ax = plt.subplots(figsize=(5, 2))
    shap_df.plot(kind="barh", ax=ax, color="tomato")
    ax.set_xlabel("SHAP |value|")
    st.pyplot(fig)
    top3 = shap_df.index.tolist()
    st.info(f"Patient's {top3[0]}, {top3[1]}, and {top3[2]} are the strongest risk drivers. "
            "A cardiologist should review these alongside the full clinical history.")
