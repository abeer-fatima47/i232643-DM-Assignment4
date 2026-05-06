# CardioAI - Heart Disease & Digit Recognition Pipeline
Assignment #4 · Abeer Fatima · i232643

---

## Project Structure

```
i232643_DM_asn4.ipynb
/Dataset & Reproducibility processed.cleveland.data · app.py · xgb_heart_model.json · scaler.pkl · feature_columns.pkl · requirements.txt
i232643_DM_asn4_Report.docx
```

---

## Dataset Setup

**Dataset 1 - UCI Heart Disease (Cleveland)**
1. Go to https://archive.ics.uci.edu/dataset/45/heart+disease
2. Download `processed.cleveland.data`
3. Place it in the same directory as the notebook

**Dataset 2 - MNIST** is loaded automatically by Keras (`tensorflow.keras.datasets.mnist`). No download needed.

---

## Running the Notebook

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm shap imbalanced-learn tensorflow streamlit joblib
jupyter notebook notebooks/i232643_DM_asn4.ipynb
```

Run all cells top to bottom.

---

## Running the Dashboard

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. The form is pre-populated - click **Predict** to run immediately.
