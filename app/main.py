import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import base64
import io

# 1. Load assets
stack = joblib.load("../outputs/models/stacking_ensemble.pkl")
best_xgb = joblib.load("../outputs/models/best_xgb.pkl")
scaler = joblib.load("../outputs/models/scaler.pkl")
feat_cols = joblib.load("../outputs/models/feature_columns.pkl")

# 2. Page config + Logo
st.set_page_config(page_title="Sydney Rain Predictor", layout="wide")
st.image("https://i.imgur.com/7nuvzR2.png", width=60)  # Add your logo or icon URL or local path
st.title("🌦️ Sydney Rain Predictor")
st.markdown("Enter today's weather conditions in the sidebar and click **Predict** to see if it will rain tomorrow in Sydney.")

# 3. Sidebar: Predict button first, then inputs
st.sidebar.title("Inputs")
if st.sidebar.button("🔍 Predict"):
    run_prediction = True
else:
    run_prediction = False

# Input features
inputs = {
    'Humidity9am':     st.sidebar.slider("Humidity9am (%)", 0, 100, 85),
    'Temp9am':         st.sidebar.slider("Temp9am (°C)", -10, 50, 18),
    'WindSpeed9am':    st.sidebar.slider("WindSpeed9am", 0, 150, 20),
    'WindGustSpeed':   st.sidebar.slider("WindGustSpeed", 0, 200, 45),
    'Rainfall':        st.sidebar.number_input("Rainfall mm", 0.0, 300.0, 2.0),
    'Evaporation':     st.sidebar.number_input("Evaporation mm", 0.0, 50.0, 3.0),
    'Sunshine':        st.sidebar.number_input("Sunshine hours", 0.0, 15.0, 5.5),
    'Cloud9am':        st.sidebar.slider("Cloud9am (0-8)", 0, 8, 4),
    'Cloud3pm':        st.sidebar.slider("Cloud3pm (0-8)", 0, 8, 5),
    'Pressure9am':     st.sidebar.number_input("Pressure9am hPa", 950.0, 1050.0, 1007.0),
    'Pressure3pm':     st.sidebar.number_input("Pressure3pm hPa", 950.0, 1050.0, 1005.0),
    'RainToday':       1 if st.sidebar.radio("RainToday?", ["No", "Yes"]) == "Yes" else 0
}

# 4. Prediction logic
if run_prediction:
    df = pd.DataFrame([inputs])
    if 'WindGustSpeed' in df and 'WindSpeed9am' in df:
        df['WindDiff'] = df['WindGustSpeed'] - df['WindSpeed9am']

    for c in feat_cols:
        if c not in df:
            df[c] = 0
    df = df[feat_cols]
    Xs = scaler.transform(df)
    proba = stack.predict_proba(Xs)[0][1]
    pred = int(proba >= 0.5)

    # 5. Columns layout for prediction and SHAP
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("🌤️ Prediction Result")
        threshold = 0.5
        if pred:
            st.success(f"It WILL rain tomorrow! (Probability: {proba*100:.2f}%)")
        else:
            st.info(f"It will NOT rain tomorrow. (Probability: {proba*100:.2f}%)")
        st.caption(f"Threshold: {threshold:.2f} → Rain probability: {proba:.2f} → {'YES' if pred else 'NO'}")
        st.markdown("**Model:** Stacking ensemble of LR, RF, XGB, and LGBM.")

    with col2:
        st.subheader("🔎 Model Explainability (SHAP)")
        explainer = shap.TreeExplainer(best_xgb)
        shap_vals = explainer.shap_values(Xs)
        shap.initjs()
        fig, ax = plt.subplots()
        shap.summary_plot(shap_vals, pd.DataFrame(Xs, columns=feat_cols), show=False)
        st.pyplot(fig)

    # 6. Download Prediction
    df['Prediction'] = ["Rain" if pred else "No Rain"]
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    st.markdown(f"📥 [Download input + prediction CSV](data:file/csv;base64,{b64})", unsafe_allow_html=True)
