import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load processed data and model once
@st.cache_resource
def load_data_model():
    df = pd.read_csv("data/processed_AAPL.csv")
    model = joblib.load("models/model.pkl")
    return df, model

df, model = load_data_model()

features = ['Return', 'MA5', 'Lag1', 'RSI', 'MACD', 'BB_%B']

st.title("📈 Stock Price Movement Prediction (AAPL)")

# Sidebar: Select Date
date = st.sidebar.selectbox("Select Date for Prediction", df['Date'].values)

# Get row for selected date
row = df[df['Date'] == date]
if row.empty:
    st.error("Date not found in dataset!")
    st.stop()

X_input = row[features]

st.write(f"### Features for {date}")
st.write(X_input.T)

# Predict button
if st.button("Predict Movement"):
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]
    movement = "UP" if pred == 1 else "DOWN"
    st.write(f"### Predicted Movement: {movement}")
    st.write(f"Probability UP: {proba[1]:.2f}, DOWN: {proba[0]:.2f}")

    # SHAP explainability
    explainer = shap.Explainer(model, df[features])
    shap_values = explainer(X_input)

    pred_class = pred

    class_shap = shap.Explanation(
        values=shap_values.values[0][:, pred_class],
        base_values=shap_values.base_values[0][pred_class],
        data=X_input.values[0],
        feature_names=features
    )

    st.write("#### SHAP Explanation for Prediction")
    fig, ax = plt.subplots()
    shap.plots.waterfall(class_shap, show=False)
    st.pyplot(fig)
