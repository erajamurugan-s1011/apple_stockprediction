# scripts/explain_shap.py

import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and data
model = joblib.load("models/model.pkl")
df = pd.read_csv("data/processed_AAPL.csv")

# Define features used
features = ['Return', 'MA5', 'Lag1', 'RSI', 'MACD', 'BB_%B']
X = df[features]

# Use only a small sample for speed
X_sample = X.sample(100, random_state=42)

# Create SHAP explainer
explainer = shap.Explainer(model, X_sample)
shap_values = explainer(X_sample)

# Summary plot (feature importance for all predictions)
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig("models/shap_summary.png")
print("📊 SHAP summary plot saved to models/shap_summary.png")

# Force plot for latest prediction
# Force plot for latest prediction
latest = X.iloc[[-1]]
latest_shap = explainer(latest)

pred_class = model.predict(latest)[0]

class_shap = shap.Explanation(
    values=latest_shap.values[0][:, pred_class],
    base_values=latest_shap.base_values[0][pred_class],
    data=latest.values[0],
    feature_names=X.columns
)

shap.plots.waterfall(class_shap, show=False)
plt.tight_layout()
plt.savefig("models/shap_latest_prediction.png")
print("📉 SHAP waterfall plot for latest prediction saved to models/shap_latest_prediction.png")
