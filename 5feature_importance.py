# scripts/feature_importance.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and data
model = joblib.load("models/model.pkl")
df = pd.read_csv("data/processed_AAPL.csv")

# Define features used in training
features = ['Return', 'MA5', 'Lag1', 'RSI', 'MACD', 'BB_%B']

# Get feature importances from model
importances = model.feature_importances_

# Create DataFrame for plotting
feat_imp_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig("models/feature_importance.png")
plt.show()

print("✅ Feature importance chart saved to models/feature_importance.png")
