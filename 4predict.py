# scripts/predict.py

import pandas as pd
import joblib

# Load model and processed data
model = joblib.load("models/model.pkl")
df = pd.read_csv("data/processed_AAPL.csv")

# Select latest row and features
features = ['Return', 'MA5', 'Lag1', 'RSI', 'MACD', 'BB_%B']
latest = df[features].iloc[-1:]
pred = model.predict(latest)[0]
proba = model.predict_proba(latest)[0]

print("📈 Last Known Data:")
print(df.iloc[-1][['Date', 'Close'] + features])

print("\n🔮 Predicted movement for next day:")
print(f"➡️ Movement: {'UP' if pred == 1 else 'DOWN'}")
print(f"🧮 Probability - DOWN: {proba[0]:.2f}, UP: {proba[1]:.2f}")
