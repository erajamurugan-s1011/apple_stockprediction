# main.py
import joblib
import pandas as pd

model = joblib.load("models/model.pkl")
data = pd.read_csv("data/processed_aapl.csv")
X = data[['Return', 'MA5', 'RSI']]
pred = model.predict([X.iloc[-1]])
print("📈 Prediction for next day:", "Up" if pred[0] == 1 else "Down")
