# scripts/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load updated data
df = pd.read_csv("data/processed_AAPL.csv")

# Select new features
features = ['Return', 'MA5', 'Lag1', 'RSI', 'MACD', 'BB_%B']
X = df[features]
y = df['Target']

# Time-aware split (no shuffle)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/model.pkl")

# Evaluate
y_pred = model.predict(X_test)
print("\n✅ Classification Report:\n")
print(classification_report(y_test, y_pred))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
