# scripts/preprocess.py

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

# Load data
df = pd.read_csv("data/AAPL.csv")

# Ensure date is sorted
df = df.sort_values("Date")
df.reset_index(drop=True, inplace=True)

# Calculate basic features
df["Return"] = df["Close"].pct_change()
df["MA5"] = df["Close"].rolling(window=5).mean()
df["Lag1"] = df["Return"].shift(1)

# RSI (14-day)
df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()

# MACD
macd = MACD(close=df["Close"])
df["MACD"] = macd.macd_diff()

# Bollinger Bands %B
bb = BollingerBands(close=df["Close"])
df["BB_%B"] = bb.bollinger_pband()

# Drop NaNs
df.dropna(inplace=True)

# Create target: 1 if tomorrow's close > today's, else 0
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

# Save processed data
df.to_csv("data/processed_AAPL.csv", index=False)
print("✅ Processed data with new features saved to data/processed_AAPL.csv")
