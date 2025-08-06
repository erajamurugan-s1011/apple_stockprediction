# scripts/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    return df

def plot_close_price(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'])
    plt.title('APPLE Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_data('data/aapl.csv')
    print(df.head())
    plot_close_price(df)
