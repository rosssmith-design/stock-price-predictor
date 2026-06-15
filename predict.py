import pandas as pd
import yfinance as yf

# Download stock data
df = yf.download('AAPL', start='2020-01-01', end='2024-01-01')

# Create lag features
df['yesterday'] = df['Close'].shift(1)
df['2_days_ago'] = df['Close'].shift(2)
df['3_days_ago'] = df['Close'].shift(3)

# Rolling averages
df['5_day_avg'] = df['Close'].rolling(5).mean()
df['20_day_avg'] = df['Close'].rolling(20).mean()

# Momentum features
df['volume_change'] = df['Volume'].pct_change()
df['price_change'] = df['Close'].pct_change()

# Target
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Drop NaN rows
df = df.dropna()

 # Features and target
features = ['yesterday', '2_days_ago', '3_days_ago',
             '5_day_avg', '20_day_avg', 'volume_change', 'price_change']

X = df[features]
y = df['target']

# Split by time - 80:20
split = int(len(df) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training rows: {len(X_train)}")
print(f"Test rows: {len(X_test)}")

#print(df[['Close', 'volume_change', 'price_change', 'target']].head(25))