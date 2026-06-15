import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Download stock data
df = yf.download('AMD', start='2020-01-01', end='2024-01-01')

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

# Daily range
df['daily_range'] = df['High'] - df['Low']

# Price position
df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])

# Calculate RSI
def calculate_rsi(prices):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['rsi'] = calculate_rsi(df['Close'])

# MACD
df['macd'] = df['Close'].rolling(12).mean() - df['Close'].rolling(26).mean()


# Drop NaN rows
df = df.dropna()

 # Features and target
features = ['yesterday', '2_days_ago', '3_days_ago',
             '5_day_avg', '20_day_avg', 'volume_change',
               'price_change', 'daily_range', 'price_position', 'rsi', 'macd']

X = df[features]
y = df['target']

# Split by time - 80:20
split = int(len(df) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training rows: {len(X_train)}")
print(f"Test rows: {len(X_test)}")

# Create model and train on train dataset
model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
model.fit(X_train, y_train)
print("Model trained!")

# Shows training accuracy
train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
print(f"Training Accuracy: {train_acc * 100:.2f}%")

# Shows test accuracy
test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

#print(df[['Close', 'volume_change', 'price_change', 'target']].head(25))