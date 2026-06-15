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



print(df[['Close', 'yesterday', '5_day_avg', '20_day_avg']].head(25))