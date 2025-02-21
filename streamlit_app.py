import streamlit as st
import pandas as pd
st.title('🎈 Nifty Range Prediction ')

st.write('this data is based on ML prediction')

df = pd.read_csv('https://raw.githubusercontent.com/vipulpisal/csv-files/refs/heads/main/NIFTY%2050_Historical_PR_01012025to21022025.csv')
# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Filter for NIFTY 50 data
df = df[df['Index Name'] == 'NIFTY 50']

# Calculate the daily range (High - Low)
df['Range'] = df['High'] - df['Low']

# Calculate the previous day's range
df['Prev_Range'] = df['Range'].shift(1)

# Calculate moving averages of the Close prices for smoothing
df['SMA_5'] = df['Close'].rolling(window=5).mean()
df['SMA_20'] = df['Close'].rolling(window=20).mean()

# Calculate volatility (Standard deviation of daily returns)
df['Volatility'] = df['Close'].pct_change().rolling(window=14).std()

# Drop the rows with NaN values (due to rolling calculations)
df = df.dropna()

# Check the updated dataframe
print(df.head())
