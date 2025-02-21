import streamlit as st
import pandas as pd
st.title('🎈 Nifty Range Prediction ')

st.write('this data is based on ML prediction')
# Step 1: Load the CSV Data
df = pd.read_csv('https://raw.githubusercontent.com/vipulpisal/csv-files/refs/heads/main/NIFTY%2050_Historical_PR_01012025to21022025.csv')

# Step 2: Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')

# Step 3: Sort the Data by Date in ascending order
df = df.sort_values(by='Date', ascending=True)

# Display the sorted data (Optional)
st.write(df.head())



# df = pd.read_csv('https://raw.githubusercontent.com/vipulpisal/csv-files/refs/heads/main/NIFTY%2050_Historical_PR_01012025to21022025.csv')
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
st.write(df.head())
# Check the updated dataframe
#print(df.head())

###################

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Step 10: Prepare the data for training the model
X = df[['Prev_Range', 'SMA_5', 'SMA_20', 'Volatility']]  # Features
y = df['Range']  # Target (Range)

# Step 11: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 12: Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 13: Predict the range for the next day using the most recent data
latest_data = X.tail(1)  # The most recent row of data
predicted_range = model.predict(latest_data)

# Step 14: Display the predicted range for tomorrow
st.write(f"Predicted Range for the Next Day: {predicted_range[0]}")

# Step 15: Display Model Evaluation (Optional)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
st.write(f"Mean Absolute Error (MAE) on test data: {mae}")

