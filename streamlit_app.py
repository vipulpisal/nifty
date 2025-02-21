import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.title('🎈 Nifty Range Prediction')

st.write('This data is based on ML prediction')

# Step 1: Load the CSV Data
df = pd.read_csv('https://raw.githubusercontent.com/vipulpisal/csv-files/refs/heads/main/NIFTY%2050_Historical_PR_01012025to21022025.csv')

# Step 2: Convert 'Date' column to datetime format (Ensuring correct date format)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Added errors='coerce' to handle any errors gracefully

# Step 3: Strip out the time component from the 'Date' column (for proper sorting)
df['Date'] = df['Date'].dt.date

# Step 4: Sort the Data by Date in ascending order
df = df.sort_values(by='Date', ascending=True)

# Step 5: Display the full DataFrame in a table format (all rows will now be visible)
st.write(f"Displaying {len(df)} rows of data:")
st.dataframe(df)  # Displaying all rows interactively in a table format

# Filter for NIFTY 50 data (just in case there are multiple indices in your data)
df = df[df['Index Name'] == 'NIFTY 50']

# Step 6: Calculate the daily range (High - Low)
df['Range'] = df['High'] - df['Low']

# Step 7: Calculate the previous day's range
df['Prev_Range'] = df['Range'].shift(1)

# Step 8: Calculate moving averages of the Close prices for smoothing
df['SMA_5'] = df['Close'].rolling(window=5).mean()
df['SMA_20'] = df['Close'].rolling(window=20).mean()

# Step 9: Calculate volatility (Standard deviation of daily returns)
df['Volatility'] = df['Close'].pct_change().rolling(window=14).std()

# Step 10: Drop the rows with NaN values (due to rolling calculations)
df = df.dropna()

# Display the updated DataFrame in a table format (after feature engineering)
st.write(f"Data after feature calculations:")
st.dataframe(df)  # This will show the updated dataframe with features like 'Range', 'Prev_Range', etc.

###################
# Step 11: Prepare the data for training the model
X = df[['Prev_Range', 'SMA_5', 'SMA_20', 'Volatility']]  # Features
y = df['Range']  # Target (Range)

# Step 12: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 13: Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 14: Predict the range for the next day using the most recent data
latest_data = X.tail(1)  # The most recent row of data
predicted_range = model.predict(latest_data)

# Step 15: Display the predicted range for tomorrow
st.write(f"Predicted Range for the Next Day: {predicted_range[0]}")

# Step 16: Display Model Evaluation (Optional)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
st.write(f"Mean Absolute Error (MAE) on test data: {mae}")
