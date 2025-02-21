import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Title and description
st.title('🎈 Nifty Range Prediction')
st.write('This model predicts the Nifty 50 range based on historical data and ML prediction.')

# Step 1: Load the CSV Data (Allow users to upload CSV)

df = pd.read_csv('https://raw.githubusercontent.com/vipulpisal/csv-files/refs/heads/main/NIFTY%2050_Historical_PR_01012025to21022025.csv')

# Step 2: Convert 'Date' column to datetime format (Ensuring correct date format)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Step 3: Strip out the time component from the 'Date' column (for proper sorting)
df['Date'] = df['Date'].dt.date

# Step 4: Sort the Data by Date in ascending order
df = df.sort_values(by='Date', ascending=True)

# Step 5: Display the full DataFrame in a table format (all rows will now be visible)
st.write(f"Displaying {len(df)} rows of data:")
st.dataframe(df, height=600)  # Displaying all rows interactively in a table format

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
st.dataframe(df, height=600)

###################
# Prepare the data for training the models
X = df[['Prev_Range', 'SMA_5', 'SMA_20', 'Volatility']]  # Features

# Step 11: Train separate models for predicting 'High' and 'Low'
# Predicting High
y_high = df['High']
X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(X, y_high, test_size=0.2, shuffle=False)

model_high = RandomForestRegressor(n_estimators=100, random_state=42)
model_high.fit(X_train_high, y_train_high)

# Predicting Low
y_low = df['Low']
X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X, y_low, test_size=0.2, shuffle=False)

model_low = RandomForestRegressor(n_estimators=100, random_state=42)
model_low.fit(X_train_low, y_train_low)

# Step 12: Predict the High and Low for the next day using the most recent data
latest_data = X.tail(1)  # The most recent row of data (latest data from the most recent day)

# Predict High and Low
predicted_high = model_high.predict(latest_data)
predicted_low = model_low.predict(latest_data)

# Step 13: Display the predicted High and Low for tomorrow
st.write(f"Predicted High for the Next Day: {predicted_high[0]:.2f}")
st.write(f"Predicted Low for the Next Day: {predicted_low[0]:.2f}")

# Step 14: Display Model Evaluation (Optional)
y_pred_high = model_high.predict(X_test_high)
mae_high = mean_absolute_error(y_test_high, y_pred_high)
st.write(f"Mean Absolute Error (MAE) for High predictions: {mae_high:.2f}")

y_pred_low = model_low.predict(X_test_low)
mae_low = mean_absolute_error(y_test_low, y_pred_low)
st.write(f"Mean Absolute Error (MAE) for Low predictions: {mae_low:.2f}")

# Additional visualizations (Optional)
import matplotlib.pyplot as plt

# Plot feature importance from RandomForest models
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Feature Importance for High prediction
ax[0].barh(X.columns, model_high.feature_importances_)
ax[0].set_title('Feature Importance (High)')

# Feature Importance for Low prediction
ax[1].barh(X.columns, model_low.feature_importances_)
ax[1].set_title('Feature Importance (Low)')

st.pyplot(fig)
