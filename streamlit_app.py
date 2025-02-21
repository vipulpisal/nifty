import streamlit as st
import pandas as pd
st.title('🎈 Nifty Range Prediction ')

st.write('this data is based on ML prediction')

df = pd.read_csv('https://raw.githubusercontent.com/vipulpisal/csv-files/refs/heads/main/NIFTY%2050_Historical_PR_01012025to21022025.csv')
df
