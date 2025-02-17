import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
import joblib
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# Setting page configuration with dark theme
st.set_page_config(
    page_title="BTC Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and elegant styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stSlider>div>div>div>div {
        background-color: #4CAF50;
    }
    .stDateInput>div>div>input {
        background-color: #2e2e2e;
        color: #ffffff;
    }
    .stDataFrame {
        background-color: #2e2e2e;
        color: #ffffff;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the application
st.title("BTC Price Prediction and SMA Analysis")

# Loading Prophet model
@st.cache_resource  # Use st.cache_resource to load the model
def load_model():
    model = joblib.load("Model/best_model.joblib")  # Usar barra diagonal
    return model

model = load_model()

# Loading historical BTC data
@st.cache_data  # Use st.cache_data to load the data
def load_data():
    data = yf.download("BTC-USD", start="2020-01-01", end="2023-12-31")
    return data

data = load_data()

# Loading 2-year predictions
@st.cache_data
def load_predictions():
    predictions = pd.read_csv("Predictions_Datasets/prediccion_2_anos.csv")  # Usar barra diagonal
    
    # Renaming columns for clarity
    predictions = predictions.rename(columns={
        "ds": "Date",
        "yhat": "Close",
        "yhat_lower": "Low",
        "yhat_upper": "High"
    })
    
    # Converting the date column to datetime.date
    predictions['Date'] = pd.to_datetime(predictions['Date']).dt.date
    return predictions

predictions = load_predictions()

# Loading when_to_buy.csv
@st.cache_data
def load_when_to_buy():
    when_to_buy = pd.read_csv("Predictions_Datasets/when_to_buy.csv")  # Usar barra diagonal
    
    # Converting the date column to datetime.date
    when_to_buy['Date'] = pd.to_datetime(when_to_buy['Date']).dt.date
    return when_to_buy

when_to_buy = load_when_to_buy()

# Combining historical data and predictions
combined_data = pd.concat([
    data.reset_index().rename(columns={"Date": "Date", "Close": "Close"}),
    predictions.rename(columns={"Close": "Close"})
])

# Ensuring that the 'Date' column is in the correct format and has no invalid values
combined_data['Date'] = pd.to_datetime(combined_data['Date'], errors='coerce').dt.date
combined_data = combined_data.dropna(subset=['Date'])  # Remove rows with invalid dates

# Seasonal decomposition
st.write("### Seasonal Decomposition")
st.write("""
Seasonal decomposition breaks down a time series into three components:
1. **Trend**: The overall direction of the data over time.
2. **Seasonality**: Patterns that repeat at regular intervals.
3. **Residual**: What remains after removing the trend and seasonality.
""")

decomposition = seasonal_decompose(data['Close'], model='multiplicative', period=30)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
decomposition.observed.plot(ax=ax1)
ax1.set_ylabel('Observed')
decomposition.trend.plot(ax=ax2)
ax2.set_ylabel('Trend')
decomposition.seasonal.plot(ax=ax3)
ax3.set_ylabel('Seasonality')
decomposition.resid.plot(ax=ax4)
ax4.set_ylabel('Residual')
st.pyplot(fig)

# Display historical data
st.write("### Historical BTC Data")
st.write(data)

# Display predictions
st.write("### Predictions for the Next 2 Years")
st.write(predictions)

# Interactive time filter
st.write("### Prediction Chart")
min_date = predictions['Date'].min()
max_date = predictions['Date'].max()
selected_range = st.slider(
    "Select a time range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date)
)

# Filter data based on the selected range
filtered_predictions = predictions[(predictions['Date'] >= selected_range[0]) & (predictions['Date'] <= selected_range[1])]

# Prediction chart
plt.figure(figsize=(12, 6))
plt.plot(filtered_predictions['Date'], filtered_predictions['Close'], label="BTC Prediction", color="#4CAF50")
plt.fill_between(filtered_predictions['Date'], filtered_predictions['Low'], filtered_predictions['High'], alpha=0.2, color="#4CAF50")
plt.title("BTC Price Predictions", color="#ffffff")
plt.xlabel("Date", color="#ffffff")
plt.ylabel("BTC Price", color="#ffffff")
plt.legend()
plt.gca().set_facecolor("#2e2e2e")
plt.gcf().set_facecolor("#1e1e1e")
plt.tick_params(colors="#ffffff")
st.pyplot(plt)

# SMA analysis and buy signals
st.write("### SMA Analysis and Buy Signals")

# Calculating SMA and buy signals
combined_data['SMA_7'] = combined_data['Close'].rolling(window=7).mean()
combined_data['SMA_30'] = combined_data['Close'].rolling(window=30).mean()
combined_data['Buy_Signal'] = np.where(combined_data['SMA_7'] > combined_data['SMA_30'], "Buy", "Sell")

# Displaying buy signals
st.write("#### Buy Signals (SMA 7 > SMA 30)")
buy_signals = combined_data[combined_data['SMA_7'] > combined_data['SMA_30']]
st.write(buy_signals[['Date', 'Close', 'SMA_7', 'SMA_30', 'Buy_Signal']])

# Date selection for recommendation
st.write("#### Buy/Sell Recommendation")
selected_date = st.date_input("Select a date", value=combined_data['Date'].min())

# Checking if the selected date is a buy signal
if selected_date in when_to_buy['Date'].values:
    recommendation = "Buy"
else:
    recommendation = "Sell"

st.write(f"Recommendation for {selected_date}: **{recommendation}**")