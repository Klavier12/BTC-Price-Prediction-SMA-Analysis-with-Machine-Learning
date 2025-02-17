# BTC Price Prediction and SMA Analysis

## 📌 Table of Contents
1. [Introduction](#introduction)
2. [Project Objective](#project-objective)
3. [Model Explanation](#model-explanation)
4. [Results](#results)
5. [How to Use](#how-to-use)
6. [Deployment](#deployment)
7. [Contributing](#contributing)
8. [License](#license)

---

## 🌟 Introduction
This project focuses on predicting the price of Bitcoin (BTC) using the **Prophet** forecasting model and analyzing **Simple Moving Averages (SMA)** to identify optimal buy/sell signals. The application is built using **Streamlit**, making it interactive and user-friendly.

---

## 🎯 Project Objective
The main objectives of this project are:
- **Predict BTC prices** for the next 2 years using historical data.
- **Identify buy/sell signals** based on SMA crossovers (SMA 7 > SMA 30).
- Provide an **interactive dashboard** for users to explore predictions and recommendations.

---

## 🧠 Model Explanation
### 1. **Prophet Model**
- **Prophet** is a forecasting tool developed by Facebook that is designed for time series data.
- It decomposes the time series into **trend**, **seasonality**, and **holidays** components.
- The model was trained on historical BTC price data and fine-tuned using **hyperparameter optimization** to achieve the best results.

### 2. **Hyperparameter Tuning**
- To optimize the Prophet model, a **hyperparameter search** was performed, adjusting parameters such as:
  - `changepoint_prior_scale`: Controls the flexibility of the trend.
  - `seasonality_prior_scale`: Adjusts the strength of seasonality.
  - `holidays_prior_scale`: Controls the impact of holidays.
- The best model was saved as `best_model.joblib` for deployment.

### 3. **SMA Analysis**
- **Simple Moving Averages (SMA)** are used to smooth out price data and identify trends.
- The application calculates **SMA 7** and **SMA 30** to determine buy/sell signals:
  - **Buy Signal**: When SMA 7 crosses above SMA 30.
  - **Sell Signal**: When SMA 7 crosses below SMA 30.

---

## 📊 Results
### 1. **Predictions**
- The model provides **BTC price predictions** for the next 2 years, including:
  - Predicted price (`yhat`).
  - Lower and upper confidence intervals (`yhat_lower`, `yhat_upper`).

### 2. **Buy/Sell Signals**
- The application identifies specific dates where **buy signals** are triggered based on SMA crossovers.
- **18 days** were identified within the 2-year prediction period where buying is recommended due to an upward trend.
- Users can select a date to see whether the recommendation is to **Buy** or **Sell**.

### 3. **Interactive Dashboard**
- The Streamlit app allows users to:
  - View historical BTC data.
  - Explore predictions with an interactive time range slider.
  - Check buy/sell recommendations for specific dates.

---

