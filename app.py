import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

# Page config
st.set_page_config(page_title="Stock Predictor", layout="wide")

st.title("📊 AI Stock Price Prediction Dashboard")

# Dropdown stocks
stocks = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Apple": "AAPL",
    "Tesla": "TSLA"
}

selected_stock = st.selectbox("Select Stock", list(stocks.keys()))
ticker = stocks[selected_stock]

if st.button("Analyze"):

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)

    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        st.error("No data found!")
    else:
        prices = data['Close'].values.reshape(-1,1)

        # Scaling
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(prices)

        # Prepare dataset
        X, y = [], []
        for i in range(30, len(scaled)):
            X.append(scaled[i-30:i, 0])
            y.append(scaled[i, 0])

        X, y = np.array(X), np.array(y)

        # Model (Linear Regression)
        model = LinearRegression()
        model.fit(X, y)

        # Predictions
        pred = model.predict(X)
        pred = pred.reshape(-1,1)
        pred = scaler.inverse_transform(pred)

        actual = prices[30:]

        # RMSE
        rmse = math.sqrt(mean_squared_error(actual, pred))
        st.metric("📉 RMSE", f"{rmse:.2f}")

        # Monthly Analysis (FIXED 'ME')
        df = pd.DataFrame({
            'Actual': actual.flatten(),
            'Predicted': pred.flatten()
        }, index=data.index[30:])

        monthly = df.resample('ME').mean()

        # Graph
        fig, ax = plt.subplots()
        ax.plot(monthly.index.strftime('%b-%y'), monthly['Actual'], marker='o', label='Actual')
        ax.plot(monthly.index.strftime('%b-%y'), monthly['Predicted'], marker='o', label='Predicted')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()
        st.pyplot(fig)

        # Next Day Prediction
        last_30 = scaled[-30:]
        next_input = last_30.reshape(1, -1)
        next_pred = model.predict(next_input)
        next_pred = scaler.inverse_transform(next_pred)

        st.subheader("🔮 Next Day Prediction")
        st.write(f"₹ {next_pred[0][0]:.2f}")

        # Investment Suggestion
        last_actual = monthly['Actual'].iloc[-1]
        last_pred = monthly['Predicted'].iloc[-1]

        change = ((last_pred - last_actual) / last_actual) * 100

        st.subheader("📊 Investment Suggestion")

        if change > 5:
            st.success("🚀 Strong Buy")
        elif change > 1:
            st.info("👍 Buy")
        elif -1 <= change <= 1:
            st.warning("🟡 Hold")
        elif change > -5:
            st.warning("⚠️ Sell")
        else:
            st.error("❌ Strong Sell")
