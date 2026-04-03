import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
import math

# Page config
st.set_page_config(page_title="Stock Predictor", layout="wide")

st.title("📊 AI Stock Price Prediction using ANN")

# Dropdown
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

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(prices)

        X, y = [], []
        for i in range(30, len(scaled)):
            X.append(scaled[i-30:i, 0])
            y.append(scaled[i, 0])

        X, y = np.array(X), np.array(y)

        # ANN Model
        model = Sequential()
        model.add(Dense(50, activation='relu', input_dim=30))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=10, batch_size=16, verbose=0)

        # Prediction
        pred = model.predict(X)
        pred = scaler.inverse_transform(pred)

        actual = prices[30:]

        # RMSE
        rmse = math.sqrt(mean_squared_error(actual, pred))
        st.metric("RMSE", f"{rmse:.2f}")

        # Monthly graph
        df = pd.DataFrame({
            'Actual': actual.flatten(),
            'Predicted': pred.flatten()
        }, index=data.index[30:])

        monthly = df.resample('M').mean()

        fig, ax = plt.subplots()
        ax.plot(monthly.index.strftime('%b-%y'), monthly['Actual'], marker='o', label='Actual')
        ax.plot(monthly.index.strftime('%b-%y'), monthly['Predicted'], marker='o', label='Predicted')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()
        st.pyplot(fig)

        # Next day prediction
        last_30 = scaled[-30:]
        next_input = last_30.reshape(1, -1)
        next_pred = model.predict(next_input)
        next_pred = scaler.inverse_transform(next_pred)

        st.subheader("Next Day Prediction")
        st.write(f"₹ {next_pred[0][0]:.2f}")

        # Decision
        last_actual = monthly['Actual'].iloc[-1]
        last_pred = monthly['Predicted'].iloc[-1]

        change = ((last_pred - last_actual) / last_actual) * 100

        st.subheader("Investment Suggestion")

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
