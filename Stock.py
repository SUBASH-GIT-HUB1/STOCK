# -*- coding: utf-8 -*-
"""
Institutional AI Stock Forecasting Platform (Streamlit App)
Supports dynamic symbol searching for NSE/BSE stocks.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================
st.set_page_config(page_title="AI Stock Forecasting App", layout="wide", page_icon="📈")

st.title("📈 Institutional AI Stock Forecasting Platform")
st.markdown("Search or enter any **NSE/BSE 500** ticker symbol to run the **Ultimate Ensemble Forecast Model** (XGBoost + Random Forest + VWAP + Bollinger Bands).")

# ==========================================================
# CORE STOCKS & DYNAMIC SEARCH SETUP
# ==========================================================
# Major benchmark list for quick selection
popular_stocks_nse = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Infosys": "INFY.NS",
    "State Bank of India": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Larsen & Toubro": "LT.NS",
    "ITC": "ITC.NS",
    "Axis Bank": "AXISBANK.NS",
    "Cipla": "CIPLA.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "NTPC": "NTPC.NS",
    "ONGC": "ONGC.NS"
}

popular_stocks_bse = {
    "Reliance Industries": "RELIANCE.BO",
    "TCS": "TCS.BO",
    "HDFC Bank": "HDFCBANK.BO",
    "ICICI Bank": "ICICIBANK.BO",
    "Infosys": "INFY.BO",
    "State Bank of India": "SBIN.BO",
    "Aayush Wellness (539528)": "539528.BO",
    "ONGC": "ONGC.BO",
    "Cipla": "CIPLA.BO"
}

col1, col2 = st.columns([1, 2])

with col1:
    exchange = st.selectbox("1. Select Exchange", ["NSE", "BSE"])

stock_dict = popular_stocks_nse if exchange == "NSE" else popular_stocks_bse

with col2:
    selected_stock_name = st.selectbox(
        "2. Quick Select Popular Stock or Type Below:", 
        list(stock_dict.keys())
    )

# Direct custom text search input to handle any of the 500+ stocks instantly
st.markdown("---")
custom_search = st.text_input(
    f"🔍 **Direct Search / Enter Any {exchange} 500 Ticker Symbol:**", 
    placeholder="e.g. WIPRO, TATAPOWER, ADANIENT, 500312"
).strip().upper()

if custom_search:
    # Format symbol automatically with appropriate suffix (.NS or .BO)
    suffix = ".NS" if exchange == "NSE" else ".BO"
    if not custom_search.endswith(".NS") and not custom_search.endswith(".BO"):
        ticker = f"{custom_search}{suffix}"
    else:
        ticker = custom_search
    display_name = custom_search
else:
    ticker = stock_dict[selected_stock_name]
    display_name = selected_stock_name

run_button = st.button("🚀 Run AI Forecast")

# ==========================================================
# FORECAST ENGINE FUNCTION
# ==========================================================
def forecast_stock(ticker_symbol):
    market_ticker = "^NSEI"  # Nifty 50 Index reference

    with st.spinner(f"Fetching institutional market data & running AI models for {ticker_symbol}..."):
        tk = yf.Ticker(ticker_symbol)
        data = tk.history(start="2021-01-01", interval="1d")

        tk_mkt = yf.Ticker(market_ticker)
        market = tk_mkt.history(start="2021-01-01", interval="1d")

    if data.empty or len(data) < 60:
        st.error(f"Error: Unable to fetch sufficient history for '{ticker_symbol}'. Please check if the company symbol is correct for the selected exchange.")
        return

    # Column cleanup
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if isinstance(market.columns, pd.MultiIndex):
        market.columns = market.columns.get_level_values(0)

    # Feature Engineering Suite
    data["Market_Return"] = market["Close"].pct_change(1)
    data["Return_1d"] = data["Close"].pct_change(1)
    data["Return_2d"] = data["Close"].pct_change(2)

    data["EMA_9"] = data["Close"].ewm(span=9, adjust=False).mean()
    data["EMA_21"] = data["Close"].ewm(span=21, adjust=False).mean()

    exp1 = data["Close"].ewm(span=12, adjust=False).mean()
    exp2 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = exp1 - exp2
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    high_low = data["High"] - data["Low"]
    high_close = np.abs(data["High"] - data["Close"].shift())
    low_close = np.abs(data["Low"] - data["Close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data["ATR"] = true_range.rolling(14).mean()

    data["BB_Mid"] = data["Close"].rolling(20).mean()
    bb_std = data["Close"].rolling(20).std()
    data["BB_High"] = data["BB_Mid"] + (bb_std * 2)
    data["BB_Low"] = data["BB_Mid"] - (bb_std * 2)
    data["BB_Width"] = (data["BB_High"] - data["BB_Low"]) / data["BB_Mid"]

    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    data["VWAP"] = (typical_price * data["Volume"]).cumsum() / data["Volume"].cumsum()
    data["Price_to_VWAP"] = (data["Close"] - data["VWAP"]) / data["VWAP"]

    data["Target_Open_Delta"] = data["Open"].shift(-1) - data["Close"]
    data["Target_Close_Delta"] = data["Close"].shift(-1) - data["Close"]

    clean_data = data.replace([np.inf, -np.inf], np.nan).dropna()

    features = [
        "Close", "Return_1d", "Return_2d", "Market_Return",
        "EMA_9", "EMA_21", "MACD", "MACD_Signal", "RSI", "ATR",
        "BB_Width", "Price_to_VWAP"
    ]
    X = clean_data[features]

    X_train, X_test, y_train_open, y_test_open = train_test_split(X, clean_data["Target_Open_Delta"], test_size=0.2, shuffle=False)
    _, _, y_train_close, y_test_close = train_test_split(X, clean_data["Target_Close_Delta"], test_size=0.2, shuffle=False)

    # Ensemble Training
    xgb_open = XGBRegressor(n_estimators=300, learning_rate=0.01, max_depth=4, subsample=0.8, colsample_bytree=0.8, gamma=0.1, random_state=42)
    rf_open = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    xgb_open.fit(X_train, y_train_open)
    rf_open.fit(X_train, y_train_open)

    xgb_close = XGBRegressor(n_estimators=300, learning_rate=0.01, max_depth=4, subsample=0.8, colsample_bytree=0.8, gamma=0.1, random_state=42)
    rf_close = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    xgb_close.fit(X_train, y_train_close)
    rf_close.fit(X_train, y_train_close)

    def predict_ensemble(X_in):
        op_d = (xgb_open.predict(X_in) * 0.6) + (rf_open.predict(X_in) * 0.4)
        cl_d = (xgb_close.predict(X_in) * 0.6) + (rf_close.predict(X_in) * 0.4)
        return op_d[0], cl_d[0]

    # Model Backtest Plot
    pred_close_deltas = (xgb_close.predict(X_test) * 0.6) + (rf_close.predict(X_test) * 0.4)
    actual_test_close = X_test["Close"].values + y_test_close.values
    predicted_test_close = X_test["Close"].values + pred_close_deltas

    st.subheader("📊 Model Validation (Test Data Performance)")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(X_test.index, actual_test_close, label="Actual Closing Price", color="#1f77b4", linewidth=1.8)
    ax.plot(X_test.index, predicted_test_close, label="AI Predicted Close", color="#ff7f0e", linestyle="--", linewidth=1.5)
    ax.set_title(f"{ticker_symbol} - Model Performance Backtest", fontsize=12)
    ax.set_ylabel("Price (INR)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Recent Historical Evaluation Table
    st.subheader("🗓️ Recent Historical Days Performance")
    recent_subset = data.iloc[-5:].copy()
    eval_rows = []

    for i in range(1, len(recent_subset)):
        prev_row = recent_subset.iloc[i-1]
        curr_row = recent_subset.iloc[i]
        curr_date = recent_subset.index[i].date()
        curr_close = prev_row["Close"]

        X_eval = pd.DataFrame([[
            curr_close, prev_row["Return_1d"], prev_row["Return_2d"], prev_row["Market_Return"],
            prev_row["EMA_9"], prev_row["EMA_21"], prev_row["MACD"], prev_row["MACD_Signal"],
            prev_row["RSI"], prev_row["ATR"], prev_row["BB_Width"], prev_row["Price_to_VWAP"]
        ]], columns=features)

        op_d, cl_d = predict_ensemble(X_eval)
        eval_rows.append({
            "Date": curr_date,
            "Actual Open": f"₹{curr_row['Open']:.2f}",
            "Pred Open": f"₹{curr_close + op_d:.2f}",
            "Actual Close": f"₹{curr_row['Close']:.2f}",
            "Pred Close": f"₹{curr_close + cl_d:.2f}"
        })

    st.table(pd.DataFrame(eval_rows))

    # Tomorrow Forecast Prediction
    latest_row = data.iloc[-1]
    latest_close = latest_row["Close"]
    tomorrow_date = pd.bdate_range(start=data.index[-1] + pd.Timedelta(days=1), periods=1)[0].date()

    X_tomorrow = pd.DataFrame([[
        latest_close, latest_row["Return_1d"], latest_row["Return_2d"], latest_row["Market_Return"],
        latest_row["EMA_9"], latest_row["EMA_21"], latest_row["MACD"], latest_row["MACD_Signal"],
        latest_row["RSI"], latest_row["ATR"], latest_row["BB_Width"], latest_row["Price_to_VWAP"]
    ]], columns=features)

    tom_op_d, tom_cl_d = predict_ensemble(X_tomorrow)
    tom_open = latest_close + tom_op_d
    tom_close = latest_close + tom_cl_d

    # Highlights UI Banner
    st.subheader(f"🎯 Forecast for Tomorrow ({tomorrow_date})")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Prior Close", f"₹{latest_close:.2f}")
    m2.metric("Predicted Open", f"₹{tom_open:.2f}", f"{tom_op_d:+.2f}")
    m3.metric("Predicted Close", f"₹{tom_close:.2f}", f"{tom_cl_d:+.2f}")
    m4.metric("Expected Movement", "Bullish 📈" if tom_cl_d > 0 else "Bearish 📉")

# ==========================================================
# EXECUTION TRIGGER
# ==========================================================
if run_button:
    if not ticker:
        st.warning("Please enter or select a valid stock ticker symbol.")
    else:
        forecast_stock(ticker)