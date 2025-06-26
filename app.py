import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import io

# Set the Streamlit page configuration.
st.set_page_config(page_title="Stock Price Forecasting", layout="wide")

# -------------------------------
# Utility Functions
# -------------------------------
def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetches historical stock data using yfinance.
    """
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            st.error("No data fetched. Please check the stock symbol and date range.")
        return data
    except Exception as e:
        st.error("Error fetching data: " + str(e))
        return None

def add_moving_averages(data):
    """
    Computes 100-day and 200-day moving averages.
    """
    data['MA100'] = data['Close'].rolling(window=100).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    return data

def build_lstm_model(input_shape):
    """
    Constructs the LSTM model for time-series forecasting.
    """
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=80, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data(data, scaler, window_size=100):
    """
    Scales the closing prices data and segments it into sequences.
    """
    raw_data = data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(raw_data)
    x, y = [], []
    for i in range(window_size, len(scaled_data)):
        x.append(scaled_data[i - window_size:i])
        y.append(scaled_data[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x, y, scaled_data

def train_test_split_data(data, split_ratio=0.75):
    """
    Splits the dataset into training and testing sets.
    """
    split_index = int(len(data) * split_ratio)
    train = data[:split_index]
    test = data[split_index:]
    return train, test

def train_and_predict(train_data, test_data, window_size=100, epochs=50, batch_size=32):
    """
    Trains the LSTM model using the training data and computes predictions on the test data.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train, y_train, _ = prepare_data(train_data, scaler, window_size)
    model = build_lstm_model((x_train.shape[1], 1))
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Prepare test data by appending the last `window_size` points of train_data to ensure continuity.
    last_train_values = train_data['Close'][-window_size:].values.reshape(-1, 1)
    combined_data = np.concatenate((last_train_values, test_data['Close'].values.reshape(-1, 1)), axis=0)
    test_scaled = scaler.transform(combined_data)
    
    x_test, y_test = [], []
    for i in range(window_size, len(test_scaled)):
        x_test.append(test_scaled[i - window_size:i])
        y_test.append(test_scaled[i, 0])
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    return model, history, predictions, y_test_inv, scaler

# -------------------------------
# Page Functions
# -------------------------------
def home_page():
    st.title("🏠 Home")
    st.markdown("""
    Welcome to the **Stock Price Forecasting App**.

    This web app leverages an LSTM model to forecast future stock prices based on historical closing prices.
    Use the sidebar to navigate between pages. Please enter the required values below.
    """)
    # Stock Symbol Input (user must type and not see a default)
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL or CL=F)")
    
    # Start Date Input using separate day, month, year select boxes
    st.markdown("### Start Date")
    col1, col2, col3 = st.columns(3)
    with col1:
        start_day = st.selectbox("Day", ["Day"] + list(range(1, 32)), key="start_day")
    with col2:
        start_month = st.selectbox("Month", ["Month"] + list(range(1, 13)), key="start_month")
    with col3:
        start_year = st.selectbox("Year", ["Year"] + list(range(2000, 2031)), key="start_year")
    
    # End Date Input using separate day, month, year select boxes
    st.markdown("### End Date")
    col4, col5, col6 = st.columns(3)
    with col4:
        end_day = st.selectbox("Day", ["Day"] + list(range(1, 32)), key="end_day")
    with col5:
        end_month = st.selectbox("Month", ["Month"] + list(range(1, 13)), key="end_month")
    with col6:
        end_year = st.selectbox("Year", ["Year"] + list(range(2000, 2031)), key="end_year")
    
    if st.button("Submit Home Inputs"):
        if stock_symbol.strip() == "":
            st.error("Please enter a stock symbol.")
            return

        # Validate that the user has made proper selections
        if start_day == "Day" or start_month == "Month" or start_year == "Year":
            st.error("Please select a valid start date.")
            return
        if end_day == "Day" or end_month == "Month" or end_year == "Year":
            st.error("Please select a valid end date.")
            return

        try:
            start_date_obj = datetime(int(start_year), int(start_month), int(start_day))
            end_date_obj = datetime(int(end_year), int(end_month), int(end_day))
            if start_date_obj >= end_date_obj:
                st.error("Start date must be earlier than end date.")
                return
            start_date = start_date_obj.strftime("%Y-%m-%d")
            end_date = end_date_obj.strftime("%Y-%m-%d")
            st.session_state['stock_symbol'] = stock_symbol
            st.session_state['start_date'] = start_date
            st.session_state['end_date'] = end_date
            st.success("Input values saved successfully!")
        except Exception as e:
            st.error("Error processing dates: " + str(e))

def stock_overview_page():
    st.title("📈 Stock Overview")
    symbol = st.session_state.get('stock_symbol', None)
    start_date = st.session_state.get('start_date', None)
    end_date = st.session_state.get('end_date', None)
    
    if not symbol or not start_date or not end_date:
        st.error("Please enter stock symbol and date values on the Home page first.")
        return
    
    data = fetch_stock_data(symbol, start_date, end_date)
    if data is None or data.empty:
        return
    data.reset_index(inplace=True)
    data = add_moving_averages(data)
    
    st.subheader(f"{symbol} Closing Prices and Moving Averages")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Date'], data['Close'], label='Closing Price', color='royalblue')
    ax.plot(data['Date'], data['MA100'], label='100-Day MA', color='green')
    ax.plot(data['Date'], data['MA200'], label='200-Day MA', color='orange')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(f"{symbol} Price Overview")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    st.subheader("Descriptive Statistics")
    st.write(data.describe())

def predict_future_prices_page():
    st.title("🔮 Predict Future Prices")
    symbol = st.session_state.get('stock_symbol', None)
    start_date = st.session_state.get('start_date', None)
    end_date = st.session_state.get('end_date', None)
    
    if not symbol or not start_date or not end_date:
        st.error("Please enter stock symbol and date values on the Home page first.")
        return
    
    data = fetch_stock_data(symbol, start_date, end_date)
    if data is None or data.empty:
        return
    data.reset_index(inplace=True)
    data.dropna(inplace=True)
    
    # Split data into training and testing sets.
    train_data, test_data = train_test_split_data(data, split_ratio=0.75)
    
    if st.button("Train LSTM Model and Forecast"):
        with st.spinner("Training the model. This may take a minute..."):
            model, history, predictions, y_test_inv, scaler = train_and_predict(
                train_data, test_data, window_size=100, epochs=50, batch_size=32
            )
        st.success("Model training and forecasting completed.")
        
        st.subheader("Predicted vs. Actual Prices")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_test_inv, label="Actual Price", color="royalblue")
        ax.plot(predictions, label="Predicted Price", color="tomato")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Price")
        ax.set_title("Predicted vs Actual Prices")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        total_points = len(y_test_inv)
        window = st.slider("Select Time Window for Zoomed View", min_value=50, max_value=total_points, value=total_points)
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(y_test_inv[-window:], label="Actual Price", color="royalblue")
        ax2.plot(predictions[-window:], label="Predicted Price", color="tomato")
        ax2.set_xlabel("Time Steps")
        ax2.set_ylabel("Price")
        ax2.set_title("Zoomed View: Predicted vs Actual Prices")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)
        
        st.subheader("Training Loss Curve")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.plot(history.history['loss'], label='Training Loss', color="purple")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")
        ax3.set_title("Training Loss over Epochs")
        ax3.legend()
        ax3.grid(True)
        st.pyplot(fig3)
    else:
        st.info("Click the button above to train the model and generate predictions.")

def model_insight_page():
    st.title("📊 Model Insight")
    
    st.subheader("LSTM Model Architecture")
    # Build a dummy model instance to show its architecture details.
    model = build_lstm_model((100, 1))
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_str = stream.getvalue()
    st.text(summary_str)
    
    st.markdown("""
    **Model Details:**
    - 4 LSTM layers with the following units: 50, 60, 80, and 120.
    - Dropout layers applied after each LSTM layer with dropout rates: 0.2, 0.3, 0.4, and 0.5.
    - A final Dense layer outputs a single value (forecasted price).
    - Compiled with the Adam optimizer and Mean Squared Error (MSE) loss.
    """)
    
    st.subheader("Data Preprocessing & Scaling")
    st.markdown("""
    The historical closing prices are first scaled to a range of 0 to 1 using MinMaxScaler.
    A sliding window (size = 100) is then used to generate sequences for training the LSTM model.
    """)
    
    st.subheader("Training Loss Curve")
    st.markdown("""
    After model training, the loss curve (viewable on the Predict Future Prices page) depicts the model's convergence.
    A lower loss indicates better model performance.
    """)

# -------------------------------
# Main Navigation and Page Routing
# -------------------------------
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Stock Overview", "Predict Future Prices", "Model Insight"])
    
    if page == "Home":
        home_page()
    elif page == "Stock Overview":
        stock_overview_page()
    elif page == "Predict Future Prices":
        predict_future_prices_page()
    elif page == "Model Insight":
        model_insight_page()

if __name__ == '__main__':
    main()