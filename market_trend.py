import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import joblib
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ---------- CONFIGURATION ----------
file_paths = ['data/20.csv', 'data/21.csv', 'data/22.csv', 'data/23.csv', 'data/24.csv']
time_step = 10
epochs = 100
batch_size = 32
commodity_name = 'Apple'   # update this as needed
market_name = 'Kolhapur'   # update this as needed

# ---------- LOAD & FILTER ----------
def filter_chunks(file, commodity, market):
    chunk_filtered = pd.DataFrame()
    for chunk in pd.read_csv(file, chunksize=100000):
        filtered = chunk[(chunk['Commodity name'] == commodity) & (chunk['Market name'] == market)]
        chunk_filtered = pd.concat([chunk_filtered, filtered], ignore_index=True)
    return chunk_filtered

def load_filtered_data(file_paths, commodity_name, market_name):
    filtered_data = pd.DataFrame()
    for file in file_paths:
        try:
            filtered = filter_chunks(file, commodity_name, market_name)
            filtered_data = pd.concat([filtered_data, filtered], ignore_index=True)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    return filtered_data

# ---------- PREPROCESS ----------
def preprocess_data(df):
    df = df[['Modal price for the commodity', 'Calendar Day']].dropna()
    df['Calendar Day'] = pd.to_datetime(df['Calendar Day'])
    df = df.sort_values('Calendar Day')
    df.set_index('Calendar Day', inplace=True)

    # Filter to last 6 months
    recent_date = df.index.max()
    six_months_ago = recent_date - pd.DateOffset(months=6)
    df = df[df.index >= six_months_ago]

    # Clip outliers (e.g., ₹500 to ₹10,000)
    df['Modal price for the commodity'] = df['Modal price for the commodity'].clip(lower=500, upper=10000)
    return df

def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaler, scaled_data

def create_dataset(dataset, time_step):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

# ---------- MODEL ----------
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ---------- TRAIN & FORECAST ----------
def train_and_forecast():
    df = load_filtered_data(file_paths, commodity_name, market_name)
    if df.empty:
        print("No data found.")
        return

    df = preprocess_data(df)
    scaler, scaled_data = scale_data(df)

    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = build_lstm_model((X.shape[1], 1))

    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])

    os.makedirs('models', exist_ok=True)
    model.save("models/lstm_model.h5")
    joblib.dump(scaler, "models/scaler.pkl")

    print("Model and scaler saved.")

    # ---------- Forecast ----------
    last_input = scaled_data[-time_step:]
    input_seq = last_input.reshape(1, time_step, 1)

    forecast_scaled = []
    for _ in range(7):
        pred = model.predict(input_seq)[0][0]
        forecast_scaled.append(pred)

        # update input sequence
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

    forecast_actual = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
    current_price = df['Modal price for the commodity'].iloc[-1]

    # ---------- Result ----------
    print(f"\n📊 Price Forecast for {commodity_name} in {market_name}")
    for i, price in enumerate(forecast_actual, start=1):
        print(f"Day {i}: ₹{price:.2f}")
    print(f"\n🟢 Current Actual Price: ₹{current_price:.2f}")

    # ---------- Plot ----------
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 8), forecast_actual, label='Forecast', marker='o')
    plt.axhline(y=current_price, color='green', linestyle='--', label='Current Price')
    plt.title(f'7-Day Price Forecast for {commodity_name} in {market_name}')
    plt.xlabel('Day')
    plt.ylabel('Price (₹)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_and_forecast()
