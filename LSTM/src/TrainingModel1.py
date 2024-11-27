import os
import numpy as np
import pandas as pd
from keras.src.layers import BatchNormalization
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from tensorflow.python.keras.regularizers import l2

from LSTM.src.DataPreprocessModel1 import preprocess_data


# Create sequences for LSTM
def create_sequences(data, time_steps):
    X, y = [], []
    feature_columns = data.select_dtypes(include=[np.number]).columns.drop('ridership')

    if len(data) < time_steps:
        raise ValueError("Dataset is too short for the specified time_steps.")

    for i in range(len(data) - time_steps):
        X.append(data.iloc[i:i + time_steps][feature_columns].values)
        y.append(data.iloc[i + time_steps]['ridership'])

    print(f"Generated {len(X)} sequences with {time_steps} time steps each.")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# Build optimized LSTM model
def build_optimized_lstm(input_shape):
    model = Sequential([
        LSTM(128, activation='tanh', return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, activation='tanh', return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='tanh', return_sequences=False),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Analyze and optimize
def analyze_data_optimized(pro_data, time_steps):
    timestamps = pro_data['transit_timestamp']
    pro_data = pro_data.drop(columns=['transit_timestamp'])

    # Scale numerical data
    num_columns = [col for col in pro_data.columns if col != 'ridership']
    if not num_columns:
        raise ValueError("No numerical columns available for scaling.")

    pro_data, feature_scaler = scale_data(pro_data, num_columns)

    # Create sequences
    X, y = create_sequences(pro_data, time_steps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if len(X_test) == 0 or len(y_test) == 0:
        raise ValueError("Not enough data for testing.")

    # Build and train the model
    model = build_optimized_lstm((X_train.shape[1], X_train.shape[2]))
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

    # Save the model and scaler
    os.makedirs('model', exist_ok=True)
    model.save('model/my_lstm_model_1_1.keras')
    joblib.dump(feature_scaler, 'model/feature_scaler.pkl')
    print("Model and scaler saved successfully!")

    # Evaluate the model
    mse, mae = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Mean Squared Error: {mse}")
    print(f"Test Mean Absolute Error: {mae}")

    # Make predictions
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R2): {r2}")

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(y_test[:100], label='Actual')
    plt.plot(predictions[:100], label='Predicted')
    plt.title(f'Actual vs Predicted Ridership (Time Steps={time_steps})')
    plt.xlabel('Time Steps')
    plt.ylabel('Ridership')
    plt.legend()
    plt.show()

    # Plot residuals
    residuals = y_test - predictions
    plt.hist(residuals, bins=30, edgecolor='k')
    plt.title('Residuals Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.show()


# Scale data
def scale_data(data, num_columns):
    scaler = MinMaxScaler()
    data[num_columns] = scaler.fit_transform(data[num_columns])
    return data, scaler


# Process and optimize
def process_and_optimize():
    data_path = '../data/processed_dataset_new.csv'
    preprocessed_df = pd.read_csv(data_path)
    processed_df = preprocess_data(preprocessed_df)

    processed_df['transit_timestamp'] = pd.to_datetime(processed_df['transit_timestamp'])

    # # Filter data for the last 7 days
    last_date = processed_df['transit_timestamp'].max()
    start_date = last_date - pd.Timedelta(days=90)
    processed_df = processed_df[processed_df['transit_timestamp'] >= start_date]

    if processed_df.empty:
        raise ValueError("No data available for the last 7 days.")

    print(f"Data filtered from {start_date} to {last_date}.")

    # Encode cyclical time features
    processed_df['hour'] = processed_df['transit_timestamp'].dt.hour
    processed_df['hour_sin'] = np.sin(2 * np.pi * processed_df['hour'] / 24)
    processed_df['hour_cos'] = np.cos(2 * np.pi * processed_df['hour'] / 24)

    # One-hot encode station_complex_id
    if 'station_complex_id' in processed_df.columns:
        processed_df = pd.get_dummies(processed_df, columns=['station_complex_id'], drop_first=False)

    return processed_df


if __name__ == "__main__":
    processed_data = process_and_optimize()
    print("Data processing completed. Optimized analysis starting...")
    analyze_data_optimized(processed_data, time_steps=24)
