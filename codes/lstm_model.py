import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

data = pd.read_csv('/content/Final_Filtered_Data_with_Only_Individual_Countries.csv')
feature_column = 'Energy_Consumption'
data = data[[feature_column]]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

sequence_length = 5
X, y = create_sequences(scaled_data, sequence_length)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(sequence_length, X.shape[2])),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)

val_predictions = model.predict(X_val)
val_predictions_rescaled = scaler.inverse_transform(val_predictions)
y_val_rescaled = scaler.inverse_transform(y_val.reshape(-1, 1))

val_rmse = np.sqrt(mean_squared_error(y_val_rescaled, val_predictions_rescaled))
val_mae = mean_absolute_error(y_val_rescaled, val_predictions_rescaled)
print(f"Validation RMSE: {val_rmse}")
print(f"Validation MAE: {val_mae}")
