import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, Flatten
from tensorflow.keras.models import Model

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

class TransformerTimeSeries(tf.keras.Model):
    def __init__(self, embed_dim, num_heads, ff_dim, sequence_length):
        super(TransformerTimeSeries, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(0.1)
        self.dropout2 = Dropout(0.1)

    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

embed_dim = X_train.shape[2]
num_heads = 2
ff_dim = 64

inputs = Input(shape=(sequence_length, embed_dim))
transformer_block = TransformerTimeSeries(embed_dim, num_heads, ff_dim, sequence_length)
x = transformer_block(inputs)
x = Dense(20, activation='relu')(x)
x = Flatten()(x)
outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)

val_predictions = model.predict(X_val)
val_predictions_rescaled = scaler.inverse_transform(val_predictions)
y_val_rescaled = scaler.inverse_transform(y_val.reshape(-1, 1))

val_rmse = np.sqrt(mean_squared_error(y_val_rescaled, val_predictions_rescaled))
val_mae = mean_absolute_error(y_val_rescaled, val_predictions_rescaled)
print(f"Validation RMSE: {val_rmse}")
print(f"Validation MAE: {val_mae}")
