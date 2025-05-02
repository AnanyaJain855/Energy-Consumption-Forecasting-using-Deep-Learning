# ⚡ Energy Consumption Forecasting for Smart Grids using Deep Learning

This project applies and compares three advanced deep learning models — **LSTM**, **Bidirectional LSTM (BiLSTM)**, and **Transformer** — to forecast annual energy consumption from historical smart grid data.

## 🎯 Objective

To improve energy management and grid stability by:
* Accurately forecasting energy consumption
* Supporting demand response planning
* Enhancing renewable integration and reducing costs

## 🛠️ Technologies Used

* Python (Google Colab)
* TensorFlow / Keras
* Pandas, NumPy, Scikit-learn
* Matplotlib
* MinMaxScaler
* Dataset Source: Kaggle (preprocessed energy data till 2024)

## 🧠 Model Architectures

### LSTM Model

* 2 stacked LSTM layers (64 units)
* Dropout layers (0.2) to prevent overfitting
* Dense output layer

```python
# Refer to: lstm_model.py
```

### BiLSTM Model

* 2 Bidirectional LSTM layers with dropout
* Dense layers for final regression output
* Achieved **best performance** in our comparison

```python
# Refer to: bilstm_model.py
```

### Transformer Model

* Custom self-attention encoder with:
   * Multi-head attention
   * Position-wise feed-forward layers
* More scalable, but slightly less accurate in this setup

```python
# Refer to: transformer_model.py
```

## 📈 Model Visualizations

### LSTM Model

#### Training and Validation Loss
![LSTM Training Loss](https://github.com/user-attachments/assets/1912dfbe-9a67-401b-911c-a369adf6a6b3)

#### Actual vs Predicted Energy Consumption
![LSTM Actual vs Predicted](https://github.com/user-attachments/assets/4f4df85b-d2b6-4eea-85df-11e4c6d563fe)

#### Forecast for 2025–2035
![LSTM Forecast](https://github.com/user-attachments/assets/7dbae833-3923-4c51-bbe9-0c5e8cb57bb3)


### BiLSTM Model

#### Training and Validation Loss
![BiLSTM Training Loss](https://github.com/user-attachments/assets/a57e1c40-9083-48f2-807c-5e704c3bbcc5)

#### Actual vs Predicted Energy Consumption
![BiLSTM Actual vs Predicted](https://github.com/user-attachments/assets/910ef8af-0889-4063-a739-2a0f9b5a5e4b)

#### Forecast for 2025–2035
![BiLSTM Forecast](https://github.com/user-attachments/assets/7b22b86a-6ede-466b-b401-5b5c4d621b9c)


### Transformer Model

#### Training and Validation Loss
![Transformer Training Loss](https://github.com/user-attachments/assets/36fbf8c0-3733-4f15-9ae1-63cac718aa59)

#### Actual vs Predicted Energy Consumption
![Transformer Actual vs Predicted](https://github.com/user-attachments/assets/da8f59a7-a66c-4e8d-9b59-c8252c1471ed)

#### Forecast for 2025–2035
![Transformer Forecast](https://github.com/user-attachments/assets/b9d306c3-627c-4387-8584-76749a9aa063)


## 🔎 Notes

* **Loss curves** help visualize convergence and detect overfitting.
* **Actual vs Predicted** plots demonstrate model generalization on validation data.
* **Forecast graphs** illustrate long-term predictions (2025–2035), useful for energy planning.

## 📊 Results

| Model | RMSE | MAE |
|-------|------|-----|
| LSTM | 0.0352 | 0.0129 |
| BiLSTM | 0.0323 | 0.0128 |
| Transformer | 0.0633 | 0.0291 |

📌 BiLSTM emerged as the most accurate model for smart grid demand forecasting.

## 🧪 Evaluation Metrics

* **RMSE**: Root Mean Squared Error — penalizes larger errors more
* **MAE**: Mean Absolute Error — provides average error in predictions
* All values computed on the **validation set** using `scikit-learn` metrics.

## 🧠 Future Work

* Integrate external factors like weather, population, or economic activity
* Apply hybrid architectures or ensemble approaches
* Optimize models for real-time deployment
* Improve model interpretability with attention visualization
