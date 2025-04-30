import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pdb

# Load data
df = pd.read_csv('export.csv')

# Prepare date features
df['date'] = pd.to_datetime(df['date'], format = "%d-%m-%Y %H:%M")
df['dayofyear'] = df['date'].dt.dayofyear
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['year'] = df['date'].dt.year

# Drop 'date' column
df = df.drop(columns=['date'])

# Clean missing values
df = df.replace([float('inf'), float('-inf')], pd.NA)
df = df.fillna(df.mean(numeric_only=True))

# Features and target
X = df.drop(columns=['tavg'])
y = df['tavg']

# Normalize features
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1,1))

# Reshape for LSTM [samples, time steps, features]
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train
history = model.fit(X_train, y_train, epochs=80, batch_size=32, validation_data=(X_test, y_test))

# Predict
y_pred = model.predict(X_test)

# Inverse transform to original scale
y_pred_inv = scaler_y.inverse_transform(y_pred)
y_test_inv = scaler_y.inverse_transform(y_test)

#pdb.set_trace()

# Metrics

mae = mean_absolute_error(y_test_inv, y_pred_inv)
mse = mean_squared_error(y_test_inv, y_pred_inv)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# directional accuracy
# Calculate difference between consecutive actual and predicted

#need 1D arrays
y_test_flat = y_test.ravel()
y_pred_flat = y_pred.ravel()

actual_diff = np.sign(np.diff(y_test_flat))
predicted_diff = np.sign(np.diff(y_pred_flat))

# Compare directions
directional_accuracy = np.mean(actual_diff == predicted_diff)

print(f"Directional Accuracy: {directional_accuracy * 100:.2f}%")

# After you have y_test and y_pred
results = pd.DataFrame({
    'Actual': y_test_flat,
    'Predicted': y_pred_flat
})

# Save to CSV
results.to_csv('predictions_lstm.csv', index=False)

print("Predictions saved to predictions_lstm.csv!")


# Plot
plt.figure(figsize=(10,6))
plt.scatter(y_test_inv, y_pred_inv, alpha=0.6)
plt.plot([y_test_inv.min(), y_test_inv.max()], [y_test_inv.min(), y_test_inv.max()], 'r--')
plt.xlabel('Actual t-avg')
plt.ylabel('Predicted t-avg')
plt.title('LSTM Prediction: Actual vs Predicted tavg')
plt.grid(True)
plt.show()

