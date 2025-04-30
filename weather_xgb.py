import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load your data
df = pd.read_csv('export.csv')

# Convert 'date' to datetime and extract useful features
df['date'] = pd.to_datetime(df['date'], format = '%d-%m-%Y %H:%M')
df['dayofyear'] = df['date'].dt.dayofyear
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['year'] = df['date'].dt.year

# Drop original 'date' column
df = df.drop(columns=['date'])

df = df.dropna() #clears NAN values 

# Define features and target
X = df.drop(columns=['tavg'])  # Features
y = df['tavg']                 # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.1, max_depth=5)

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# After you have y_test and y_pred
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

# Save to CSV
results.to_csv('predictions.csv', index=False)

print("Predictions saved to predictions.csv!")

# directional accuracy
# Calculate difference between consecutive actual and predicted
actual_diff = np.sign(np.diff(y_test))
predicted_diff = np.sign(np.diff(y_pred))

# Compare directions
directional_accuracy = np.mean(actual_diff == predicted_diff)

print(f"Directional Accuracy: {directional_accuracy * 100:.2f}%")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual t-avg')
plt.ylabel('Predicted t-avg')
plt.title('Actual vs Predicted Average Temperature')
plt.grid(True)
plt.show()

