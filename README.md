# Weather Prediction

A time series machine learning project that predicts average daily temperature using Indian weather data. This project implements and compares two different machine learning approaches: **XGBoost Regression** and **LSTM (Long Short-Term Memory) Neural Networks**.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Project Files](#project-files)
- [Results](#results)
- [Evaluation Metrics](#evaluation-metrics)
- [Key Insights](#key-insights)
- [License](#license)

## 🎯 Overview

The goal of this project is to predict the average daily temperature (`tavg`) based on historical weather data. The project evaluates various meteorological features including temperature extremes, precipitation, wind patterns, and atmospheric pressure to make accurate predictions.

## ✨ Features

- **Dual Model Implementation**: Compare XGBoost and LSTM model performance
- **Comprehensive Feature Engineering**: Extract temporal features (day, month, year, day of year)
- **Multiple Evaluation Metrics**: MAE, MSE, R² Score, and Directional Accuracy
- **Correlation Analysis**: Understand relationships between weather parameters
- **Visualization**: Actual vs. Predicted scatter plots for model evaluation
- **CSV Export**: Save predictions for further analysis

## 📊 Dataset

The project uses Indian weather data stored in `export.csv` with the following features:

| Feature | Description |
|---------|-------------|
| `date` | Date and time of observation (DD-MM-YYYY HH:MM) |
| `tavg` | Average temperature (target variable) |
| `tmin` | Minimum temperature |
| `tmax` | Maximum temperature |
| `prcp` | Precipitation |
| `wdir` | Wind direction |
| `wspd` | Wind speed |
| `pres` | Atmospheric pressure |

**Data Processing**:
- Temporal features are extracted from dates (day, month, year, day of year)
- Missing values are handled using mean imputation
- Features are normalized using MinMaxScaler for LSTM
- Data is split 80/20 for training and testing

## 🤖 Models

### 1. XGBoost Regression (`weather_xgb.py`)

A gradient boosting model optimized for regression tasks.

**Configuration**:
- Objective: `reg:squarederror`
- N Estimators: 1000
- Learning Rate: 0.1
- Max Depth: 5

**Advantages**:
- Fast training and prediction
- Handles non-linear relationships well
- Robust to outliers

### 2. LSTM Neural Network (`weather_lstm.py`)

A deep learning approach using recurrent neural networks for time series prediction.

**Architecture**:
- Input Layer: Reshaped to [samples, 1, features]
- LSTM Layer: 64 units with ReLU activation
- Dense Output Layer: 1 unit (temperature prediction)
- Optimizer: Adam
- Loss: Mean Squared Error (MSE)
- Epochs: 80
- Batch Size: 32

**Advantages**:
- Captures temporal dependencies
- Learns complex patterns in sequential data
- Effective for time series forecasting

## 🚀 Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Required Libraries

```bash
pip install pandas numpy matplotlib scikit-learn xgboost tensorflow seaborn
```

Or install from a requirements file:

```bash
# Create requirements.txt with:
pandas
numpy
matplotlib
scikit-learn
xgboost
tensorflow
seaborn
```

```bash
pip install -r requirements.txt
```

## 💻 Usage

### 1. Correlation Analysis

Analyze correlations between weather parameters:

```bash
python correlation.py
```

**Output**:
- `correlation_matrix.csv`: Full correlation matrix
- Console output: Correlation values with `tavg`
- Visualization: Heatmap of correlations (displayed)

### 2. XGBoost Model

Train and evaluate the XGBoost regression model:

```bash
python weather_xgb.py
```

**Output**:
- `predictions.csv`: Actual vs. predicted temperatures
- Console metrics: MAE, MSE, R² Score, Directional Accuracy
- Visualization: Scatter plot of actual vs. predicted values

### 3. LSTM Model

Train and evaluate the LSTM neural network:

```bash
python weather_lstm.py
```

**Output**:
- `predictions_lstm.csv`: Actual vs. predicted temperatures
- Console metrics: MAE, MSE, Directional Accuracy
- Visualization: Scatter plot of actual vs. predicted values

## 📁 Project Files

| File | Description |
|------|-------------|
| `export.csv` | Raw weather dataset |
| `weather_xgb.py` | XGBoost regression implementation |
| `weather_lstm.py` | LSTM neural network implementation |
| `correlation.py` | Correlation analysis script |
| `predictions.csv` | XGBoost model predictions |
| `predictions_lstm.csv` | LSTM model predictions |
| `correlation_matrix.csv` | Correlation matrix of all features |
| `README.md` | Project documentation |

## 📈 Results

Both models generate predictions saved to CSV files that can be used for:
- Comparing model performance
- Further statistical analysis
- Visualization and reporting
- Ensemble modeling

The scatter plots show the relationship between actual and predicted temperatures, with the red dashed line representing perfect predictions.

## 📏 Evaluation Metrics

The project uses multiple metrics to evaluate model performance:

1. **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual values
2. **Mean Squared Error (MSE)**: Average squared difference (penalizes larger errors more)
3. **R² Score** (XGBoost only): Coefficient of determination (proportion of variance explained)
4. **Directional Accuracy**: Percentage of correct trend predictions (up/down movements)

These metrics provide a comprehensive view of:
- Prediction accuracy (MAE, MSE)
- Model fit quality (R²)
- Trend prediction capability (Directional Accuracy)

## 🔍 Key Insights

- The correlation analysis reveals which weather parameters most strongly influence average temperature
- XGBoost typically offers faster training times and interpretable feature importance
- LSTM can capture temporal patterns and sequential dependencies in the data
- Comparing both models helps identify the best approach for temperature prediction

## 📝 License

This project is open source and available for educational and research purposes.

---

**Note**: Make sure you have the `export.csv` dataset in the project root directory before running any scripts. 
