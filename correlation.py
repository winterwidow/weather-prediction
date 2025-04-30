'''import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load your dataset
df = pd.read_csv('export.csv')

# Optional: convert date to datetime, drop it if not needed
df['date'] = pd.to_datetime(df['date'], format = "%d-%m-%Y %H:%M")
df = df.drop(columns=['date'])  # drop non-numeric column for correlation

# Clean missing values (important for correlation matrix)
df = df.fillna(df.mean(numeric_only=True))

# Compute correlation matrix
corr = df.select_dtypes(include=[np.number])  # only numeric columns

# Plot heatmap
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr.corr(), dtype=bool))  # mask upper triangle for readability

#sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True, linewidths=0.5)

sns.heatmap(corr.corr(), 
            mask=mask, 
            annot=True, 
            cmap='coolwarm', 
            fmt=".2f", 
            linewidths=0.5,
            vmin=-1, vmax=1)

plt.title('Correlation Matrix of Weather Parameters')
plt.tight_layout()
plt.show()

# Calculate correlation matrix
corr_matrix = df.corr(numeric_only=True)

# Extract only the correlations with 'tavg'
tavg_corr = corr_matrix[['tavg']].drop('tavg')

# Plot as heatmap
plt.figure(figsize=(5,6))
sns.heatmap(tavg_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation of tavg with Other Parameters")
plt.show()'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv("export.csv")

# Ensure 'date' column is removed if present (since it's not numeric)
if 'date' in df.columns:
    df = df.drop(columns=['date'])

# Drop non-numeric columns, if any
df_numeric = df.select_dtypes(include=['number'])

# Compute correlation matrix
corr_matrix = df_numeric.corr()

# Print full correlation matrix
print("Full Correlation Matrix:\n")
print(corr_matrix)

# Save full matrix to CSV
corr_matrix.to_csv("correlation_matrix.csv")
print("\nCorrelation matrix saved as 'correlation_matrix.csv'.")

# Optional: Correlation of tavg with others
if 'tavg' in corr_matrix.columns:
    print("\nCorrelation of 'tavg' with other features:")
    print(corr_matrix['tavg'].drop('tavg').sort_values(ascending=False))

# Optional: Plot full heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlation Matrix of Weather Parameters")
plt.tight_layout()
plt.show()
