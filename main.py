
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# Load the first worksheet of the power plant dataset
df = pd.read_excel("data/Folds5x2_pp.xlsx", sheet_name=0)

# Display basic information about the dataset
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe())

# Calculate correlations between variables
correlations = df.corr()

print("\nCorrelation with electrical power output:")
print(correlations["PE"].sort_values())

# Create a correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlations, annot=True, cmap="coolwarm",
            fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Between Power Plant Variables")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nHeatmap saved successfully!")