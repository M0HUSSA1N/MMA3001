# MMA3001 Project: Combined Cycle Power Plant Output Prediction

# Import libraries
import time
import matplotlib
matplotlib.use("Agg")  # Save plots without opening a graphical window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# ============================================================
# Reusable functions
# ============================================================

def evaluate_model(y_true, y_pred):
    """Calculate MAE, RMSE and R² for regression predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


# ============================================================
# 1. Load and inspect the dataset
# ============================================================

# Load the first worksheet containing the power plant observations.
df = pd.read_excel("data/Folds5x2_pp.xlsx", sheet_name=0)
print("Original dataset shape:", df.shape)

# Check for missing values and repeated observations.
print("\nMissing values:")
print(df.isnull().sum())

duplicate_count = df.duplicated().sum()
print("\nDuplicate rows found:", duplicate_count)

# Remove exact duplicates to reduce potential overlap between data splits.
# The original Excel file remains unchanged.
df = df.drop_duplicates().reset_index(drop=True)

print("Dataset shape after cleaning:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nStatistical summary:")
print(df.describe())


# ============================================================
# 2. Exploratory data analysis
# ============================================================

# Calculate correlations to examine relationships between variables.
correlations = df.corr()

print("\nCorrelation with electrical power output:")
print(correlations["PE"].sort_values())

# Visualise correlations using Matplotlib.
fig, ax = plt.subplots(figsize=(8, 6))
heatmap = ax.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)

# Label each variable and display its correlation coefficient.
ax.set_xticks(range(len(correlations.columns)))
ax.set_yticks(range(len(correlations.columns)))
ax.set_xticklabels(correlations.columns)
ax.set_yticklabels(correlations.columns)

for i in range(len(correlations)):
    for j in range(len(correlations.columns)):
        value = correlations.iloc[i, j]
        text_color = "white" if abs(value) > 0.5 else "black"
        ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                color=text_color)

fig.colorbar(heatmap, ax=ax, label="Pearson correlation")
ax.set_title("Correlation Between Power Plant Variables")
fig.tight_layout()
fig.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nHeatmap saved successfully!")


# ============================================================
# 3. Prepare training, validation and testing datasets
# ============================================================

# Separate the four operating conditions from electrical power output.
X = df[["AT", "V", "AP", "RH"]]
y = df["PE"]

# Reserve 15% for final testing using a fixed seed for reproducibility.
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# Divide the remaining observations into 70% training and 15% validation.
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15 / 0.85, random_state=42
)

# Verify the number of observations in each dataset.
print("\nDataset split:")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))
print("Total samples:", len(X_train) + len(X_val) + len(X_test))


# ============================================================
# 4. Multiple Linear Regression (MLR)
# ============================================================

# Use MLR as an interpretable baseline for comparison with SVR and ANN.
mlr_model = LinearRegression()

# Measure the time required to fit the model to the training data.
start_time = time.perf_counter()
mlr_model.fit(X_train, y_train)
mlr_train_time = time.perf_counter() - start_time

# Predict electrical output using the validation dataset.
start_time = time.perf_counter()
mlr_val_pred = mlr_model.predict(X_val)
mlr_predict_time = time.perf_counter() - start_time

# Evaluate the model using the same metrics planned for SVR and ANN.
mlr_mae, mlr_rmse, mlr_r2 = evaluate_model(y_val, mlr_val_pred)

print("\nMultiple Linear Regression - Validation Results:")
print(f"MAE: {mlr_mae:.4f} MW")
print(f"RMSE: {mlr_rmse:.4f} MW")
print(f"R²: {mlr_r2:.4f}")
print(f"Training time: {mlr_train_time:.6f} seconds")
print(f"Prediction time: {mlr_predict_time:.6f} seconds")

# Display the fitted coefficients in their original engineering units.
print("\nMLR intercept:", mlr_model.intercept_)

print("\nMLR coefficients:")
for feature, coefficient in zip(X.columns, mlr_model.coef_):
    print(f"{feature}: {coefficient:.4f}")