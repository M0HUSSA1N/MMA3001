# MMA3001 Project: Combined Cycle Power Plant Output Prediction

import time
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# ============================================================
# Reusable Functions
# ============================================================

def evaluate_model(y_true, y_pred):
    """
    The evaluate_model function evaluates the accuracy of a regression
    model by comparing actual and predicted electrical power output.

    Args:
        y_true: Vector containing actual electrical power output values (MW).
        y_pred: Vector containing predicted electrical power output values (MW).

    Returns:
        mae: Mean Absolute Error (MW), measuring average absolute error.
        rmse: Root Mean Squared Error (MW), giving greater weight to larger prediction errors.
        r2: Coefficient of determination, measuring how well the model explains variation in electrical power output.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


def make_ann(hidden_layers, alpha, learning_rate):
    """
    The make_ann function creates an Artificial Neural Network regression
    model with the specified hyperparameters and feature standardisation.

    Args:
        hidden_layers: Tuple specifying the number of neurons in each hidden layer of the neural network.
        alpha: L2 regularisation parameter used during ANN training.
        learning_rate: Initial learning rate used by the Adam optimiser.

    Returns:
        ann_model: Scikit-learn pipeline containing a StandardScaler and an MLPRegressor with the selected hyperparameters.
    """
    ann_model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=learning_rate,
            max_iter=2500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=35,
            tol=0.00001,
            random_state=42
        )
    )

    return ann_model


# ============================================================
# Project Heading
# ============================================================

print("=" * 65)
print("       MMA3001 | POWER PLANT OUTPUT PREDICTION")
print("=" * 65)


# ============================================================
# 1. Load and Inspect Dataset
# ============================================================

df = pd.read_excel("data/Folds5x2_pp.xlsx", sheet_name=0)

original_samples = len(df)
missing_values = df.isnull().sum().sum()
duplicate_count = df.duplicated().sum()

# Remove exact duplicate observations.
df = df.drop_duplicates().reset_index(drop=True)

print("\n[1] DATASET")
print("-" * 65)
print(f"Original samples       : {original_samples:,}")
print(f"Duplicate rows removed : {duplicate_count:,}")
print(f"Cleaned samples        : {len(df):,}")
print(f"Missing values         : {missing_values}")
print(f"Input features         : AT, V, AP, RH")
print(f"Target variable        : PE (MW)")


# ============================================================
# 2. Exploratory Data Analysis
# ============================================================

correlations = df.corr()

# Generate a correlation heatmap.
fig, ax = plt.subplots(figsize=(8, 6))
heatmap = ax.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)

ax.set_xticks(range(len(correlations.columns)))
ax.set_yticks(range(len(correlations.columns)))
ax.set_xticklabels(correlations.columns)
ax.set_yticklabels(correlations.columns)

for i in range(len(correlations)):
    for j in range(len(correlations.columns)):
        value = correlations.iloc[i, j]
        text_color = "white" if abs(value) > 0.5 else "black"
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color)

fig.colorbar(heatmap, ax=ax, label="Pearson correlation")
ax.set_title("Correlation Between Power Plant Variables")

fig.tight_layout()
fig.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("\n[2] EXPLORATORY DATA ANALYSIS")
print("-" * 65)
print("Correlation with electrical power output:")

for feature in ["AT", "V", "AP", "RH"]: 
    print(f"  {feature:<3} : {correlations.loc[feature, 'PE']:+.4f}")

print("\nCorrelation heatmap    : correlation_heatmap.png")


# ============================================================
# 3. Prepare Training, Validation and Testing Datasets
# ============================================================

X = df[["AT", "V", "AP", "RH"]]
y = df["PE"]

# Reserve 15% of observations for independent final testing.
X_temp, X_test, y_temp, y_test = train_test_split(X,y,test_size=0.15,random_state=42)

# Divide the remaining observations into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(X_temp,y_temp,test_size=0.15 / 0.85,random_state=42)

print("\n[3] DATA SPLIT")
print("-" * 65)
print(f"Training samples       : {len(X_train):,} (70%)")
print(f"Validation samples     : {len(X_val):,} (15%)")
print(f"Testing samples        : {len(X_test):,} (15%)")
print("Random state           : 42")


# ============================================================
# 4. Multiple Linear Regression (MLR)
# ============================================================

mlr_model = LinearRegression()

# Measure model training time.
start_time = time.perf_counter()
mlr_model.fit(X_train, y_train)
mlr_train_time = time.perf_counter() - start_time

# Measure validation prediction time.
start_time = time.perf_counter()
mlr_val_pred = mlr_model.predict(X_val)
mlr_predict_time = time.perf_counter() - start_time

mlr_mae, mlr_rmse, mlr_r2 = evaluate_model(y_val,mlr_val_pred)

print("\n[4] MULTIPLE LINEAR REGRESSION")
print("-" * 65)
print(f"Validation MAE         : {mlr_mae:.4f} MW")
print(f"Validation RMSE        : {mlr_rmse:.4f} MW")
print(f"Validation R²          : {mlr_r2:.4f}")
print(f"Training time          : {mlr_train_time:.6f} s")
print(f"Prediction time        : {mlr_predict_time:.6f} s")

print("\nRegression coefficients:")
print(f"  Intercept : {mlr_model.intercept_:.4f}")

for feature, coefficient in zip(X.columns, mlr_model.coef_):
    print(f"  {feature:<3}       : {coefficient:+.4f}")


# ============================================================
# 5. Support Vector Regression (SVR)
# ============================================================

# Hyperparameters investigated using validation RMSE.
c_values = [1, 10, 100]
epsilon_values = [0.1, 1.0]
gamma_values = [0.01, 0.1, 0.3, "scale"]

svr_results = []

best_svr_rmse = float("inf")
best_svr_model = None
best_svr_settings = None

print("\n[5] SUPPORT VECTOR REGRESSION")
print("-" * 65)
print("Tuning SVR hyperparameters...")

svr_tuning_start = time.perf_counter()

for c in c_values:
    for epsilon in epsilon_values:
        for gamma in gamma_values:

            # Standardise features before applying the RBF kernel.
            svr_model = make_pipeline(
                StandardScaler(),
                SVR(kernel="rbf", C=c, epsilon=epsilon, gamma=gamma)
            )

            # Train the SVR model.
            start_time = time.perf_counter()
            svr_model.fit(X_train, y_train)
            train_time = time.perf_counter() - start_time

            # Evaluate on validation data.
            start_time = time.perf_counter()
            val_pred = svr_model.predict(X_val)
            predict_time = time.perf_counter() - start_time

            mae, rmse, r2 = evaluate_model(
                y_val,
                val_pred
            )

            support_vectors = len(
                svr_model.named_steps["svr"].support_
            )

            # Store the complete tuning results.
            svr_results.append({
                "C": c,
                "epsilon": epsilon,
                "gamma": gamma,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "Support_Vectors": support_vectors,
                "Train_Time": train_time,
                "Predict_Time": predict_time
            })

            # Select the configuration with the lowest validation RMSE.
            if rmse < best_svr_rmse:
                best_svr_rmse = rmse
                best_svr_model = svr_model
                best_svr_settings = (c, epsilon, gamma)

svr_tuning_time = time.perf_counter() - svr_tuning_start

# Save all configurations for analysis and documentation.
svr_results_df = pd.DataFrame(svr_results).sort_values("RMSE")
svr_results_df.to_csv("svr_tuning_results.csv", index=False)

best_svr_mae, best_svr_rmse, best_svr_r2 = evaluate_model(y_val, best_svr_model.predict(X_val))

best_svr_row = svr_results_df.iloc[0]

best_support_vectors = len(best_svr_model.named_steps["svr"].support_)

support_percentage = (best_support_vectors / len(X_train) * 100)

print(f"Configurations tested  : {len(svr_results)}")
print(f"Best C                 : {best_svr_settings[0]}")
print(f"Best epsilon           : {best_svr_settings[1]}")
print(f"Best gamma             : {best_svr_settings[2]}")
print(f"Validation MAE         : {best_svr_mae:.4f} MW")
print(f"Validation RMSE        : {best_svr_rmse:.4f} MW")
print(f"Validation R²          : {best_svr_r2:.4f}")
print(f"Support vectors        : {best_support_vectors:,}")
print(f"Support vector ratio   : {support_percentage:.2f}%")
print(f"Selected training time : {best_svr_row['Train_Time']:.4f} s")
print(f"Total tuning time      : {svr_tuning_time:.2f} s")
print("Detailed results       : svr_tuning_results.csv")


# ============================================================
# 6. Artificial Neural Network (ANN)
# ============================================================

# Compare architecture, regularisation and learning rate.
hidden_layer_options = [(20,), (50,), (50, 25)]
alpha_values = [0.0001, 0.01]
learning_rates = [0.001, 0.01]

ann_results = []

best_ann_rmse = float("inf")
best_ann_model = None
best_ann_settings = None

print("\n[6] ARTIFICIAL NEURAL NETWORK")
print("-" * 65)
print("Tuning ANN hyperparameters...")

ann_tuning_start = time.perf_counter()

for hidden_layers in hidden_layer_options:
    for alpha in alpha_values:
        for learning_rate in learning_rates:

            ann_model = make_ann(hidden_layers=hidden_layers, alpha=alpha, learning_rate=learning_rate)

            # Train ANN and record convergence warnings.
            start_time = time.perf_counter()

            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", ConvergenceWarning)
                ann_model.fit(X_train, y_train)

            train_time = time.perf_counter() - start_time

            # Evaluate training performance.
            train_pred = ann_model.predict(X_train)

            train_mae, train_rmse, train_r2 = evaluate_model(y_train, train_pred)

            # Evaluate validation performance.
            start_time = time.perf_counter()
            val_pred = ann_model.predict(X_val)
            predict_time = time.perf_counter() - start_time

            mae, rmse, r2 = evaluate_model(y_val, val_pred)

            # Record ANN training iterations and parameter count.
            fitted_ann = ann_model.named_steps["mlpregressor"]

            iterations = fitted_ann.n_iter_

            parameter_count = sum(
                weights.size for weights in fitted_ann.coefs_
            ) + sum(
                biases.size for biases in fitted_ann.intercepts_
            )

            convergence_warning = any(
                issubclass(w.category, ConvergenceWarning)
                for w in caught_warnings
            )

            # Store complete ANN tuning results.
            ann_results.append({
                "Hidden_Layers": str(hidden_layers),
                "Alpha": alpha,
                "Learning_Rate": learning_rate,
                "Train_MAE": train_mae,
                "Train_RMSE": train_rmse,
                "Train_R2": train_r2,
                "Val_MAE": mae,
                "Val_RMSE": rmse,
                "Val_R2": r2,
                "Parameters": parameter_count,
                "Iterations": iterations,
                "Convergence_Warning": convergence_warning,
                "Train_Time": train_time,
                "Predict_Time": predict_time
            })

            # Select the configuration with the lowest validation RMSE.
            if rmse < best_ann_rmse:
                best_ann_rmse = rmse
                best_ann_model = ann_model
                best_ann_settings = (hidden_layers, alpha, learning_rate)

ann_tuning_time = time.perf_counter() - ann_tuning_start

# Save all ANN configurations for analysis and documentation.
ann_results_df = pd.DataFrame(ann_results).sort_values("Val_RMSE")
ann_results_df.to_csv("ann_tuning_results.csv", index=False)

best_ann_row = ann_results_df.iloc[0]

best_ann_mae, best_ann_rmse, best_ann_r2 = evaluate_model(y_val, best_ann_model.predict(X_val))

print(f"Configurations tested  : {len(ann_results)}")
print(f"Best architecture      : {best_ann_settings[0]}")
print(f"Best alpha             : {best_ann_settings[1]}")
print(f"Best learning rate     : {best_ann_settings[2]}")
print(f"Validation MAE         : {best_ann_mae:.4f} MW")
print(f"Validation RMSE        : {best_ann_rmse:.4f} MW")
print(f"Validation R²          : {best_ann_r2:.4f}")
print(f"Trainable parameters   : {int(best_ann_row['Parameters']):,}")
print(f"Training iterations    : {int(best_ann_row['Iterations'])}")
print(f"Convergence warning    : {best_ann_row['Convergence_Warning']}")
print(f"Selected training time : {best_ann_row['Train_Time']:.4f} s")
print(f"Total tuning time      : {ann_tuning_time:.2f} s")
print("Detailed results       : ann_tuning_results.csv")

# ============================================================
# Validation Model Comparison
# ============================================================

comparison_df = pd.DataFrame({
    "Model": ["MLR", "SVR", "ANN"],
    "MAE (MW)": [mlr_mae, best_svr_mae, best_ann_mae],
    "RMSE (MW)": [mlr_rmse, best_svr_rmse, best_ann_rmse],
    "R2": [mlr_r2, best_svr_r2, best_ann_r2]
})

comparison_df = comparison_df.sort_values("RMSE (MW)")

print("\n" + "=" * 65)
print("                  VALIDATION MODEL COMPARISON")
print("=" * 65)

print(comparison_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

best_model_name = comparison_df.iloc[0]["Model"]

print(f"\nBest validation model  : {best_model_name}")
print("Final test evaluation  : Not performed")
print("=" * 65)