
"""
MMA3001 Project: Combined Cycle Power Plant Output Prediction

Description:
    This program predicts electrical power output using Multiple Linear
    Regression (MLR), Support Vector Regression (SVR) and Artificial
    Neural Networks (ANN).

    The workflow includes dataset validation, exploratory data analysis,
    hyperparameter tuning, model comparison and independent testing.
    Reusable functions are imported from main_model_functions.py.

Outputs:
    CSV files containing model performance and hyperparameter tuning results.
    PNG figures showing correlations, predictions and residuals.

Execution:
    python main.py
"""

import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression

from main_model_functions import (
    load_and_clean_data,
    split_dataset,
    evaluate_model,
    make_svr,
    make_ann
)

# ============================================================
# Project Heading
# ============================================================

print("=" * 65)
print("       MMA3001 | POWER PLANT OUTPUT PREDICTION")
print("=" * 65)

# ============================================================
# 1. Load and Inspect Dataset
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
dataset_path = PROJECT_DIR / "data" / "Folds5x2_pp.xlsx"

df, original_samples, duplicate_count, missing_values = (load_and_clean_data(dataset_path))

print("\n[1] DATASET")
print("-" * 65)
print(f"Original samples       : {original_samples:,}")
print(f"Duplicate rows removed : {duplicate_count:,}")
print(f"Cleaned samples        : {len(df):,}")
print(f"Missing values         : {missing_values}")
print("Input features         : AT, V, AP, RH")
print("Target variable        : PE (MW)")

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
fig.savefig(PROJECT_DIR / "correlation_heatmap.png", dpi=300, bbox_inches="tight")
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

X_temp, X_train, X_val, X_test, y_temp, y_train, y_val, y_test = split_dataset(df)

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

for feature, coefficient in zip(X_train.columns, mlr_model.coef_):
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
            svr_model = make_svr(c, epsilon, gamma)

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
svr_results_df.to_csv(PROJECT_DIR / "svr_tuning_results.csv", index=False)

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
ann_results_df.to_csv(PROJECT_DIR / "ann_tuning_results.csv", index=False)

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
print("=" * 65)

# ============================================================
# 7. Final Test Evaluation
# ============================================================

# Refit all three models on the combined training and validation data.
# The independent test set is used only for final performance evaluation.
final_models = {
    "MLR": LinearRegression(),
    "SVR": make_svr(best_svr_settings[0], best_svr_settings[1], best_svr_settings[2]),
    "ANN": make_ann(hidden_layers=best_ann_settings[0], alpha=best_ann_settings[1], learning_rate=best_ann_settings[2])
}

final_results = []
final_predictions = {}

print("\n[7] FINAL TEST EVALUATION")
print("-" * 65)
print(f"Final training samples : {len(X_temp):,} (85%)")
print(f"Independent test samples: {len(X_test):,} (15%)")

for model_name, model in final_models.items():
    start_time = time.perf_counter()
    model.fit(X_temp, y_temp)
    train_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    test_pred = model.predict(X_test)
    predict_time = time.perf_counter() - start_time

    mae, rmse, r2 = evaluate_model(y_test, test_pred)
    final_predictions[model_name] = test_pred
    final_results.append({
        "Model": model_name,
        "MAE (MW)": mae,
        "RMSE (MW)": rmse,
        "R2": r2,
        "Train Time (s)": train_time,
        "Predict Time (s)": predict_time
    })
    print(f"{model_name:<3} | MAE: {mae:.4f} MW | RMSE: {rmse:.4f} MW | R²: {r2:.4f}")

final_results_df = pd.DataFrame(final_results).sort_values("RMSE (MW)")
final_results_df.to_csv(PROJECT_DIR / "model_performance_results.csv", index=False)

print("\n" + "=" * 65)
print("                     FINAL TEST COMPARISON")
print("=" * 65)
print(final_results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print(f"\nSelected model         : {best_model_name} (based on validation RMSE)")
print(f"Lowest test RMSE       : {final_results_df.iloc[0]['Model']}")
print("Detailed results       : model_performance_results.csv")


# ============================================================
# 8. Final Test Performance Visualisation
# ============================================================

print("\n[8] FINAL TEST VISUALISATION")
print("-" * 65)

# Compare final test RMSE across the three regression models.
fig, ax = plt.subplots(figsize=(7, 5))
model_colors = {"MLR": "#4472C4", "SVR": "#ED7D31", "ANN": "#70AD47"}

ax.bar(final_results_df["Model"], final_results_df["RMSE (MW)"], color=[model_colors[model] for model in final_results_df["Model"]])
ax.set_ylabel("Test RMSE (MW)")
ax.set_xlabel("Regression Model")
ax.set_title("Final Test RMSE Comparison")
ax.set_ylim(0, final_results_df["RMSE (MW)"].max() * 1.2)
for i, rmse in enumerate(final_results_df["RMSE (MW)"]):
    ax.text(i, rmse, f"{rmse:.3f}", ha="center", va="bottom")
fig.tight_layout()
fig.savefig(PROJECT_DIR / "model_rmse_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("RMSE comparison        : model_rmse_comparison.png")

# Plot actual versus predicted output for each model.
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
minimum = min(y_test.min(), *(pred.min() for pred in final_predictions.values()))
maximum = max(y_test.max(), *(pred.max() for pred in final_predictions.values()))
for ax, (model_name, prediction) in zip(axes, final_predictions.items()):
    ax.scatter(y_test, prediction, s=10, alpha=0.35)
    ax.plot([minimum, maximum], [minimum, maximum], "r--", linewidth=1.5)
    ax.set_title(model_name)
    ax.set_xlabel("Actual PE (MW)")
    ax.set_ylabel("Predicted PE (MW)")
    ax.set_xlim(minimum, maximum)
    ax.set_ylim(minimum, maximum)
fig.suptitle("Actual vs Predicted Power Output (Final Test Set)")
fig.tight_layout()
fig.savefig(PROJECT_DIR / "model_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Actual vs predicted    : model_actual_vs_predicted.png")

# Residual = actual output minus predicted output.
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, (model_name, prediction) in zip(axes, final_predictions.items()):
    residuals = y_test.to_numpy() - prediction
    ax.scatter(prediction, residuals, s=10, alpha=0.35)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_title(model_name)
    ax.set_xlabel("Predicted PE (MW)")
    ax.set_ylabel("Residual (MW)")
fig.suptitle("Residuals of Regression Models (Final Test Set)")
fig.tight_layout()
fig.savefig(PROJECT_DIR / "model_residual_plots.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Residual plots         : model_residual_plots.png")
print("=" * 65)