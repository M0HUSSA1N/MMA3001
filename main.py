"""
MMA3001 Project: Combined Cycle Power Plant Output Prediction

Description:
    This program predicts electrical power output using Multiple Linear
    Regression (MLR), Support Vector Regression (SVR) and Artificial
    Neural Networks (ANN).

    The workflow includes dataset validation, exploratory data analysis,
    hyperparameter tuning, model comparison and independent testing.
    Reusable functions are imported from main_model_functions.py.

    Additional analyses investigate computational complexity, approximate
    floating-point operations, accuracy versus computational cost,
    prediction errors across ambient-temperature regions, and the
    physical plausibility of model predictions.

Outputs:
    CSV files containing hyperparameter tuning results, model performance,
    computational cost estimates and temperature-region evaluation.

    PNG figures showing correlations, model predictions, residuals,
    accuracy-cost trade-offs and temperature-dependent model behaviour.

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
import numpy as np

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
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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
fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=300, bbox_inches="tight")
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
svr_results_df.to_csv(RESULTS_DIR / "svr_tuning_results.csv", index=False)

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
ann_results_df.to_csv(RESULTS_DIR / "ann_tuning_results.csv", index=False)

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
final_results_df.to_csv(RESULTS_DIR / "model_performance_results.csv", index=False)

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
fig.savefig(FIGURES_DIR / "model_rmse_comparison.png", dpi=300, bbox_inches="tight")
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
fig.savefig(FIGURES_DIR / "model_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
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
fig.savefig(FIGURES_DIR / "model_residual_plots.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Residual plots         : model_residual_plots.png")
print("=" * 65)

# ============================================================
# 9. Computational Efficiency and Accuracy Trade-off
# ============================================================

print("\n[9] COMPUTATIONAL EFFICIENCY AND ACCURACY TRADE-OFF")
print("-" * 65)

# Number of input features used by each regression model.
n_features = X_test.shape[1]

# Retrieve the fitted ANN architecture and parameter counts.
fitted_ann = final_models["ANN"].named_steps["mlpregressor"]

ann_weights = sum(weights.size for weights in fitted_ann.coefs_)
ann_biases = sum(biases.size for biases in fitted_ann.intercepts_)
ann_parameters = ann_weights + ann_biases

# Retrieve the number of support vectors in the final fitted SVR.
fitted_svr = final_models["SVR"].named_steps["svr"]
svr_support_vectors = len(fitted_svr.support_)

# Estimate basic floating-point arithmetic per prediction.
# MLR: one multiplication and one addition per input feature.
mlr_flops = 2 * n_features

# ANN: approximately one multiplication and one addition per weight.
ann_flops = 2 * ann_weights

# RBF-SVR: distance, kernel scaling and weighted summation.
# Exponential evaluations are counted separately.
svr_flops = svr_support_vectors * (3 * n_features + 2)
svr_exponentials = svr_support_vectors

flop_estimates = {
    "MLR": mlr_flops,
    "ANN": ann_flops,
    "SVR": svr_flops
}

# Combine computational estimates with measured test performance.
cost_results = []

for model_name in ["MLR", "ANN", "SVR"]:

    model_row = final_results_df[
        final_results_df["Model"] == model_name
    ].iloc[0]

    predict_time = model_row["Predict Time (s)"]

    cost_results.append({
        "Model": model_name,
        "Test MAE (MW)": model_row["MAE (MW)"],
        "Test RMSE (MW)": model_row["RMSE (MW)"],
        "Test R2": model_row["R2"],
        "Train Time (s)": model_row["Train Time (s)"],
        "Predict Time (s)": predict_time,
        "Amortized Predict Time (ms/sample)": (
            predict_time * 1000 / len(X_test)
        ),
        "Approx Basic FLOPs per Prediction": flop_estimates[model_name]
    })

cost_df = pd.DataFrame(cost_results)

cost_df.to_csv(RESULTS_DIR / "computational_cost_analysis.csv",index=False)

print("Estimated arithmetic operations per prediction:")
print(f"MLR                   : {mlr_flops:,}")
print(f"ANN                   : {ann_flops:,}")
print(f"SVR                   : {svr_flops:,}")

print(f"\nANN trainable parameters : {ann_parameters:,}")
print(f"ANN weights              : {ann_weights:,}")
print(f"ANN biases               : {ann_biases:,}")

print(f"\nFinal SVR support vectors: {svr_support_vectors:,}")
print(f"SVR exponentials/sample  : {svr_exponentials:,}")

print("\nAccuracy and computational cost:")
print(cost_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

# Calculate the accuracy improvement and prediction time trade-off.
cost_indexed = cost_df.set_index("Model")

mlr_rmse = cost_indexed.loc["MLR", "Test RMSE (MW)"]
ann_rmse = cost_indexed.loc["ANN", "Test RMSE (MW)"]
svr_rmse = cost_indexed.loc["SVR", "Test RMSE (MW)"]

ann_time = cost_indexed.loc["ANN", "Predict Time (s)"]
svr_time = cost_indexed.loc["SVR", "Predict Time (s)"]

svr_improvement_mlr = (mlr_rmse - svr_rmse) / mlr_rmse * 100
svr_improvement_ann = (ann_rmse - svr_rmse) / ann_rmse * 100

print(f"\nSVR RMSE improvement over MLR : {svr_improvement_mlr:.2f}%")
print(f"SVR RMSE improvement over ANN : {svr_improvement_ann:.2f}%")

if ann_time > 0:
    print(f"SVR / ANN prediction time ratio: {svr_time / ann_time:.1f}x")

# Visualise the relationship between prediction accuracy and runtime.
fig, ax = plt.subplots(figsize=(8, 5))

comparison_colors = {
    "MLR": "#4472C4",
    "SVR": "#ED7D31",
    "ANN": "#70AD47"
}

for _, row in cost_df.iterrows():

    model_name = row["Model"]
    predict_time_ms = row["Predict Time (s)"] * 1000
    rmse = row["Test RMSE (MW)"]

    ax.scatter(
        predict_time_ms,
        rmse,
        color=comparison_colors[model_name],
        s=100,
        label=model_name
    )

    ax.annotate(
        model_name,
        (predict_time_ms, rmse),
        xytext=(7, 5),
        textcoords="offset points"
    )

ax.set_xscale("log")
ax.set_xlabel("Prediction Time for Test Batch (ms, logarithmic scale)")
ax.set_ylabel("Test RMSE (MW)")
ax.set_title("Prediction Accuracy vs Computational Cost")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()

fig.savefig(
    FIGURES_DIR / "accuracy_vs_computational_cost.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\nDetailed results       : computational_cost_analysis.csv")
print("Accuracy-cost plot     : accuracy_vs_computational_cost.png")
print("Note: FLOP estimates exclude scaling and nonlinear function costs.")
print("Note: Runtime values are from a single measurement per model.")
print("=" * 65)


# ============================================================
# 10. Operating-Region Performance and Physical Plausibility
# ============================================================

print("\n[10] OPERATING-REGION PERFORMANCE AND PHYSICAL PLAUSIBILITY")
print("-" * 65)

# ============================================================
# 10.1 Temperature-Region Performance and Data Coverage
# ============================================================

# Define three equal-width temperature regions using training data only.
# This avoids using test data to determine the region boundaries.
temperature_min = X_train["AT"].min()
temperature_max = X_train["AT"].max()

temperature_step = (temperature_max - temperature_min) / 3

low_boundary = temperature_min + temperature_step
high_boundary = temperature_min + 2 * temperature_step

print("\nTemperature-region boundaries:")
print(f"Low                  : AT <= {low_boundary:.2f} °C")
print(f"Moderate             : {low_boundary:.2f} < AT <= {high_boundary:.2f} °C")
print(f"High                 : AT > {high_boundary:.2f} °C")

# Count observations in each region using training and test data.
train_temperature = X_train["AT"].to_numpy()
test_temperature = X_test["AT"].to_numpy()

train_masks = {
    "Low": train_temperature <= low_boundary,
    "Moderate": (
        (train_temperature > low_boundary) &
        (train_temperature <= high_boundary)
    ),
    "High": train_temperature > high_boundary
}

test_masks = {
    "Low": test_temperature <= low_boundary,
    "Moderate": (
        (test_temperature > low_boundary) &
        (test_temperature <= high_boundary)
    ),
    "High": test_temperature > high_boundary
}

region_names = ["Low", "Moderate", "High"]
model_names = ["MLR", "ANN", "SVR"]

regional_results = []

for region in region_names:

    train_count = int(train_masks[region].sum())
    test_mask = test_masks[region]
    test_count = int(test_mask.sum())

    print(f"\n{region.upper()} TEMPERATURE REGION")
    print(f"Training observations : {train_count:,}")
    print(f"Testing observations  : {test_count:,}")

    for model_name in model_names:

        if test_count == 0:
            mae = np.nan
            rmse = np.nan
        else:
            # Evaluate the existing final predictions in this region.
            actual = y_test.to_numpy()[test_mask]
            predicted = np.asarray(
                final_predictions[model_name]
            )[test_mask]

            mae, rmse, _ = evaluate_model(actual, predicted)

        regional_results.append({
            "Temperature Region": region,
            "Model": model_name,
            "Training Samples": train_count,
            "Testing Samples": test_count,
            "Test MAE (MW)": mae,
            "Test RMSE (MW)": rmse
        })

        print(f"{model_name:<3} | MAE: {mae:.4f} MW | RMSE: {rmse:.4f} MW")

regional_df = pd.DataFrame(regional_results)

regional_df.to_csv(RESULTS_DIR / "temperature_region_performance.csv",index=False)

# Visualise RMSE differences across temperature regions.
fig, ax = plt.subplots(figsize=(9, 5))

region_positions = np.arange(len(region_names))
bar_width = 0.25

model_colors = {
    "MLR": "#4472C4",
    "SVR": "#ED7D31",
    "ANN": "#70AD47"
}

for i, model_name in enumerate(model_names):

    model_data = regional_df[
        regional_df["Model"] == model_name
    ].set_index("Temperature Region")

    rmse_values = [
        model_data.loc[region, "Test RMSE (MW)"]
        for region in region_names
    ]

    ax.bar(
        region_positions + (i - 1) * bar_width,
        rmse_values,
        width=bar_width,
        label=model_name,
        color=model_colors[model_name]
    )

# Display the sample count beneath each region.
region_labels = []

for region in region_names:
    sample_count = int(test_masks[region].sum())
    region_labels.append(f"{region}\n(n={sample_count})")

ax.set_xticks(region_positions)
ax.set_xticklabels(region_labels)

ax.set_xlabel("Ambient Temperature Region")
ax.set_ylabel("Test RMSE (MW)")
ax.set_title("Model Prediction Error Across Temperature Regions")
ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

fig.savefig(
    FIGURES_DIR / "temperature_region_rmse.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# 10.2 Physical Plausibility of Predicted Power Output
# ============================================================

print("\n[10.2] PHYSICAL PLAUSIBILITY")
print("-" * 65)

# Use the development dataset to establish the observed PE range.
# Predictions outside this range warrant inspection but are not
# automatically physically impossible.
pe_min = y_temp.min()
pe_max = y_temp.max()

print(f"Development PE range : {pe_min:.2f} to {pe_max:.2f} MW")

for model_name in model_names:

    predictions = np.asarray(final_predictions[model_name])

    outside_range = int((
        (predictions < pe_min) |
        (predictions > pe_max)
    ).sum())

    percentage = outside_range / len(predictions) * 100

    print(
        f"{model_name:<3} | Predictions outside range: "
        f"{outside_range:,} ({percentage:.2f}%)"
    )

# Investigate how predicted power output responds to temperature.
# Hold V, AP and RH constant at their training-set median values.
# This is a sensitivity check, not a causal thermodynamic simulation.
reference_inputs = X_train.median()

temperature_values = np.linspace(
    X_train["AT"].quantile(0.05),
    X_train["AT"].quantile(0.95),
    80
)

sweep_inputs = pd.DataFrame({
    column: np.full(
        len(temperature_values),
        float(reference_inputs[column])
    )
    for column in X_train.columns
})

sweep_inputs["AT"] = temperature_values

fig, ax = plt.subplots(figsize=(8, 5))

print("\nPredicted response to increasing ambient temperature:")

for model_name in model_names:

    model = final_models[model_name]
    sweep_predictions = model.predict(sweep_inputs)

    # Positive changes indicate local increases in predicted output.
    increasing_steps = int(np.sum(np.diff(sweep_predictions) > 0))

    total_change = (sweep_predictions[-1] - sweep_predictions[0])

    print(
        f"{model_name:<3} | Overall PE change: "
        f"{total_change:+.3f} MW | "
        f"Increasing intervals: {increasing_steps}"
    )

    ax.plot(
        temperature_values,
        sweep_predictions,
        label=model_name,
        color=model_colors[model_name],
        linewidth=2
    )

ax.set_xlabel("Ambient Temperature (°C)")
ax.set_ylabel("Predicted Electrical Power Output (MW)")
ax.set_title("Predicted Power Output Response to Ambient Temperature")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()

fig.savefig(
    FIGURES_DIR / "temperature_response_check.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\nDetailed results       : temperature_region_performance.csv")
print("Regional RMSE plot     : temperature_region_rmse.png")
print("Temperature response   : temperature_response_check.png")
print("Note: Temperature sweeps hold other variables at training medians.")
print("Note: Such input combinations may not represent actual operating states.")
print("=" * 65)