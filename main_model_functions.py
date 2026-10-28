
"""
MMA3001 Project: Reusable Model Functions

Description:
    This module provides reusable functions for the combined cycle power
    plant output prediction project.

    Functions include dataset loading and cleaning, training-validation-test
    splitting, regression performance evaluation, and the creation of
    Support Vector Regression (SVR) and Artificial Neural Network (ANN) models.

Usage:
    Functions are imported and used by main.py and test_main.py.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def load_and_clean_data(dataset_path):
    """
    Load and validate the power plant dataset before model training.

    Args:
        dataset_path: Path to the Excel dataset containing AT, V, AP, RHand PE columns.

    Returns:
        df: Cleaned dataset containing the required columns.
        original_samples: Number of observations before cleaning.
        duplicate_count: Number of duplicate observations removed.
        missing_values: Number of missing values detected.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If required columns are missing, values are invalid, or no observations remain after cleaning.
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_excel(dataset_path, sheet_name=0)

    # Verify the required input features and target variable.
    required_columns = ["AT", "V", "AP", "RH", "PE"]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[required_columns].copy()

    # Reject non-numeric or missing observations.
    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    missing_values = int(df.isnull().sum().sum())

    if missing_values > 0:
        raise ValueError(
            f"Dataset contains {missing_values} missing values."
        )

    # Reject infinite numerical values.
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("Dataset contains infinite values.")

    original_samples = len(df)
    duplicate_count = int(df.duplicated().sum())

    # Avoid repeated identical observations in the dataset.
    df = df.drop_duplicates().reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid observations remain after cleaning.")

    return df, original_samples, duplicate_count, missing_values

def split_dataset(df):
    """
    Split the cleaned dataset into training, validation and testing sets.

    Args:
        df: Cleaned power plant dataset containing AT, V, AP, RH and PE.

    Returns:
        X_temp: Combined training and validation input features (85%).
        X_train: Training input features (70%).
        X_val: Validation input features (15%).
        X_test: Testing input features (15%).
        y_temp: Combined training and validation target values (85%).
        y_train: Training target values (70%).
        y_val: Validation target values (15%).
        y_test: Testing target values (15%).
    """
    X = df[["AT", "V", "AP", "RH"]]
    y = df["PE"]

    # Reserve an independent test set before model selection.
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Use the remaining observations for training and validation.
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15 / 0.85, random_state=42)

    return (
        X_temp, X_train, X_val, X_test,
        y_temp, y_train, y_val, y_test
    )

def evaluate_model(y_true, y_pred):
    """
    Evaluate regression accuracy using actual and predicted power output.

    Args:
        y_true: Actual electrical power output values (MW).
        y_pred: Predicted electrical power output values (MW).

    Returns:
        mae: Mean Absolute Error (MW).
        rmse: Root Mean Squared Error (MW).
        r2: Coefficient of determination.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2

def make_svr(c, epsilon, gamma):
    """
    Create an RBF Support Vector Regression model.

    Feature standardisation is included because the RBF kernel depends on distances between observations.

    Args:
        c: Regularisation parameter controlling the penalty for errors.
        epsilon: Width of the epsilon-insensitive error region.
        gamma: RBF kernel coefficient controlling model sensitivity.

    Returns:
        svr_model: Pipeline containing a StandardScaler and SVR.
    """
    svr_model = make_pipeline(
        StandardScaler(),
        SVR(kernel="rbf", C=c, epsilon=epsilon, gamma=gamma)
    )

    return svr_model

def make_ann(hidden_layers, alpha, learning_rate):
    """
    Create an ANN regression model with feature standardisation.

    Args:
        hidden_layers: Tuple specifying neurons in each hidden layer.
        alpha: L2 regularisation parameter used during ANN training.
        learning_rate: Initial learning rate for the Adam optimiser.

    Returns:
        ann_model: Pipeline containing a StandardScaler and MLPRegressor.
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