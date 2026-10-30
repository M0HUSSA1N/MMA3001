"""
MMA3001 Project: Reusable Model Functions

Purpose:
    Provide the reusable data-processing, evaluation, and model-construction
    functions used to predict combined cycle power plant electrical output.
    The main analysis and its visualisations are implemented in main.py.

Functions:
    load_and_clean_data() - Load and validate the Excel dataset, then remove duplicate records before data splitting.
    split_dataset()       - Create reproducible training, validation, and independent test sets (70% / 15% / 15%).
    evaluate_model()      - Calculate MAE, RMSE, and R² for regression results.
    make_svr()            - Create a scaled RBF Support Vector Regression model.
    make_ann()            - Create a scaled feedforward neural network with reproducible training and early stopping.

Implementation notes:
    Scaling is included inside the SVR and ANN pipelines so that the scaler
    is fitted only when the corresponding model is fitted. During tuning in
    main.py, this prevents information from the validation and test sets
    from entering the training-data preprocessing stage.

    This module builds and evaluates models; it does not tune hyperparameters
    or choose the final model. Those decisions are made in main.py using
    validation RMSE, before an independent test-set evaluation.

Usage:
    Imported by main.py for analysis and by test_main.py for automated tests.
    These functions are documented separately using pdoc.
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
    Load, check, and clean the power plant dataset before model development.

    The dataset must contain four input features (AT, V, AP, RH) and the
    target (PE). Missing, non-numeric, or infinite values are rejected rather
    than silently replaced. Exact duplicate rows are removed before splitting
    to reduce the risk of identical observations appearing in different sets.

    This function validates numerical data but does not establish whether a
    particular observation is physically realistic or within the operating
    domain of the power plant.

    Args:
        dataset_path: Path or path-like value for the Excel workbook. 
        The required columns are AT (°C), V (cm Hg), AP (mbar), RH (%), and PE (MW).

    Returns:
        df: DataFrame containing the five required columns, with duplicate rows removed and the index reset.
        original_samples: Row count before duplicate removal.
        duplicate_count: Number of exact duplicate rows removed.
        missing_values: Number of missing entries found (zero for a successfully returned dataset, since missing data raises an error).

    Raises:
        FileNotFoundError: If the dataset path does not identify a file.
        ValueError: If required columns are missing, numeric conversion fails, missing or non-finite values occur, or no rows remain.
    """
    dataset_path = Path(dataset_path)

    # Fail early with a clear error instead of relying on pandas' file error.
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Use the first worksheet, matching the supplied UCI Excel workbook.
    df = pd.read_excel(dataset_path, sheet_name=0)

    # All four predictors and the power-output target are required by the models.
    required_columns = ["AT", "V", "AP", "RH", "PE"]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Fix the feature order and exclude unrelated columns, if any are present.
    df = df[required_columns].copy()

    # Raise on invalid text instead of silently coercing it into missing data.
    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    # Missing measurements should not be silently filled or dropped because either choice could change the observations used for model comparison.
    missing_values = int(df.isnull().sum().sum())

    if missing_values > 0:
        raise ValueError(
            f"Dataset contains {missing_values} missing values."
        )

    # Infinity passes ordinary numeric conversion but is unsuitable for fitting.
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("Dataset contains infinite values.")

    original_samples = len(df)
    duplicate_count = int(df.duplicated().sum())

    # Remove identical rows before splitting to limit duplicate leakage between training, validation, and test subsets.
    df = df.drop_duplicates().reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid observations remain after cleaning.")

    return df, original_samples, duplicate_count, missing_values


def split_dataset(df):
    """
    Create reproducible, non-overlapping training, validation, and test sets.

    The test set is reserved first and is not used for tuning. The remaining
    observations are then divided into training and validation subsets.
    The second test_size is relative to the remaining 85%, so 0.15 / 0.85
    produces approximately 15% validation data overall.

    Args:
        df: Cleaned DataFrame containing AT, V, AP, RH, and PE.

    Returns:
        X_temp: Combined training and validation input features (~85%).
        X_train: Training input features (~70%).
        X_val: Validation input features (~15%).
        X_test: Independent test input features (~15%).
        y_temp: Combined training and validation PE values (~85%).
        y_train: Training PE values (~70%).
        y_val: Validation PE values (~15%).
        y_test: Independent test PE values (~15%).

    Notes:
        random_state=42 makes the split reproducible for the same input data
        and software environment. This is a random, not chronological, split.
    """
    # The predictors are measured inputs; PE is the output to be estimated.
    X = df[["AT", "V", "AP", "RH"]]
    y = df["PE"]

    # Reserve the untouched 15% test set before any model selection.
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Allocate 15% of the original dataset to validation and 70% to training.
    # The denominator accounts for the 15% already held out for final testing.
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15 / 0.85, random_state=42)

    # Returning X_temp/y_temp permits a final refit on the full 85% development subset after hyperparameters have been chosen using validation data.
    return (
        X_temp, X_train, X_val, X_test,
        y_temp, y_train, y_val, y_test
    )


def evaluate_model(y_true, y_pred):
    """
    Calculate complementary regression metrics for electrical power output.

    MAE describes average absolute error; RMSE penalises large errors more
    heavily; R² measures explained variation relative to predicting the mean.
    These metrics describe the supplied observations and do not establish
    reliability outside the evaluated operating conditions.

    Args:
        y_true: Measured electrical power output values (MW).
        y_pred: Model-predicted electrical power output values (MW).

    Returns:
        mae: Mean Absolute Error (MW).
        rmse: Root Mean Squared Error (MW).
        r2: Coefficient of determination (dimensionless).
    """
    mae = mean_absolute_error(y_true, y_pred)

    # Take the square root of MSE so that RMSE is expressed in MW, not MW².
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


def make_svr(c, epsilon, gamma):
    """
    Construct a StandardScaler + RBF Support Vector Regression pipeline.

    The RBF kernel compares distances between input observations. Scaling
    prevents variables with larger numerical ranges from dominating these
    distances. The pipeline fits its scaler on whichever dataset is passed
    to .fit(), preventing validation/test leakage during model tuning.

    Args:
        c: Positive regularisation parameter controlling the penalty for
            predictions outside the epsilon-insensitive region.
        epsilon: Width of the error-insensitive region in target units (MW).
        gamma: RBF kernel coefficient, supplied as a number or 'scale'.

    Returns:
        svr_model: Unfitted pipeline containing StandardScaler and RBF SVR.

    Notes:
        Model fitting and hyperparameter selection take place in main.py.
        Larger epsilon values can reduce the number of support vectors, but
        their effect on accuracy must be assessed using validation data.
    """
    # Place scaling in the pipeline so each candidate learns its own scaling parameters exclusively from the observations used to fit that candidate.
    svr_model = make_pipeline(
        StandardScaler(),
        SVR(kernel="rbf", C=c, epsilon=epsilon, gamma=gamma)
    )

    return svr_model


def make_ann(hidden_layers, alpha, learning_rate):
    """
    Construct a scaled feedforward neural network regression pipeline.

    The ANN uses ReLU activations and Adam optimisation. Standardisation helps
    gradient-based training, while L2 regularisation and early stopping help
    limit overfitting. The random seed fixes the ANN initialisation and its
    internal validation split for reproducibility.

    Args:
        hidden_layers: Tuple giving the number of neurons in each hidden layer, e.g. (50, 25) for two hidden layers.
        alpha: L2 regularisation strength; larger values penalise large weights more strongly.
        learning_rate: Initial Adam learning rate, passed as learning_rate_init to MLPRegressor.

    Returns:
        ann_model: Unfitted pipeline containing StandardScaler and MLPRegressor.

    Notes:
        Early stopping reserves 15% of the data passed to .fit() for
        internal training monitoring. This differs from the separate 15%
        validation subset used in main.py for comparing configurations.
        Training stops after sufficient lack of improvement or after the
        maximum of 2,500 iterations; the actual iteration count may vary.
    """
    # Keep feature standardisation inside the pipeline to avoid fitting a scaler on validation or test data during model selection.
    ann_model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=learning_rate,
            max_iter=2500,
            # Reserve part of each fit's data for early-stopping monitoring.
            early_stopping=True,
            validation_fraction=0.15,
            # Stop if the monitored score does not improve sufficiently.
            n_iter_no_change=35,
            tol=0.00001,
            random_state=42
        )
    )

    return ann_model