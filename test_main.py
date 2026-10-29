
"""
MMA3001 Project: Software Testing

Description:
    This module contains automated tests for the combined cycle power
    plant output prediction project.

    The tests verify dataset structure, data cleaning, data splitting,
    model performance results, baseline comparison, regression metrics,
    SVR and ANN model configuration, and invalid dataset handling.

Testing Framework:
    pytest

Execution:
    python -m pytest test_main.py -v
"""

from pathlib import Path

import pandas as pd
import pytest

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from main_model_functions import (
    load_and_clean_data,
    split_dataset,
    evaluate_model,
    make_svr,
    make_ann
)

# ============================================================
# Project Files
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "Folds5x2_pp.xlsx"
RESULTS_FILE = PROJECT_DIR / "results" / "model_performance_results.csv"

FEATURES = ["AT", "V", "AP", "RH"]
TARGET = "PE"

# ============================================================
# 1. Dataset Structure
# ============================================================

def test_dataset_columns():
    """
    The test_dataset_columns function verifies that the dataset contains all required input features and the target variable.

    Verification: 
        Confirms that AT, V, AP, RH and PE are present in the dataset.
    """
    df = pd.read_excel(DATA_FILE, sheet_name=0)
    assert all(column in df.columns for column in FEATURES + [TARGET])

# ============================================================
# 2. Dataset Cleaning
# ============================================================

def test_dataset_cleaning():
    """
    Verify that the project's cleaning function processes the dataset.

    Verification:
        Confirms the original sample count, duplicate removal,
        cleaned sample count and absence of missing values.
    """
    df, original_samples, duplicate_count, missing_values = (
        load_and_clean_data(DATA_FILE)
    )

    assert original_samples == 9568
    assert duplicate_count == 41
    assert len(df) == 9527
    assert df.duplicated().sum() == 0
    assert df.isnull().sum().sum() == 0
    assert missing_values == 0

# ============================================================
# 3. Data Splitting
# ============================================================

def test_data_split():
    """
    Verify that the project's splitting function creates
    separate training, validation and testing datasets.

    Verification:
        Confirms the partition sizes, feature dimensions,
        target alignment and absence of overlapping indices.
    """
    df, _, _, _ = load_and_clean_data(DATA_FILE)

    X_temp, X_train, X_val, X_test, y_temp, y_train, y_val, y_test = (
        split_dataset(df)
    )

    assert len(X_temp) == 8097
    assert len(X_train) == 6668
    assert len(X_val) == 1429
    assert len(X_test) == 1430

    assert X_train.shape[1] == 4
    assert X_val.shape[1] == 4
    assert X_test.shape[1] == 4

    assert set(X_train.index).isdisjoint(X_val.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_val.index).isdisjoint(X_test.index)

    assert X_train.index.equals(y_train.index)
    assert X_val.index.equals(y_val.index)
    assert X_test.index.equals(y_test.index)
    assert X_temp.index.equals(y_temp.index)

# ============================================================
# 4. Final Results File
# ============================================================

def test_final_results_exist():
    """
    The test_final_results_exist function verifies that the main program has generated the final model performance results file.

    Verification:
        Confirms that model_performance_results.csv exists.
    """
    assert RESULTS_FILE.is_file()

# ============================================================
# 5. Model Performance Metrics
# ============================================================

def test_final_results():
    """
    The test_final_results function verifies that the final results contain all three models and valid performance metrics.

    Verification:
        Confirms that MLR, SVR and ANN are present, with non-negative MAE and RMSE values and R2 values no greater than one.
    """
    results = pd.read_csv(RESULTS_FILE)

    assert set(results["Model"]) == {"MLR", "SVR", "ANN"}
    assert len(results) == 3

    for column in ["MAE (MW)", "RMSE (MW)", "R2"]:
        assert results[column].notna().all()

    assert (results["MAE (MW)"] >= 0).all()
    assert (results["RMSE (MW)"] >= 0).all()
    assert (results["R2"] <= 1).all()

# ============================================================
# 6. Baseline Comparison
# ============================================================

def test_svr_outperforms_baseline():
    """
    The test_svr_outperforms_baseline function verifies that the selected SVR model improves upon the MLR baseline.

    Verification:
        Confirms that SVR achieves a lower final test RMSE than MLR.
    """
    results = pd.read_csv(RESULTS_FILE)
    results = results.set_index("Model")
    svr_rmse = float(results.loc["SVR", "RMSE (MW)"])
    mlr_rmse = float(results.loc["MLR", "RMSE (MW)"])

    assert svr_rmse < mlr_rmse

# ============================================================
# 7. Test Model Evaluation Function
# ============================================================

def test_evaluate_model():
    """
    Verify that the model evaluation function calculates the expected regression metrics.

    Verification:
        MAE, RMSE and R2 match independently calculated values.
    """
    y_true = [10, 20, 30]
    y_pred = [12, 18, 30]
    mae, rmse, r2 = evaluate_model(y_true, y_pred)

    assert abs(mae - (4 / 3)) < 1e-6
    assert abs(rmse - (8 / 3) ** 0.5) < 1e-6
    assert abs(r2 - 0.96) < 1e-6

# ============================================================
# 8. Test ANN Model Creation Function
# ============================================================

def test_make_ann():
    """
    Verify that the ANN creation function produces the expected preprocessing and regression pipeline.

    Verification:
        The pipeline contains a StandardScaler and MLPRegressor, with the specified hyperparameters.
    """
    ann_model = make_ann((50, 25), 0.01, 0.001)

    assert isinstance(ann_model.steps[0][1], StandardScaler)
    assert isinstance(ann_model.steps[1][1], MLPRegressor)

    mlp = ann_model.steps[1][1]

    assert mlp.hidden_layer_sizes == (50, 25)
    assert mlp.alpha == 0.01
    assert mlp.learning_rate_init == 0.001
    assert mlp.early_stopping is True
    assert mlp.random_state == 42

# ============================================================
# 9. Test SVR Model Creation Function
# ============================================================

def test_make_svr():
    """
    Verify that the SVR creation function produces the
    expected preprocessing and regression pipeline.

    Verification:
        Confirms the scaler, RBF kernel and hyperparameters.
    """
    svr_model = make_svr(100, 1.0, 0.3)

    assert isinstance(svr_model.steps[0][1], StandardScaler)
    assert isinstance(svr_model.steps[1][1], SVR)

    svr = svr_model.steps[1][1]

    assert svr.kernel == "rbf"
    assert svr.C == 100
    assert svr.epsilon == 1.0
    assert svr.gamma == 0.3


# ============================================================
# 10. Missing Dataset Columns
# ============================================================

def test_missing_dataset_column(tmp_path):
    """
    Verify that the loading function rejects a dataset
    with a missing required column.

    Verification:
        Confirms that ValueError is raised when PE is absent.
    """
    df = pd.DataFrame({
        "AT": [20],
        "V": [40],
        "AP": [1010],
        "RH": [60]
    })

    invalid_file = tmp_path / "missing_column.xlsx"
    df.to_excel(invalid_file, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_and_clean_data(invalid_file)


# ============================================================
# 11. Missing Dataset Values
# ============================================================

def test_missing_dataset_value(tmp_path):
    """
    Verify that the loading function rejects missing values.

    Verification:
        Confirms that ValueError is raised when an
        observation contains a missing value.
    """
    df = pd.DataFrame({
        "AT": [None],
        "V": [40],
        "AP": [1010],
        "RH": [60],
        "PE": [450]
    })

    invalid_file = tmp_path / "missing_value.xlsx"
    df.to_excel(invalid_file, index=False)

    with pytest.raises(ValueError, match="missing values"):
        load_and_clean_data(invalid_file)


# ============================================================
# 12. Non-Numeric Dataset Values
# ============================================================

def test_non_numeric_dataset_value(tmp_path):
    """
    Verify that the loading function rejects non-numeric inputs.

    Verification:
        Confirms that ValueError is raised when a numerical
        feature contains text instead of a number.
    """
    df = pd.DataFrame({
        "AT": ["invalid"],
        "V": [40],
        "AP": [1010],
        "RH": [60],
        "PE": [450]
    })

    invalid_file = tmp_path / "non_numeric_value.xlsx"
    df.to_excel(invalid_file, index=False)

    with pytest.raises(ValueError):
        load_and_clean_data(invalid_file)