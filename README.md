# Combined Cycle Power Plant Electrical Output Prediction

**MMA3001: Numerical Methods and Machine Learning | Monash University Malaysia**

## Project overview

This project predicts the net electrical power output of a combined cycle power plant (CCPP) from four measured operating variables. Three regression methods are implemented and compared:

- **Multiple Linear Regression (MLR):** an interpretable, computationally inexpensive baseline.
- **Support Vector Regression (SVR):** nonlinear regression using a radial basis function (RBF) kernel.
- **Artificial Neural Network (ANN):** a feedforward multilayer perceptron for nonlinear regression.

The project compares prediction accuracy, hyperparameter configurations, computational requirements and behaviour across ambient-temperature regions. Its intended application is *offline estimation and performance assessment* for operating conditions similar to those in the historical dataset. The code is a reproducible experiment, not a real-time power plant controller or a command-line service for predicting arbitrary new inputs.

## Dataset

The [Combined Cycle Power Plant dataset](https://archive.ics.uci.edu/dataset/294/combined%2Bcycle%2Bpower%2Bplant) from the UCI Machine Learning Repository contains 9,568 observations collected during full-load operation between 2006 and 2011. The five required columns are:

| Variable | Description | Unit | Recorded range | Role |
| --- | --- | --- | --- | --- |
| `AT` | Ambient temperature | °C | 1.81–37.11 | Input |
| `V` | Exhaust vacuum | cm Hg | 25.36–81.56 | Input |
| `AP` | Atmospheric pressure | mbar | 992.89–1033.30 | Input |
| `RH` | Relative humidity | % | 25.56–100.16 | Input |
| `PE` | Net electrical power output | MW | 420.26–495.76 | Prediction target |

The program loads the first sheet of `data/Folds5x2_pp.xlsx`, checks the required columns and numerical values, and removes 41 exact duplicate observations, leaving 9,527 unique rows. It rejects missing files, missing required columns, non-numeric entries and missing/non-finite values. It does **not** fully enforce physical operating limits or verify whether all combinations of otherwise valid inputs are represented in the training data.

### Dataset location

The file must be at the following path **relative to `main.py`**:

```text
data/Folds5x2_pp.xlsx
```

If the workbook is not included in your copy of the repository, download the `.xlsx` dataset from the [UCI source](https://archive.ics.uci.edu/dataset/294/combined%2Bcycle%2Bpower%2Bplant) and place `Folds5x2_pp.xlsx` inside the `data` folder. Keep its original column names (`AT`, `V`, `AP`, `RH`, `PE`).

## Quick start

From a fresh clone, create a Python virtual environment, install `requirements.txt`, confirm that the workbook is at `data/Folds5x2_pp.xlsx`, then run `python main.py`. The script performs the complete training and evaluation workflow and saves CSV results in `results/` and figures in `figures/`. Run `python -m pytest test_main.py -v` **after** `main.py` because some tests inspect generated results.

The code uses paths relative to `main.py`, so these commands should be run from the repository root. The project is a complete analysis script, **not** an interactive prediction application.

## Repository files

```text
MMA3001/
├── main.py                         # Runs the ten-stage analysis
├── main_model_functions.py         # Dataset, splitting, evaluation and model builders
├── test_main.py                    # Automated tests (pytest)
├── README.md                       # Project documentation and reproduction steps
├── requirements.txt                # Pinned dependencies for the tested environment
├── LICENSE                         # Source-code licence
├── .gitignore                      # Excludes virtual environments and cache files
├── test_report.txt                 # Saved pytest verification output
├── data/
│   └── Folds5x2_pp.xlsx            # Original UCI dataset (CC BY 4.0)
├── results/                         # Five CSV analysis outputs
│   ├── svr_tuning_results.csv
│   ├── ann_tuning_results.csv
│   ├── model_performance_results.csv
│   ├── computational_cost_analysis.csv
│   └── temperature_region_performance.csv
├── figures/                         # Seven PNG visualisations
│   ├── correlation_heatmap.png
│   ├── model_rmse_comparison.png
│   ├── model_actual_vs_predicted.png
│   ├── model_residual_plots.png
│   ├── accuracy_vs_computational_cost.png
│   ├── temperature_region_rmse.png
│   └── temperature_response_check.png
└── docs/                            # Generated HTML API documentation + workflow overview
```


The script writes CSV results into `results/` and PNG figures into `figures/`. The `docs/` folder contains generated HTML documentation once created. Keeping source, data, results and documentation separate makes the repository easier to review. This structure assumes the output paths in `main.py` and the output checks in `test_main.py` have been updated to match.

## Getting the repository

Clone the repository using its URL from GitHub:

```bash
git clone https://github.com/M0HUSSA1N/MMA3001.git
cd MMA3001
```

Alternatively, download the repository ZIP from GitHub and extract it before running the installation commands. Work from the directory containing `main.py`.

## Environment and installation

Use Python 3 and a virtual environment. The project depends on **NumPy, pandas, scikit-learn, Matplotlib and openpyxl**; **pytest** is needed for the automated tests. Exact versions should be recorded in `requirements.txt` for reproducibility.

From a terminal opened in the repository's root directory:

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, the environment's Python executable can be used directly instead of changing the system execution policy. For example: `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

## Running the project

From the repository root, with the virtual environment activated, run:

```bash
python main.py
```

`main.py` is a sequential script. It performs the entire workflow directly when executed; **there is no `main()` function or interactive menu**. It does not need additional command-line arguments. The dataset path and output paths are resolved relative to the script's location.

The script performs these stages:

1. Load and validate the dataset; remove exact duplicates.
2. Generate a correlation analysis and heatmap.
3. Split data into training (70%), validation (15%) and independent testing (15%) subsets using random state 42.
4. Fit the MLR baseline.
5. Evaluate 24 SVR hyperparameter configurations on validation data.
6. Evaluate 12 ANN hyperparameter configurations on validation data.
7. Select the model by validation RMSE, refit all three methods on the combined training and validation data, and evaluate them on the untouched test set.
8. Generate the final prediction and residual figures.
9. Estimate computational requirements and compare accuracy with training and prediction costs.
10. Evaluate errors across ambient-temperature regions and examine the predicted temperature response for physical plausibility.

Standardisation for SVR and ANN is fitted inside scikit-learn pipelines, using training data during tuning. The test set is kept separate from model selection. The script may take some time to finish because it trains all 24 SVR and 12 ANN configurations again on each run.

## Expected results

The following values were obtained from the final reported run. Minor changes in runtime are expected across systems and executions; the fixed random state and dataset split support reproducible prediction results.

| Model | Test MAE (MW) | Test RMSE (MW) | Test R² |
| --- | ---: | ---: | ---: |
| MLR | 3.6399 | 4.6365 | 0.9268 |
| ANN | 3.2697 | 4.2314 | 0.9390 |
| **SVR** | **2.9440** | **4.0275** | **0.9447** |

The selected configurations were:

- **SVR:** RBF kernel, `C=100`, `epsilon=1.0`, `gamma=0.3`.
- **ANN:** hidden layers `(50, 25)`, `alpha=0.01`, initial learning rate `0.001`.

SVR had the lowest validation and final test RMSE. ANN required substantially less inference computation in the reported run, so it remains a credible alternative where prediction frequency or computational resources are important. Timings are hardware-dependent and were not obtained from repeated controlled benchmarks.

### Computational performance

The final fitted models produced the following **single-run** timing measurements for the batch of 1,430 test observations:

| Model | Final training time (s) | Batch prediction time (s) | Approx. basic arithmetic operations per prediction |
| --- | ---: | ---: | ---: |
| MLR | 0.0018 | 0.0009 | 8 |
| ANN | 4.4349 | 0.0011 | 2,950 |
| SVR | 2.7432 | 0.3596 | 86,982 |

The final SVR retained 6,213 support vectors, whereas the selected ANN contained 1,551 trainable parameters. The simplified operation estimates exclude some work, including RBF exponential evaluations, ANN activation functions and feature scaling. Runtime measurements depend on hardware and execution conditions; these results are **not** controlled performance benchmarks.

### Engineering checks

SVR achieved the lowest test RMSE across the low-, moderate- and high-temperature regions (4.7091, 3.5802 and 4.0203 MW, respectively). All three models predicted decreasing electrical output during an ambient-temperature sweep with the other inputs fixed at training-data medians. SVR produced one test prediction outside the observed development-data PE range (420.26–495.76 MW). These checks provide additional diagnostic evidence but do not guarantee physical validity or performance outside the measured domain.

## Generated outputs

Running `main.py` creates or overwrites the following generated files (within `results/` and `figures/`):

**CSV results**

| File | Contents |
| --- | --- |
| `results/svr_tuning_results.csv` | Validation metrics, support-vector counts and timings for 24 SVR configurations |
| `results/ann_tuning_results.csv` | Training/validation metrics, iterations and timings for 12 ANN configurations |
| `results/model_performance_results.csv` | Independent test metrics and final-model timings |
| `results/computational_cost_analysis.csv` | Test accuracy, prediction time and approximate arithmetic-operation estimates |
| `results/temperature_region_performance.csv` | Sample counts and test MAE/RMSE for low, moderate and high ambient-temperature regions |

**PNG figures**

| File | Contents |
| --- | --- |
| `figures/correlation_heatmap.png` | Dataset correlations |
| `figures/model_rmse_comparison.png` | Final test RMSE by model |
| `figures/model_actual_vs_predicted.png` | Predicted versus measured PE |
| `figures/model_residual_plots.png` | Final test residuals |
| `figures/accuracy_vs_computational_cost.png` | Accuracy versus prediction-time comparison |
| `figures/temperature_region_rmse.png` | Test RMSE across ambient-temperature regions |
| `figures/temperature_response_check.png` | Predicted PE response as ambient temperature increases with other inputs fixed |

Existing CSV/PNG files with these names are overwritten when the analysis is run again. Model errors should remain consistent for the same dataset and environment, but runtime measurements can vary.

### Preview of the generated figures

These are selected figure previews. All seven PNG files are available in `figures/`; they are generated by the project rather than downloaded images.

When the generated PNGs are included in the repository, GitHub displays these figures directly:

**Final test comparison**

![RMSE comparison of MLR, ANN and SVR](figures/model_rmse_comparison.png)

**Prediction residuals**

![Residual plots for all three regression models](figures/model_residual_plots.png)

**Operating-region accuracy**

![RMSE comparison across ambient-temperature regions](figures/temperature_region_rmse.png)

**Physical plausibility check**

![Predicted electrical output versus ambient temperature](figures/temperature_response_check.png)

The image previews work once the PNGs have been generated inside `figures/` and committed to the repository. The complete list of all seven figures appears above.

## Automated testing

After running `main.py` once to generate the result files, execute the test suite from the repository root:

```bash
python -m pytest test_main.py -v
```

To save the test output to a report:

```bash
python -m pytest test_main.py -v > test_report.txt
```

The project has **12 pytest tests**, covering dataset validation/cleaning, disjoint data splitting, evaluation metrics, model construction, invalid inputs and selected output checks. All 12 tests passed in the final reported verification. Some tests check files produced by `main.py`, so **run `main.py` before pytest** on a fresh clone.

Passing tests verify the behaviour covered by these cases; they do not independently validate every supplementary analysis, prove physical accuracy, or guarantee reliability for unsupported operating conditions.

### How to verify the reproduced results

On a fresh clone, verify that the script completes all ten stages, produces five CSV files in `results/` and seven PNG files in `figures/`, and reports the test RMSE values in the table above (subject to dependency/version compatibility). Then run the test suite and confirm all 12 tests pass. The split uses `random_state=42`, while measured execution times will vary with hardware and software environment. Some test cases depend on the CSV results existing, so their order matters.

## Source-code documentation

The project includes docstrings and inline comments. Reusable functions are defined in `main_model_functions.py`, while `main.py` coordinates the ten-stage analysis. HTML documentation for the helper module can be generated using **pdoc**:

```bash
python -m pip install pdoc
python -m pdoc main_model_functions -o docs
```

Open the generated `docs/main_model_functions.html` in a browser. A separate overview in `docs/` can explain the high-level workflow in `main.py`. **Do not import `main.py` into pdoc directly**: it is a sequential executable script, so importing it would run the training workflow. The submitted `docs/` directory should contain the generated pages and any main-script overview actually prepared for the project.

## Software environment

Python 3 and the packages listed in `requirements.txt` are needed to reproduce the workflow. The core packages are NumPy, pandas, scikit-learn, Matplotlib and openpyxl; pytest is used for testing. The optional HTML generator is pdoc. The versions in the final `requirements.txt` should match a working, tested environment rather than being guessed.

To inspect the Python version and installed dependency versions:

```bash
python --version
python -m pip show numpy pandas scikit-learn matplotlib openpyxl pytest
```

The project was developed in VS Code on Windows; the code uses script-relative data and output paths to avoid dependence on a particular user's computer. Exact training and prediction times may change with the operating system, hardware and installed library versions.

## Troubleshooting

- **`FileNotFoundError` for `Folds5x2_pp.xlsx`:** verify the original file is at `data/Folds5x2_pp.xlsx` and is named exactly as shown.
- **`ModuleNotFoundError`:** activate the intended virtual environment and install `requirements.txt` using the same Python interpreter that runs `main.py`.
- **Tests cannot find a generated CSV:** run `main.py` first and check that the CSVs are under `results/`; tests need to reference the same directory.
- **README image previews are missing:** ensure the PNGs are committed under `figures/` with the exact referenced filenames.
- **Runtime values differ:** timing depends on machine load, hardware and installed libraries. Compare prediction-error metrics separately from measured timings.

## Scope and limitations

- The models were developed using one historical CCPP dataset representing **full-load** operation. Generalisation to other plants, later years and part-load operation has not been demonstrated.
- Validation uses a single reproducible random split; cross-validation or chronological evaluation would provide additional evidence of generalisation.
- Inputs beyond the historical operating domain, or unusual combinations of otherwise valid inputs, may lead to unreliable extrapolation.
- Regional analysis showed larger errors at low ambient temperatures. The models should not be treated as thermodynamic simulators or used for safety-critical control without further independent validation.
- Arithmetic-operation estimates omit some nonlinear and preprocessing costs. Reported prediction timings are single-run measurements and vary with execution conditions.

## Design and validation decisions

The three methods were chosen to compare an interpretable linear baseline (MLR) against two credible nonlinear alternatives (RBF SVR and feedforward ANN). Hyperparameter selection uses validation RMSE rather than test metrics. Model performance is evaluated using MAE, RMSE, R², residual plots and additional analyses of temperature-region error and temperature-response plausibility. The preferred model for the current *accuracy-first, non-time-critical offline* use is SVR; ANN remains a credible choice if prediction speed matters more. None of these checks establishes reliable behaviour outside the historical plant/domain.

## Dataset attribution and licensing

**Dataset:** Tfekci, P., & Kaya, H. (2014). *Combined Cycle Power Plant* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5002N

The source dataset is distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** licence: https://creativecommons.org/licenses/by/4.0/. The original workbook is used as the source; the program removes duplicate rows in memory for analysis. Dataset licensing is separate from the source-code licence in `LICENSE`.

**Related study:** Tüfekci, P. (2014). Prediction of full load electrical power output of a base load operated combined cycle power plant using machine learning methods. *International Journal of Electrical Power & Energy Systems, 60*, 126–140. https://doi.org/10.1016/j.ijepes.2014.02.027

## References

- Tüfekci, P. (2014). Prediction of full load electrical power output of a base load operated combined cycle power plant using machine learning methods. *International Journal of Electrical Power & Energy Systems, 60*, 126–140. https://doi.org/10.1016/j.ijepes.2014.02.027
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html
- Tfekci, P., & Kaya, H. (2014). *Combined Cycle Power Plant* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5002N

## AI assistance

ChatGPT was used to assist with Python development, debugging, automated testing, understanding implementation concepts, supplementary analyses and drafting/revising parts of the project documentation. Suggestions and generated code were reviewed and adapted, and the tested functionality was checked using program execution and pytest. See the accompanying project report for the full AI-use reflection.

## Project report

The accompanying MMA3001 project report contains the full methodology, validation, model-selection justification, computational analysis, engineering interpretation, limitations and AI-use reflection.