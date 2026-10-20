# FinResilience AI
### Financial Stress Prediction and Financial Resilience Assessment

## 1. Project Overview

FinResilience AI is a machine-learning-based academic project that explores credit repayment stress and financial resilience.

The project investigates whether historical credit-behaviour information can improve repayment-stress prediction beyond conventional application features. It also provides a separate rule-based Financial Resilience Score based on user-entered income, expenses, EMI, savings, debt, and recent monthly expenses.

The application is built using Python and Streamlit, with model training, evaluation, and SHAP explainability performed in Jupyter notebooks.

**Important:** The Financial Resilience Score is independent of the machine-learning prediction. The two outputs use different inputs and should not be treated as interchangeable.

## 2. Objectives

- Build a conventional credit-stress prediction model.
- Evaluate whether behavioural credit-history features improve prediction performance.
- Compare models using suitable classification metrics, particularly PR-AUC.
- Use SHAP to understand the features influencing model predictions.
- Develop a transparent 0–100 Financial Resilience Score.
- Present project results through a simple, interactive dashboard.
- Replicate the feature-group experiment using a second public dataset.

## 3. Datasets

### 3.1 Home Credit Default Risk

Home Credit is the primary dataset used for model development and evaluation.

**Target:** `TARGET`

The target represents the repayment-difficulty outcome defined by the dataset.

Two feature groups are investigated:

**Group A — Conventional features**

Uses 25 application and financial features, including income, credit amount, annuity, family information, employment information, contract type, and derived financial ratios.

**Group B — Conventional + behavioural features**

Uses 57 features, combining the conventional features with aggregated indicators from historical instalment payments, credit-card records, and bureau credit history.

The selected model recorded in the project metadata is XGBoost for both Group A and Group B.

### 3.2 Give Me Some Credit

Give Me Some Credit is used as a secondary replication dataset.

**Target:** `SeriousDlqin2yrs`

The dataset has a different target and feature space from Home Credit. Its results must therefore be interpreted separately rather than as a direct comparison of predictions on the same population.

The current application displays the saved replication results. A separate saved prediction model is required before individual Give Me Some Credit predictions can be offered.

## 4. Financial Resilience Score

The dashboard includes an independent rule-based score ranging from 0 to 100.

The score combines five components:

| Component | Weight | Full marks | Zero marks |
|---|---:|---|---|
| EMI burden | 25% | EMI/income ≤ 20% | EMI/income ≥ 50% |
| Debt-to-income proxy | 20% | ≤ 20% | ≥ 60% |
| Savings rate | 20% | ≥ 20% | ≤ 0% |
| Emergency coverage | 25% | At least 6 months | 0 months |
| Expense volatility | 10% | CV ≤ 0.05 | CV ≥ 0.30 |

Each component is scaled between its full-mark and zero-mark thresholds using piecewise-linear scaling.

The score categories are:

- **Fragile:** 0–39
- **Watchful:** 40–69
- **Resilient:** 70–100

The implementation uses total outstanding debt divided by annual income as its debt-to-income proxy. This definition should be documented consistently in the project report.

Expense volatility is measured using the coefficient of variation:

`CV = Standard deviation of monthly expenses / Mean monthly expenses`

When insufficient expense history is provided, the current implementation excludes the volatility component and proportionally rescales the remaining component weights.

The score is an educational indicator, not a credit score or financial recommendation.

## 5. Machine Learning and Evaluation

The project compares conventional and behavioural feature groups using classification models.

The saved Home Credit model bundle contains calibrated versions of:

- Logistic Regression
- Random Forest
- XGBoost

The recorded primary evaluation metric is **PR-AUC**, which is useful when the positive class is relatively uncommon.

Other reported metrics include:

- ROC-AUC
- Precision
- Recall
- F1-score
- Brier score
- Calibration analysis
- Confidence intervals and model-comparison results

The saved final validation summary reports the following Home Credit results:

| Metric | Reported value |
|---|---:|
| PR-AUC | 0.2062 |
| ROC-AUC | 0.7250 |
| Brier score | 0.0703 |
| Precision | 21.14% |
| Recall | 40.52% |
| F1-score | 0.2779 |

These are the recorded results in the project artifacts. They should not be interpreted as guaranteed performance on new data.

The reported recall is below the project's aspirational target of approximately 70%. This limitation should be disclosed in the final report and presentation.

The locked test set must not be used for further model selection or threshold tuning.

## 6. Explainable AI Using SHAP

SHAP (SHapley Additive exPlanations) is used in Notebook 2 to analyze the selected Group B model pipeline.

The existing workflow:

1. Selects a model based on cross-validation PR-AUC.
2. Samples background rows from the training data.
3. Samples rows for explanation from the calibration-validation data.
4. Applies the fitted preprocessing pipeline.
5. Calculates SHAP values.
6. Generates a global beeswarm plot.
7. Generates a global mean-absolute-SHAP bar plot.

The plots are saved as:

- `results/shap_global_beeswarm.png`
- `results/shap_global_importance.png`

The Streamlit dashboard displays these plots in the **About & Project Details → SHAP Explainability** section when the files are available.

SHAP explains how features contribute to the underlying model output. It does not establish causal relationships. The current dashboard does not calculate a local waterfall plot for an individual applicant.

## 7. Application Features

The Streamlit application contains two main navigation sections.

### Financial Resilience Score

- User input for income and monthly expenses.
- EMI, savings, and outstanding debt inputs.
- Recent monthly expense history.
- Calculation of the 0–100 score.
- Score category and progress indicator.
- Component-level score breakdown.
- Financial indicator summaries.
- Calculation details and score interpretation.

### About & Project Details

Contains the following tabs:

- **Project Overview:** Objectives, datasets, and model design.
- **Model Evaluation:** Saved Home Credit evaluation metrics, secondary-dataset results, and available research plots.
- **SHAP Explainability:** Existing global SHAP plots and interpretation guidance.
- **Score Methodology:** Component weights, thresholds, and score bands.
- **Limitations:** Dataset, model, privacy, and responsible-use limitations.

## 8. Project Structure

The main project files are organized as follows:

```text
mini_project/
│
├── app.py
├── README.md
├── requirements.txt
│
├── 01_data_preparation.ipynb
├── 02_model_training_evaluation.ipynb
├── 03_resilience_score_validation.ipynb
│
├── data/
│   ├── home_credit/
│   └── give_me_some_credit/
│       ├── cs-training.csv
│       ├── cs-test.csv
│       ├── Data Dictionary.xls
│       └── sampleEntry.csv
│
├── artifacts/
│   ├── all_calibrated_models.joblib
│   ├── best_group_a_calibrated_model.joblib
│   ├── best_group_b_calibrated_model.joblib
│   ├── model_metadata.json
│   ├── home_credit_feature_table.csv
│   ├── home_credit_group_b.csv
│   └── additional aggregated data files
│
└── results/
    ├── final_locked_test_results.csv
    ├── secondary_dataset_replication.csv
    ├── calibration_curves.png
    ├── roc_pr_curves.png
    ├── shap_global_beeswarm.png
    ├── shap_global_importance.png
    └── final_report/
        ├── final_validation_summary.json
        ├── confusion_matrix.png
        └── additional validation artifacts
```

The exact contents of the folders may vary depending on which notebook outputs have been generated.

## 9. Technologies Used

- **Python:** Main programming language
- **Pandas:** Data loading and manipulation
- **NumPy:** Numerical calculations
- **Scikit-learn:** Model evaluation and probability calibration
- **XGBoost:** Gradient-boosted tree model
- **SHAP:** Model explainability
- **Matplotlib:** Research plots
- **Streamlit:** Interactive web dashboard
- **Jupyter Notebook:** Data preparation, model training, evaluation, and validation
- **Joblib:** Saving and loading trained models

## 10. Installation and Setup

### Prerequisites

Install Python and Visual Studio Code. Ensure that the project datasets and generated artifacts are present in their expected folders.

### Step 1: Open the project folder

Open the `mini_project` folder in VS Code.

### Step 2: Activate the virtual environment

In PowerShell, run:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell does not allow script execution, use a suitable environment activation method for your system.

### Step 3: Install dependencies

Run:

```powershell
python -m pip install -r requirements.txt
```

Make sure `requirements.txt` contains the packages needed by the application and notebooks, including Streamlit, Pandas, NumPy, Scikit-learn, Matplotlib, Joblib, XGBoost, and SHAP.

### Step 4: Run the application

From the project root, run:

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal if the browser does not open automatically.

## 11. How to Use the Dashboard

1. Open the Financial Resilience Score page.
2. Enter your monthly income.
3. Enter essential and total monthly expenses.
4. Enter total monthly EMI, available savings, and outstanding debt.
5. Enter available recent monthly expense figures.
6. Click **Calculate my resilience score**.
7. Review the score, category, component breakdown, and financial indicators.
8. Open About & Project Details to review model evaluation, SHAP plots, score methodology, and limitations.

Use approximate or sample figures for demonstrations. Do not enter bank passwords, account credentials, or unnecessary identifying information.

## 12. Limitations and Responsible Use

- The project is an educational prototype, not a lending decision system.
- Model probabilities are estimates, not guarantees.
- Home Credit and Give Me Some Credit represent different populations and targets.
- The datasets may not represent Indian consumers.
- Historical behavioural information may not be available for a new applicant.
- Predictive associations do not establish causation.
- The rule-based score depends on the accuracy of user-entered financial information.
- The score's weights and thresholds are documented assumptions, not universal financial standards.
- The current dashboard includes global SHAP plots but not individual applicant-level SHAP waterfall explanations.
- The debt-to-income ratio definition should remain consistent between the implementation and project report.
- Privacy and security should be reviewed before public deployment.

## 13. Future Improvements

Possible improvements include:

- Adding a local SHAP waterfall plot and a plain-language explanation for an individual applicant.
- Saving and integrating a separate Give Me Some Credit prediction model.
- Verifying training-time preprocessing and categorical encoding for any live prediction interface.
- Improving recall while maintaining a transparent and properly validated evaluation protocol.
- Adding automated tests for score bounds, monotonicity, missing values, and division-by-zero handling.
- Conducting further fairness and robustness analysis.
- Improving accessibility and deployment readiness.

## 14. Conclusion

FinResilience AI combines a credit-stress research experiment with a transparent financial-resilience calculator. The project evaluates conventional and behavioural feature groups, uses SHAP for global model interpretation, and presents financial indicators through a Streamlit dashboard.

The two main outputs serve different purposes: the machine-learning experiment estimates repayment stress from historical credit data, while the rule-based score summarizes a user's reported financial position. Both require careful interpretation and should not be used as the sole basis for financial decisions.

---
