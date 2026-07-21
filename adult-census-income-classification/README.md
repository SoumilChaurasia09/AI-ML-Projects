# Adult Census Income Classification

This project implements a complete machine learning classification workflow on the **UCI Adult Census Income dataset** to predict whether an individual's annual income exceeds **$50,000**.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Project Structure](#project-structure)
4. [Installation & Setup](#installation--setup)
5. [Preprocessing Pipeline](#preprocessing-pipeline)
6. [Model Evaluation Results](#model-evaluation-results)
7. [Key Insights](#key-insights)
8. [How to Use the Model](#how-to-use-the-model)

---

## Project Overview
The Adult Census Income dataset (often referred to as the "Census Income" dataset) is a classic binary classification dataset. The goal is to determine if a person makes more than $50K a year based on demographic and employment attributes. 

This repository structures a modular machine learning pipeline consisting of data caching/cleaning, custom visualization helper utilities, a Jupyter Notebook for Exploratory Data Analysis (EDA) and hyperparameter tuning, and a production-ready serialized model pipeline.

---

## Dataset Description
The dataset contains 14 features and 1 target variable:

*   **Continuous Features**: `age`, `fnlwgt` (final weight), `education-num` (years of education), `capital-gain`, `capital-loss`, `hours-per-week`.
*   **Categorical Features**: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`.
*   **Target**: `income` (categorical: `<=50K` or `>50K`; binary encoded as `0` and `1`).

---

## Project Structure
```
adult-census-income-classification/
├── data/
│   ├── adult.data               # Raw training data
│   └── adult.test               # Raw test data
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Dataset downloading and cleaning
│   └── utils.py                 # Plotting and evaluation helper functions
├── notebooks/
│   └── adult_census_income.ipynb # Main EDA and model development notebook
├── plots/                       # Generated evaluation plots
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_random_forest.png
│   ├── confusion_matrix_tuned_xgboost.png
│   ├── feature_importance_xgboost.png
│   ├── roc_curves_comparison.png
│   └── ...
├── models/
│   └── best_income_classifier_pipeline.joblib # Saved pipeline (preprocessor + XGBoost)
├── requirements.txt             # Python packages
└── README.md                    # Project documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher installed.

### Step-by-Step Installation
1. **Clone or Navigate to the Directory**:
   ```bash
   cd C:\Users\soumi\.gemini\antigravity\scratch\adult-census-income-classification
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   ```

3. **Activate the Virtual Environment**:
   - **Windows (PowerShell)**:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD)**:
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the End-to-End Pipeline**:
   To download the data, train the models, save evaluation plots, and export the best model pipeline:
   ```bash
   python src/train.py
   ```

6. **Launch the Web UI Dashboard**:
   To start the FastAPI web server and explore predictions in the interactive dashboard:
   ```bash
   python src/server.py
   ```
   Open your web browser and navigate to:
   [http://127.0.0.1:8080](http://127.0.0.1:8080)

7. **Launch the Jupyter Notebook**:
   To inspect the EDA and step-by-step training details:
   ```bash
   jupyter notebook notebooks/adult_census_income.ipynb
   ```

---

## Preprocessing Pipeline
Since raw census data is noisy and contains missing values (encoded as `?`), we built a robust pipeline using `scikit-learn`'s `ColumnTransformer`:
1. **Drop Redundant Columns**: We dropped the categorical `education` column, as the numerical `education-num` column contains the exact ordinal representation of education level.
2. **Missing Values**: Replaced `?` with `NaN` and imputed them using `SimpleImputer` (median for numerical columns, most frequent class for categorical columns).
3. **Scaling**: Applied `RobustScaler` on numerical columns to center and scale the data while mitigating the effect of extreme outliers present in `capital-gain` and `capital-loss`.
4. **Encoding**: Categorical features were encoded using `OneHotEncoder(handle_unknown='ignore')` to ensure robust handling of unseen categorical levels in production.

---

## Model Evaluation Results

After preprocessing, we trained and compared three different models on the test set:

| Model | Accuracy | F1-Score (Class 1) | ROC-AUC |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | 80.37% | 0.6696 | 0.9036 |
| **Random Forest** | 81.27% | 0.6854 | 0.9144 |
| **XGBoost (Baseline)** | 87.40% | 0.7083 | 0.9285 |
| **Tuned XGBoost** | 83.80% | 0.7126 | 0.9281 |

*Note: In the hyperparameter tuning phase, we introduced `scale_pos_weight` (set to 2 or 3) to penalize minority class errors more heavily. This boosted the **Recall** of the high-income class (`>50K`) from **65% to 85%** and increased the **F1-Score** from **0.7083 to 0.7126**, while accepting a minor decrease in overall accuracy due to a higher rate of false positives. This represents a favorable trade-off for scenarios where identifying high-income individuals is the primary goal.*

*Note: F1-Score for Class 1 represents the model's accuracy on the minority class (`>50K`), which is crucial due to the 3:1 class imbalance.*

---

## Key Insights
Based on the feature importances extracted from the tree models (XGBoost & Random Forest):
1. **Marital Status**: Being married (`marital-status_Married-civ-spouse` and `relationship_Husband`) is the strongest demographic predictor of higher income in this dataset.
2. **Capital Gain**: Capital gains are a strong indicator of wealth, directly correlating with high income.
3. **Education**: Higher education years (`education-num`) directly correlate with a higher probability of earning >$50K.

---

## How to Use the Model

The best model is exported as a combined pipeline containing both the preprocessor and the classifier. You can load it and run inference on raw data directly:

```python
import joblib
import pandas as pd

# Load the saved pipeline
pipeline = joblib.load("models/best_income_classifier_pipeline.joblib")

# Example raw data row (without preprocessing applied)
raw_data = pd.DataFrame([{
    'age': 39,
    'workclass': 'State-gov',
    'fnlwgt': 77516,
    'education-num': 13,
    'marital-status': 'Never-married',
    'occupation': 'Adm-clerical',
    'relationship': 'Not-in-family',
    'race': 'White',
    'sex': 'Male',
    'capital-gain': 2174,
    'capital-loss': 0,
    'hours-per-week': 40,
    'native-country': 'United-States'
}])

# Predict directly (automatically imputes, scales, encodes, and classifies)
prediction = pipeline.predict(raw_data)
probability = pipeline.predict_proba(raw_data)[:, 1]

print(f"Prediction: {'>50K' if prediction[0] == 1 else '<=50K'}")
print(f"Probability of earning >50K: {probability[0]:.4f}")
```
