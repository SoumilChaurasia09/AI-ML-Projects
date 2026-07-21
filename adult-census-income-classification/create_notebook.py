import json
import os

notebook = {
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Adult Census Income Classification\n",
        "\n",
        "This notebook presents an end-to-end Machine Learning pipeline to predict whether an individual's income exceeds $50,000 per year based on census features. We will use the **Adult Census Income dataset** from the UCI Machine Learning Repository.\n",
        "\n",
        "## Structure of the Notebook:\n",
        "1. **Environment Setup & Data Loading**\n",
        "2. **Exploratory Data Analysis (EDA)**\n",
        "3. **Data Preprocessing & Pipeline Construction**\n",
        "4. **Model Training & Initial Comparison**\n",
        "5. **Hyperparameter Tuning**\n",
        "6. **Model Evaluation & Visualizations**\n",
        "7. **Saving the Pipeline**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 1. Environment Setup & Data Loading\n",
        "\n",
        "First, we install/import the required libraries. We will import pandas, numpy, matplotlib, seaborn, scikit-learn, and xgboost."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "import os\n",
        "import sys\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "# Add src/ directory to the path so we can import modules\n",
        "sys.path.append(os.path.abspath('../src'))\n",
        "\n",
        "from data_loader import load_and_clean_data\n",
        "from utils import plot_confusion_matrix, plot_roc_curves, plot_feature_importance\n",
        "\n",
        "# Set plotting aesthetics\n",
        "sns.set_theme(style=\"whitegrid\")\n",
        "plt.rcParams[\"figure.figsize\"] = (10, 6)\n",
        "plt.rcParams[\"font.size\"] = 12\n",
        "\n",
        "print(\"Libraries loaded successfully.\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### Load Dataset\n",
        "Let's load the training and test sets using our helper function in `data_loader.py` which downloads the files if they are not cached."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "train_df, test_df = load_and_clean_data(\"../data\")\n",
        "print(f\"Train dataset shape: {train_df.shape}\")\n",
        "print(f\"Test dataset shape: {test_df.shape}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "Let's print the first few rows of the training dataset:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "train_df.head()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "Let's check the schema and types of features:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "train_df.info()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "Check basic statistics of numerical columns:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "train_df.describe()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 2. Exploratory Data Analysis (EDA)\n",
        "\n",
        "### Target Variable Distribution\n",
        "Let's see if the class label is balanced."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "fig, ax = plt.subplots(figsize=(6, 4))\n",
        "sns.countplot(data=train_df, x='income', palette='viridis', ax=ax)\n",
        "plt.title('Distribution of Target (Income)')\n",
        "plt.ylabel('Count')\n",
        "plt.xlabel('Income Class')\n",
        "\n",
        "# Show percentages\n",
        "total = len(train_df)\n",
        "for p in ax.patches:\n",
        "    percentage = f'{100 * p.get_height() / total:.1f}%'\n",
        "    ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height() + 200), \n",
        "                ha='center', va='center', xytext=(0, 5), textcoords='offset points', weight='bold')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "**Observation**: The dataset is heavily imbalanced, with roughly 76% of individuals earning `<=50K` and only 24% earning `>50K`. We will need to take this class imbalance into account when building models (e.g. using `class_weight='balanced'` or stratified CV)."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### Numerical Features Analysis\n",
        "Let's look at the correlation matrix of numerical variables."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']\n",
        "corr_matrix = train_df[num_cols + ['income_bin']].corr()\n",
        "\n",
        "plt.figure(figsize=(8, 6))\n",
        "sns.heatmap(corr_matrix, annot=True, fmt=\".2f\", cmap=\"coolwarm\", square=True, linewidths=0.5)\n",
        "plt.title(\"Correlation Matrix of Numerical Features\")\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "**Observation**: `education-num`, `age`, `hours-per-week`, and `capital-gain` have the highest positive correlation with `income_bin`. `fnlwgt` (final weight) shows close to zero correlation and is likely not very predictive."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### Age Distribution by Income\n",
        "Let's see if age plays a significant role in income levels."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "plt.figure(figsize=(10, 5))\n",
        "sns.histplot(data=train_df, x='age', hue='income', kde=True, bins=30, multiple='stack', palette='viridis')\n",
        "plt.title('Age Distribution by Income')\n",
        "plt.xlabel('Age')\n",
        "plt.ylabel('Count')\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "**Observation**: The proportion of individuals earning `>50K` peaks between the ages of 35 and 55. Younger individuals (under 25) rarely earn `>50K`."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### Education Num vs Income\n",
        "Let's see if more years of education relates directly to higher income."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "plt.figure(figsize=(10, 5))\n",
        "sns.kdeplot(data=train_df, x='education-num', hue='income', fill=True, common_norm=False, palette='mako')\n",
        "plt.title('Years of Education Density by Income')\n",
        "plt.xlabel('Education Num (Years of Education)')\n",
        "plt.ylabel('Density')\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "**Observation**: Individuals with higher education years (specifically > 12 years, corresponding to Bachelor's, Master's, Doctorate) have a much higher likelihood of earning `>50K`."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### Categorical Features Analysis\n",
        "Let's analyze the relationship between categorical features (`workclass`, `sex`, `marital-status`, `occupation`) and income."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n",
        "\n",
        "# Workclass\n",
        "sns.countplot(data=train_df, y='workclass', hue='income', palette='viridis', ax=axes[0, 0])\n",
        "axes[0, 0].set_title('Income by Workclass')\n",
        "axes[0, 0].set_xlabel('Count')\n",
        "\n",
        "# Sex\n",
        "sns.countplot(data=train_df, x='sex', hue='income', palette='mako', ax=axes[0, 1])\n",
        "axes[0, 1].set_title('Income by Gender')\n",
        "axes[0, 1].set_ylabel('Count')\n",
        "\n",
        "# Marital Status\n",
        "sns.countplot(data=train_df, y='marital-status', hue='income', palette='flare', ax=axes[1, 0])\n",
        "axes[1, 0].set_title('Income by Marital Status')\n",
        "axes[1, 0].set_xlabel('Count')\n",
        "\n",
        "# Occupation\n",
        "sns.countplot(data=train_df, y='occupation', hue='income', palette='crest', ax=axes[1, 1])\n",
        "axes[1, 1].set_title('Income by Occupation')\n",
        "axes[1, 1].set_xlabel('Count')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "**Observations**:\n",
        "- **Workclass**: The private sector employs the vast majority of individuals.\n",
        "- **Gender**: A higher proportion of males earn `>50K` compared to females in this dataset.\n",
        "- **Marital Status**: Individuals who are \"Married-civ-spouse\" have a significantly higher rate of earning `>50K` compared to other categories.\n",
        "- **Occupation**: Professional specialty and Executive/Managerial roles have the highest density of `>50K` earners."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 3. Data Preprocessing & Pipeline Construction\n",
        "\n",
        "We will build a preprocessing pipeline using `scikit-learn`'s `ColumnTransformer`. \n",
        "- For **numerical** columns, we impute missing values with the median and scale using `RobustScaler` (handles outliers in capital gain/loss).\n",
        "- For **categorical** columns, we impute with the most frequent value (mode) and encode using `OneHotEncoder`.\n",
        "- We drop the column `education` because it is redundant with `education-num`."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.impute import SimpleImputer\n",
        "from sklearn.preprocessing import RobustScaler, OneHotEncoder\n",
        "\n",
        "# Define features and target\n",
        "target_col = 'income_bin'\n",
        "exclude_cols = ['income', 'income_bin', 'education']\n",
        "feature_cols = [col for col in train_df.columns if col not in exclude_cols]\n",
        "\n",
        "numerical_cols = train_df[feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()\n",
        "categorical_cols = train_df[feature_cols].select_dtypes(include=['object']).columns.tolist()\n",
        "\n",
        "X_train = train_df[feature_cols]\n",
        "y_train = train_df[target_col]\n",
        "X_test = test_df[feature_cols]\n",
        "y_test = test_df[target_col]\n",
        "\n",
        "# Construct Pipelines\n",
        "num_pipeline = Pipeline([\n",
        "    ('imputer', SimpleImputer(strategy='median')),\n",
        "    ('scaler', RobustScaler())\n",
        "])\n",
        "\n",
        "cat_pipeline = Pipeline([\n",
        "    ('imputer', SimpleImputer(strategy='most_frequent')),\n",
        "    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))\n",
        "])\n",
        "\n",
        "preprocessor = ColumnTransformer(transformers=[\n",
        "    ('num', num_pipeline, numerical_cols),\n",
        "    ('cat', cat_pipeline, categorical_cols)\n",
        "])\n",
        "\n",
        "# Fit preprocessor and transform features\n",
        "X_train_proc = preprocessor.fit_transform(X_train)\n",
        "X_test_proc = preprocessor.transform(X_test)\n",
        "\n",
        "# Extract feature names\n",
        "cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']\n",
        "try:\n",
        "    cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()\n",
        "except AttributeError:\n",
        "    cat_feature_names = cat_encoder.get_feature_names(categorical_cols).tolist()\n",
        "all_feature_names = numerical_cols + cat_feature_names\n",
        "\n",
        "print(f\"X_train shape after processing: {X_train_proc.shape}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 4. Model Training & Initial Comparison\n",
        "\n",
        "We will train three different algorithms:\n",
        "1. **Logistic Regression** (Linear baseline)\n",
        "2. **Random Forest Classifier** (Tree-based ensemble)\n",
        "3. **XGBoost Classifier** (Gradient boosting ensemble)\n",
        "\n",
        "We'll adjust class weights or settings to handle the imbalance where possible."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from xgboost import XGBClassifier\n",
        "from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score\n",
        "\n",
        "models = {\n",
        "    \"Logistic Regression\": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),\n",
        "    \"Random Forest\": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, class_weight='balanced'),\n",
        "    \"XGBoost\": XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')\n",
        "}\n",
        "\n",
        "trained_models = {}\n",
        "metrics = []\n",
        "\n",
        "for name, model in models.items():\n",
        "    print(f\"Training {name}...\")\n",
        "    model.fit(X_train_proc, y_train)\n",
        "    trained_models[name] = model\n",
        "    \n",
        "    y_pred = model.predict(X_test_proc)\n",
        "    y_prob = model.predict_proba(X_test_proc)[:, 1]\n",
        "    \n",
        "    acc = accuracy_score(y_test, y_pred)\n",
        "    f1 = f1_score(y_test, y_pred)\n",
        "    auc_score = roc_auc_score(y_test, y_prob)\n",
        "    \n",
        "    metrics.append({\n",
        "        \"Model\": name,\n",
        "        \"Accuracy\": acc,\n",
        "        \"F1-Score\": f1,\n",
        "        \"ROC-AUC\": auc_score\n",
        "    })\n",
        "    \n",
        "    print(f\"--- {name} Results ---\")\n",
        "    print(f\"Accuracy: {acc:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {auc_score:.4f}\")\n",
        "    print(classification_report(y_test, y_pred))\n",
        "    print(\"\\n\")\n",
        "    \n",
        "metrics_df = pd.DataFrame(metrics)\n",
        "print(\"=== Summary of Initial Performance ===\")\n",
        "print(metrics_df)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 5. Hyperparameter Tuning\n",
        "\n",
        "Let's optimize our best model (XGBoost) using `RandomizedSearchCV` to fine-tune the parameters and get even better performance."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "from sklearn.model_selection import RandomizedSearchCV\n",
        "\n",
        "param_dist = {\n",
        "    'n_estimators': [100, 150, 200, 250],\n",
        "    'max_depth': [3, 4, 5, 6, 8],\n",
        "    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],\n",
        "    'subsample': [0.7, 0.8, 0.9, 1.0],\n",
        "    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],\n",
        "    'scale_pos_weight': [1, 2, 3] # To help with class imbalance\n",
        "}\n",
        "\n",
        "xgb_base = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')\n",
        "\n",
        "print(\"Starting RandomizedSearchCV for XGBoost...\")\n",
        "xgb_search = RandomizedSearchCV(\n",
        "    estimator=xgb_base,\n",
        "    param_distributions=param_dist,\n",
        "    n_iter=10,\n",
        "    scoring='f1',\n",
        "    cv=3,\n",
        "    verbose=1,\n",
        "    random_state=42,\n",
        "    n_jobs=-1\n",
        ")\n",
        "\n",
        "xgb_search.fit(X_train_proc, y_train)\n",
        "\n",
        "print(f\"Best F1-Score on CV: {xgb_search.best_score_:.4f}\")\n",
        "print(f\"Best Parameters: {xgb_search.best_params_}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "Let's evaluate the tuned XGBoost model on the test set:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "best_xgb = xgb_search.best_estimator_\n",
        "trained_models[\"Tuned XGBoost\"] = best_xgb\n",
        "\n",
        "y_pred_tuned = best_xgb.predict(X_test_proc)\n",
        "y_prob_tuned = best_xgb.predict_proba(X_test_proc)[:, 1]\n",
        "\n",
        "acc_t = accuracy_score(y_test, y_pred_tuned)\n",
        "f1_t = f1_score(y_test, y_pred_tuned)\n",
        "auc_t = roc_auc_score(y_test, y_prob_tuned)\n",
        "\n",
        "print(\"--- Tuned XGBoost Results ---\")\n",
        "print(f\"Accuracy: {acc_t:.4f} | F1-Score: {f1_t:.4f} | ROC-AUC: {auc_t:.4f}\")\n",
        "print(classification_report(y_test, y_pred_tuned))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 6. Model Evaluation & Visualizations\n",
        "\n",
        "### Confusion Matrix Comparison\n",
        "Let's plot confusion matrices for our classifiers."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "for model_name in [\"Logistic Regression\", \"Random Forest\", \"Tuned XGBoost\"]:\n",
        "    plot_confusion_matrix(y_test, trained_models[model_name].predict(X_test_proc), title=f\"Confusion Matrix: {model_name}\")\n",
        "    plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### ROC Curves\n",
        "Let's plot ROC-AUC curves for all models side-by-side to visually compare performance."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "plot_roc_curves(trained_models, X_test_proc, y_test, title=\"ROC Curves Comparison\")\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### Feature Importance\n",
        "Let's inspect the feature importance of our Tuned XGBoost model to find which variables drive the prediction."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "xgb_importances = best_xgb.feature_importances_\n",
        "plot_feature_importance(all_feature_names, xgb_importances, title=\"Feature Importance - Tuned XGBoost\", top_n=15)\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "**Observation**: The top features predicting higher income are **marital-status (Married-civ-spouse)**, **capital-gain**, **education-num** (years of education), and **relationship (husband)**. These align well with our EDA findings."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 7. Saving the Pipeline\n",
        "\n",
        "We'll serialize the best performing model pipeline (preprocessor + Tuned XGBoost Classifier) so it can be easily deployed in production."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "import joblib\n",
        "\n",
        "full_pipeline = Pipeline([\n",
        "    ('preprocessor', preprocessor),\n",
        "    ('classifier', best_xgb)\n",
        "])\n",
        "\n",
        "# Save\n",
        "model_file = \"../models/best_income_classifier_pipeline.joblib\"\n",
        "os.makedirs(\"../models\", exist_ok=True)\n",
        "joblib.dump(full_pipeline, model_file)\n",
        "print(f\"Full end-to-end model pipeline saved successfully to {model_file}!\")"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 2
}

os.makedirs("notebooks", exist_ok=True)
with open("notebooks/adult_census_income.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)
print("Notebook generated successfully under notebooks/adult_census_income.ipynb!")
