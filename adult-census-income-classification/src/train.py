import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
from xgboost import XGBClassifier

from data_loader import load_and_clean_data
from utils import plot_confusion_matrix, plot_roc_curves, plot_feature_importance

def build_preprocessing_pipeline(categorical_cols, numerical_cols):
    """
    Creates a preprocessor using ColumnTransformer.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])
    
    return preprocessor

def main():
    print("Step 1: Loading and cleaning data...")
    train_df, test_df = load_and_clean_data("data")
    
    # Define features and target
    # 'education' is redundant with 'education-num', and 'income'/'income_bin' are target cols
    target_col = 'income_bin'
    exclude_cols = ['income', 'income_bin', 'education']
    
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Identify numerical and categorical columns
    numerical_cols = train_df[feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = train_df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    
    print(f"Features list: {feature_cols}")
    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    print("\nStep 2: Fitting preprocessing pipeline...")
    preprocessor = build_preprocessing_pipeline(categorical_cols, numerical_cols)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Retrieve feature names for importance plot
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    try:
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    except AttributeError:
        # Fallback for older scikit-learn versions
        cat_feature_names = cat_encoder.get_feature_names(categorical_cols).tolist()
        
    all_feature_names = numerical_cols + cat_feature_names
    print(f"Processed feature matrix shape: {X_train_proc.shape} (Number of features: {len(all_feature_names)})")
    
    print("\nStep 3: Training models...")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    trained_models = {}
    metrics = []
    
    # Create directory for saving plots and models
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_proc, y_train)
        trained_models[name] = model
        
        # Predictions
        y_pred = model.predict(X_test_proc)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_proc)[:, 1]
        else:
            y_prob = model.decision_function(X_test_proc)
            
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_prob)
        
        print(f"{name} Classification Report:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {acc:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {auc_score:.4f}")
        
        metrics.append({
            "Model": name,
            "Accuracy": acc,
            "F1-Score": f1,
            "ROC-AUC": auc_score
        })
        
        # Save Confusion Matrix Plot
        fig_cm = plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix - {name}")
        fig_cm.savefig(f"plots/confusion_matrix_{name.lower().replace(' ', '_')}.png")
        plt.close(fig_cm)
        
    # Step 4: Plot and save ROC Curves
    print("\nStep 4: Generating ROC curves plot...")
    fig_roc = plot_roc_curves(trained_models, X_test_proc, y_test, title="ROC Curves Comparison")
    fig_roc.savefig("plots/roc_curves_comparison.png")
    plt.close(fig_roc)
    
    # Step 5: Feature Importance for best non-linear model
    print("\nStep 5: Extracting and plotting feature importances...")
    # XGBoost
    xgb_model = trained_models["XGBoost"]
    xgb_importances = xgb_model.feature_importances_
    fig_imp = plot_feature_importance(all_feature_names, xgb_importances, title="Feature Importance (XGBoost)")
    fig_imp.savefig("plots/feature_importance_xgboost.png")
    plt.close(fig_imp)
    
    # Random Forest
    rf_model = trained_models["Random Forest"]
    rf_importances = rf_model.feature_importances_
    fig_imp_rf = plot_feature_importance(all_feature_names, rf_importances, title="Feature Importance (Random Forest)")
    fig_imp_rf.savefig("plots/feature_importance_rf.png")
    plt.close(fig_imp_rf)
    
    # Save best model based on F1-Score and ROC-AUC
    # Let's say XGBoost is our selected model
    best_model_name = "XGBoost"
    best_model = trained_models[best_model_name]
    
    # We will save the preprocessing pipeline along with the model
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', best_model)
    ])
    
    joblib.dump(full_pipeline, "models/best_income_classifier_pipeline.joblib")
    print(f"\nSaved the best model pipeline ({best_model_name}) to models/best_income_classifier_pipeline.joblib")
    
    # Print metrics table
    metrics_df = pd.DataFrame(metrics)
    print("\n=== Model Evaluation Summary ===")
    print(metrics_df.to_string(index=False))

if __name__ == "__main__":
    main()
