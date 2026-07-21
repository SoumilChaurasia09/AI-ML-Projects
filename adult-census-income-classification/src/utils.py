import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import numpy as np

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", cmap="Blues"):
    """
    Plots a confusion matrix using seaborn heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
    plt.title(title, fontsize=14, pad=15)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    return fig

def plot_roc_curves(models_dict, X_test, y_test, title="ROC Curves"):
    """
    Plots ROC curves for multiple models.
    models_dict: dict of {model_name: trained_model_object}
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Draw diagonal line
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    
    for name, model in models_dict.items():
        # Get probability estimates
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_probs = model.decision_function(X_test)
        else:
            continue
            
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
        
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig

def plot_feature_importance(feature_names, importances, title="Feature Importance", top_n=15):
    """
    Plots a bar chart of the top feature importances.
    """
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    # Elegant color palette
    colors = sns.color_palette("viridis", len(top_features))
    sns.barplot(x=top_importances, y=top_features, palette=colors, ax=ax)
    
    ax.set_title(f"{title} (Top {top_n})", fontsize=14, pad=15)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig
