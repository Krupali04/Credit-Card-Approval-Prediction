import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_correlation_matrix(df, figsize=(10, 8)):
    """Plot correlation matrix of numerical features."""
    plt.figure(figsize=figsize)
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    return plt.gcf()

def plot_feature_importance(feature_names, importances, title="Feature Importance"):
    """Plot feature importance from model."""
    importance_df = pd.DataFrame({
        'features': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='features', data=importance_df)
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate classification metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    
    return metrics
