"""Model visualization utilities for logging to MLflow."""

import io
import os
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def create_predictions_plot(y_true, y_pred, model_name="Model"):
    """
    Create actual vs predicted scatter plot and save to temp file.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        model_name: Name of the model
        
    Returns:
        Path to the saved plot file
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(f'{model_name}: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return _fig_to_file(fig, "predictions_plot.png")


def create_residuals_plot(y_true, y_pred, model_name="Model"):
    """
    Create residuals vs predicted values plot and save to temp file.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        model_name: Name of the model
        
    Returns:
        Path to the saved plot file
    """
    residuals = y_true - y_pred
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    
    ax.set_xlabel('Predicted Values', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title(f'{model_name}: Residuals Plot', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    return _fig_to_file(fig, "residuals_plot.png")


def create_residuals_distribution(y_true, y_pred, model_name="Model"):
    """
    Create residuals distribution histogram with normal distribution overlay and save to temp file.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        model_name: Name of the model
        
    Returns:
        Path to the saved plot file
    """
    residuals = y_true - y_pred
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.hist(residuals, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
    
    # Overlay normal distribution
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
            'r-', lw=2, label='Normal Distribution')
    
    ax.set_xlabel('Residuals', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{model_name}: Residuals Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    return _fig_to_file(fig, "residuals_distribution.png")


def create_coefficients_plot(coef_dict, model_name="Model"):
    """
    Create feature coefficients bar plot and save to temp file.
    
    Args:
        coef_dict: Dictionary with feature names as keys and coefficients as values
        model_name: Name of the model
        
    Returns:
        Path to the saved plot file
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    features = list(coef_dict.keys())
    coefficients = list(coef_dict.values())
    colors = ['green' if c > 0 else 'red' for c in coefficients]
    
    ax.barh(features, coefficients, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Coefficient Value', fontsize=12)
    ax.set_title(f'{model_name}: Feature Coefficients', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', lw=0.8)
    ax.grid(True, alpha=0.3, axis='x')
    
    return _fig_to_file(fig, "coefficients_plot.png")


def create_metrics_summary(y_true, y_pred, model_name="Model"):
    """
    Create a text summary of model metrics and save to temp file.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        model_name: Name of the model
        
    Returns:
        Path to the saved plot file
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    n = len(y_true)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - y_pred.shape[0] - 1))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    summary_text = f"""
    {model_name} - Performance Metrics
    {'=' * 50}
    
    R² Score:              {r2:.4f}
    Adjusted R²:           {adj_r2:.4f}
    Root Mean Squared Error (RMSE):  {rmse:.4f}
    Mean Squared Error (MSE):        {mse:.4f}
    Mean Absolute Error (MAE):       {mae:.4f}
    
    Dataset Size:          {n}
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    return _fig_to_file(fig, "metrics_summary.png")


def _fig_to_file(fig, filename):
    """Convert matplotlib figure to a file in temp directory."""
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, filename)
    fig.savefig(filepath, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    return filepath

