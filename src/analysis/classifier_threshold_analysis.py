#!/usr/bin/env python3
"""
Classifier Feature Response Analysis for OCO-3 Swath Bias Correction

This script creates plots showing the relationship between each feature value 
and the classifier's prediction probability using real data points.

Usage:
    python -m src.analysis.classifier_threshold_analysis
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.utils.config_paths import PathConfig

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_model_and_features():
    """Load the trained RF model and sample features"""
    config = PathConfig()
    
    # Load the final model
    model_path = config.get_model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    rf_model = joblib.load(model_path)
    
    # Load SAM features
    features_path = config.PROCESSED_FINAL_DIR / "sam_features.parquet" 
    if not features_path.exists():
        raise FileNotFoundError(f"SAM features not found at {features_path}")
    
    sam_features_df = pd.read_parquet(features_path)
    
    # Extract the features used by the model
    feature_names = list(rf_model.feature_names_in_) if hasattr(rf_model, 'feature_names_in_') else []
    
    return rf_model, sam_features_df, feature_names

def create_feature_response_plots_with_real_data(rf_model, sam_features_df, feature_names, output_dir):
    """Create plots showing feature value vs classification probability using real data"""
    
    model_threshold = rf_model.rf_prediction_threshold_ if hasattr(rf_model, 'rf_prediction_threshold_') else 0.5
    
    # Get predictions for all real data points
    X_data = sam_features_df[feature_names].dropna()
    predictions = rf_model.predict_proba(X_data.values)[:, 1]
    
    # Create subplots
    n_features = len(feature_names)
    cols = min(3, n_features)
    rows = (n_features + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    if n_features == 1:
        axes = [axes]
    elif rows == 1 and cols > 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten() if n_features > 1 else [axes[0]]
    
    for i, feature_name in enumerate(feature_names):
        if i >= len(axes_flat):
            break
            
        ax = axes_flat[i]
        
        # Get feature values and corresponding predictions
        feature_values = X_data[feature_name].values
        
        # Create scatter plot with real data
        scatter = ax.scatter(feature_values, predictions, alpha=0.6, s=20)
        
        # Add decision threshold line
        ax.axhline(y=model_threshold, color='red', linestyle='--', alpha=0.7, 
                  label=f'Decision Threshold ({model_threshold})')
        
        # Add trend line using local regression (lowess)
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            # Sort by feature values for smooth line
            sorted_indices = np.argsort(feature_values)
            sorted_features = feature_values[sorted_indices]
            sorted_predictions = predictions[sorted_indices]
            
            # Apply lowess smoothing
            smoothed = lowess(sorted_predictions, sorted_features, frac=0.3)
            ax.plot(smoothed[:, 0], smoothed[:, 1], 'orange', linewidth=2, 
                   label='Trend Line')
        except ImportError:
            # Fallback to simple binned averages if statsmodels not available
            n_bins = 20
            bin_edges = np.linspace(feature_values.min(), feature_values.max(), n_bins + 1)
            bin_centers = []
            bin_means = []
            
            for j in range(n_bins):
                mask = (feature_values >= bin_edges[j]) & (feature_values < bin_edges[j + 1])
                if np.sum(mask) > 0:
                    bin_centers.append((bin_edges[j] + bin_edges[j + 1]) / 2)
                    bin_means.append(predictions[mask].mean())
            
            if bin_centers:
                ax.plot(bin_centers, bin_means, 'orange', linewidth=2, marker='o', 
                       markersize=4, label='Binned Average')
        
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Classification Probability')
        ax.set_title(f'{feature_name}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add legend only to first plot
        if i == 0:
            ax.legend()
    
    # Hide unused subplots
    for i in range(n_features, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_response_real_data.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def main():
    """Main function"""
    # Setup
    config = PathConfig()
    output_dir = config.FIGURES_DIR / "classifier_threshold_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and data
    rf_model, sam_features_df, feature_names = load_model_and_features()
    
    # Create feature response plots using real data
    create_feature_response_plots_with_real_data(rf_model, sam_features_df, feature_names, output_dir)
    
    return 0

if __name__ == '__main__':
    exit(main()) 