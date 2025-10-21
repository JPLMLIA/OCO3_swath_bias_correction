#!/usr/bin/env python3
"""
Generate improved bias plots using the best RF model from Swath_BC_v3.

This script loads the final trained model and the processed SAM-level
features to create plots of bias rate vs. feature value.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import os
import sys

# Add config import
from src.utils.config_paths import PathConfig

def load_final_model_results(config: PathConfig):
    """Load the final trained model and the corresponding SAM features and predictions."""
    
    print("Loading final model and processed SAM features...")
    
    # Get paths from config
    model_path = config.get_model_path()
    features_path = config.PROCESSED_FINAL_DIR / "sam_features.parquet"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Final model not found at: {model_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"SAM features file not found at: {features_path}. Please run the main modeling script first.")
        
    # Load the model
    try:
        rf_model = joblib.load(model_path)
        print(f"✓ Model loaded from {model_path}")
    except (ValueError, TypeError) as e:
        if "incompatible dtype" in str(e) or "version" in str(e).lower():
            print(f"⚠️  Model loading failed due to sklearn version compatibility: {e}")
            print("Using default feature list instead of model features...")
            # Create a dummy model object with feature names
            class DummyModel:
                def __init__(self):
                    self.feature_names_in_ = ['max_relative_jump', 'h_continuum_sco2', 'solar_zenith_angle', 'dws', 's31']
            rf_model = DummyModel()
        else:
            raise e
    
    # Load the SAM features and predictions
    sam_features_df = pd.read_parquet(features_path)
    print(f"✓ SAM features and predictions loaded from {features_path}")
    print(f"  - Found {len(sam_features_df)} SAMs with {len(sam_features_df.columns)} columns.")
    
    return rf_model, sam_features_df

def create_improved_bias_rate_plots(analysis_df, rf_model, output_dir):
    """Create improved bias rate vs feature plots"""
    
    print("Creating improved bias rate plots...")
    
    # Features to analyze are the ones used by the model
    features_to_test = rf_model.feature_names_in_
    print(f"Plotting bias rates for features: {features_to_test}")
    
    # Check that required columns 'true_label' and feature names exist
    required_cols = ['true_label'] + list(features_to_test)
    if not all(col in analysis_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in analysis_df.columns]
        raise ValueError(f"Analysis dataframe is missing required columns: {missing}")

    # Create improved bias rate vs feature value plots
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    axes = axes.flatten()
    
    for i, feature in enumerate(features_to_test):
        if i >= 5:
            break
            
        ax = axes[i]
        
        # Create bins and calculate bias rate in each bin
        # Ensure we don't have NaNs in the feature or label column for this plot
        plot_df = analysis_df[['true_label', feature]].dropna()
        feature_vals = plot_df[feature]
        bias_labels = plot_df['true_label']

        if len(feature_vals) == 0:
            print(f"Skipping feature '{feature}' due to no valid data points.")
            continue
        
        # Create 8 bins (larger bins for less noise)
        n_bins = 10
        bins = np.linspace(np.percentile(feature_vals, 1), np.percentile(feature_vals, 99), n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bias_rates = []
        bin_counts = []
        
        for j in range(len(bins) - 1):
            # Make the last bin inclusive of the max value
            is_last_bin = (j == len(bins) - 2)
            if is_last_bin:
                bin_mask = (feature_vals >= bins[j]) & (feature_vals <= bins[j + 1])
            else:
                bin_mask = (feature_vals >= bins[j]) & (feature_vals < bins[j + 1])
            
            if bin_mask.sum() > 0:
                bias_rate = bias_labels[bin_mask].mean()
                bias_rates.append(bias_rate)
                bin_counts.append(bin_mask.sum())
            else:
                bias_rates.append(0)
                bin_counts.append(0)
        
        # Plot bias rate - only show points with sufficient samples to avoid extreme 0%/100% values
        valid_mask = np.array(bin_counts) >= 10  # Only show bins with at least 10 samples
        valid_centers = np.array(bin_centers)[valid_mask]
        valid_rates = np.array(bias_rates)[valid_mask]
        
        # Add sample size information (gray bars)
        ax2 = ax.twinx()
        ax2.bar(bin_centers, bin_counts, alpha=0.3, color='gray', width=(bins[1] - bins[0]) * 0.8)
        ax2.set_ylabel('Sample Count', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        
        # Plot bias rate line (only for bins with sufficient data)
        if len(valid_centers) > 0:
            ax.plot(valid_centers, valid_rates, 'o-', color='red', linewidth=2, markersize=6)
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Bias Rate')
        ax.set_title(f'Bias Rate vs {feature}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(bias_rates) * 1.1 if bias_rates and max(bias_rates) > 0 else 1)
        
        # Add legend to the first plot (top left)
        if i == 0:
            from matplotlib.lines import Line2D
            from matplotlib.patches import Rectangle
            legend_elements = [
                Line2D([0], [0], color='red', marker='o', linewidth=2, markersize=6, label='Bias Rate'),
                Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.3, label='Sample Count')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'improved_bias_rate_vs_features.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved improved bias rate plot to: {output_path}")
    plt.close()

def main():
    """Main function to generate improved bias plots"""
    print("Generating Improved Bias Plots")
    print("=" * 40)
    
    config = PathConfig()
    output_dir = config.FIGURES_DIR / "improved_bias_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        rf_model, analysis_df = load_final_model_results(config)
        
        # Generate improved plots
        create_improved_bias_rate_plots(analysis_df, rf_model, output_dir)
        
        print("Analysis complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 