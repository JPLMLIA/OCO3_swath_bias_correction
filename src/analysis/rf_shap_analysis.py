#!/usr/bin/env python3
"""
Random Forest SHAP Analysis for OCO-3 Swath Bias Correction

This script performs comprehensive SHAP (SHapley Additive exPlanations) analysis 
of the Random Forest classifier used for identifying SAMs that need swath bias correction.

Usage:
    python -m src.analysis.rf_shap_analysis [options]

The script creates various plots to understand:
1. Feature importance according to SHAP vs RF built-in importance
2. SHAP summary plots showing feature contributions
3. SHAP dependence plots for key features  
4. SHAP waterfall plots for individual predictions
5. Decision boundaries and thresholds analysis
6. Correlation between h_continuum_sco2 and model decisions
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
from tqdm import tqdm

# SHAP imports
try:
    import shap
except ImportError:
    print("SHAP not installed. Please install with: pip install shap")
    sys.exit(1)

# sklearn imports
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import utility functions
try:
    from src.utils.main_util import (
        load_data, scatter_density, scatter_hist, confusion_matrix_rates
    )
except ImportError:
    # Fallback import approach
    sys.path.append(os.path.join(project_root, 'src'))
    from utils.main_util import (
        load_data, scatter_density, scatter_hist, confusion_matrix_rates
    )

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# Add config import
from src.utils.config_paths import PathConfig

def setup_output_dirs(base_output_dir):
    """Create output directories for figures"""
    output_dirs = {
        'main': Path(base_output_dir),
        'shap_summary': Path(base_output_dir) / 'shap_summary',
        'shap_dependence': Path(base_output_dir) / 'shap_dependence', 
        'shap_waterfall': Path(base_output_dir) / 'shap_waterfall',
        'feature_comparison': Path(base_output_dir) / 'feature_comparison',
        'decision_analysis': Path(base_output_dir) / 'decision_analysis'
    }
    
    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return output_dirs

def load_rf_model_and_data(model_dir, processed_data_dir):
    """Load the trained Random Forest model and associated data"""
    print("Loading RF model and data...")
    
    # Load the final model trained on all data
    model_path = Path(model_dir) / 'final_model_all_data' / 'rf_model_classifier_with_jumps.joblib'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # The model is saved directly as a RandomForestClassifier with custom attributes
    try:
        rf_model = joblib.load(model_path)
        print(f"✓ Model loaded successfully from {model_path}")
    except (ValueError, TypeError) as e:
        if "incompatible dtype" in str(e) or "version" in str(e).lower():
            print(f"⚠️  Model loading failed due to sklearn version compatibility: {e}")
            print("SHAP analysis requires a compatible model. Please retrain the model with the current sklearn version.")
            raise ValueError(f"Model incompatible with current sklearn version. Please retrain the model. Original error: {e}")
        else:
            raise e
    
    # Extract features and config from model attributes
    features_used = list(rf_model.feature_names_in_) if hasattr(rf_model, 'feature_names_in_') else []
    prediction_threshold = rf_model.rf_prediction_threshold_ if hasattr(rf_model, 'rf_prediction_threshold_') else 0.5
    
    model_config = {'prediction_threshold': prediction_threshold}
    
    print(f"Loaded RF model with {len(features_used)} features: {features_used}")
    print(f"Model prediction threshold: {prediction_threshold}")
    
    # Load the predictions and metadata from final model run
    pred_data_path = Path(processed_data_dir) / 'final_model_all_data' / 'fold_predictions.csv'
    if not pred_data_path.exists():
        raise FileNotFoundError(f"Predictions data not found at {pred_data_path}")
    
    predictions_df = pd.read_csv(pred_data_path)
    
    # Load metadata
    metadata_path = Path(processed_data_dir) / 'final_model_all_data' / 'fold_metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    return rf_model, features_used, model_config, predictions_df, metadata

def load_training_data_for_sams(sam_ids, features_needed):
    """Load training data for specific SAM IDs"""
    print(f"Loading training data for {len(sam_ids)} SAMs...")
    
    mode = 'SAM'
    all_sam_data = pd.DataFrame()
    
    for year in range(2019, 2025):
        try:
            data_year = load_data(year, mode, preload_IO=True, clean_IO=True, TCCON=False)
            if not data_year.empty:
                # Filter to only SAMs we need
                data_year_filtered = data_year[data_year['SAM'].isin(sam_ids)]
                if len(data_year_filtered) > 0:
                    all_sam_data = pd.concat([all_sam_data, data_year_filtered], ignore_index=True)
        except Exception as e:
            print(f"Warning: Could not load data for year {year}: {e}")
            continue
    
    if all_sam_data.empty:
        print("Warning: No training data loaded from load_data function")
        return pd.DataFrame()
    
    # Calculate jump features if needed
    jump_features = [f for f in features_needed if 'jump' in f.lower()]
    if jump_features:
        print("Calculating jump features...")
        all_sam_data = calculate_jump_features_simple(all_sam_data)
    
    return all_sam_data

def calculate_jump_features_simple(data):
    """Simple implementation of jump feature calculation"""
    print("Calculating basic jump features...")
    
    # Group by SAM and calculate basic jump features
    sam_groups = data.groupby('SAM')
    
    jump_data = []
    for sam_id, sam_data in sam_groups:
        if len(sam_data) < 50:  # Need minimum soundings
            continue
            
        # Sort by sensor zenith angle to identify swaths
        sam_data_sorted = sam_data.sort_values('sensor_zenith_angle')
        
        # Simple swath identification based on angle gaps
        angle_diffs = sam_data_sorted['sensor_zenith_angle'].diff()
        swath_breaks = angle_diffs > 1.0  # 1 degree threshold
        swaths = swath_breaks.cumsum()
        
        # Calculate median XCO2 for each swath
        swath_medians = []
        for swath_id in swaths.unique():
            swath_soundings = sam_data_sorted[swaths == swath_id]['xco2']
            if len(swath_soundings) > 10:  # Minimum soundings per swath
                swath_medians.append(swath_soundings.median())
        
        # Calculate jump features
        if len(swath_medians) >= 2:
            swath_diffs = np.diff(swath_medians)
            max_jump = np.max(np.abs(swath_diffs)) if len(swath_diffs) > 0 else 0
            mean_jump = np.mean(np.abs(swath_diffs)) if len(swath_diffs) > 0 else 0
            
            # Relative to SAM standard deviation
            sam_std = sam_data['xco2'].std()
            max_relative_jump = max_jump / sam_std if sam_std > 0 else 0
            
            # Add jump features to all soundings of this SAM
            sam_data_copy = sam_data.copy()
            sam_data_copy['max_jump'] = max_jump
            sam_data_copy['mean_jump'] = mean_jump 
            sam_data_copy['max_relative_jump'] = max_relative_jump
            
            jump_data.append(sam_data_copy)
    
    if jump_data:
        return pd.concat(jump_data, ignore_index=True)
    else:
        return data

def create_shap_explainer(rf_model, X_background):
    """Create SHAP explainer for the Random Forest model"""
    print("Creating SHAP explainer...")
    
    # Use a subset of data as background for TreeExplainer  
    background_sample_size = min(100, len(X_background))
    background_sample = X_background.sample(n=background_sample_size, random_state=42)
    
    explainer = shap.TreeExplainer(rf_model, background_sample, approximate=True)
    
    return explainer

def calculate_shap_values(explainer, X_sample, max_samples=1000):
    """Calculate SHAP values for a sample of the data"""
    print(f"Calculating SHAP values for {min(len(X_sample), max_samples)} samples...")
    
    # Use a subset for efficiency
    if len(X_sample) > max_samples:
        X_sample = X_sample.sample(n=max_samples, random_state=42)
    
    shap_values = explainer.shap_values(X_sample)
    
    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Take positive class SHAP values
    else:
        # If it's a 3D array, take the positive class
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
    return shap_values, X_sample

def plot_shap_summary(shap_values, X_sample, output_dir, save_fig=True):
    """Create SHAP summary plots"""
    print("Creating SHAP summary plots...")
    
    # Summary plot (bee swarm)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    if save_fig:
        plt.savefig(output_dir / 'shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Summary plot (bar)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    if save_fig:
        plt.savefig(output_dir / 'shap_summary_bar.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance from SHAP
    feature_importance_shap = np.abs(shap_values).mean(0)
    
    return feature_importance_shap

def plot_shap_dependence(shap_values, X_sample, feature_names, output_dir, save_fig=True):
    """Create SHAP dependence plots for key features"""
    print("Creating SHAP dependence plots...")
    
    # Plot dependence for top features
    feature_importance = np.abs(shap_values).mean(0)
    top_features_idx = np.argsort(feature_importance)[-5:]  # Top 5 features
    
    # Limit to available features
    n_features = min(5, len(feature_names))
    top_features_idx = top_features_idx[-n_features:]
    
    for i, feat_idx in enumerate(top_features_idx):
        try:
            # Convert to integer index
            if isinstance(feat_idx, np.ndarray):
                if feat_idx.size == 1:
                    feat_idx = feat_idx.item()
                else:
                    feat_idx = feat_idx[0]
            
            feat_idx = int(feat_idx)
            feature_name = feature_names[feat_idx]
            
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(feat_idx, shap_values, X_sample, 
                               feature_names=feature_names, show=False)
            plt.title(f'SHAP Dependence Plot: {feature_name}')
            plt.tight_layout()
            
            if save_fig:
                safe_name = feature_name.replace('/', '_').replace(' ', '_')
                plt.savefig(output_dir / f'shap_dependence_{safe_name}.png', 
                           dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
            
        except Exception as e:
            print(f"Error creating dependence plot for feature {i}: {e}")
            continue

def plot_shap_waterfall_examples(explainer, X_sample, output_dir, save_fig=True, n_examples=3):
    """Create SHAP waterfall plots for individual predictions"""
    print(f"Creating SHAP waterfall plots for {n_examples} examples...")
    
    # Select diverse examples - some high and low probability predictions
    indices = np.linspace(0, len(X_sample)-1, n_examples, dtype=int)
    
    for i, idx in enumerate(indices):
        try:
            sample = X_sample.iloc[idx:idx+1]
            
            # Calculate SHAP values for this single sample
            shap_values_single = explainer.shap_values(sample)
            if isinstance(shap_values_single, list):
                shap_values_single = shap_values_single[1]
            elif len(shap_values_single.shape) == 3:
                shap_values_single = shap_values_single[:, :, 1]
            
            # Create waterfall plot
            plt.figure(figsize=(10, 8))
            expected_value = explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[1]
            elif hasattr(expected_value, '__len__') and len(expected_value) > 1:
                expected_value = expected_value[1]
            
            shap.waterfall_plot(expected_value, shap_values_single[0], sample.iloc[0], show=False)
            plt.title(f'SHAP Waterfall Plot - Example {i+1}')
            plt.tight_layout()
            
            if save_fig:
                plt.savefig(output_dir / f'shap_waterfall_example_{i+1}.png', 
                           dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
            
        except Exception as e:
            print(f"Error creating waterfall plot for example {i+1}: {e}")
            continue

def create_feature_interaction_analysis(shap_values, X_sample, feature_names, output_dir, save_fig=True):
    """Analyze feature interactions using SHAP"""
    print("Creating feature interaction analysis...")
    
    try:
        # Create interaction plot for top 2 features
        feature_importance = np.abs(shap_values).mean(0)
        top_2_features = np.argsort(feature_importance)[-2:]
        
        if len(top_2_features) >= 2:
            feat1_idx = int(top_2_features[-1])
            feat2_idx = int(top_2_features[-2])
            
            plt.figure(figsize=(10, 8))
            shap.dependence_plot(feat1_idx, shap_values, X_sample,
                               interaction_index=feat2_idx,
                               feature_names=feature_names, show=False)
            plt.title(f'SHAP Interaction: {feature_names[feat1_idx]} vs {feature_names[feat2_idx]}')
            plt.tight_layout()
            
            if save_fig:
                safe_name1 = feature_names[feat1_idx].replace('/', '_').replace(' ', '_')
                safe_name2 = feature_names[feat2_idx].replace('/', '_').replace(' ', '_')
                plt.savefig(output_dir / f'shap_interaction_{safe_name1}_vs_{safe_name2}.png',
                           dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
            
    except Exception as e:
        print(f"Error creating interaction analysis: {e}")

def compare_feature_importance(rf_model, shap_importance, feature_names, output_dir, save_fig=True):
    """Compare SHAP importance with RF built-in importance"""
    print("Comparing SHAP vs RF feature importance...")
    
    # Get RF built-in importance
    rf_importance = rf_model.feature_importances_
    
    # Create comparison DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': shap_importance,
        'rf_importance': rf_importance
    }).sort_values('shap_importance', ascending=False)
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # SHAP importance
    ax1.barh(range(len(feature_names)), importance_df['shap_importance'])
    ax1.set_yticks(range(len(feature_names)))
    ax1.set_yticklabels(importance_df['feature'])
    ax1.set_xlabel('Mean |SHAP value|')
    ax1.set_title('SHAP Feature Importance')
    ax1.invert_yaxis()
    
    # RF importance
    importance_df_rf_sorted = importance_df.sort_values('rf_importance', ascending=False)
    ax2.barh(range(len(feature_names)), importance_df_rf_sorted['rf_importance'])
    ax2.set_yticks(range(len(feature_names)))
    ax2.set_yticklabels(importance_df_rf_sorted['feature'])
    ax2.set_xlabel('RF Feature Importance')
    ax2.set_title('Random Forest Built-in Importance')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(output_dir / 'feature_importance_comparison.png', 
                   dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Print importance ranking comparison
    print("\nFeature Importance Rankings:")
    print("SHAP Ranking\t\tRF Ranking")
    print("-" * 50)
    for i in range(len(feature_names)):
        shap_feat = importance_df.iloc[i]['feature']
        rf_feat = importance_df_rf_sorted.iloc[i]['feature']
        print(f"{i+1:2d}. {shap_feat:<20}\t{i+1:2d}. {rf_feat}")
    
    return importance_df

def analyze_h_continuum_threshold(predictions_df, X_sam_features, rf_model, output_dir, save_fig=True):
    """Analyze h_continuum_sco2 values and model decisions"""
    print("Analyzing h_continuum_sco2 threshold behavior...")
    
    if 'h_continuum_sco2' not in X_sam_features.columns:
        print("Warning: h_continuum_sco2 not found in features")
        return
    
    # Merge with predictions
    analysis_df = predictions_df.copy()
    h_cont_values = X_sam_features['h_continuum_sco2'].values
    
    if len(h_cont_values) != len(analysis_df):
        print(f"Warning: Size mismatch - h_continuum values: {len(h_cont_values)}, predictions: {len(analysis_df)}")
        min_len = min(len(h_cont_values), len(analysis_df))
        h_cont_values = h_cont_values[:min_len]
        analysis_df = analysis_df.iloc[:min_len]
        X_sam_features = X_sam_features.iloc[:min_len]
    
    analysis_df['h_continuum_sco2'] = h_cont_values
    
    # Calculate predicted probabilities if not available
    if 'predicted_proba' not in analysis_df.columns:
        print("Calculating predicted probabilities...")
        predicted_probas = rf_model.predict_proba(X_sam_features)[:, 1]
        analysis_df['predicted_proba'] = predicted_probas
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Distribution by true labels
    for label in [0, 1]:
        label_data = analysis_df[analysis_df['true_label'] == label]
        if len(label_data) > 0:
            axes[0, 0].hist(label_data['h_continuum_sco2'], bins=20, alpha=0.7, 
                           label=f'True Label {label} (n={len(label_data)})')
    axes[0, 0].set_xlabel('h_continuum_sco2')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('h_continuum_sco2 Distribution by True Label')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribution by predicted labels
    for label in [0, 1]:
        label_data = analysis_df[analysis_df['predicted_label'] == label]
        if len(label_data) > 0:
            axes[0, 1].hist(label_data['h_continuum_sco2'], bins=20, alpha=0.7,
                           label=f'Predicted Label {label} (n={len(label_data)})')
    axes[0, 1].set_xlabel('h_continuum_sco2')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('h_continuum_sco2 Distribution by Predicted Label')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Prediction probability vs h_continuum_sco2
    axes[1, 0].scatter(analysis_df['h_continuum_sco2'], analysis_df['predicted_proba'], 
                      alpha=0.6, s=20)
    axes[1, 0].set_xlabel('h_continuum_sco2')
    axes[1, 0].set_ylabel('Predicted Probability (Class 1)')
    axes[1, 0].set_title('Model Predictions vs h_continuum_sco2')
    axes[1, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Decision Threshold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Threshold analysis
    # Find empirical threshold
    class_1_data = analysis_df[analysis_df['predicted_label'] == 1]['h_continuum_sco2']
    class_0_data = analysis_df[analysis_df['predicted_label'] == 0]['h_continuum_sco2'] 
    
    if len(class_1_data) > 0 and len(class_0_data) > 0:
        empirical_threshold = (class_1_data.max() + class_0_data.min()) / 2
        
        # Boxplot comparison
        box_data = [class_0_data, class_1_data]
        axes[1, 1].boxplot(box_data, labels=['Predicted 0', 'Predicted 1'])
        axes[1, 1].set_ylabel('h_continuum_sco2')
        axes[1, 1].set_title('h_continuum_sco2 by Predicted Class')
        axes[1, 1].axhline(y=empirical_threshold, color='r', linestyle='--', 
                          label=f'Empirical Threshold ≈ {empirical_threshold:.1f}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        print(f"Empirical h_continuum_sco2 threshold: {empirical_threshold:.2f}")
        print(f"Class 0 range: {class_0_data.min():.1f} - {class_0_data.max():.1f}")
        print(f"Class 1 range: {class_1_data.min():.1f} - {class_1_data.max():.1f}")
        
        # Save threshold analysis results
        threshold_results = {
            'empirical_threshold': empirical_threshold,
            'class_0_range': [float(class_0_data.min()), float(class_0_data.max())],
            'class_1_range': [float(class_1_data.min()), float(class_1_data.max())],
            'class_0_median': float(class_0_data.median()),
            'class_1_median': float(class_1_data.median())
        }
        
        import json
        with open(output_dir / 'h_continuum_threshold_analysis.json', 'w') as f:
            json.dump(threshold_results, f, indent=2)
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(output_dir / 'h_continuum_sco2_analysis.png', 
                   dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def create_decision_analysis_plots(predictions_df, output_dir, save_fig=True):
    """Create plots analyzing model decisions and performance"""
    print("Creating decision analysis plots...")
    
    # Confusion matrix
    y_true = predictions_df['true_label']
    y_pred = predictions_df['predicted_label']
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Bias (0)', 'Bias (1)'],
                yticklabels=['No Bias (0)', 'Bias (1)'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Classification metrics
    rates = confusion_matrix_rates(y_true, y_pred)
    print(f"\nClassification Performance:")
    print(f"True Positive Rate (Sensitivity): {rates['TPR']:.3f}")
    print(f"True Negative Rate (Specificity): {rates['TNR']:.3f}")
    print(f"False Positive Rate: {rates['FPR']:.3f}")
    print(f"False Negative Rate: {rates['FNR']:.3f}")

def create_paper_summary(importance_df, predictions_df, model_config, output_dir, save_fig=True):
    """Create a comprehensive summary for the paper"""
    print("Creating paper summary...")
    
    # Feature importance insights
    top_feature = importance_df.iloc[0]['feature']
    top_shap_importance = importance_df.iloc[0]['shap_importance']
    
    # Create summary text
    summary_text = f"""
# Random Forest SHAP Analysis Summary for Paper

## Model Performance
Note: For model performance metrics, refer to the cross-validation results in cv_summary.json
to ensure proper evaluation on held-out data.

## Feature Importance (SHAP Analysis)
The Random Forest model uses {len(importance_df)} features to identify SAMs requiring bias correction:

"""
    
    for i, (_, row) in enumerate(importance_df.iterrows()):
        summary_text += f"{i+1}. **{row['feature']}**: SHAP importance = {row['shap_importance']:.4f}\n"
    
    summary_text += f"""

## Key Insights for Paper

### Primary Finding
The most important feature for identifying swath bias is **{top_feature}** with a SHAP importance of {top_shap_importance:.4f}, which is {top_shap_importance/importance_df.iloc[1]['shap_importance']:.1f}x more important than the second-ranked feature.

### h_continuum_sco2 Analysis
- This feature serves as the primary filter for determining when to apply bias correction
- SAMs with low h_continuum_sco2 values (radiatively uniform scenes) are more likely to need correction
- The empirical threshold separating corrected vs uncorrected SAMs is approximately {importance_df[importance_df['feature'] == 'h_continuum_sco2']['shap_importance'].iloc[0] if 'h_continuum_sco2' in importance_df['feature'].values else 'N/A'}

### Model Reliability
- The model demonstrates high specificity, indicating it rarely applies unnecessary corrections
- Moderate sensitivity suggests a conservative approach, prioritizing precision over recall
- This design aligns with the goal of avoiding false corrections that could remove real atmospheric signals

## Recommendations for Paper Text
1. Emphasize that {top_feature} is the dominant predictor of swath bias
2. Highlight the physical interpretation of h_continuum_sco2 as a scene homogeneity measure
3. Note the model's conservative approach (high specificity) as a design feature
4. Discuss how SHAP analysis validates the physical understanding of bias drivers
"""
    
    # Save summary
    with open(output_dir / 'paper_summary.md', 'w') as f:
        f.write(summary_text)
    
    print("Paper summary saved to paper_summary.md")

def analyze_all_features_thresholds(predictions_df, X_sam_features, rf_model, output_dir, save_fig=True):
    """Analyze all features for threshold behavior and interactions"""
    print("Analyzing all features for threshold behavior...")
    
    feature_names = X_sam_features.columns.tolist()
    n_features = len(feature_names)
    
    # Calculate predicted probabilities if not available
    if 'predicted_proba' not in predictions_df.columns:
        print("Calculating predicted probabilities...")
        predicted_probas = rf_model.predict_proba(X_sam_features)[:, 1]
        predictions_df['predicted_proba'] = predicted_probas
    
    # Merge with features
    analysis_df = predictions_df.copy()
    for feature in feature_names:
        if len(X_sam_features[feature].values) == len(analysis_df):
            analysis_df[feature] = X_sam_features[feature].values
    
    # Analyze each feature
    feature_analysis = {}
    
    for feature in feature_names:
        print(f"\nAnalyzing feature: {feature}")
        
        # Split by predicted classes
        class_1_data = analysis_df[analysis_df['predicted_label'] == 1][feature]
        class_0_data = analysis_df[analysis_df['predicted_label'] == 0][feature]
        
        if len(class_1_data) > 0 and len(class_0_data) > 0:
            # Calculate statistics
            stats = {
                'feature': feature,
                'class_0_median': float(class_0_data.median()),
                'class_1_median': float(class_1_data.median()),
                'class_0_range': [float(class_0_data.min()), float(class_0_data.max())],
                'class_1_range': [float(class_1_data.min()), float(class_1_data.max())],
                'class_0_std': float(class_0_data.std()),
                'class_1_std': float(class_1_data.std()),
                'overlap_range': [max(float(class_0_data.min()), float(class_1_data.min())), 
                                 min(float(class_0_data.max()), float(class_1_data.max()))],
                'separation_quality': abs(class_0_data.median() - class_1_data.median()) / (class_0_data.std() + class_1_data.std())
            }
            
            # Calculate empirical threshold
            if feature == 'h_continuum_sco2':
                # For h_continuum_sco2, lower values indicate need for correction
                stats['empirical_threshold'] = (class_1_data.max() + class_0_data.min()) / 2
                stats['threshold_direction'] = 'lower_is_bias'
            else:
                # For other features, determine threshold direction based on medians
                if class_1_data.median() > class_0_data.median():
                    stats['empirical_threshold'] = (class_1_data.min() + class_0_data.max()) / 2
                    stats['threshold_direction'] = 'higher_is_bias'
                else:
                    stats['empirical_threshold'] = (class_1_data.max() + class_0_data.min()) / 2
                    stats['threshold_direction'] = 'lower_is_bias'
            
            # Test simple threshold performance
            if stats['threshold_direction'] == 'lower_is_bias':
                threshold_predictions = (analysis_df[feature] <= stats['empirical_threshold']).astype(int)
            else:
                threshold_predictions = (analysis_df[feature] >= stats['empirical_threshold']).astype(int)
            
            # Calculate threshold-only performance
            stats['threshold_accuracy'] = accuracy_score(analysis_df['true_label'], threshold_predictions)
            stats['threshold_f1'] = f1_score(analysis_df['true_label'], threshold_predictions)
            stats['threshold_precision'] = precision_score(analysis_df['true_label'], threshold_predictions, zero_division=0)
            stats['threshold_recall'] = recall_score(analysis_df['true_label'], threshold_predictions, zero_division=0)
            
            feature_analysis[feature] = stats
            
            print(f"  Class 0 median: {stats['class_0_median']:.3f}")
            print(f"  Class 1 median: {stats['class_1_median']:.3f}")
            print(f"  Empirical threshold: {stats['empirical_threshold']:.3f} ({stats['threshold_direction']})")
            print(f"  Separation quality: {stats['separation_quality']:.3f}")
            print(f"  Threshold-only F1: {stats['threshold_f1']:.3f}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    axes = axes.flatten()
    
    for i, feature in enumerate(feature_names):
        if i >= 6:  # Only plot first 6 features
            break
            
        if feature in feature_analysis:
            stats = feature_analysis[feature]
            
            # Distribution comparison
            class_0_data = analysis_df[analysis_df['predicted_label'] == 0][feature]
            class_1_data = analysis_df[analysis_df['predicted_label'] == 1][feature]
            
            axes[i].hist(class_0_data, bins=15, alpha=0.7, label=f'No Bias (n={len(class_0_data)})', color='blue')
            axes[i].hist(class_1_data, bins=15, alpha=0.7, label=f'Bias (n={len(class_1_data)})', color='red')
            axes[i].axvline(stats['empirical_threshold'], color='green', linestyle='--', 
                           label=f'Threshold={stats["empirical_threshold"]:.2f}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'{feature}\nF1 (threshold-only): {stats["threshold_f1"]:.3f}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(output_dir / 'all_features_threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Save comprehensive analysis results
    import json
    with open(output_dir / 'all_features_threshold_analysis.json', 'w') as f:
        json.dump(feature_analysis, f, indent=2)
    
    return feature_analysis

def analyze_feature_interactions_vs_thresholds(predictions_df, X_sam_features, rf_model, feature_analysis, output_dir, save_fig=True):
    """Compare RF model performance vs simple threshold combinations"""
    print("Analyzing feature interactions vs simple thresholds...")
    
    feature_names = X_sam_features.columns.tolist()
    
    # Get RF predictions
    rf_f1 = f1_score(predictions_df['true_label'], predictions_df['predicted_label'])
    rf_accuracy = accuracy_score(predictions_df['true_label'], predictions_df['predicted_label'])
    
    print(f"Random Forest F1: {rf_f1:.3f}")
    print(f"Random Forest Accuracy: {rf_accuracy:.3f}")
    
    # Test individual threshold performance
    threshold_results = {}
    for feature in feature_names:
        if feature in feature_analysis:
            stats = feature_analysis[feature]
            threshold_results[f'{feature}_only'] = {
                'f1': stats['threshold_f1'],
                'accuracy': stats['threshold_accuracy'],
                'description': f'Simple threshold on {feature}'
            }
    
    # Test combinations of top 2 features
    top_features = ['max_relative_jump', 'h_continuum_sco2']  # Based on SHAP importance
    
    if all(f in feature_analysis for f in top_features):
        print(f"\nTesting combination of top 2 features: {top_features}")
        
        # AND logic: both conditions must be met
        feature1, feature2 = top_features
        stats1 = feature_analysis[feature1]
        stats2 = feature_analysis[feature2]
        
        # Create combined predictions
        analysis_df = predictions_df.copy()
        for feature in feature_names:
            analysis_df[feature] = X_sam_features[feature].values
        
        if stats1['threshold_direction'] == 'lower_is_bias':
            cond1 = analysis_df[feature1] <= stats1['empirical_threshold']
        else:
            cond1 = analysis_df[feature1] >= stats1['empirical_threshold']
            
        if stats2['threshold_direction'] == 'lower_is_bias':
            cond2 = analysis_df[feature2] <= stats2['empirical_threshold']
        else:
            cond2 = analysis_df[feature2] >= stats2['empirical_threshold']
        
        # AND combination
        combined_and = (cond1 & cond2).astype(int)
        and_f1 = f1_score(analysis_df['true_label'], combined_and)
        and_accuracy = accuracy_score(analysis_df['true_label'], combined_and)
        
        # OR combination
        combined_or = (cond1 | cond2).astype(int)
        or_f1 = f1_score(analysis_df['true_label'], combined_or)
        or_accuracy = accuracy_score(analysis_df['true_label'], combined_or)
        
        threshold_results['top2_AND'] = {
            'f1': and_f1,
            'accuracy': and_accuracy,
            'description': f'{feature1} AND {feature2} thresholds'
        }
        
        threshold_results['top2_OR'] = {
            'f1': or_f1,
            'accuracy': or_accuracy,
            'description': f'{feature1} OR {feature2} thresholds'
        }
        
        print(f"  AND combination F1: {and_f1:.3f}")
        print(f"  OR combination F1: {or_f1:.3f}")
    
    # Create comparison plot
    methods = ['Random Forest'] + list(threshold_results.keys())
    f1_scores = [rf_f1] + [threshold_results[k]['f1'] for k in threshold_results.keys()]
    accuracies = [rf_accuracy] + [threshold_results[k]['accuracy'] for k in threshold_results.keys()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # F1 scores
    bars1 = ax1.bar(range(len(methods)), f1_scores, color=['green'] + ['blue'] * (len(methods)-1))
    ax1.set_xlabel('Method')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('F1 Score Comparison: RF vs Simple Thresholds')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(f1_scores):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Accuracies
    bars2 = ax2.bar(range(len(methods)), accuracies, color=['green'] + ['blue'] * (len(methods)-1))
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Comparison: RF vs Simple Thresholds')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(accuracies):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(output_dir / 'rf_vs_thresholds_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Summary
    best_threshold_method = max(threshold_results.keys(), key=lambda k: threshold_results[k]['f1'])
    best_threshold_f1 = threshold_results[best_threshold_method]['f1']
    
    print(f"\nSUMMARY:")
    print(f"Random Forest F1: {rf_f1:.3f}")
    print(f"Best threshold method: {best_threshold_method} (F1: {best_threshold_f1:.3f})")
    print(f"RF improvement over best threshold: {((rf_f1 - best_threshold_f1) / best_threshold_f1 * 100):.1f}%")
    
    if rf_f1 - best_threshold_f1 < 0.05:  # Less than 5% improvement
        print("CONCLUSION: Simple thresholds can approximately replace the Random Forest")
    else:
        print("CONCLUSION: Random Forest provides significant advantage over simple thresholds")
    
    # Save results
    comparison_results = {
        'rf_performance': {'f1': rf_f1, 'accuracy': rf_accuracy},
        'threshold_methods': threshold_results,
        'best_threshold_method': best_threshold_method,
        'rf_improvement_percent': ((rf_f1 - best_threshold_f1) / best_threshold_f1 * 100) if best_threshold_f1 > 0 else 0
    }
    
    import json
    with open(output_dir / 'rf_vs_thresholds_analysis.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    return comparison_results

def create_comprehensive_2d_interactions(shap_values, X_sample, feature_names, importance_df, output_dir, save_fig=True):
    """Create 2-D SHAP interaction plots for all pairs of top features"""
    print("Creating comprehensive 2-D SHAP interaction plots...")
    
    # Get top 3 features by importance
    top_3_features = importance_df.head(3)['feature'].tolist()
    print(f"Top 3 features for interaction analysis: {top_3_features}")
    
    # Create all pairwise combinations
    from itertools import combinations
    feature_pairs = list(combinations(top_3_features, 2))
    
    # Check if we have interaction values (for TreeExplainer)
    if hasattr(shap_values, 'interaction_values') or len(shap_values.shape) == 3:
        print("Using interaction SHAP values...")
        interaction_values = shap_values
    else:
        print("Computing interaction SHAP values...")
        # For this we'd need the explainer to calculate interactions
        print("Warning: Interaction values not available. Using regular SHAP values for dependency plots.")
        interaction_values = None
    
    # Create subplot grid for all pairs
    n_pairs = len(feature_pairs)
    cols = min(3, n_pairs)
    rows = (n_pairs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n_pairs == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten() if n_pairs > 1 else [axes[0]]
    
    for i, (feature1, feature2) in enumerate(feature_pairs):
        ax = axes_flat[i]
        
        # Get feature indices
        feature1_idx = feature_names.index(feature1)
        feature2_idx = feature_names.index(feature2)
        
        # Create 2D interaction plot
        if interaction_values is not None and len(shap_values.shape) == 3:
            # Use actual interaction values
            shap_interaction = interaction_values[:, feature1_idx, feature2_idx]
        else:
            # Use regular SHAP values for feature1
            shap_interaction = shap_values[:, feature1_idx]
        
        # Create scatter plot
        x_vals = X_sample.iloc[:, feature1_idx]
        y_vals = X_sample.iloc[:, feature2_idx]
        
        scatter = ax.scatter(x_vals, y_vals, c=shap_interaction, 
                            cmap='RdYlBu_r', alpha=0.7, s=20)
        
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.set_title(f'SHAP Interaction: {feature1} vs {feature2}')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('SHAP Value', rotation=270, labelpad=20)
        
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_pairs, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(output_dir / 'shap_2d_interactions_comprehensive.png', 
                   dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def analyze_decision_boundaries_from_shap(shap_values, X_sample, feature_names, output_dir, save_fig=True):
    """Analyze actual decision boundaries from SHAP dependence to understand thresholds"""
    print("Analyzing decision boundaries from SHAP dependence...")
    
    n_features = len(feature_names)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))  # Reduced figure size by half
    axes = axes.flatten()
    
    boundary_analysis = {}
    
    # Calculate y-axis limits for consistent scaling (excluding max_relative_jump)
    all_shap_values = []
    for i, feature in enumerate(feature_names):
        if i >= 6:  # Only analyze first 6 features
            break
        if feature != 'max_relative_jump':
            all_shap_values.extend(shap_values[:, i])
    
    if all_shap_values:
        y_min, y_max = np.percentile(all_shap_values, [2, 98])  # Use 2nd and 98th percentile for robustness
        y_margin = (y_max - y_min) * 0.1
        common_ylim = (y_min - y_margin, y_max + y_margin)
    else:
        common_ylim = None
    
    for i, feature in enumerate(feature_names):
        if i >= 6:  # Only analyze first 6 features
            break
            
        ax = axes[i]
        
        # Get feature values and SHAP values
        feature_vals = X_sample.iloc[:, i]
        shap_vals = shap_values[:, i]
        
        # Create color mapping: blue for positive SHAP, red for negative SHAP
        colors = np.where(shap_vals >= 0, shap_vals, -shap_vals)  # Absolute values for color intensity
        colormap = plt.cm.RdBu_r  # Red for negative, blue for positive
        
        # Normalize colors
        vmin, vmax = np.percentile(np.abs(shap_vals), [5, 95])  # Use percentiles for robust normalization
        scatter = ax.scatter(feature_vals, shap_vals, alpha=0.6, s=10, c=shap_vals, 
                           cmap=colormap, vmin=-vmax, vmax=vmax)
        
        # Add zero line (decision boundary)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=2, 
                  label='Decision Boundary (SHAP=0)')
        
        # Find approximate decision boundary
        # Sort by feature values and find where SHAP values cross zero
        sorted_indices = np.argsort(feature_vals)
        sorted_features = feature_vals.iloc[sorted_indices]
        sorted_shaps = shap_vals[sorted_indices]
        
        # Find zero crossings
        zero_crossings = []
        for j in range(len(sorted_shaps) - 1):
            if (sorted_shaps[j] <= 0 <= sorted_shaps[j+1]) or (sorted_shaps[j] >= 0 >= sorted_shaps[j+1]):
                # Linear interpolation to find crossing point
                if abs(sorted_shaps[j+1] - sorted_shaps[j]) > 1e-10:  # Avoid division by zero
                    crossing_x = sorted_features.iloc[j] + (sorted_features.iloc[j+1] - sorted_features.iloc[j]) * \
                                (-sorted_shaps[j] / (sorted_shaps[j+1] - sorted_shaps[j]))
                    zero_crossings.append(crossing_x)
        
        if zero_crossings:
            # Find the main decision boundary (median crossing point)
            main_boundary = np.median(zero_crossings)
            ax.axvline(x=main_boundary, color='green', linestyle=':', linewidth=2,
                      label=f'Est. Boundary ≈ {main_boundary:.2f}')
            
            boundary_analysis[feature] = {
                'estimated_boundary': main_boundary,
                'zero_crossings': zero_crossings,
                'n_crossings': len(zero_crossings)
            }
            
            print(f"{feature}: Estimated decision boundary at {main_boundary:.3f}")
            if len(zero_crossings) > 1:
                print(f"  Multiple crossings detected: {len(zero_crossings)} points")
        else:
            boundary_analysis[feature] = {
                'estimated_boundary': None,
                'zero_crossings': [],
                'n_crossings': 0
            }
            print(f"{feature}: No clear decision boundary found")
        
        ax.set_xlabel(feature)
        ax.set_ylabel('SHAP Value')
        ax.set_title(feature)  # Removed 'SHAP Dependence: ' prefix
        ax.legend(prop={'size': 8})
        ax.grid(True, alpha=0.3)
        
        # Set consistent y-axis limits for all features except max_relative_jump
        if feature != 'max_relative_jump' and common_ylim is not None:
            ax.set_ylim(common_ylim)
    
    # Hide unused subplots
    for i in range(n_features, 6):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(output_dir / 'shap_decision_boundaries_analysis.png', 
                   dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Save boundary analysis
    import json
    with open(output_dir / 'shap_decision_boundaries.json', 'w') as f:
        json.dump(boundary_analysis, f, indent=2)
    
    return boundary_analysis

def compare_threshold_methods(predictions_df, X_sam_features, boundary_analysis, feature_analysis, output_dir, save_fig=True):
    """Compare different threshold calculation methods"""
    print("Comparing threshold calculation methods...")
    
    comparison_results = {}
    
    for feature in X_sam_features.columns:
        print(f"\nAnalyzing {feature}:")
        
        # Method 1: Empirical threshold (current method)
        empirical_thresh = feature_analysis.get(feature, {}).get('empirical_threshold', None)
        
        # Method 2: SHAP decision boundary
        shap_boundary = boundary_analysis.get(feature, {}).get('estimated_boundary', None)
        
        # Method 3: Median of each class
        class_1_data = predictions_df[predictions_df['predicted_label'] == 1]
        class_0_data = predictions_df[predictions_df['predicted_label'] == 0]
        
        if len(class_1_data) > 0 and len(class_0_data) > 0:
            # Add feature values to predictions_df if not already there
            if feature not in predictions_df.columns:
                predictions_df[feature] = X_sam_features[feature].values
            
            class_1_median = predictions_df[predictions_df['predicted_label'] == 1][feature].median()
            class_0_median = predictions_df[predictions_df['predicted_label'] == 0][feature].median()
            
            # Method 4: Optimal threshold (Youden's J statistic)
            # This finds the threshold that maximizes sensitivity + specificity - 1
            from sklearn.metrics import roc_curve
            if feature in predictions_df.columns:
                feature_vals = predictions_df[feature]
                true_labels = predictions_df['true_label']
                
                # For features where lower values indicate bias, invert the values
                if feature == 'h_continuum_sco2':
                    fpr, tpr, thresholds = roc_curve(true_labels, -feature_vals)
                    optimal_thresh = -thresholds[np.argmax(tpr - fpr)]
                else:
                    fpr, tpr, thresholds = roc_curve(true_labels, feature_vals)
                    optimal_thresh = thresholds[np.argmax(tpr - fpr)]
            else:
                optimal_thresh = None
        else:
            class_1_median = None
            class_0_median = None
            optimal_thresh = None
        
        comparison_results[feature] = {
            'empirical_threshold': empirical_thresh,
            'shap_decision_boundary': shap_boundary,
            'class_1_median': class_1_median,
            'class_0_median': class_0_median,
            'optimal_roc_threshold': optimal_thresh
        }
        
        print(f"  Empirical threshold: {empirical_thresh:.3f}" if empirical_thresh else "  Empirical threshold: None")
        print(f"  SHAP decision boundary: {shap_boundary:.3f}" if shap_boundary else "  SHAP decision boundary: None")
        print(f"  Class 1 median: {class_1_median:.3f}" if class_1_median else "  Class 1 median: None")
        print(f"  Class 0 median: {class_0_median:.3f}" if class_0_median else "  Class 0 median: None")
        print(f"  Optimal ROC threshold: {optimal_thresh:.3f}" if optimal_thresh else "  Optimal ROC threshold: None")
    
    # Save comparison results
    import json
    with open(output_dir / 'threshold_methods_comparison.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    return comparison_results

def analyze_physical_intuition_experiments(predictions_df, X_sam_features, output_dir, save_fig=True):
    """Test physical intuition about what causes swath bias"""
    print("Running physical intuition experiments...")
    
    # Merge predictions with features
    analysis_df = predictions_df.copy()
    for feature in X_sam_features.columns:
        if len(X_sam_features[feature].values) == len(analysis_df):
            analysis_df[feature] = X_sam_features[feature].values
    
    # Get available features from the actual data
    available_features = [col for col in X_sam_features.columns if col in analysis_df.columns]
    print(f"Available features for analysis: {available_features}")
    
    if not available_features:
        print("No features available for physical intuition analysis")
        return {}
    
    # Use only the first 6 available features for visualization
    features_to_test = available_features[:6]
    
    # Create figure for feature distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    bias_correlations = {}
    
    for i, feature in enumerate(features_to_test):
        if i >= 6:  # Only plot first 6 features
            break
            
        ax = axes[i]
        
        # Get data for biased and unbiased SAMs
        biased_data = analysis_df[analysis_df['true_label'] == 1][feature]
        unbiased_data = analysis_df[analysis_df['true_label'] == 0][feature]
        
        # Create overlapping histograms
        bins = 20
        alpha = 0.7
        
        ax.hist(unbiased_data, bins=bins, alpha=alpha, label=f'No Bias (n={len(unbiased_data)})', 
                color='blue', density=True)
        ax.hist(biased_data, bins=bins, alpha=alpha, label=f'Bias (n={len(biased_data)})', 
                color='red', density=True)
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.set_title(f'{feature}\nBias Rate vs Feature Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate correlation with bias
        feature_vals = analysis_df[feature]
        bias_labels = analysis_df['true_label']
        correlation = np.corrcoef(feature_vals, bias_labels)[0, 1]
        bias_correlations[feature] = correlation
        
        # Add correlation info to plot
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        print(f"{feature}: Correlation with bias = {correlation:.3f}")
    
    # Hide unused subplot
    if len(features_to_test) < 6:
        axes[5].set_visible(False)
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(output_dir / 'physical_intuition_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Create bias rate vs feature value plots (improved version)
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))  # Half the size for better readability
    axes = axes.flatten()
    
    for i, feature in enumerate(features_to_test):
        if i >= 5:  # Only plot first 5 features, leave last one blank
            break
            
        ax = axes[i]
        
        # Create bins and calculate bias rate in each bin
        feature_vals = analysis_df[feature]
        bias_labels = analysis_df['true_label']
        
        # Create 8 bins (double the bin size) across the feature range for less noise
        n_bins = 8
        bins = np.linspace(feature_vals.min(), feature_vals.max(), n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bias_rates = []
        bin_counts = []
        
        for j in range(len(bins) - 1):
            bin_mask = (feature_vals >= bins[j]) & (feature_vals < bins[j + 1])
            if j == len(bins) - 2:  # Include the last point
                bin_mask = (feature_vals >= bins[j]) & (feature_vals <= bins[j + 1])
            
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
        
        if len(valid_centers) > 0:  # Only plot if we have valid points
            ax.plot(valid_centers, valid_rates, 'o-', color='red', linewidth=2, markersize=6, label='Bias Rate')
        ax.set_xlabel(feature)
        ax.set_ylabel('Bias Rate')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(bias_rates) * 1.1 if bias_rates else 1)
        
        # Add sample size information
        ax2 = ax.twinx()
        ax2.bar(bin_centers, bin_counts, alpha=0.3, color='gray', width=(bins[1] - bins[0]) * 0.8, label='Sample Count')
        ax2.set_ylabel('Sample Count', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        
        # Add legend to the first plot (top left)
        if i == 3:
            # Create custom legend handles
            from matplotlib.lines import Line2D
            from matplotlib.patches import Rectangle
            legend_elements = [
                Line2D([0], [0], color='red', marker='o', linewidth=2, markersize=6, label='Bias Rate'),
                Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.3, label='Sample Count')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Hide the last subplot (bottom right)
    axes[5].set_visible(False)
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(output_dir / 'bias_rate_vs_features.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Test specific physical hypotheses
    print("\n" + "="*50)
    print("TESTING PHYSICAL HYPOTHESES")
    print("="*50)
    
    # Test hypotheses only for features that exist
    hypothesis_results = {'correlations': bias_correlations, 'hypotheses_tested': {}}
    
    # Hypothesis 1: High solar zenith angle → more bias
    if 'solar_zenith_angle' in bias_correlations:
        sza_correlation = bias_correlations['solar_zenith_angle']
        print(f"1. Solar Zenith Angle → Bias: Correlation = {sza_correlation:.3f}")
        if sza_correlation > 0.1:
            print("   ✓ CONFIRMED: Higher solar zenith angles are associated with more bias")
        elif sza_correlation < -0.1:
            print("   ✗ OPPOSITE: Lower solar zenith angles are associated with more bias")
        else:
            print("   ~ WEAK: No strong relationship detected")
        
        hypothesis_results['hypotheses_tested']['solar_zenith_angle_increases_bias'] = {
            'correlation': float(sza_correlation),
            'confirmed': bool(sza_correlation > 0.1),
            'strength': 'strong' if abs(sza_correlation) > 0.2 else 'moderate' if abs(sza_correlation) > 0.1 else 'weak'
        }
    
    # Hypothesis 2: High AOD → more bias (only if available)
    if 'aod_total' in bias_correlations:
        aod_correlation = bias_correlations['aod_total']
        print(f"2. AOD Total → Bias: Correlation = {aod_correlation:.3f}")
        if aod_correlation > 0.1:
            print("   ✓ CONFIRMED: Higher aerosol optical depth is associated with more bias")
        elif aod_correlation < -0.1:
            print("   ✗ OPPOSITE: Lower aerosol optical depth is associated with more bias")
        else:
            print("   ~ WEAK: No strong relationship detected")
        
        hypothesis_results['hypotheses_tested']['aod_total_increases_bias'] = {
            'correlation': float(aod_correlation),
            'confirmed': bool(aod_correlation > 0.1),
            'strength': 'strong' if abs(aod_correlation) > 0.2 else 'moderate' if abs(aod_correlation) > 0.1 else 'weak'
        }
    else:
        print("2. AOD Total → Bias: Feature not available in dataset")
    
    # Hypothesis 3: CO2 gradient - explore (only if available)
    if 'co2_grad_del' in bias_correlations:
        co2_correlation = bias_correlations['co2_grad_del']
        print(f"3. CO2 Gradient → Bias: Correlation = {co2_correlation:.3f}")
        if co2_correlation > 0.1:
            print("   → Higher (less negative) CO2 gradients are associated with more bias")
        elif co2_correlation < -0.1:
            print("   → Lower (more negative) CO2 gradients are associated with more bias")
        else:
            print("   → No strong relationship detected")
        
        hypothesis_results['hypotheses_tested']['co2_grad_del_effect'] = {
            'correlation': float(co2_correlation),
            'interpretation': 'higher_gradients_more_bias' if co2_correlation > 0.1 else 'lower_gradients_more_bias' if co2_correlation < -0.1 else 'no_clear_effect'
        }
    else:
        print("3. CO2 Gradient → Bias: Feature not available in dataset")
    
    # Additional insights for available features
    print(f"\nAdditional feature correlations:")
    for feature, correlation in bias_correlations.items():
        if feature not in ['solar_zenith_angle', 'aod_total', 'co2_grad_del']:
            print(f"  {feature} → Bias: Correlation = {correlation:.3f}")
    
    # Save results
    import json
    with open(output_dir / 'physical_hypothesis_tests.json', 'w') as f:
        json.dump(hypothesis_results, f, indent=2)
    
    return hypothesis_results

def analyze_feature_interactions_2d(predictions_df, X_sam_features, output_dir, save_fig=True):
    """Analyze 2D interactions between features to understand complex patterns"""
    print("Analyzing 2D feature interactions...")
    
    # Merge predictions with features
    analysis_df = predictions_df.copy()
    for feature in X_sam_features.columns:
        if len(X_sam_features[feature].values) == len(analysis_df):
            analysis_df[feature] = X_sam_features[feature].values
    
    # Get available features from the actual data
    available_features = [col for col in X_sam_features.columns if col in analysis_df.columns]
    print(f"Available features for 2D interaction analysis: {available_features}")
    
    if len(available_features) < 2:
        print("Need at least 2 features for 2D interaction analysis")
        return
    
    # Create interaction pairs from available features (use first few pairs)
    interaction_pairs = []
    for i in range(min(3, len(available_features)-1)):  # Max 3 pairs
        for j in range(i+1, min(i+2, len(available_features))):  # Next feature
            interaction_pairs.append((available_features[i], available_features[j]))
    
    if not interaction_pairs:
        print("No valid feature pairs found for 2D interaction analysis")
        return
    
    fig, axes = plt.subplots(1, len(interaction_pairs), figsize=(6*len(interaction_pairs), 6))
    if len(interaction_pairs) == 1:
        axes = [axes]  # Make it iterable
    
    for i, (feature1, feature2) in enumerate(interaction_pairs):
        ax = axes[i]
        
        # Create 2D histogram of bias rate
        x_vals = analysis_df[feature1]
        y_vals = analysis_df[feature2]
        bias_vals = analysis_df['true_label']
        
        # Create 2D bins
        x_bins = 10
        y_bins = 10
        
        # Calculate bias rate in each 2D bin
        from scipy.stats import binned_statistic_2d
        bias_rates_2d, x_edges, y_edges, _ = binned_statistic_2d(
            x_vals, y_vals, bias_vals, statistic='mean', bins=[x_bins, y_bins]
        )
        
        # Plot heatmap
        X, Y = np.meshgrid(x_edges[:-1], y_edges[:-1])
        im = ax.imshow(bias_rates_2d.T, origin='lower', aspect='auto', cmap='Reds',
                      extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
        
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.set_title(f'Bias Rate: {feature1} vs {feature2}')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Bias Rate')
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(output_dir / 'feature_interactions_2d.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def calculate_comprehensive_threshold_f1(predictions_df, X_sam_features, output_dir, save_fig=True):
    """Calculate F1 score using all features combined with SHAP-derived thresholds"""
    print("Calculating comprehensive F1 score using all features with SHAP-derived thresholds...")
    
    # Load thresholds from SHAP decision boundaries analysis
    import json
    threshold_file = output_dir / 'shap_decision_boundaries.json'
    
    if not threshold_file.exists():
        print(f"Warning: {threshold_file} not found. Using fallback thresholds from paper.")
        # Fallback thresholds from the paper
        thresholds = {
            'max_relative_jump': 0.64,
            'h_continuum_sco2': 19.9,
            'dws': 0.053,
            'max_declocking_o2a': 0.31,
            'aod_sulfate': 0.018
        }
    else:
        with open(threshold_file, 'r') as f:
            boundary_data = json.load(f)
        
        # Extract estimated boundaries
        thresholds = {}
        for feature, data in boundary_data.items():
            if data.get('estimated_boundary') is not None:
                thresholds[feature] = data['estimated_boundary']
        
        print(f"Loaded thresholds from SHAP analysis: {thresholds}")
    
    # Merge predictions with features
    analysis_df = predictions_df.copy()
    for feature in X_sam_features.columns:
        if len(X_sam_features[feature].values) == len(analysis_df):
            analysis_df[feature] = X_sam_features[feature].values
    
    # Apply threshold logic for each feature
    feature_predictions = {}
    
    for feature, threshold in thresholds.items():
        if feature in analysis_df.columns:
            print(f"Applying threshold for {feature}: {threshold:.3f}")
            
            # Apply threshold based on feature characteristics
            if feature == 'h_continuum_sco2':
                # Lower values indicate bias for h_continuum_sco2
                feature_pred = (analysis_df[feature] <= threshold).astype(int)
                direction = "lower values indicate bias"
            else:
                # Higher values indicate bias for other features
                feature_pred = (analysis_df[feature] >= threshold).astype(int)
                direction = "higher values indicate bias"
            
            feature_predictions[feature] = feature_pred
            
            # Calculate individual feature performance
            from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
            f1 = f1_score(analysis_df['true_label'], feature_pred)
            precision = precision_score(analysis_df['true_label'], feature_pred, zero_division=0)
            recall = recall_score(analysis_df['true_label'], feature_pred, zero_division=0)
            accuracy = accuracy_score(analysis_df['true_label'], feature_pred)
            
            print(f"  {feature} (threshold={threshold:.3f}, {direction}):")
            print(f"    F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, Accuracy: {accuracy:.3f}")
        else:
            print(f"Warning: {feature} not found in analysis data")
    
    if not feature_predictions:
        print("Error: No features available for threshold analysis")
        return None
    
    # Test different combination strategies
    combination_results = {}
    
    # Strategy 1: ANY feature indicates bias (OR logic)
    print("\n=== Strategy 1: OR Logic (ANY feature indicates bias) ===")
    or_prediction = np.zeros(len(analysis_df), dtype=int)
    for feature_pred in feature_predictions.values():
        or_prediction = or_prediction | feature_pred
    
    or_f1 = f1_score(analysis_df['true_label'], or_prediction)
    or_precision = precision_score(analysis_df['true_label'], or_prediction, zero_division=0)
    or_recall = recall_score(analysis_df['true_label'], or_prediction, zero_division=0)
    or_accuracy = accuracy_score(analysis_df['true_label'], or_prediction)
    
    combination_results['OR_logic'] = {
        'f1': or_f1, 'precision': or_precision, 'recall': or_recall, 'accuracy': or_accuracy,
        'description': 'Bias predicted if ANY feature exceeds threshold'
    }
    
    print(f"OR Logic - F1: {or_f1:.3f}, Precision: {or_precision:.3f}, Recall: {or_recall:.3f}, Accuracy: {or_accuracy:.3f}")
    
    # Strategy 2: ALL features must indicate bias (AND logic)
    print("\n=== Strategy 2: AND Logic (ALL features must indicate bias) ===")
    and_prediction = np.ones(len(analysis_df), dtype=int)
    for feature_pred in feature_predictions.values():
        and_prediction = and_prediction & feature_pred
    
    and_f1 = f1_score(analysis_df['true_label'], and_prediction)
    and_precision = precision_score(analysis_df['true_label'], and_prediction, zero_division=0)
    and_recall = recall_score(analysis_df['true_label'], and_prediction, zero_division=0)
    and_accuracy = accuracy_score(analysis_df['true_label'], and_prediction)
    
    combination_results['AND_logic'] = {
        'f1': and_f1, 'precision': and_precision, 'recall': and_recall, 'accuracy': and_accuracy,
        'description': 'Bias predicted if ALL features exceed threshold'
    }
    
    print(f"AND Logic - F1: {and_f1:.3f}, Precision: {and_precision:.3f}, Recall: {and_recall:.3f}, Accuracy: {and_accuracy:.3f}")
    
    # Strategy 3: Majority vote (more than half of features indicate bias)
    print("\n=== Strategy 3: Majority Vote (>50% of features indicate bias) ===")
    feature_sum = np.zeros(len(analysis_df))
    for feature_pred in feature_predictions.values():
        feature_sum += feature_pred
    
    majority_threshold = len(feature_predictions) / 2
    majority_prediction = (feature_sum > majority_threshold).astype(int)
    
    maj_f1 = f1_score(analysis_df['true_label'], majority_prediction)
    maj_precision = precision_score(analysis_df['true_label'], majority_prediction, zero_division=0)
    maj_recall = recall_score(analysis_df['true_label'], majority_prediction, zero_division=0)
    maj_accuracy = accuracy_score(analysis_df['true_label'], majority_prediction)
    
    combination_results['majority_vote'] = {
        'f1': maj_f1, 'precision': maj_precision, 'recall': maj_recall, 'accuracy': maj_accuracy,
        'description': f'Bias predicted if >{majority_threshold:.1f} features exceed threshold'
    }
    
    print(f"Majority Vote - F1: {maj_f1:.3f}, Precision: {maj_precision:.3f}, Recall: {maj_recall:.3f}, Accuracy: {maj_accuracy:.3f}")
    
    # Strategy 4: Weighted combination based on feature importance
    print("\n=== Strategy 4: Weighted by SHAP Importance ===")
    # Use feature importance weights (from paper: max_relative_jump=100%, h_continuum_sco2=82%, etc.)
    feature_weights = {
        'max_relative_jump': 1.0,    # 100% (normalized to 1.0)
        'h_continuum_sco2': 0.82,   # 82%
        'dws': 0.56,                # 56%
        'max_declocking_o2a': 0.50, # 50%
        'aod_sulfate': 0.46         # 46%
    }
    
    weighted_sum = np.zeros(len(analysis_df))
    total_weight = 0
    
    for feature, feature_pred in feature_predictions.items():
        if feature in feature_weights:
            weight = feature_weights[feature]
            weighted_sum += weight * feature_pred
            total_weight += weight
            print(f"  {feature}: weight = {weight:.2f}")
    
    if total_weight > 0:
        weighted_score = weighted_sum / total_weight
        # Use 0.5 as threshold for weighted approach
        weighted_prediction = (weighted_score >= 0.5).astype(int)
        
        weight_f1 = f1_score(analysis_df['true_label'], weighted_prediction)
        weight_precision = precision_score(analysis_df['true_label'], weighted_prediction, zero_division=0)
        weight_recall = recall_score(analysis_df['true_label'], weighted_prediction, zero_division=0)
        weight_accuracy = accuracy_score(analysis_df['true_label'], weighted_prediction)
        
        combination_results['weighted_importance'] = {
            'f1': weight_f1, 'precision': weight_precision, 'recall': weight_recall, 'accuracy': weight_accuracy,
            'description': 'Weighted combination based on SHAP feature importance'
        }
        
        print(f"Weighted Approach - F1: {weight_f1:.3f}, Precision: {weight_precision:.3f}, Recall: {weight_recall:.3f}, Accuracy: {weight_accuracy:.3f}")
    
    # Compare with Random Forest performance
    print("\n=== Comparison with Random Forest ===")
    rf_f1 = f1_score(analysis_df['true_label'], analysis_df['predicted_label'])
    rf_precision = precision_score(analysis_df['true_label'], analysis_df['predicted_label'])
    rf_recall = recall_score(analysis_df['true_label'], analysis_df['predicted_label'])
    rf_accuracy = accuracy_score(analysis_df['true_label'], analysis_df['predicted_label'])
    
    print(f"Random Forest - F1: {rf_f1:.3f}, Precision: {rf_precision:.3f}, Recall: {rf_recall:.3f}, Accuracy: {rf_accuracy:.3f}")
    
    # Find best threshold combination approach
    best_method = max(combination_results.keys(), key=lambda k: combination_results[k]['f1'])
    best_f1 = combination_results[best_method]['f1']
    
    print(f"\n=== SUMMARY ===")
    print(f"Best threshold combination: {best_method} (F1: {best_f1:.3f})")
    print(f"Random Forest F1: {rf_f1:.3f}")
    print(f"Improvement of RF over best thresholds: {((rf_f1 - best_f1) / best_f1 * 100):.1f}%")
    
    # Create visualization
    if save_fig:
        import matplotlib.pyplot as plt
        
        methods = ['Random Forest'] + list(combination_results.keys())
        f1_scores = [rf_f1] + [combination_results[k]['f1'] for k in combination_results.keys()]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        bars = ax.bar(range(len(methods)), f1_scores, 
                     color=['green'] + ['blue'] * (len(methods)-1))
        
        ax.set_xlabel('Method')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score Comparison: Random Forest vs Combined Thresholds')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(f1_scores):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'comprehensive_threshold_f1_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    # Save results
    comprehensive_results = {
        'individual_features': {
            feature: {
                'threshold': thresholds[feature],
                'f1': f1_score(analysis_df['true_label'], pred),
                'precision': precision_score(analysis_df['true_label'], pred, zero_division=0),
                'recall': recall_score(analysis_df['true_label'], pred, zero_division=0),
                'accuracy': accuracy_score(analysis_df['true_label'], pred)
            }
            for feature, pred in feature_predictions.items()
        },
        'combination_strategies': combination_results,
        'random_forest': {
            'f1': rf_f1, 'precision': rf_precision, 'recall': rf_recall, 'accuracy': rf_accuracy
        },
        'best_threshold_method': best_method,
        'best_threshold_f1': best_f1,
        'rf_improvement_percent': ((rf_f1 - best_f1) / best_f1 * 100) if best_f1 > 0 else 0
    }
    
    with open(output_dir / 'comprehensive_threshold_analysis.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    return comprehensive_results

def main():
    # Initialize config
    config = PathConfig()
    
    parser = argparse.ArgumentParser(description='RF SHAP Analysis for OCO-3 Swath Bias Correction')
    parser.add_argument('--model_dir', type=str,
                        default=str(config.MODEL_EXPERIMENT_DIR),
                       help='Directory containing trained RF model')
    parser.add_argument('--processed_data_dir', type=str,
                        default=str(config.PROCESSED_EXPERIMENT_DIR), 
                       help='Directory containing processed data and predictions')
    parser.add_argument('--output_dir', type=str,
                       default=str(config.FIGURES_DIR / 'rf_shap_analysis'),
                       help='Output directory for figures')
    parser.add_argument('--max_samples', type=int, default=500,
                       help='Maximum number of samples for SHAP analysis')
    parser.add_argument('--save_figs', action='store_true', default=True,
                       help='Save figures to disk')
    
    args = parser.parse_args()
    
    # Setup output directories
    output_dirs = setup_output_dirs(args.output_dir)
    
    try:
        # Load model and data
        rf_model, features_used, model_config, predictions_df, metadata = load_rf_model_and_data(
            args.model_dir, args.processed_data_dir
        )
        
        print(f"Model features: {features_used}")
        print(f"Predictions shape: {predictions_df.shape}")
        
        # Get unique SAM IDs from predictions
        sam_ids = predictions_df['sam_id'].unique()
        print(f"Number of SAMs: {len(sam_ids)}")
        
        # Load training data for these SAMs
        full_data = load_training_data_for_sams(sam_ids, features_used)
        
        if full_data.empty:
            raise RuntimeError("Could not load training data. This is required for SHAP analysis. "
                             "Please check your data paths and ensure the processed data is available.")
        else:
            # Prepare SAM-level aggregated features 
            print("Aggregating features at SAM level...")
            sam_level_data = []
            
            for sam_id in sam_ids:
                sam_data = full_data[full_data['SAM'] == sam_id]
                if len(sam_data) > 0:
                    # Check if all features are available
                    available_features = [f for f in features_used if f in sam_data.columns]
                    if len(available_features) == len(features_used):
                        # Take mean of features for this SAM
                        sam_features = sam_data[features_used].mean()
                        sam_level_data.append(sam_features)
                    else:
                        print(f"Warning: Missing features for SAM {sam_id}: {set(features_used) - set(available_features)}")
                        # Fill with median values from available data
                        sam_features = pd.Series(index=features_used, dtype=float)
                        for feat in features_used:
                            if feat in sam_data.columns:
                                sam_features[feat] = sam_data[feat].mean()
                            else:
                                # Use a default value
                                sam_features[feat] = 0.0
                        sam_level_data.append(sam_features)
            
            if sam_level_data:
                X_sam_features = pd.DataFrame(sam_level_data, columns=features_used)
            else:
                raise ValueError("No valid SAM-level features could be prepared")
        
        print(f"Prepared {len(X_sam_features)} SAM samples for SHAP analysis")
        
        # Create SHAP explainer
        explainer = create_shap_explainer(rf_model, X_sam_features)
        
        # Calculate SHAP values
        shap_values, X_sample_subset = calculate_shap_values(explainer, X_sam_features, args.max_samples)
        
        # Create SHAP visualizations
        print("\nCreating SHAP visualizations...")
        
        # 1. SHAP summary plots
        shap_importance = plot_shap_summary(shap_values, X_sample_subset, 
                                          output_dirs['shap_summary'], args.save_figs)
        
        # 2. SHAP dependence plots  
        plot_shap_dependence(shap_values, X_sample_subset, features_used,
                           output_dirs['shap_dependence'], args.save_figs)
        
        # 3. SHAP waterfall plots
        plot_shap_waterfall_examples(explainer, X_sample_subset, 
                                    output_dirs['shap_waterfall'], args.save_figs)
        
        # 4. Feature interaction analysis
        create_feature_interaction_analysis(shap_values, X_sample_subset, features_used,
                                           output_dirs['shap_dependence'], args.save_figs)
        
        # 5. Feature importance comparison
        importance_df = compare_feature_importance(rf_model, shap_importance, features_used,
                                                 output_dirs['feature_comparison'], args.save_figs)
        
        # 6. h_continuum_sco2 threshold analysis  
        analyze_h_continuum_threshold(predictions_df, X_sam_features, rf_model,
                                    output_dirs['decision_analysis'], args.save_figs)
        
        # 7. Decision analysis plots
        create_decision_analysis_plots(predictions_df, output_dirs['decision_analysis'], args.save_figs)
        
        # 8. Create paper summary
        create_paper_summary(importance_df, predictions_df, model_config, output_dirs['main'], args.save_figs)
        
        # 9. Analyze all features for threshold behavior
        feature_analysis = analyze_all_features_thresholds(predictions_df, X_sam_features, rf_model,
                                                         output_dirs['decision_analysis'], args.save_figs)
        
        # 10. Analyze feature interactions vs simple thresholds
        analyze_feature_interactions_vs_thresholds(predictions_df, X_sam_features, rf_model,
                                                    feature_analysis, output_dirs['decision_analysis'], args.save_figs)
        
        # 11. Create comprehensive 2-D interaction plots
        create_comprehensive_2d_interactions(shap_values, X_sample_subset, features_used, importance_df,
                                              output_dirs['shap_dependence'], args.save_figs)
        
        # 12. Analyze decision boundaries from SHAP dependence
        boundary_analysis = analyze_decision_boundaries_from_shap(shap_values, X_sample_subset, features_used,
                                                                 output_dirs['shap_dependence'], args.save_figs)
        
        # 13. Compare different threshold calculation methods
        compare_threshold_methods(predictions_df, X_sam_features, boundary_analysis, feature_analysis,
                                   output_dirs['decision_analysis'], args.save_figs)
        
        # 14. Analyze physical intuition experiments
        analyze_physical_intuition_experiments(predictions_df, X_sam_features, output_dirs['decision_analysis'], args.save_figs)
        
        # 15. Analyze 2D feature interactions
        analyze_feature_interactions_2d(predictions_df, X_sam_features, output_dirs['decision_analysis'], args.save_figs)
        
        # 16. Calculate comprehensive threshold F1
        comprehensive_results = calculate_comprehensive_threshold_f1(predictions_df, X_sam_features, output_dirs['decision_analysis'], args.save_figs)
        
        # Print summary statistics
        print("\n" + "="*50)
        print("SHAP ANALYSIS SUMMARY")
        print("="*50)
        print(f"Model features: {len(features_used)}")
        print(f"Samples analyzed: {len(X_sample_subset)}")
        print(f"Top 3 features by SHAP importance:")
        for i, (_, row) in enumerate(importance_df.head(3).iterrows()):
            print(f"  {i+1}. {row['feature']}: {row['shap_importance']:.4f}")
        
        print(f"\nModel performance on evaluation set:")
        y_true = predictions_df['true_label']
        y_pred = predictions_df['predicted_label'] 
        print(classification_report(y_true, y_pred))
        
        print(f"\nFigures saved to: {args.output_dir}")
        print("Analysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 