#!/usr/bin/env python3
"""
ROC Curve Generation for OCO-3 Swath Bias Correction

This script generates a publication-ready ROC curve for the Random Forest classifier
using cross-validation results to ensure no data leakage occurs in performance reporting.

Key Features:
- Uses stored cross-validation results for exact SAM assignments
- Generates probabilities using appropriate fold models for proper ROC curve
- Reports true out-of-sample performance metrics
- Creates clean ROC curve for publication

Usage:
    python -m src.analysis.classifier_performance_curves
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, classification_report
import joblib

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.utils.config_paths import PathConfig

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_cv_results():
    """Load cross-validation results from fold directories"""
    config = PathConfig()
    
    # Path to CV results
    cv_base_path = config.PROCESSED_EXPERIMENT_DIR
    
    # Load CV summary
    cv_summary_file = cv_base_path / "cv_summary.json"
    if not cv_summary_file.exists():
        raise FileNotFoundError(f"CV summary file not found: {cv_summary_file}")
    
    import json
    with open(cv_summary_file, 'r') as f:
        cv_summary = json.load(f)
    
    # Load SAM-level predictions from all folds
    fold_dirs = [d for d in cv_base_path.iterdir() if d.name.startswith('final_best_config_fold_')]
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found in {cv_base_path}")
    
    all_cv_results = []
    for fold_dir in fold_dirs:
        sam_features_file = fold_dir / 'sam_features.parquet'
        if sam_features_file.exists():
            fold_results = pd.read_parquet(sam_features_file)
            all_cv_results.append(fold_results)
    
    if not all_cv_results:
        raise FileNotFoundError("No CV results found in fold directories")
    
    # Combine all CV results
    combined_cv_results = pd.concat(all_cv_results, ignore_index=False)
    
    print(f"Loaded CV results: {len(combined_cv_results)} SAMs from {len(fold_dirs)} folds")
    
    return combined_cv_results, cv_summary

def load_fold_models():
    """Load pre-trained models from cross-validation folds"""
    config = PathConfig()
    
    # Path to pre-trained models
    models_base_path = config.MODEL_BASE_DIR / "Swath_BC_v4.0_th_06"
    
    # Load fold models
    fold_models = {}
    for fold_num in range(1, 5):  # folds 1-4
        fold_path = models_base_path / f"final_best_config_fold_{fold_num}" / "rf_model_classifier_with_jumps.joblib"
        if not fold_path.exists():
            raise FileNotFoundError(f"Fold {fold_num} model not found at {fold_path}")
        fold_models[fold_num] = joblib.load(fold_path)
    
    print(f"Loaded {len(fold_models)} fold models")
    return fold_models

def load_labeled_data():
    """Load labeled data for curve analysis"""
    config = PathConfig()
    
    # Load labeled data
    labels_path = config.LABELS_FILE
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found at {labels_path}")
    
    labels_df = pd.read_csv(labels_path)
    
    # Load SAM features
    features_path = config.PROCESSED_FINAL_DIR / "sam_features.parquet" 
    if not features_path.exists():
        raise FileNotFoundError(f"SAM features not found at {features_path}")
    
    sam_features_df = pd.read_parquet(features_path)
    
    return labels_df, sam_features_df

def prepare_data_for_analysis(final_model, labels_df, sam_features_df):
    """Prepare data for performance analysis"""
    
    # Extract the features used by the model
    feature_names = list(final_model.feature_names_in_) if hasattr(final_model, 'feature_names_in_') else []
    
    # Filter labels to exclude uncertain (class 2) and get binary labels
    binary_labels = labels_df[labels_df['label'].isin([0, 1])].copy()
    
    # Handle duplicate labels by taking the most recent/last label for each SAM
    binary_labels = binary_labels.drop_duplicates(subset=['identifier'], keep='last')
    
    # Get corresponding SAM features
    labeled_sams = binary_labels['identifier'].values
    sam_features_filtered = sam_features_df[sam_features_df.index.isin(labeled_sams)]
    
    # Align labels and features
    common_sams = set(binary_labels['identifier']).intersection(set(sam_features_filtered.index))
    
    # Create aligned datasets
    binary_labels = binary_labels[binary_labels['identifier'].isin(common_sams)].set_index('identifier')
    sam_features_filtered = sam_features_filtered.loc[list(common_sams)]
    
    # Sort to ensure alignment
    binary_labels = binary_labels.sort_index()
    sam_features_filtered = sam_features_filtered.sort_index()
    
    # Get X and y - ensure they have the same indices
    X = sam_features_filtered[feature_names].dropna()
    
    # Filter labels to match X exactly
    y = binary_labels.loc[X.index, 'label']
    
    print(f"Final dataset: X shape = {X.shape}, y shape = {y.shape}")
    print(f"Positive class ratio: {y.mean()*100:.1f}%")
    
    # Double check alignment
    assert len(X) == len(y), f"Mismatch: X has {len(X)} samples, y has {len(y)} samples"
    assert (X.index == y.index).all(), "Indices are not aligned"
    
    return X, y, feature_names



def create_roc_curve_cv(cv_results, fold_models, X, operational_threshold=0.6, output_dir=None, save_fig=True):
    """Create ROC curve using cross-validation results (no data leakage)"""
    
    # We'll use the exact SAMs from stored CV results but generate probabilities using the models
    # Group CV results by fold to match with the corresponding fold models
    config = PathConfig()
    cv_base_path = config.PROCESSED_EXPERIMENT_DIR
    
    all_y_true = []
    all_y_proba = []
    
    # Process each fold to get probabilities for the exact SAMs in the stored CV results
    for fold_num in range(1, 5):  # folds 1-4
        fold_dir = cv_base_path / f"final_best_config_fold_{fold_num}"
        fold_cv_file = fold_dir / 'sam_features.parquet'
        
        if fold_cv_file.exists():
            # Load this fold's CV results
            fold_cv_data = pd.read_parquet(fold_cv_file)
            fold_model = fold_models[fold_num]
            
            # Get the SAM identifiers for this fold
            fold_sam_ids = fold_cv_data.index.tolist()
            
            # Filter X to only include SAMs from this fold, maintain order
            fold_X = X.loc[X.index.intersection(fold_sam_ids)]
            
            if len(fold_X) > 0:
                # Generate probabilities using this fold's model
                fold_proba = fold_model.predict_proba(fold_X)[:, 1]
                
                # Get corresponding true labels from stored CV results
                fold_true_labels = fold_cv_data.loc[fold_X.index, 'true_label'].values
                
                # Store results
                all_y_true.extend(fold_true_labels)
                all_y_proba.extend(fold_proba)
    
    # Convert to numpy arrays
    all_y_true = np.array(all_y_true)
    all_y_proba = np.array(all_y_proba)
    
    print(f"ROC curve: Using {len(all_y_true)} SAMs from stored CV results")
    
    # Generate ROC curve using the probabilities
    fpr, tpr, roc_thresholds = roc_curve(all_y_true, all_y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create smaller figure
    fig, ax = plt.subplots(figsize=(4, 3))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', alpha=0.5, label='Random Classifier')
    
    # Mark operational threshold
    threshold_idx = np.argmin(np.abs(roc_thresholds - operational_threshold))
    if threshold_idx < len(fpr):
        ax.plot(fpr[threshold_idx], tpr[threshold_idx], 'ro', markersize=6,
                label=f'Threshold = {operational_threshold}')
    
    # Verify this matches the stored CV results by calculating performance at operational threshold
    y_pred_at_threshold = (all_y_proba >= operational_threshold).astype(int)
    tp = ((all_y_true == 1) & (y_pred_at_threshold == 1)).sum()
    fn = ((all_y_true == 1) & (y_pred_at_threshold == 0)).sum()
    calculated_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"ROC curve calculated recall at threshold {operational_threshold}: {calculated_recall:.3f}")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Make ticks smaller
    ax.tick_params(axis='both', which='major', labelsize=9)
    
    plt.tight_layout()
    
    if save_fig and output_dir:
        plt.savefig(output_dir / 'roc_curve_cv.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    
    return roc_auc

def main():
    """Main function"""
    # Setup
    config = PathConfig()
    output_dir = config.FIGURES_DIR / "classifier_performance_curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading fold models...")
    fold_models = load_fold_models()
    
    print("Loading labeled data...")
    labels_df, sam_features_df = load_labeled_data()
    
    # Get final model for feature extraction (we'll load just the first one)
    config = PathConfig()
    final_model_path = config.MODEL_BASE_DIR / "Swath_BC_v4.0_th_06" / "final_model_all_data" / "rf_model_classifier_with_jumps.joblib"
    final_model = joblib.load(final_model_path)
    
    print("Preparing data for analysis...")
    X, y, feature_names = prepare_data_for_analysis(final_model, labels_df, sam_features_df)
    
    print(f"Analysis dataset: {len(X)} SAMs, {y.mean()*100:.1f}% positive class")
    print(f"Model features: {feature_names}")
    
    # Load cross-validation results
    print("\nLoading cross-validation results...")
    cv_results, cv_summary = load_cv_results()
    
    # Get operational threshold
    operational_threshold = getattr(final_model, 'rf_prediction_threshold_', 0.6)
    print(f"Operational threshold: {operational_threshold}")
    
    print("\nGenerating ROC curve using cross-validation results...")
    roc_auc = create_roc_curve_cv(cv_results, fold_models, X, operational_threshold, output_dir, save_fig=True)
    
    # Calculate true CV performance metrics
    y_true = cv_results['true_label'].values
    y_pred = cv_results['rf_prediction'].values
    
    # Calculate confusion matrix elements
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    
    cv_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    cv_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    cv_f1 = 2 * cv_precision * cv_recall / (cv_precision + cv_recall) if (cv_precision + cv_recall) > 0 else 0
    
    # Print summary
    print(f"\n=== PERFORMANCE SUMMARY ===")
    print(f"ROC AUC (Cross-Validation): {roc_auc:.3f}")
    print(f"Operational threshold: {operational_threshold}")
    
    print(f"\n=== TRUE CROSS-VALIDATION PERFORMANCE (No Data Leakage) ===")
    print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Precision: {cv_precision:.3f}")
    print(f"Recall: {cv_recall:.3f}")
    print(f"F1-Score: {cv_f1:.3f}")
    
    # Create detailed classification report using CV results
    print(f"\nDetailed Performance Report (Cross-Validation Results):")
    print(classification_report(y_true, y_pred, target_names=['No Bias', 'Bias']))
    
    print(f"\nROC curve saved to: {output_dir / 'roc_curve_cv.png'}")

    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 