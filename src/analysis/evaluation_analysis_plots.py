# Comprehensive Analysis and Visualization Script for Swath Bias Correction Evaluation
# Generates plots for paper figures and detailed analysis

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import warnings
import argparse
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Import functions from main_util
from ..utils.main_util import plot_SAM, plot_map, scatter_hist, load_data, SAM_enhancement
from ..utils.config_paths import PathConfig

# Plot configuration
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'

# Control which plots to generate
PLOT_CONFIG = {
    'dataset_statistics': False,           # Print general dataset statistics
    'confusion_matrix_examples': True,    # Examples of TP, TN, FP, FN SAMs
    'confusion_matrix_heatmap': False,     # Nice confusion matrix visualization
    'feature_distributions': False,        # Errors vs each predictor feature
    'performance_by_sam_category': False,  # Break down by fossil, volcano, etc.
    'spatial_error_distribution': False,   # Geographic distribution of errors
    'correction_magnitude_distributions': False,  # Histograms of correction sizes
    'correction_application_map': False,   # Geographic map of where correction is applied
    'sam_count_map': False,               # Geographic map showing total SAM counts
    'tp_fp_correction_magnitude_standalone': False,
    'f1_score_by_sam_category_standalone': False,
    'correction_magnitude_by_category_standalone': False,
    'labeled_data_distribution_by_category': False,
    'enhancement_analysis': False
}

def load_evaluation_data(base_dir):
    """Load and combine evaluation results from the k-fold cross-validation output."""
    print(f"Loading evaluation data from: {base_dir}")

    # Load CV summary for metadata
    cv_summary_file = os.path.join(base_dir, "cv_summary.json")
    if not os.path.exists(cv_summary_file):
        raise FileNotFoundError(f"CV summary file not found: {cv_summary_file}")
    import json
    with open(cv_summary_file, 'r') as f:
        cv_summary = json.load(f)

    # Find all fold directories
    fold_dirs = [d for d in os.listdir(base_dir) if d.startswith('final_best_config_fold_')]
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found in {base_dir}")
    print(f"Found {len(fold_dirs)} fold directories")

    # Load SAM-level features and predictions from all folds
    all_sam_features = []
    for fold_dir in fold_dirs:
        features_file = os.path.join(base_dir, fold_dir, 'sam_features.parquet')
        if os.path.exists(features_file):
            fold_features = pd.read_parquet(features_file)
            fold_features['fold_name'] = fold_dir
            all_sam_features.append(fold_features)
    
    if not all_sam_features:
        raise FileNotFoundError("No 'sam_features.parquet' files found. Please ensure the modeling script has been run successfully.")
    
    # Combine all SAM feature data
    sam_features_df = pd.concat(all_sam_features, ignore_index=False)  # Keep the index
    
    # The SAM identifier in sam_features.parquet is in the index, not a column
    # Reset index to make SAM ID a column
    sam_features_df = sam_features_df.reset_index()
    if 'SAM' in sam_features_df.columns:
        # SAM column exists after reset_index - keep it as 'SAM' for merging
        pass
    elif sam_features_df.index.name == 'SAM':
        # Index was already reset, so SAM is now a column
        pass
    else:
        raise KeyError("SAM identifier not found in sam_features.parquet. Check the modeling script output.")

    # Create confusion matrix categories
    sam_features_df['confusion_category'] = sam_features_df.apply(
        lambda row: 'TP' if row['true_label'] == 1 and row['rf_prediction'] == 1 else
                    'TN' if row['true_label'] == 0 and row['rf_prediction'] == 0 else
                    'FP' if row['true_label'] == 0 and row['rf_prediction'] == 1 else
                    'FN', axis=1
    )

    # Load sounding-level data (with corrected XCO2) from all folds
    all_sounding_data = []
    for fold_dir in fold_dirs:
        sounding_file = os.path.join(base_dir, fold_dir, 'eval_data_corrected.parquet')
        if os.path.exists(sounding_file):
            fold_soundings = pd.read_parquet(sounding_file)
            all_sounding_data.append(fold_soundings)
    
    # Create summary statistics from the SAM-level data
    summary_stats = create_summary_stats_from_folds(cv_summary, sam_features_df)

    if not all_sounding_data:
        print("Warning: No 'eval_data_corrected.parquet' files found. Sounding-level plots will be skipped.")
        # Return SAM-level data if no sounding data is available
        # Rename column for consistency with downstream functions
        sam_features_df = sam_features_df.rename(columns={'SAM': 'SAM'})
        return sam_features_df, summary_stats

    # Combine all sounding data
    sounding_df = pd.concat(all_sounding_data, ignore_index=True)
    if 'SAM' not in sounding_df.columns:
        raise KeyError("'SAM' column not found in eval_data_corrected.parquet. Check the modeling script output.")

    # Merge SAM-level features and predictions into the sounding-level dataframe
    # This creates one large dataframe for consistent use in plotting functions
    combined_data = sounding_df.merge(sam_features_df, left_on='SAM', right_on='SAM', how='left')

    # Drop the redundant sam_id column if it exists
    if 'sam_id' in combined_data.columns:
        combined_data = combined_data.drop(columns=['sam_id'])

    print(f"Successfully loaded and merged data. Total soundings: {len(combined_data)}, Unique SAMs: {combined_data['SAM'].nunique()}")
    print(f"DEBUG: Available columns after merge: {[c for c in combined_data.columns if 'confusion' in c or 'label' in c or 'prediction' in c]}")

    return combined_data, summary_stats

def create_summary_stats_from_folds(cv_summary, all_predictions):
    """Create summary statistics compatible with old format."""
    # Count confusion matrix elements
    confusion_counts = all_predictions['confusion_category'].value_counts()
    
    tp = confusion_counts.get('TP', 0)
    tn = confusion_counts.get('TN', 0) 
    fp = confusion_counts.get('FP', 0)
    fn = confusion_counts.get('FN', 0)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = cv_summary['mean_f1_score']
    
    summary_stats = {
        'mean_f1_score': cv_summary['mean_f1_score'],
        'std_f1_score': cv_summary['std_f1_score'],
        'fold_f1_scores': cv_summary['fold_f1_scores'],
        'n_folds': cv_summary['n_folds'],
        'total_labeled_sams': cv_summary['total_labeled_sams'],
        'selected_features': cv_summary['selected_features'],
        'tp_per_fold': [tp / cv_summary['n_folds']] * cv_summary['n_folds'],  # Approximate
        'tn_per_fold': [tn / cv_summary['n_folds']] * cv_summary['n_folds'],
        'fp_per_fold': [fp / cv_summary['n_folds']] * cv_summary['n_folds'], 
        'fn_per_fold': [fn / cv_summary['n_folds']] * cv_summary['n_folds'],
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'total_tp': tp,
        'total_tn': tn,
        'total_fp': fp,
        'total_fn': fn
    }
    
    return summary_stats

def save_summary_stats_to_csv(summary_stats, output_dir):
    """Save the overall summary statistics to a CSV file."""
    print("Saving overall evaluation summary statistics...")
    
    # Convert summary_stats dict to a DataFrame (single row)
    stats_df = pd.DataFrame([summary_stats])
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'evaluation_summary_stats.csv')
    stats_df.to_csv(csv_path, index=False)
    
    print(f"Overall evaluation statistics saved to: {os.path.basename(csv_path)}")
    return csv_path

def create_sam_category_mapping(data):
    """Create mapping of SAM categories based on SAM names."""
    def categorize_sam(sam_name):
        sam_lower = str(sam_name).lower()
        if 'fossil' in sam_lower:
            return 'Fossil'
        elif 'volcano' in sam_lower:
            return 'Volcano'
        elif 'sif' in sam_lower:
            return 'SIF'
        elif 'texmex' in sam_lower or 'tex' in sam_lower:
            return 'TexMex'
        elif 'ecostress' in sam_lower:
            return 'ECOSTRESS'
        elif 'desert' in sam_lower:
            return 'Desert'
        else:
            return 'Other'
    
    data['sam_category'] = data['SAM'].apply(categorize_sam)
    return data

def plot_confusion_matrix_examples(data, output_dir, n_examples=4):
    """Plot example SAM maps for each confusion matrix category using plot_SAM function."""
    if not PLOT_CONFIG['confusion_matrix_examples']:
        return
        
    print("Creating confusion matrix examples...")
    examples_dir = os.path.join(output_dir, 'examples')
    os.makedirs(examples_dir, exist_ok=True)
    
    categories = ['TP', 'TN', 'FP', 'FN']
    category_names = {
        'TP': 'True Positive', 
        'TN': 'True Negative', 
        'FP': 'False Positive', 
        'FN': 'False Negative'
    }
    
    # Handle confusion_category column naming (may have _x suffix from merge)
    confusion_col = 'confusion_category'
    if confusion_col not in data.columns:
        if 'confusion_category_x' in data.columns:
            confusion_col = 'confusion_category_x'
        elif 'confusion_category_y' in data.columns:
            confusion_col = 'confusion_category_y'
        else:
            print(f"Warning: No confusion_category column found. Available columns: {[c for c in data.columns if 'confusion' in c]}")
            return
    
    all_example_paths = {}
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    for category in categories:
        print(f"Processing {category} examples...")
        category_data = data[data[confusion_col] == category]
        if category_data.empty:
            continue
            
        # Get unique SAMs for this category and randomly select from them
        unique_sams = category_data['SAM'].unique()
        n_available = len(unique_sams)
        n_to_select = min(n_examples, n_available)  # Don't select more than available
        
        if n_to_select > 0:
            # Randomly select SAMs without replacement
            sams_in_category = np.random.choice(unique_sams, size=n_to_select, replace=False)
        else:
            sams_in_category = []
            
        example_paths = []
        
        for i, sam_id in enumerate(sams_in_category):
            sam_data = category_data[category_data['SAM'] == sam_id]
            
            if len(sam_data) < 10:  # Skip if too few soundings
                continue
                
            # Create plots for original, corrected, and difference
            variables = ['xco2', 'xco2_swath-BC']
            var_names = ['Original', 'Corrected']
            
            for var, var_name in zip(variables, var_names):
                if var in sam_data.columns:
                    title_addition = f'{category_names[category]} Example {i+1}\n{var_name} XCO₂'
                    plot_name = f'{category}_example_{i+1}_{var_name}_{sam_id}'
                    
                    plot_SAM(sam_data, var, 
                           save_fig=True, 
                           name=plot_name, 
                           path=examples_dir,
                           title_addition=title_addition,
                           simplified_title=True)
                    
                    plot_path = os.path.join(examples_dir, f'{sam_id}_{var}_{plot_name}.png')
                    if os.path.exists(plot_path):
                        example_paths.append(plot_path)
            
            # Create difference plot using plot_SAM function
            if 'xco2' in sam_data.columns and 'xco2_swath-BC' in sam_data.columns:
                # Add difference column to the data
                sam_data_diff = sam_data.copy()
                sam_data_diff['xco2_correction_applied'] = sam_data_diff['xco2_swath-BC'] - sam_data_diff['xco2']
                
                # Use plot_SAM to create the difference plot
                title_addition_diff = f'{category_names[category]} Example {i+1}\nCorrection Applied (Bias Corrected - Original)'
                plot_name_diff = f'{category}_example_{i+1}_difference_{sam_id}'
                
                plot_SAM(sam_data_diff, 'xco2_correction_applied',
                       vmin=-1, vmax=1,
                       save_fig=True, 
                       name=plot_name_diff, 
                       path=examples_dir,
                       title_addition=title_addition_diff,
                       simplified_title=True)
                
                diff_path = os.path.join(examples_dir, f'{sam_id}_xco2_correction_applied_{plot_name_diff}.png')
                if os.path.exists(diff_path):
                    example_paths.append(diff_path)
                    
        all_example_paths[category] = example_paths[:12]  # Keep up to 12 plots per category (4 SAMs × 3 plots each)
    
    # Create separate figures for each confusion matrix category
    print("Creating separate figures for each confusion matrix category...")
    
    for category in categories:
        category_paths = all_example_paths.get(category, [])
        
        if len(category_paths) > 0:
            # Calculate grid dimensions for this category
            n_plots = len(category_paths)
            if n_plots <= 3:
                # Single row for up to 3 plots
                n_rows, n_cols = 1, n_plots
                figsize = (5*n_cols, 5)
            else:
                # Multiple rows for more plots
                n_cols = 3  # Max 3 columns
                n_rows = int(np.ceil(n_plots / n_cols))
                figsize = (5*n_cols, 5*n_rows)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            
            # Handle single plot case
            if n_plots == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes if n_plots > 1 else [axes]
            else:
                axes = axes.flatten()
            
            # Plot each example
            for i, plot_path in enumerate(category_paths):
                if os.path.exists(plot_path):
                    img = plt.imread(plot_path)
                    axes[i].imshow(img)
                    axes[i].axis('off')
                    
            # Hide unused subplots
            for i in range(len(category_paths), len(axes)):
                axes[i].axis('off')
            
            # Set main title for the figure
            fig.suptitle(f'{category_names[category]} Examples', 
                        fontsize=18, fontweight='bold', y=0.95)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.90)
            
            # Save with category-specific name
            category_path = os.path.join(output_dir, f'confusion_matrix_examples_{category}.png')
            plt.savefig(category_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"{category_names[category]} examples saved to: {os.path.basename(category_path)}")
        else:
            print(f"No examples found for {category_names[category]}")

def plot_confusion_matrix_heatmap(summary_stats, output_dir):
    """Create a publication-ready confusion matrix heatmap."""
    if not PLOT_CONFIG['confusion_matrix_heatmap']:
        return
        
    print("Creating confusion matrix heatmap...")
    
    # Extract confusion matrix data from new format
    tp = summary_stats.get('total_tp', 0)
    tn = summary_stats.get('total_tn', 0) 
    fp = summary_stats.get('total_fp', 0)
    fn = summary_stats.get('total_fn', 0)
    
    # Create confusion matrix
    cm = np.array([[tn, fp], [fn, tp]])
    cm_percent = cm / cm.sum() * 100
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Predicted: No Bias', 'Predicted: Bias'],
                yticklabels=['True: No Bias', 'True: Bias'],
                cbar_kws={'label': 'Count'})
    
    # Add percentages
    for i in range(2):
        for j in range(2):
            ax.text(j+0.5, i+0.7, f'({cm_percent[i,j]:.1f}%)', 
                   ha='center', va='center', fontsize=10, color='gray')
    
    plt.title('Confusion Matrix: Swath Bias Detection', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_distributions(data, summary_stats, output_dir):
    """Plot distribution of errors and features, broken down by confusion matrix category."""
    if not PLOT_CONFIG['feature_distributions']:
        return
        
    print("Creating feature distribution plots...")
    
    # Create feature distribution subdirectory 
    feature_dir = os.path.join(output_dir, 'feature_distributions')
    os.makedirs(feature_dir, exist_ok=True)
    
    # Get selected features from summary stats
    selected_features = summary_stats.get('selected_features', [])
    if not selected_features:
        selected_features = ['max_relative_jump', 'h_continuum_sco2', 'solar_zenith_angle', 'dws', 's31']
        print("Warning: No selected features found in summary stats, using default list")
    
    # Aggregate to SAM level (one row per SAM) and create confusion_category on the fly
    # Handle potential _x/_y suffixes from merge
    true_label_col = 'true_label' if 'true_label' in data.columns else 'true_label_x'
    rf_prediction_col = 'rf_prediction' if 'rf_prediction' in data.columns else 'rf_prediction_x'
    
    sam_summary = data.groupby('SAM').agg({
        true_label_col: 'first',
        rf_prediction_col: 'first',
        **{feature: 'first' for feature in selected_features if feature in data.columns}
    }).reset_index()
    
    # Rename columns to standard names
    sam_summary = sam_summary.rename(columns={
        true_label_col: 'true_label',
        rf_prediction_col: 'rf_prediction'
    })

    # Create confusion category
    sam_summary['confusion_category'] = sam_summary.apply(
        lambda row: 'TP' if row['true_label'] == 1 and row['rf_prediction'] == 1 else
                    'TN' if row['true_label'] == 0 and row['rf_prediction'] == 0 else
                    'FP' if row['true_label'] == 0 and row['rf_prediction'] == 1 else
                    'FN', axis=1
    )

    for feature in tqdm(selected_features, desc="Generating feature plots"):
        if feature not in sam_summary.columns:
            print(f"Warning: Feature '{feature}' not found in data. Skipping.")
            continue

        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        fig.suptitle(f'Analysis of Feature: {feature}', fontsize=16, fontweight='bold')

        # Plot 1: Scatter plot of classification results vs feature value
        colors = {'TP': 'green', 'TN': 'blue', 'FP': 'red', 'FN': 'orange'}
        for category in ['TP', 'TN', 'FP', 'FN']:
            cat_data = sam_summary[sam_summary['confusion_category'] == category]
            if not cat_data.empty:
                # Use random jitter on y-axis for better visualization
                jitter = np.random.normal(0, 0.1, len(cat_data))
                axes[0].scatter(cat_data[feature], jitter, 
                                label=f'{category} (n={len(cat_data)})', 
                                color=colors[category], alpha=0.6, s=30)
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel('Random Jitter')
        axes[0].set_title('Classification Results vs. Feature Value')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.5)

        # Plot 2: Box plot of feature value distribution by confusion category
        sns.boxplot(data=sam_summary, x='confusion_category', y=feature, ax=axes[1],
                    order=['TP', 'FP', 'TN', 'FN'])
        axes[1].set_title('Distribution by Confusion Category')
        axes[1].set_xlabel('Confusion Category')
        axes[1].set_ylabel(f'Value of {feature}')
        axes[1].grid(True, linestyle='--', alpha=0.5)

        # Plot 3: Histogram of feature value distribution by true label
        no_bias_data = sam_summary[sam_summary['true_label'] == 0][feature].dropna()
        bias_data = sam_summary[sam_summary['true_label'] == 1][feature].dropna()
        
        axes[2].hist(no_bias_data, alpha=0.7, label=f'No Bias (n={len(no_bias_data)})', bins=25, density=True)
        axes[2].hist(bias_data, alpha=0.7, label=f'Bias (n={len(bias_data)})', bins=25, density=True)
        axes[2].set_xlabel(feature)
        axes[2].set_ylabel('Density')
        axes[2].set_title('Distribution by True Label')
        axes[2].legend()
        axes[2].grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
        plot_filename = os.path.join(feature_dir, f'feature_analysis_{feature}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Feature analysis plots saved to: {feature_dir}")


def plot_correction_magnitude_by_category_standalone(data, output_dir):
    """Create a standalone plot of correction magnitude distribution by confusion category."""
    if not PLOT_CONFIG['correction_magnitude_by_category_standalone']:
        return
        
    print("Creating standalone correction magnitude by category plot...")
    
    # Calculate correction magnitude
    data['correction_magnitude'] = np.abs(data['xco2_swath-BC'] - data['xco2'])
    
    # Handle column naming from merge operations
    confusion_col = 'confusion_category'
    if confusion_col not in data.columns:
        if 'confusion_category_x' in data.columns:
            confusion_col = 'confusion_category_x'
        elif 'confusion_category_y' in data.columns:
            confusion_col = 'confusion_category_y'
        else:
            print("Warning: No confusion_category column found, skipping correction magnitude distributions plot")
            return
    
    sam_summary = data.groupby('SAM').agg({
        'correction_magnitude': 'mean',
        confusion_col: 'first'
    }).reset_index()
    
    # Rename the column for consistent access
    if confusion_col != 'confusion_category':
        sam_summary = sam_summary.rename(columns={confusion_col: 'confusion_category'})

    # Create standalone plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories_with_data = sam_summary[sam_summary['confusion_category'].isin(['TP', 'TN', 'FP', 'FN'])]
    if not categories_with_data.empty:
        sns.boxplot(data=categories_with_data, x='confusion_category', y='correction_magnitude', ax=ax)
        ax.set_title('Correction Magnitude Distribution by Category', fontsize=14)
        ax.set_ylabel('Mean Correction Magnitude (ppm)', fontsize=12)
        ax.set_xlabel('Confusion Matrix Category', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correction_magnitude_by_category_standalone.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_f1_score_by_sam_category_standalone(data, output_dir):
    """Create a standalone plot of F1 scores by SAM category."""
    if not PLOT_CONFIG['f1_score_by_sam_category_standalone']:
        return
        
    print("Creating standalone F1 score by SAM category plot...")
    
    # Handle different possible column names - including _x/_y suffixes from merge
    sam_col = 'SAM' if 'SAM' in data.columns else 'sam_id'
    pred_col = ('predicted_label' if 'predicted_label' in data.columns else 
                'pred_label' if 'pred_label' in data.columns else 
                'rf_prediction' if 'rf_prediction' in data.columns else 'rf_prediction_x')
    confusion_col = 'confusion_category' if 'confusion_category' in data.columns else 'confusion_category_x'
    true_label_col = 'true_label' if 'true_label' in data.columns else 'true_label_x'
    
    # Aggregate by SAM to get category-level stats
    sam_summary = data.groupby(sam_col).agg({
        'sam_category': 'first',
        confusion_col: 'first',
        true_label_col: 'first',
        pred_col: 'first'
    }).reset_index()
    
    # Standardize column names
    sam_summary = sam_summary.rename(columns={
        sam_col: 'SAM', 
        pred_col: 'pred_label',
        confusion_col: 'confusion_category',
        true_label_col: 'true_label'
    })
    
    # Calculate metrics by category
    category_stats = []
    
    for category in sam_summary['sam_category'].unique():
        cat_data = sam_summary[sam_summary['sam_category'] == category]
        
        if len(cat_data) < 5:  # Skip categories with too few samples
            continue
            
        tp = ((cat_data['true_label'] == 1) & (cat_data['pred_label'] == 1)).sum()
        tn = ((cat_data['true_label'] == 0) & (cat_data['pred_label'] == 0)).sum()
        fp = ((cat_data['true_label'] == 0) & (cat_data['pred_label'] == 1)).sum()
        fn = ((cat_data['true_label'] == 1) & (cat_data['pred_label'] == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        category_stats.append({
            'category': category,
            'n_samples': len(cat_data),
            'f1': f1
        })
    
    if not category_stats:
        print("No categories with sufficient data for analysis")
        return
        
    stats_df = pd.DataFrame(category_stats)
    
    # Create standalone F1 score plot with nice colors
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define nice colors for each category
    color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    colors = color_palette[:len(stats_df)]
    
    bars = ax.bar(stats_df['category'], stats_df['f1'], color=colors)
    ax.set_title('F1 Score by SAM Category', fontsize=16, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=14)
    ax.set_xlabel('SAM Category', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(stats_df['f1']) * 1.1)
    
    # Add value labels on bars
    for bar, value in zip(bars, stats_df['f1']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_score_by_sam_category_standalone.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_labeled_data_distribution_by_category(data, output_dir):
    """Create a stacked bar plot showing labeled data distribution by SAM category and class."""
    if not PLOT_CONFIG['labeled_data_distribution_by_category']:
        return
        
    print("Creating labeled data distribution by SAM category plot...")
    
    # Load labeled data
    try:
        labels = pd.read_csv('data/labels/Swath_Bias_labels.csv')
        # Remove duplicates
        labels = labels.drop_duplicates(subset=['identifier'])
    except FileNotFoundError:
        print("Warning: Could not find labeled data file 'data/labels/Swath_Bias_labels.csv'")
        print("Skipping labeled data distribution plot...")
        return
    
    # Define categories and classes
    categories = ['fossil', 'desert', 'ecostress', 'volcano', 'sif', 'texmex']
    classes = [0, 1, 2]
    
    # Calculate stats for labeled data by category
    category_stats = {}
    for cat in categories:
        subset = labels[labels['identifier'].str.contains(cat, case=False, na=False)]
        stats = {}
        for c in classes:
            stats[c] = subset[subset['label'] == c].shape[0]
        category_stats[cat] = stats
    
    # Filter out categories with no data
    category_stats = {cat: stats for cat, stats in category_stats.items() 
                     if sum(stats.values()) > 0}
    
    if not category_stats:
        print("No labeled data found for any category")
        return
    
    # Create stacked bar plot
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Using a colorblind-friendly palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, c in enumerate(classes):
        bar_positions = range(len(category_stats.keys()))
        bar_heights = [category_stats[cat][c] for cat in category_stats.keys()]
        bottoms = [sum(category_stats[cat][k] for k in classes[:i]) for cat in category_stats.keys()]
        
        label_names = {0: 'No Bias', 1: 'Clear Bias', 2: 'Uncertain'}
        bars = ax.bar(
            category_stats.keys(),
            bar_heights,
            label=f"Class {c}: {label_names[c]}",
            bottom=bottoms,
            color=colors[i]
        )
    
    ax.set_ylabel("Number of Labeled SAMs", fontsize=14)
    ax.set_xlabel("SAM Category", fontsize=14)
    ax.set_title("Labeled Data Distribution by Category and Class", fontsize=16, fontweight='bold')
    ax.legend(title="Label Class")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'labeled_data_distribution_by_category.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print("Labeled Data Statistics:")
    total_labeled = sum(sum(stats.values()) for stats in category_stats.values())
    for c in classes:
        count = sum(category_stats[cat][c] for cat in category_stats.keys())
        label_names = {0: 'No Clear Bias', 1: 'Clear Bias', 2: 'Uncertain'}
        print(f"  Class {c} ({label_names[c]}): {count} ({count/total_labeled*100:.1f}%)")
    
    # Save detailed stats
    stats_df = pd.DataFrame(category_stats).T
    stats_df.columns = ['No_Clear_Bias', 'Clear_Bias', 'Uncertain']
    stats_df['Total'] = stats_df.sum(axis=1)
    stats_df.to_csv(os.path.join(output_dir, 'labeled_data_stats_by_category.csv'))
    print(f"Detailed statistics saved to: labeled_data_stats_by_category.csv")

def plot_performance_by_sam_category(data, output_dir):
    """Plot performance metrics broken down by SAM category."""
    if not PLOT_CONFIG['performance_by_sam_category']:
        return
        
    print("Creating performance by SAM category plot...")
    
    # Handle different possible column names - including _x/_y suffixes from merge
    sam_col = 'SAM' if 'SAM' in data.columns else 'sam_id'
    pred_col = ('predicted_label' if 'predicted_label' in data.columns else 
                'pred_label' if 'pred_label' in data.columns else 
                'rf_prediction' if 'rf_prediction' in data.columns else 'rf_prediction_x')
    confusion_col = 'confusion_category' if 'confusion_category' in data.columns else 'confusion_category_x'
    true_label_col = 'true_label' if 'true_label' in data.columns else 'true_label_x'
    
    # Aggregate by SAM to get category-level stats
    sam_summary = data.groupby(sam_col).agg({
        'sam_category': 'first',
        confusion_col: 'first',
        true_label_col: 'first',
        pred_col: 'first'
    }).reset_index()
    
    # Standardize column names
    sam_summary = sam_summary.rename(columns={
        sam_col: 'SAM', 
        pred_col: 'pred_label',
        confusion_col: 'confusion_category',
        true_label_col: 'true_label'
    })
    
    # Calculate metrics by category
    category_stats = []
    
    for category in sam_summary['sam_category'].unique():
        cat_data = sam_summary[sam_summary['sam_category'] == category]
        
        if len(cat_data) < 5:  # Skip categories with too few samples
            continue
            
        tp = ((cat_data['true_label'] == 1) & (cat_data['pred_label'] == 1)).sum()
        tn = ((cat_data['true_label'] == 0) & (cat_data['pred_label'] == 0)).sum()
        fp = ((cat_data['true_label'] == 0) & (cat_data['pred_label'] == 1)).sum()
        fn = ((cat_data['true_label'] == 1) & (cat_data['pred_label'] == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        category_stats.append({
            'category': category,
            'n_samples': len(cat_data),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity
        })
    
    if not category_stats:
        print("No categories with sufficient data for analysis")
        return
        
    stats_df = pd.DataFrame(category_stats)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # F1 scores by category
    axes[0,0].bar(stats_df['category'], stats_df['f1'])
    axes[0,0].set_title('F1 Score by SAM Category')
    axes[0,0].set_ylabel('F1 Score')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Sample sizes
    axes[0,1].bar(stats_df['category'], stats_df['n_samples'])
    axes[0,1].set_title('Sample Size by SAM Category')
    axes[0,1].set_ylabel('Number of SAMs')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Precision vs Recall
    axes[1,0].scatter(stats_df['recall'], stats_df['precision'], s=stats_df['n_samples']*2)
    for i, row in stats_df.iterrows():
        axes[1,0].annotate(row['category'], (row['recall'], row['precision']), 
                          xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[1,0].set_xlabel('Recall (Sensitivity)')
    axes[1,0].set_ylabel('Precision')
    axes[1,0].set_title('Precision vs Recall by Category\n(Bubble size = sample size)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Confusion matrix breakdown
    confusion_by_cat = sam_summary.groupby(['sam_category', 'confusion_category']).size().unstack(fill_value=0)
    if not confusion_by_cat.empty:
        confusion_by_cat.plot(kind='bar', ax=axes[1,1], stacked=True)
        axes[1,1].set_title('Confusion Matrix Categories by SAM Type')
        axes[1,1].set_ylabel('Count')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].legend(title='Confusion Category')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_by_sam_category.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save statistics table
    stats_df.to_csv(os.path.join(output_dir, 'performance_by_sam_category.csv'), index=False)

def plot_correction_application_map(data, output_dir):
    """Plot geographic map showing where bias correction is applied using custom plot_map function."""
    if not PLOT_CONFIG['correction_application_map']:
        return
        
    print("Creating correction application map...")
    
    # Determine which SAMs had correction applied
    # Check if correction was applied by comparing original and corrected values
    data['correction_applied'] = np.abs(data['xco2_swath-BC'] - data['xco2']) > 1e-6
    
    # Aggregate by SAM to get one data point per SAM
    sam_summary = data.groupby('SAM').agg({
        'latitude': 'mean',
        'longitude': 'mean',
        'correction_applied': 'first',  # All soundings in a SAM should have same correction status
        'sam_category': 'first'
    }).reset_index()
    
    # Create correction percentage variable (1 if corrected, 0 if not)
    sam_summary['correction_percent'] = sam_summary['correction_applied'].astype(float) * 100
    
    # Use the custom plot_map function with 5-degree resolution and no title
    plot_map_custom_res(sam_summary, 
            ['correction_percent'], 
            save_fig=True, 
            path=output_dir, 
            name='correction_application_map_5deg',
            pos_neg_IO=False,  # This is percentage data, not positive/negative
            max=100,  # Maximum is 100%
            min=0,    # Minimum is 0%
            aggregate='mean',  # Average the percentages in each grid cell
            cmap=plt.cm.Reds,  # Use red colormap for intensity
            set_nan=True,
            res=5)  # Use 5-degree resolution
    
    print(f"Correction application map saved with 5-degree resolution")
    
    # Print summary statistics
    n_corrected = sam_summary['correction_applied'].sum()
    n_total = len(sam_summary)
    correction_rate = n_corrected / n_total * 100
    
    print(f"Correction Application Summary:")
    print(f"  Total SAMs: {n_total}")
    print(f"  SAMs with correction applied: {n_corrected}")
    print(f"  Overall correction rate: {correction_rate:.1f}%")
    
    # Save summary by category
    category_summary = sam_summary.groupby('sam_category').agg({
        'correction_applied': ['sum', 'count']
    }).round(1)
    category_summary.columns = ['corrected_count', 'total_count']
    category_summary['correction_rate_percent'] = (category_summary['corrected_count'] / 
                                                  category_summary['total_count'] * 100).round(1)
    
    category_summary.to_csv(os.path.join(output_dir, 'correction_application_by_category.csv'))
    print(f"Correction rates by SAM category saved to: correction_application_by_category.csv")

def plot_sam_count_map(data, output_dir):
    """Plot geographic map showing total number of SAMs in each grid cell."""
    if not PLOT_CONFIG.get('sam_count_map', True):  # Default to True if not specified
        return
        
    print("Creating SAM count map...")
    
    # Aggregate by SAM to get one data point per SAM
    sam_summary = data.groupby('SAM').agg({
        'latitude': 'mean',
        'longitude': 'mean',
        'sam_category': 'first'
    }).reset_index()
    
    # Create a count value of 1 for each SAM (so we can sum them up in each grid cell)
    sam_summary['sam_count'] = 1
    
    # Use the custom plot_map function with 5-degree resolution and no title
    plot_map_custom_res(sam_summary, 
            ['sam_count'],  # Use sam_count values to aggregate
            save_fig=True, 
            path=output_dir, 
            name='sam_count_map_5deg',
            pos_neg_IO=False,  # Count data is always positive
            max=None,  # Let it determine max from non-zero data
            min=0,     # Minimum is 0
            aggregate='sum',  # Sum the sam_count values in each grid cell
            cmap=plt.cm.Blues,  # Use blue colormap for counts
            set_nan=True,
            res=5)  # Use 5-degree resolution
    
    print(f"SAM count map saved with 5-degree resolution")
    
    # Print summary statistics
    n_total = len(sam_summary)
    print(f"SAM Count Summary:")
    print(f"  Total SAMs: {n_total}")
    
    # Print count by category
    category_counts = sam_summary['sam_category'].value_counts()
    print(f"  SAMs by category:")
    for category, count in category_counts.items():
        print(f"    {category}: {count}")


def plot_map_custom_res(data, vars, save_fig=False, path='None', name='None', pos_neg_IO=True, max=None, min=None, aggregate='mean', cmap=None, set_nan=True, res=5):
    """
    Custom plot_map function with configurable resolution and no title.
    Modified version of the original plot_map function from main_util.py
    """
    # Import required functions from main_util
    from src.utils.main_util import raster_data
    
    # make vars into a list in case we didn't pass a list
    if isinstance(vars, str):
        vars = [vars]

    for var in vars:
        raster = raster_data(data[var].to_numpy(), data['latitude'].to_numpy(), data['longitude'].to_numpy(), res=res, aggregate=aggregate, set_nan=set_nan)
        if pos_neg_IO:
            if max == None:
                MAX = np.abs(np.nanpercentile(raster, 95))
                MIN = np.abs(np.nanpercentile(raster, 5))
                MAXX = np.max([MAX, MIN])
                MIN = -MAXX
                MAX = MAXX
            else:
                MIN = min
                MAX = max
            if cmap == None:
                colormap = plt.cm.coolwarm
            else:
                colormap = cmap
            extend = 'both'
        else:
            if max == None:
                # For count data (like SAM counts), use actual maximum instead of 95th percentile
                # to show the full gradient distribution
                if aggregate == 'count' or aggregate == 'sum':
                    # For count/sum data, use only non-zero values to determine the color scale
                    non_zero_values = raster[raster > 0]
                    if len(non_zero_values) > 0:
                        # Use 95th percentile if data is highly skewed, otherwise use max
                        percentile_95 = np.percentile(non_zero_values, 95)
                        max_val = np.max(non_zero_values)
                        # If 95th percentile is much less than max, data is skewed - use 95th percentile
                        # Otherwise use the actual maximum for better color resolution
                        if max_val > 2 * percentile_95:
                            MAX = percentile_95
                        else:
                            MAX = max_val
                    else:
                        MAX = 1  # Fallback if no non-zero values
                else:
                    MAX = np.nanpercentile(raster, 95)
            else:
                MAX = max
            if min == None:
                # For count data, minimum should be 0
                if aggregate == 'count' or aggregate == 'sum':
                    MIN = 0
                else:
                    MIN = np.nanpercentile(raster, 5)
            else:
                MIN = min
            if cmap == None:
                colormap = plt.cm.get_cmap('OrRd')
            else:
                colormap = cmap
            extend = 'max'
        
        var_name = var
        # Use custom Earth_Map_Raster with no title
        Earth_Map_Raster_No_Title(raster, MIN, MAX, var_name, res=res, Save=save_fig,
                         Save_Name=path + '/' + var_name + name, colormap=colormap, extend=extend)


def Earth_Map_Raster_No_Title(raster, MIN, MAX, var_name, Save=False, Save_Name='None', res=1, colormap=plt.cm.coolwarm, extend='both'):
    """
    Modified version of Earth_Map_Raster from main_util.py that doesn't add a title.
    """
    import cartopy.crs as ccrs
    
    # New version with Cartopy
    limits = [-180, 180, -90, 90]
    offset = res / 2 * np.array([0, 0, 2, 2])
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    im = ax.imshow(np.flipud(raster), interpolation='nearest', origin='lower',
                   extent=np.array(limits)+offset, cmap=colormap, vmin=MIN, vmax=MAX,
                   transform=ccrs.PlateCarree(), alpha=0.9)

    # No title added here (this is the key change)
    plt.colorbar(im, fraction=0.066, pad=0.08, extend=extend, location='bottom', label=var_name)

    if Save:
        plt.savefig(Save_Name + '.png', dpi=200)
        plt.close()
    else:
        plt.show()
    return fig

def plot_spatial_error_distribution(data, output_dir):
    """Plot geographic distribution of classification errors."""
    if not PLOT_CONFIG['spatial_error_distribution']:
        return
        
    print("Creating spatial error distribution plot...")
    
    # Aggregate by SAM - handle potential _x/_y suffixes from merge
    confusion_col = 'confusion_category' if 'confusion_category' in data.columns else 'confusion_category_x'
    
    sam_summary = data.groupby('SAM').agg({
        'latitude': 'mean',
        'longitude': 'mean',
        confusion_col: 'first',
        'sam_category': 'first'
    }).reset_index()
    
    # Rename columns to standard names
    sam_summary = sam_summary.rename(columns={
        confusion_col: 'confusion_category'
    })

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # All classifications
    colors = {'TP': 'green', 'TN': 'blue', 'FP': 'red', 'FN': 'orange'}
    
    for category in ['TP', 'TN', 'FP', 'FN']:
        cat_data = sam_summary[sam_summary['confusion_category'] == category]
        if not cat_data.empty:
            axes[0,0].scatter(cat_data['longitude'], cat_data['latitude'], 
                            label=f'{category} (n={len(cat_data)})', 
                            color=colors[category], alpha=0.7, s=30)
    
    axes[0,0].set_xlabel('Longitude')
    axes[0,0].set_ylabel('Latitude')
    axes[0,0].set_title('Geographic Distribution of Classification Results')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Errors only
    error_data = sam_summary[sam_summary['confusion_category'].isin(['FP', 'FN'])]
    if not error_data.empty:
        for category in ['FP', 'FN']:
            cat_data = error_data[error_data['confusion_category'] == category]
            if not cat_data.empty:
                axes[0,1].scatter(cat_data['longitude'], cat_data['latitude'], 
                                label=f'{category} (n={len(cat_data)})', 
                                color=colors[category], alpha=0.7, s=50)
        
        axes[0,1].set_xlabel('Longitude')
        axes[0,1].set_ylabel('Latitude')
        axes[0,1].set_title('Geographic Distribution of Classification Errors')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    
    # By SAM category
    sam_types = sam_summary['sam_category'].unique()
    colors_sam = plt.cm.tab10(np.linspace(0, 1, len(sam_types)))
    
    for i, sam_type in enumerate(sam_types):
        type_data = sam_summary[sam_summary['sam_category'] == sam_type]
        if not type_data.empty:
            axes[1,0].scatter(type_data['longitude'], type_data['latitude'], 
                            label=f'{sam_type} (n={len(type_data)})', 
                            color=colors_sam[i], alpha=0.7, s=30)
    
    axes[1,0].set_xlabel('Longitude')
    axes[1,0].set_ylabel('Latitude')
    axes[1,0].set_title('Geographic Distribution by SAM Category')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Error rate by region (rough binning)
    lat_bins = np.linspace(sam_summary['latitude'].min(), sam_summary['latitude'].max(), 10)
    lon_bins = np.linspace(sam_summary['longitude'].min(), sam_summary['longitude'].max(), 10)
    
    sam_summary['lat_bin'] = pd.cut(sam_summary['latitude'], lat_bins)
    sam_summary['lon_bin'] = pd.cut(sam_summary['longitude'], lon_bins)
    
    error_rate_by_region = sam_summary.groupby(['lat_bin', 'lon_bin']).apply(
        lambda x: (x['confusion_category'].isin(['FP', 'FN'])).mean() if len(x) > 2 else np.nan
    ).reset_index(name='error_rate')
    
    # Create a simple heatmap-like visualization
    lat_centers = [interval.mid for interval in error_rate_by_region['lat_bin'] if pd.notna(interval)]
    lon_centers = [interval.mid for interval in error_rate_by_region['lon_bin'] if pd.notna(interval)]
    error_rates = error_rate_by_region['error_rate'].values
    
    valid_indices = ~np.isnan(error_rates)
    if valid_indices.sum() > 0:
        scatter = axes[1,1].scatter([lon_centers[i] for i in range(len(lon_centers)) if valid_indices[i]], 
                                  [lat_centers[i] for i in range(len(lat_centers)) if valid_indices[i]], 
                                  c=error_rates[valid_indices], cmap='Reds', s=100)
        plt.colorbar(scatter, ax=axes[1,1], label='Error Rate')
    
    axes[1,1].set_xlabel('Longitude')
    axes[1,1].set_ylabel('Latitude')
    axes[1,1].set_title('Error Rate by Geographic Region')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spatial_error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def print_dataset_statistics(output_dir):
    """Print general dataset statistics from all available data."""
    if not PLOT_CONFIG['dataset_statistics']:
        return

    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    # Load all years of data for comprehensive stats
    all_data = pd.DataFrame()
    mode = 'SAM'
    for year in range(2019, 2025):
        try:
            data_y = load_data(year, mode)
            all_data = pd.concat([all_data, data_y], axis=0)
        except Exception as e:
            print(f"Warning: Could not load data for year {year}: {e}")
            continue
    
    if all_data.empty:
        print("Warning: No data could be loaded for dataset statistics")
        return
    
    # Create log file for statistics
    log_file = open(os.path.join(output_dir, "dataset_statistics.txt"), "w")
    
    def dual_print(*args, **kwargs):
        print(*args, **kwargs)  # prints to console
        print(*args, file=log_file, **kwargs)
    
    # Basic statistics
    n_sams = all_data['SAM'].nunique()
    avg_soundings = all_data.groupby('SAM').size().mean()
    n_sams_500 = all_data.groupby('SAM').size().loc[lambda x: x > 500].count()
    n_unique_target_ids = all_data['target_id'].nunique()
    
    dual_print(f"Total unique SAMs: {n_sams}")
    dual_print(f"Average soundings per SAM: {avg_soundings:.1f}")
    dual_print(f"SAMs with >500 soundings: {n_sams_500}")
    dual_print(f"Number of unique target_id's: {n_unique_target_ids}")
    
    # Category breakdown
    categories = ['fossil', 'desert', 'ecostress', 'volcano', 'sif', 'texmex']
    dual_print("\nSAM counts by category:")
    for category in categories:
        count = all_data[all_data['SAM'].str.contains(category, case=False, na=False)].groupby('SAM').size().count()
        dual_print(f"  {category}: {count}")
    
    log_file.close()
    print(f"Dataset statistics saved to: {os.path.join(output_dir, 'dataset_statistics.txt')}")

def plot_enhancement_analysis(data, output_dir):
    """
    Analyzes and plots the impact of bias correction on the SAM enhancement proxy.
    """
    if not PLOT_CONFIG.get('enhancement_analysis', False):
        return
    print("Creating SAM enhancement analysis plots...")

    # Calculate enhancement for original and bias-corrected data
    # We need to do this SAM by SAM.
    enhancements = []
    
    # Group by the SAM column name that exists in the data
    sam_col = 'SAM' if 'SAM' in data.columns else 'sam_id'
    
    for sam_id, sam_df in tqdm(data.groupby(sam_col), desc="Calculating Enhancements"):
        if len(sam_df) < 50:
            continue
        
        # Check if we have the required columns
        if 'xco2' not in sam_df.columns or 'xco2_swath-BC' not in sam_df.columns:
            continue
            
        try:
            enhancement_orig = SAM_enhancement(sam_df, 'xco2', qf=None, custom_SAM=True)
            enhancement_bc = SAM_enhancement(sam_df, 'xco2_swath-BC', qf=None, custom_SAM=True)
            
            # A SAM is considered corrected if any sounding has the correction flag.
            is_corrected = sam_df.get('correction_applied', pd.Series([False]*len(sam_df))).any()

            enhancements.append({
                'SAM': sam_id,
                'enhancement_original': enhancement_orig,
                'enhancement_bc': enhancement_bc,
                'is_corrected': is_corrected,
                'n_soundings': len(sam_df)
            })
        except Exception as e:
            print(f"Warning: Could not calculate enhancement for SAM {sam_id}: {e}")
            continue
    
    if not enhancements:
        print("Warning: No enhancements could be calculated. Skipping enhancement analysis.")
        return
        
    enhancements_df = pd.DataFrame(enhancements)
    
    # --- The plotting logic from create_enhancement_comparison_plots ---
    # Remove NaN values
    clean_df = enhancements_df.dropna(subset=['enhancement_original', 'enhancement_bc'])
    
    if len(clean_df) < 5:
        print("Not enough valid enhancements for comparison plots.")
        return
    
    # Remove extreme outliers
    for col in ['enhancement_original', 'enhancement_bc']:
        q01 = clean_df[col].quantile(0.01)
        q99 = clean_df[col].quantile(0.99)
        clean_df = clean_df[(clean_df[col] >= q01) & (clean_df[col] <= q99)]
    
    if len(clean_df) < 5:
        print("Not enough valid enhancements after outlier removal.")
        return
        
    # Scatter plot comparison
    correlation = clean_df['enhancement_original'].corr(clean_df['enhancement_bc'])
    m, b = np.polyfit(clean_df['enhancement_original'], clean_df['enhancement_bc'], 1)
    
    # Summary statistics
    mean_orig = clean_df['enhancement_original'].mean()
    mean_bc = clean_df['enhancement_bc'].mean()
    print(f"\nEnhancement Summary ({len(clean_df)} SAMs):")
    print(f"Original - Mean: {mean_orig:.3f}, Std: {clean_df['enhancement_original'].std():.3f}")
    print(f"Bias Corrected - Mean: {mean_bc:.3f}, Std: {clean_df['enhancement_bc'].std():.3f}")
    print(f"Correlation: {correlation:.3f}")
    
    summary_stats = {
        'n_sams': len(clean_df),
        'mean_original': mean_orig,
        'std_original': clean_df['enhancement_original'].std(),
        'mean_bc': mean_bc,
        'std_bc': clean_df['enhancement_bc'].std(),
        'correlation': correlation,
        'regression_slope': m,
        'regression_intercept': b
    }
    
    # New calculation for standard deviation of enhancement difference on corrected SAMs
    std_diff_corrected = np.nan
    if 'is_corrected' in clean_df.columns:
        corrected_sams_df = clean_df[clean_df['is_corrected']]
        if not corrected_sams_df.empty:
            enhancement_diff = corrected_sams_df['enhancement_bc'] - corrected_sams_df['enhancement_original']
            std_diff_corrected = enhancement_diff.std()
            print(f"Standard deviation of enhancement difference for corrected SAMs: {std_diff_corrected:.3f} ppm m/s")

            # Create histogram of enhancement differences for corrected SAMs
            plt.figure(figsize=(6, 4))
            plt.hist(enhancement_diff, bins=30, alpha=0.7, color='purple', label=f'Std Dev = {std_diff_corrected:.3f} ppm m/s')
            plt.axvline(enhancement_diff.mean(), color='black', linestyle='--', label=f'Mean = {enhancement_diff.mean():.3f} ppm m/s')
            plt.xlabel('Enhancement Difference (Corrected - Original) [ppm m/s]')
            plt.ylabel('Count')
            plt.title('Distribution of Enhancement Differences for Corrected SAMs')
            plt.xlim(-7, 7)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'enhancement_difference_histogram_corrected_sams.png'), dpi=300)
            plt.close()
        else:
            print("No corrected SAMs found to calculate enhancement difference standard deviation.")
    
    summary_stats['std_enhancement_diff_corrected'] = std_diff_corrected
    
    # --- New: Relative difference analysis ---
    std_rel_diff_corrected = np.nan
    median_abs_percent_diff = np.nan
    
    if 'is_corrected' in clean_df.columns:
        corrected_sams_df = clean_df[clean_df['is_corrected']]
        # Filter for SAMs where the original enhancement is meaningful
        meaningful_enhancement_sams = corrected_sams_df[np.abs(corrected_sams_df['enhancement_original']) > 1.0]
        
        if not meaningful_enhancement_sams.empty:
            print(f"\nAnalyzing relative enhancement changes for {len(meaningful_enhancement_sams)} corrected SAMs (where |original| > 1.0 ppm m/s).")
            
            # Calculate relative difference
            relative_diff = (meaningful_enhancement_sams['enhancement_bc'] - meaningful_enhancement_sams['enhancement_original']) / meaningful_enhancement_sams['enhancement_original']
            std_rel_diff_corrected = relative_diff.std()
            
            # Calculate median absolute percentage difference
            median_abs_percent_diff = np.median(np.abs(relative_diff)) * 100

            print(f"Standard deviation of RELATIVE enhancement difference for corrected SAMs: {std_rel_diff_corrected:.2f}")
            print(f"Median ABSOLUTE PERCENTAGE enhancement difference for corrected SAMs: {median_abs_percent_diff:.1f}%")

            # Create histogram of relative enhancement differences
            plt.figure(figsize=(10, 6))
            # Cap the histogram range to something reasonable to avoid showing extreme outliers
            plt.hist(relative_diff, bins=40, range=(-2, 2), alpha=0.7, color='teal', label=f'Std Dev = {std_rel_diff_corrected:.2f}')
            plt.axvline(relative_diff.mean(), color='black', linestyle='--', label=f'Mean = {relative_diff.mean():.2f}')
            plt.xlabel('Relative Enhancement Difference ((Corrected - Original) / Original)')
            plt.ylabel('Count')
            plt.title(f'Distribution of Relative Enhancement Differences for Corrected SAMs\n(|Original Enhancement| > 1.0 ppm m/s, n={len(meaningful_enhancement_sams)})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'relative_enhancement_difference_histogram.png'), dpi=300)
            plt.close()
        else:
            print("No corrected SAMs with meaningful original enhancement (> 1.0 ppm m/s) found.")
    
    summary_stats['std_relative_enhancement_diff_corrected'] = std_rel_diff_corrected
    summary_stats['median_absolute_percentage_diff_corrected'] = median_abs_percent_diff

    # --- NEW: Create scatter plot with only corrected SAMs ---
    # Only plot corrected SAMs, since uncorrected ones just fall on the 1:1 line
    if 'is_corrected' in clean_df.columns:
        corrected_df = clean_df[clean_df['is_corrected']]
        
        if not corrected_df.empty:
            # Calculate fit parameters for corrected SAMs only
            m_corr, b_corr = np.polyfit(corrected_df['enhancement_original'], corrected_df['enhancement_bc'], 1)
            correlation_corr = corrected_df['enhancement_original'].corr(corrected_df['enhancement_bc'])
            r_squared_corr = correlation_corr ** 2
            
            # Add corrected-SAMs-only statistics to summary_stats
            summary_stats['n_corrected_sams'] = len(corrected_df)
            summary_stats['mean_original_corrected'] = corrected_df['enhancement_original'].mean()
            summary_stats['std_original_corrected'] = corrected_df['enhancement_original'].std()
            summary_stats['mean_bc_corrected'] = corrected_df['enhancement_bc'].mean()
            summary_stats['std_bc_corrected'] = corrected_df['enhancement_bc'].std()
            summary_stats['correlation_corrected'] = correlation_corr
            summary_stats['regression_slope_corrected'] = m_corr
            summary_stats['regression_intercept_corrected'] = b_corr
            summary_stats['r_squared_corrected'] = r_squared_corr
            
            # Calculate mean absolute change for corrected SAMs
            enhancement_diff_corrected = corrected_df['enhancement_bc'] - corrected_df['enhancement_original']
            mean_abs_change_corrected = enhancement_diff_corrected.abs().mean()
            summary_stats['mean_abs_enhancement_change_corrected'] = mean_abs_change_corrected
            
            print(f"\nCORRECTED SAMs ONLY Statistics ({len(corrected_df)} SAMs):")
            print(f"Original - Mean: {corrected_df['enhancement_original'].mean():.3f}, Std: {corrected_df['enhancement_original'].std():.3f}")
            print(f"Bias Corrected - Mean: {corrected_df['enhancement_bc'].mean():.3f}, Std: {corrected_df['enhancement_bc'].std():.3f}")
            print(f"Correlation: {correlation_corr:.3f}")
            print(f"Regression slope: {m_corr:.3f}")
            print(f"Regression intercept: {b_corr:.3f}")
            print(f"R²: {r_squared_corr:.3f}")
            print(f"Mean absolute enhancement change: {mean_abs_change_corrected:.3f} ppm m/s")
            
            # Make figure roughly square and 20% smaller
            plt.figure(figsize=(6.4, 6.4))  # Reduced from 8x6 to 6.4x6.4 (20% smaller, square)
            
            # Plot only corrected SAMs
            plt.scatter(corrected_df['enhancement_original'], corrected_df['enhancement_bc'],
                       alpha=0.8, s=30, color='red', label='Corrected SAMs', zorder=2)
            
            # Add 1:1 line
            plot_range = [min(corrected_df['enhancement_original'].min(), corrected_df['enhancement_bc'].min()),
                          max(corrected_df['enhancement_original'].max(), corrected_df['enhancement_bc'].max())]
            plt.plot(plot_range, plot_range, 'k--', alpha=0.5, linewidth=1, label='1:1 Line')
            
            # Add regression line with reduced precision
            x_line = np.linspace(plot_range[0], plot_range[1], 100)
            y_line = m_corr * x_line + b_corr
            plt.plot(x_line, y_line, 'b-', alpha=0.7, linewidth=2, 
                     label=f'Fit: y = {m_corr:.2f}x + {b_corr:.2f} (R² = {r_squared_corr:.2f})')
            
            plt.xlabel('Enhancement Original [ppm m/s]')
            plt.ylabel('Enhancement Bias Corrected [ppm m/s]')
            plt.title('Enhancement Proxy: Before vs After Correction')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'enhancement_scatter_corrected_highlighted.png'), dpi=300)
            plt.close()
        else:
            print("No corrected SAMs found for scatter plot.")
    else:
        print("No 'is_corrected' column found in data.")
    
    # Create enhanced CSV with clear documentation
    summary_df = pd.DataFrame([summary_stats])
    
    # Add documentation header to CSV
    documentation = [
        "# Enhancement Analysis Statistics",
        "# Overall dataset statistics (all SAMs with valid enhancement calculations):",
        "# n_sams: Total number of SAMs analyzed",
        "# mean_original, std_original: Mean and std dev of original enhancement proxy [ppm m/s]", 
        "# mean_bc, std_bc: Mean and std dev of bias-corrected enhancement proxy [ppm m/s]",
        "# correlation: Correlation between original and corrected enhancement (all SAMs)",
        "# regression_slope, regression_intercept: Linear fit parameters (all SAMs)",
        "#",
        "# Corrected SAMs only statistics (SAMs that were actually bias-corrected):",
        "# n_corrected_sams: Number of SAMs that received bias correction",
        "# mean_original_corrected, std_original_corrected: Original enhancement stats for corrected SAMs",
        "# mean_bc_corrected, std_bc_corrected: Bias-corrected enhancement stats for corrected SAMs", 
        "# correlation_corrected: Correlation between original and corrected (corrected SAMs only)",
        "# regression_slope_corrected, regression_intercept_corrected: Linear fit (corrected SAMs only)",
        "# r_squared_corrected: R² value for corrected SAMs only",
        "# mean_abs_enhancement_change_corrected: Mean absolute change in enhancement for corrected SAMs [ppm m/s]",
        "#",
        "# Other statistics:",
        "# std_enhancement_diff_corrected: Std dev of enhancement differences for corrected SAMs",
        "# std_relative_enhancement_diff_corrected: Std dev of relative enhancement differences",
        "# median_absolute_percentage_diff_corrected: Median absolute percentage difference",
        ""
    ]
    
    # Write documentation and data
    csv_path = os.path.join(output_dir, 'enhancement_summary_stats.csv')
    with open(csv_path, 'w') as f:
        for line in documentation:
            f.write(line + '\n')
        summary_df.to_csv(f, index=False)
    
    print(f"Enhancement summary stats saved to: {csv_path}")
    print(f"Enhancement scatter plot with corrected SAMs highlighted saved.")

def plot_tp_fp_correction_magnitude_standalone(data, output_dir):
    """Creates a compact plot comparing TP vs FP correction magnitudes."""
    if not PLOT_CONFIG['tp_fp_correction_magnitude_standalone']:
        return
    print("Creating standalone TP vs FP correction magnitude histogram...")
    
    # Calculate correction magnitude
    data['correction_magnitude'] = np.abs(data['xco2_swath-BC'] - data['xco2'])
    
    # Handle column naming from merge operations
    confusion_col = 'confusion_category'
    if confusion_col not in data.columns:
        if 'confusion_category_x' in data.columns:
            confusion_col = 'confusion_category_x'
        elif 'confusion_category_y' in data.columns:
            confusion_col = 'confusion_category_y'
        else:
            print("Warning: No confusion_category column found, skipping TP vs FP correction magnitude plot")
            return
    
    # Aggregate by SAM
    sam_summary = data.groupby('SAM').agg({
        'correction_magnitude': 'mean',
        confusion_col: 'first'
    }).reset_index()
    
    # Rename the column for consistent access
    if confusion_col != 'confusion_category':
        sam_summary = sam_summary.rename(columns={confusion_col: 'confusion_category'})
    
    # Get TP and FP data
    tp_data = sam_summary[sam_summary['confusion_category'] == 'TP']['correction_magnitude']
    fp_data = sam_summary[sam_summary['confusion_category'] == 'FP']['correction_magnitude']
    
    if len(tp_data) == 0 and len(fp_data) == 0:
        print("No TP or FP data available for correction magnitude comparison")
        return

    # Create compact standalone plot
    fig, ax = plt.subplots(figsize=(6, 4))  # Compact size
    
    # Use same bins for both histograms to ensure proper comparison
    if len(tp_data) > 0 and len(fp_data) > 0:
        # Determine common bin range
        all_data = np.concatenate([tp_data, fp_data])
        bins = np.linspace(all_data.min(), all_data.max(), 15)
    else:
        bins = 15
    
    # Plot histograms with counts instead of density
    if len(tp_data) > 0:
        n_tp, bins_tp, patches_tp = ax.hist(tp_data, alpha=0.7, label=f'True Positives (n={len(tp_data)})', 
                                          bins=bins, color='green')
        # Add mean line for TP
        tp_mean = tp_data.mean()
        ax.axvline(tp_mean, color='darkgreen', linestyle='--', linewidth=2, 
                  label=f'TP Mean: {tp_mean:.2f} ppm')
    
    if len(fp_data) > 0:
        n_fp, bins_fp, patches_fp = ax.hist(fp_data, alpha=0.7, label=f'False Positives (n={len(fp_data)})', 
                                          bins=bins, color='red')
        # Add mean line for FP
        fp_mean = fp_data.mean()
        ax.axvline(fp_mean, color='darkred', linestyle='--', linewidth=2,
                  label=f'FP Mean: {fp_mean:.2f} ppm')
    
    ax.set_xlabel('Mean Correction Magnitude (ppm)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)  # Changed from Density to Count
    ax.set_title('Correction Magnitude: True vs False Positives', fontsize=12, fontweight='bold')
    
    # Adjust layout for compact size
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tp_fp_correction_magnitude_standalone.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    if len(tp_data) > 0:
        print(f"  TP correction magnitude - Mean: {tp_mean:.6f}, Std: {tp_data.std():.6f} ppm")
    if len(fp_data) > 0:
        print(f"  FP correction magnitude - Mean: {fp_mean:.6f}, Std: {fp_data.std():.6f} ppm")


def plot_tp_fp_emission_proxy_change_standalone(data, output_dir):
    """Create three standalone plots comparing TP vs FP emission proxy changes in different formats."""
    if not PLOT_CONFIG['tp_fp_correction_magnitude_standalone']:
        return
    print("Creating standalone TP vs FP emission proxy change histograms (3 versions)...")
    
    # Import the SAM_enhancement function
    from src.utils.main_util import SAM_enhancement
    
    # Handle column naming from merge operations
    confusion_col = 'confusion_category'
    if confusion_col not in data.columns:
        if 'confusion_category_x' in data.columns:
            confusion_col = 'confusion_category_x'
        elif 'confusion_category_y' in data.columns:
            confusion_col = 'confusion_category_y'
        else:
            print("Warning: No confusion_category column found, skipping TP vs FP emission proxy change plot")
            return
    
    # Calculate emission enhancement proxy for each SAM
    sam_col = 'SAM' if 'SAM' in data.columns else 'sam_id'
    enhancement_results = []
    
    print("Calculating emission proxy changes for TP/FP analysis...")
    for sam_id, sam_df in tqdm(data.groupby(sam_col), desc="Processing SAMs"):
        if len(sam_df) < 100:  # Skip SAMs with insufficient data
            continue
        
        # Get confusion category
        confusion_category = sam_df[confusion_col].iloc[0]
        
        # Only process TP and FP cases
        if confusion_category not in ['TP', 'FP']:
            continue
            
        # Check if we have the required columns
        if 'xco2' not in sam_df.columns or 'xco2_swath-BC' not in sam_df.columns:
            continue
            
        try:
            # Calculate original and corrected enhancement proxies
            enhancement_orig = SAM_enhancement(sam_df, 'xco2', qf=None, custom_SAM=True)
            enhancement_bc = SAM_enhancement(sam_df, 'xco2_swath-BC', qf=None, custom_SAM=True)
            
            # Store results for all valid calculations
            if not np.isnan(enhancement_orig) and not np.isnan(enhancement_bc):
                # Calculate absolute change in ppm·m/s
                abs_change = abs(enhancement_bc - enhancement_orig)
                
                # Calculate signed change in ppm·m/s  
                signed_change = enhancement_bc - enhancement_orig
                
                # Calculate percentage change if original is not near zero
                percent_change = None
                if abs(enhancement_orig) > 0.1:  # Avoid division by very small numbers
                    percent_change = abs((enhancement_bc - enhancement_orig) / enhancement_orig) * 100
                
                enhancement_results.append({
                    'SAM': sam_id,
                    'confusion_category': confusion_category,
                    'enhancement_original': enhancement_orig,
                    'enhancement_bc': enhancement_bc,
                    'abs_change_ppm_ms': abs_change,
                    'signed_change_ppm_ms': signed_change,
                    'percent_change': percent_change
                })
                
        except Exception as e:
            print(f"Warning: Could not calculate enhancement for SAM {sam_id}: {e}")
            continue
    
    if not enhancement_results:
        print("No enhancement calculations could be completed for TP/FP analysis")
        return
        
    results_df = pd.DataFrame(enhancement_results)
    
    # Get TP and FP data for all three metrics
    tp_results = results_df[results_df['confusion_category'] == 'TP']
    fp_results = results_df[results_df['confusion_category'] == 'FP']
    
    if len(tp_results) == 0 and len(fp_results) == 0:
        print("No TP or FP data available for emission proxy change comparison")
        return

    # === PLOT 1: Percentage changes (absolute) ===
    tp_percent = tp_results.dropna(subset=['percent_change'])['percent_change']
    fp_percent = fp_results.dropna(subset=['percent_change'])['percent_change']
    
    if len(tp_percent) > 0 or len(fp_percent) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Determine common bin range for percentage
        if len(tp_percent) > 0 and len(fp_percent) > 0:
            all_percent = np.concatenate([tp_percent, fp_percent])
            max_val = min(np.percentile(all_percent, 95), 100)
            bins = np.linspace(0, max_val, 15)
        else:
            bins = np.linspace(0, 50, 15)
        
        if len(tp_percent) > 0:
            n_tp, bins_tp, patches_tp = ax.hist(tp_percent, alpha=0.7, label=f'True Positives (n={len(tp_percent)})', 
                                              bins=bins, color='green')
            tp_mean = tp_percent.mean()
            ax.axvline(tp_mean, color='darkgreen', linestyle='--', linewidth=2, 
                      label=f'TP Mean: {tp_mean:.1f}%')
        
        if len(fp_percent) > 0:
            n_fp, bins_fp, patches_fp = ax.hist(fp_percent, alpha=0.7, label=f'False Positives (n={len(fp_percent)})', 
                                              bins=bins, color='red')
            fp_mean = fp_percent.mean()
            ax.axvline(fp_mean, color='darkred', linestyle='--', linewidth=2,
                      label=f'FP Mean: {fp_mean:.1f}%')
        
        ax.set_xlabel('Absolute Percentage Change in Emission Proxy (%)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Emission Proxy Change: True vs False Positives (Percentage)', fontsize=12, fontweight='bold')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'emission_proxy_change_tp_fp.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  PERCENTAGE - TP: Mean {tp_mean:.1f}%, FP: Mean {fp_mean:.1f}%")

    # === PLOT 2: Absolute changes in ppm·m/s ===
    tp_abs = tp_results['abs_change_ppm_ms']
    fp_abs = fp_results['abs_change_ppm_ms']
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Determine common bin range for absolute changes
    if len(tp_abs) > 0 and len(fp_abs) > 0:
        all_abs = np.concatenate([tp_abs, fp_abs])
        max_val = np.percentile(all_abs, 95)  # Use 95th percentile to avoid outliers
        bins = np.linspace(0, max_val, 15)
    else:
        bins = 15
    
    if len(tp_abs) > 0:
        n_tp, bins_tp, patches_tp = ax.hist(tp_abs, alpha=0.7, label=f'True Positives (n={len(tp_abs)})', 
                                          bins=bins, color='green')
        tp_mean_abs = tp_abs.mean()
        ax.axvline(tp_mean_abs, color='darkgreen', linestyle='--', linewidth=2, 
                  label=f'TP Mean: {tp_mean_abs:.2f} ppm·m/s')
    
    if len(fp_abs) > 0:
        n_fp, bins_fp, patches_fp = ax.hist(fp_abs, alpha=0.7, label=f'False Positives (n={len(fp_abs)})', 
                                          bins=bins, color='red')
        fp_mean_abs = fp_abs.mean()
        ax.axvline(fp_mean_abs, color='darkred', linestyle='--', linewidth=2,
                  label=f'FP Mean: {fp_mean_abs:.2f} ppm·m/s')
    
    ax.set_xlabel('Absolute Change in Emission Proxy (ppm·m/s)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Emission Proxy Change: True vs False Positives', fontsize=12, fontweight='bold')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emission_proxy_change_tp_fp_absolute.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ABSOLUTE - TP: Mean {tp_mean_abs:.2f} ppm·m/s, FP: Mean {fp_mean_abs:.2f} ppm·m/s")

    # === PLOT 3: Signed changes in ppm·m/s (centered at zero) ===
    tp_signed = tp_results['signed_change_ppm_ms']
    fp_signed = fp_results['signed_change_ppm_ms']
    
    # Calculate statistics BEFORE binning
    tp_mean_signed = tp_signed.mean() if len(tp_signed) > 0 else 0
    tp_std_signed = tp_signed.std() if len(tp_signed) > 0 else 0
    fp_mean_signed = fp_signed.mean() if len(fp_signed) > 0 else 0
    fp_std_signed = fp_signed.std() if len(fp_signed) > 0 else 0
    
    fig, ax = plt.subplots(figsize=(6, 5))  # Increased height from 4 to 5 inches
    
    # Determine symmetric bin range for signed changes
    if len(tp_signed) > 0 and len(fp_signed) > 0:
        all_signed = np.concatenate([tp_signed, fp_signed])
        max_abs_val = max(abs(np.percentile(all_signed, 5)), abs(np.percentile(all_signed, 95)))
        bins = np.linspace(-max_abs_val, max_abs_val, 15)
    else:
        # Fallback symmetric range
        max_abs_val = 5.0
        bins = np.linspace(-max_abs_val, max_abs_val, 15)
    
    if len(tp_signed) > 0:
        n_tp, bins_tp, patches_tp = ax.hist(tp_signed, alpha=0.7, 
                                          label=f'TP (n={len(tp_signed)}): μ={tp_mean_signed:.2f}, σ={tp_std_signed:.2f} ppm·m/s', 
                                          bins=bins, color='green')
        # Add mean line for TP
        ax.axvline(tp_mean_signed, color='darkgreen', linestyle='--', linewidth=2)
        
        # Add standard deviation range as shaded area
        ax.axvspan(tp_mean_signed - tp_std_signed, tp_mean_signed + tp_std_signed, 
                  alpha=0.2, color='green', label=f'TP ±1σ range')
    
    if len(fp_signed) > 0:
        n_fp, bins_fp, patches_fp = ax.hist(fp_signed, alpha=0.7, 
                                          label=f'FP (n={len(fp_signed)}): μ={fp_mean_signed:.2f}, σ={fp_std_signed:.2f} ppm·m/s', 
                                          bins=bins, color='red')
        # Add mean line for FP
        ax.axvline(fp_mean_signed, color='darkred', linestyle='--', linewidth=2)
        
        # Add standard deviation range as shaded area
        ax.axvspan(fp_mean_signed - fp_std_signed, fp_mean_signed + fp_std_signed, 
                  alpha=0.2, color='red', label=f'FP ±1σ range')
    
    # Add zero line for reference
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Zero Change')
    
    ax.set_xlabel('Signed Change in Emission Proxy (ppm·m/s)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Emission Proxy Change: True vs False Positives (Signed)', fontsize=12, fontweight='bold')
    
    # Place legend in upper left to avoid overlap with data
    plt.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emission_proxy_change_tp_fp_signed.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  SIGNED - TP: Mean {tp_mean_signed:.2f} ppm·m/s (σ={tp_std_signed:.2f}), FP: Mean {fp_mean_signed:.2f} ppm·m/s (σ={fp_std_signed:.2f})")
    
    # Save the results for reference
    results_df.to_csv(os.path.join(output_dir, 'tp_fp_emission_proxy_changes.csv'), index=False)
    print(f"Emission proxy change results saved to: {os.path.join(output_dir, 'tp_fp_emission_proxy_changes.csv')}")
    print("Generated 3 emission proxy change plots:")
    print("  1. emission_proxy_change_tp_fp.png (percentage changes)")
    print("  2. emission_proxy_change_tp_fp_absolute.png (absolute changes in ppm·m/s)")
    print("  3. emission_proxy_change_tp_fp_signed.png (signed changes in ppm·m/s, centered at zero)")
    
    return results_df

def main():
    """Main function to run all analysis plots."""
    
    # Set up argument parser for CLI
    plot_choices = list(PLOT_CONFIG.keys())
    parser = argparse.ArgumentParser(description="Generate analysis plots for Swath BC Evaluation.")
    parser.add_argument(
        '--plot',
        nargs='*',
        choices=plot_choices,
        help='Specify one or more plots to generate by name. If not specified, all plots will be generated.'
    )
    args = parser.parse_args()

    # If specific plots are requested via CLI, update the global PLOT_CONFIG
    if args.plot:
        # Turn all plots off first
        for key in PLOT_CONFIG:
            PLOT_CONFIG[key] = False
        # Then turn on the requested ones
        for plot_name in args.plot:
            PLOT_CONFIG[plot_name] = True

    # Use centralized config for paths
    config = PathConfig()
    base_data_dir = config.PROCESSED_EXPERIMENT_DIR
    output_dir = config.FIGURES_DIR / "evaluation_analysis"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run functions based on config
    if PLOT_CONFIG.get('dataset_statistics'):
        print_dataset_statistics(str(output_dir))
    
    # The labeled data distribution plot loads its own data file
    if PLOT_CONFIG.get('labeled_data_distribution_by_category'):
        plot_labeled_data_distribution_by_category(None, str(output_dir))
    
    # Check if any data-dependent plots are needed before proceeding
    plots_requiring_data = [k for k, v in PLOT_CONFIG.items() if v and k not in ['dataset_statistics', 'labeled_data_distribution_by_category']]
    if not plots_requiring_data:
        print("\nNo data-dependent plots requested. Analysis complete!")
        return
        
    print("\nLoading evaluation data...")
    data, summary_stats = load_evaluation_data(str(base_data_dir))
    
    if data.empty:
        print("No data loaded. Cannot generate plots.")
        return

    print(f"Loaded data for {data['SAM'].nunique()} unique SAMs")
    if 'confusion_category' in data.columns:
        print(f"Confusion categories: {data['confusion_category'].value_counts().to_dict()}")
    
    # Save overall evaluation summary statistics to CSV
    save_summary_stats_to_csv(summary_stats, str(output_dir))
    
    # Add SAM category mapping
    data = create_sam_category_mapping(data)
    print(f"SAM categories: {data['sam_category'].value_counts().to_dict()}")
    
    # Generate plots based on PLOT_CONFIG
    # Each function now checks its own flag internally.
    print("\n" + "="*50)
    print("GENERATING ANALYSIS PLOTS")
    print("="*50)
    
    plot_confusion_matrix_examples(data, str(output_dir), n_examples=50)
    plot_confusion_matrix_heatmap(summary_stats, str(output_dir))
    plot_feature_distributions(data, summary_stats, str(output_dir))
    plot_correction_magnitude_by_category_standalone(data, str(output_dir))
    plot_performance_by_sam_category(data, str(output_dir))
    plot_f1_score_by_sam_category_standalone(data, str(output_dir))
    plot_spatial_error_distribution(data, str(output_dir))
    plot_correction_application_map(data, str(output_dir))
    plot_sam_count_map(data, str(output_dir))
    
    if PLOT_CONFIG.get('enhancement_analysis',False):
        plot_enhancement_analysis(data, str(output_dir))

    
    if PLOT_CONFIG.get('correction_magnitude_distributions'):
        print("Creating correction magnitude distributions plot...")
        if 'xco2_swath-BC' in data.columns and 'xco2' in data.columns:
            data['correction_magnitude'] = np.abs(data['xco2_swath-BC'] - data['xco2'])
            
            # Handle column naming from merge operations
            confusion_col = 'confusion_category'
            if confusion_col not in data.columns:
                if 'confusion_category_x' in data.columns:
                    confusion_col = 'confusion_category_x'
                elif 'confusion_category_y' in data.columns:
                    confusion_col = 'confusion_category_y'
                else:
                    print("Warning: No confusion_category column found, skipping correction magnitude distributions plot")
                    return
            
            sam_summary = data.groupby('SAM').agg({
                'correction_magnitude': 'mean',
                confusion_col: 'first'
            }).reset_index()
            
            # Rename the column for consistent access
            if confusion_col != 'confusion_category':
                sam_summary = sam_summary.rename(columns={confusion_col: 'confusion_category'})

            plt.figure(figsize=(12, 8))
            
            categories = ['TP', 'TN', 'FP', 'FN']
            colors = ['green', 'blue', 'red', 'orange']
            
            for i, category in enumerate(categories):
                cat_data = sam_summary[sam_summary['confusion_category'] == category]['correction_magnitude']
                if len(cat_data) > 0:
                    plt.hist(cat_data, alpha=0.7, label=f'{category} (n={len(cat_data)})', 
                            bins=30, density=True, color=colors[i])
            
            plt.xlabel('Mean Correction Magnitude (ppm)')
            plt.ylabel('Density')
            plt.title('Distribution of Correction Magnitudes by Confusion Category')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'correction_magnitude_distributions.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print("Skipping correction magnitude distribution plot: XCO2 columns not found.")
    
    plot_tp_fp_correction_magnitude_standalone(data, str(output_dir))
    plot_tp_fp_emission_proxy_change_standalone(data, str(output_dir))
    
    print(f"\nAll plots saved to: {output_dir}")
    print("Analysis complete!")

if __name__ == "__main__":
    main() 