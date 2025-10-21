# Performs simple Swath BC based on swath_BC.py
# NEW APPROACH: One-step process where jump statistics are calculated as features
# and passed to RF along with other features to maximize F1 score directly

import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
from tqdm import tqdm
import optuna

from ..utils.main_util import load_data, plot_SAM, confusion_matrix_rates

# Import shared functions from the new core module
from .swath_bc_core import (
    adjust_swath_diffs_by_proximity
)

def calculate_sam_jump_features(data_sam, var='xco2', swath_grouping_threshold_angle=1, min_soundings_for_median=50):
    """
    Calculate jump statistics for a single SAM as features for RF.
    Returns relative jump statistics without applying any corrections.
    """
    # Create swath groupings
    data_sam = data_sam.copy()
    data_sam.loc[:, 'swath'] = (data_sam['pma_elevation_angle'].diff().abs() > swath_grouping_threshold_angle).cumsum()
    
    # Calculate swath medians
    swath_medians = data_sam.groupby('swath')[var].median()
    swath_counts = data_sam.groupby('swath')[var].count()
    
    # Filter out swaths with too few soundings
    swath_medians[swath_counts < min_soundings_for_median] = np.nan  
    
    # Calculate swath differences
    swath_diffs = swath_medians.diff()
    
    # Adjust by proximity (reuse existing function)
    swath_diffs = adjust_swath_diffs_by_proximity(data_sam, swath_diffs)
    
    # Calculate scene variability for relative jump calculation
    scene_std = np.nanstd(data_sam[var])
    if scene_std == 0 or np.isnan(scene_std):
        scene_std = 1.0  # Fallback to avoid division by zero
    
    # Calculate relative jumps (difference divided by scene variability)
    relative_jumps = swath_diffs / scene_std
    
    # Extract jump features (only max_relative_jump)
    jump_features = {}
    
    if len(relative_jumps.dropna()) == 0:
        # No valid jumps
        jump_features['max_relative_jump'] = 0.0
    else:
        valid_jumps = relative_jumps.dropna()
        jump_features['max_relative_jump'] = np.abs(valid_jumps).max()
    
    return jump_features

def extract_jump_features_for_all_sams(data, var='xco2'):
    """
    Extract jump features for all SAMs in the dataset.
    """
    print("Calculating jump features for all SAMs...")
    jump_features_dict = {}
    
    for sam_id, sam_data in tqdm(data.groupby('SAM'), desc="Processing SAMs"):
        jump_features = calculate_sam_jump_features(sam_data, var=var)
        jump_features_dict[sam_id] = jump_features
    
    # Convert to DataFrame
    jump_features_df = pd.DataFrame.from_dict(jump_features_dict, orient='index')
    return jump_features_df

def train_rf_with_jump_features(X_full_labeled, y_full_labeled, X_all_sams, train_sam_ids, 
                               preselected_features, rf_config_params, 
                               rf_threshold_config, model_artifacts_output_dir, 
                               figure_output_dir=None, save_fig=False):
    """
    Train RF model using a pre-computed feature matrix.
    This function no longer computes features; it uses the provided matrices.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import make_scorer, f1_score
    import joblib
    
    model_name = 'rf_model_classifier_with_jumps'
    os.makedirs(model_artifacts_output_dir, exist_ok=True)
    
    print("Training RF model with pre-computed features...")
    
    # Get training data for this fold from the full feature matrices
    X_train = X_full_labeled.loc[train_sam_ids]
    y_train = y_full_labeled.loc[train_sam_ids]
    
    # Align them just in case
    common_idx = X_train.index.intersection(y_train.index)
    X_train = X_train.loc[common_idx]
    y_train = y_train.loc[common_idx]

    if len(X_train) < 2:
        print("Warning: Insufficient training samples for this fold.")
        # Return a dictionary with 0 for all SAMs in X_all_sams
        return {sam: 0 for sam in X_all_sams.index}

    # Use only the globally preselected features
    print(f"Using preselected features for RF model: {preselected_features}")
    X_train_final = X_train[preselected_features]
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=43,
        n_jobs=-1,
        max_depth=rf_config_params['max_depth'],
        class_weight={0: rf_config_params['class_weight_zero'], 1: rf_config_params['class_weight_one']},
        min_weight_fraction_leaf=rf_config_params['min_weight_fraction_leaf'],
        bootstrap=True
    )
    
    model.fit(X_train_final, y_train)
    
    # Tune threshold if requested
    current_rf_threshold = 0.5
    if rf_threshold_config['method'] == 'tune' and len(X_train) >= 4:
        print("Tuning RF prediction threshold...")
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        best_f1 = -1
        tuned_threshold = 0.5
        
        for th_candidate in np.linspace(0.1, 0.9, 17):
            f1_scores = []
            for train_idx, test_idx in skf.split(X_train_final, y_train):
                X_fold_train, X_fold_test = X_train_final.iloc[train_idx], X_train_final.iloc[test_idx]
                y_fold_train, y_fold_test = y_train.iloc[train_idx], y_train.iloc[test_idx]
                
                temp_model = RandomForestClassifier(
                    n_estimators=50, random_state=43, n_jobs=-1,
                    max_depth=rf_config_params['max_depth'],
                    class_weight={0: rf_config_params['class_weight_zero'], 1: rf_config_params['class_weight_one']},
                    min_weight_fraction_leaf=rf_config_params['min_weight_fraction_leaf']
                )
                temp_model.fit(X_fold_train, y_fold_train)
                probs = temp_model.predict_proba(X_fold_test)[:, 1]
                y_pred = (probs >= th_candidate).astype(int)
                f1_scores.append(f1_score(y_fold_test, y_pred))
            
            avg_f1 = np.mean(f1_scores)
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                tuned_threshold = th_candidate
        
        current_rf_threshold = tuned_threshold
        print(f"Tuned RF threshold to: {current_rf_threshold:.2f} (achieved F1: {best_f1:.3f})")
    else:
        current_rf_threshold = rf_threshold_config.get('fixed_value', 0.5)
    
    # Save model with metadata
    model.feature_names_in_ = preselected_features
    model.rf_prediction_threshold_ = current_rf_threshold
    joblib.dump(model, os.path.join(model_artifacts_output_dir, f'{model_name}.joblib'))
    
    # Make predictions for ALL SAMs using the full feature set
    # Ensure all required features are present in the dataset for all SAMs
    X_all_sams_final = X_all_sams.copy()
    for feat in preselected_features:
        if feat not in X_all_sams_final.columns:
            X_all_sams_final[feat] = 0
            print(f"Warning: Feature {feat} missing from prediction data, filled with 0.")
    
    X_all_final_pred = X_all_sams_final[preselected_features]
    probas_all = model.predict_proba(X_all_final_pred)[:, 1]
    y_pred_all = (probas_all >= current_rf_threshold).astype(int)
    
    predictions_dict = dict(zip(X_all_final_pred.index, y_pred_all))
    
    # Return predictions for all SAMs in the input data
    final_predictions = {sam: predictions_dict.get(sam, 0) for sam in X_all_sams.index}
    return final_predictions

def select_features_for_fold(X_train_fold, y_train_fold, feature_config):
    """
    Select features for a single CV fold using only the training data from that fold.
    This prevents data leakage by not using test fold information in feature selection.
    """
    from .swath_bc_core import iterative_feature_selection
    
    if feature_config['method'] == 'iterative':
        selected_features = iterative_feature_selection(
            X_train_fold, y_train_fold,
            candidate_features=list(X_train_fold.columns),
            max_features=feature_config['iterative_max'],
            cv=3, random_state=11
        )
        if not selected_features:
            # Fallback if no features selected
            selected_features = list(X_train_fold.columns)[:5]
    else:
        # Default to using all features
        selected_features = list(X_train_fold.columns)
    
    return selected_features

def combine_features_from_folds(fold_feature_lists, top_n=5):
    """
    Combine features selected from multiple CV folds by selecting the most frequently occurring ones.
    
    Args:
        fold_feature_lists: List of lists, where each inner list contains features selected from one fold
        top_n: Number of top features to select based on frequency
    
    Returns:
        List of the most frequently selected features across folds
    """
    from collections import Counter
    
    # Count frequency of each feature across all folds
    feature_counts = Counter()
    for features in fold_feature_lists:
        feature_counts.update(features)
    
    # Get the most common features
    most_common_features = [feature for feature, count in feature_counts.most_common(top_n)]
    
    print(f"Feature selection across {len(fold_feature_lists)} folds:")
    print("Feature frequency ranking:")
    for i, (feature, count) in enumerate(feature_counts.most_common()):
        print(f"  {i+1:2d}. {feature:<25} (selected in {count}/{len(fold_feature_lists)} folds)")
    
    print(f"\nSelected top {top_n} features: {most_common_features}")
    return most_common_features

def evaluate_single_config_fold(
    full_data_df_original, 
    all_labels_df, 
    train_sam_ids_for_rf, 
    eval_sam_ids_for_rf,  
    config, 
    X_full_labeled,
    y_full_labeled,
    X_all_sams_features,
    base_model_dir_path=None,
    base_processed_data_dir_path=None,
    base_figure_dir_path=None,
    project_tmp_dir_path=None,
    run_name_for_trial="default_trial_run",
    preselected_features=None
):
    """
    Evaluates a single pipeline configuration for one fold of a cross-validation.
    NEW APPROACH: Uses jump features directly in RF, but still applies swath-by-swath corrections.
    """
    print(f"--- Evaluating Config for Fold (Eval SAMs: {len(eval_sam_ids_for_rf)}) ---")
    print(f"Config: {config}")

    data = full_data_df_original.copy()

    trial_model_artifacts_output_dir = None
    trial_figure_output_dir = None
    save_trial_figs = config.get('save_fig_trial', False)

    if save_trial_figs and base_model_dir_path and base_figure_dir_path:
        trial_model_artifacts_output_dir = os.path.join(base_model_dir_path, run_name_for_trial)
        trial_figure_output_dir = os.path.join(base_figure_dir_path, run_name_for_trial)
        os.makedirs(trial_model_artifacts_output_dir, exist_ok=True)
        os.makedirs(trial_figure_output_dir, exist_ok=True)
    
    temp_model_dir = os.path.join(project_tmp_dir_path or '.', 'temp_model_artifacts_opt')
    os.makedirs(temp_model_dir, exist_ok=True)

    # STEP 1: Train RF to decide which SAMs need correction
    from .swath_bc_core import correct_swath_bias
    
    print("Training RF to identify SAMs needing correction...")
    predictions = train_rf_with_jump_features(
        X_full_labeled=X_full_labeled,
        y_full_labeled=y_full_labeled,
        X_all_sams=X_all_sams_features,
        train_sam_ids=train_sam_ids_for_rf,
        preselected_features=preselected_features,
        rf_config_params=config['rf_config_params'],
        rf_threshold_config=config['rf_threshold_config'],
        model_artifacts_output_dir=trial_model_artifacts_output_dir if trial_model_artifacts_output_dir else temp_model_dir,
        figure_output_dir=trial_figure_output_dir,
        save_fig=save_trial_figs
    )
    
    # STEP 2: Apply corrections ONLY to SAMs that RF identified as needing correction
    sams_needing_correction = [sam for sam, pred in predictions.items() if pred]
    print(f"Applying swath bias correction to {len(sams_needing_correction)} SAMs identified by RF...")
    
    if sams_needing_correction:
        # Create corrected column for all data (initialize as copy of original)
        data['xco2_swath-BC'] = data['xco2'].copy()
        
        # Apply corrections only to identified SAMs
        data_subset = data[data['SAM'].isin(sams_needing_correction)].copy()
        data_subset = correct_swath_bias(
            data=data_subset,
            var='xco2',  # Will modify 'xco2_swath-BC' column
            swath_grouping_threshold_angle=1.0,
            jump_significance_threshold_value=0.6,
            min_soundings_for_median=50
        )
        
        # Update main dataframe with corrections
        data.loc[data['SAM'].isin(sams_needing_correction), 'xco2_swath-BC'] = data_subset['xco2_swath-BC']
    else:
        # No SAMs need correction - just create the columns
        data['xco2_swath-BC'] = data['xco2'].copy()
        print("No SAMs identified for correction by RF.")

    # Evaluate on the eval SAMs (note: data now contains corrected XCO2 where appropriate)
    eval_data_subset = data[data['SAM'].isin(eval_sam_ids_for_rf)]
    if eval_data_subset.empty:
        print(f"Warning: No data for evaluation SAMs in this fold. Returning F1=0.")
        return 0.0
    
    # Check if corrections were applied
    corrected_sams = eval_data_subset[eval_data_subset['xco2_swath-BC'] != eval_data_subset['xco2']]['SAM'].nunique()
    total_eval_sams = eval_data_subset['SAM'].nunique()
    print(f"Evaluation SAMs: {total_eval_sams}, RF predicted for correction: {corrected_sams}, No correction needed: {total_eval_sams - corrected_sams}")

    true_labels_eval_sams_df = all_labels_df[all_labels_df['identifier'].isin(eval_sam_ids_for_rf)].copy()
    true_labels_eval_sams_df = true_labels_eval_sams_df[true_labels_eval_sams_df['label'] != 2]
    
    y_true_for_eval = []
    y_pred_for_eval = []

    for sam_id in eval_sam_ids_for_rf:
        sam_true_label_series = true_labels_eval_sams_df[true_labels_eval_sams_df['identifier'] == sam_id]['label']
        if sam_true_label_series.empty:
            continue
        y_true_for_eval.append(sam_true_label_series.iloc[0])
        
        # Check if this SAM actually had corrections applied
        sam_data = eval_data_subset[eval_data_subset['SAM'] == sam_id]
        if len(sam_data) > 0:
            actually_corrected = (sam_data['xco2_swath-BC'] != sam_data['xco2']).any()
            y_pred_for_eval.append(1 if actually_corrected else 0)
        else:
            y_pred_for_eval.append(0)

    if not y_true_for_eval or len(y_true_for_eval) != len(y_pred_for_eval):
        print("Warning: Mismatch in evaluation data. Returning F1=0.")
        return 0.0

    f1_score_fold = f1_score(y_true_for_eval, y_pred_for_eval, zero_division=0)
    print(f"F1 Score for this fold: {f1_score_fold:.4f} (True: {sum(y_true_for_eval)}/{len(y_true_for_eval)}, Pred: {sum(y_pred_for_eval)}/{len(y_pred_for_eval)})")
    
    # Save fold results for analysis if directories are provided
    if base_processed_data_dir_path and config.get('save_data_trial', False):
        fold_results = {
            'sam_id': [],
            'true_label': [],
            'predicted_label': [],
            'train_sam_ids': list(train_sam_ids_for_rf),
            'eval_sam_ids': list(eval_sam_ids_for_rf),
            'f1_score': f1_score_fold,
            'run_name': run_name_for_trial
        }
        
        # Add individual SAM results
        for i, sam_id in enumerate(eval_sam_ids_for_rf):
            sam_true_label_series = true_labels_eval_sams_df[true_labels_eval_sams_df['identifier'] == sam_id]['label']
            if not sam_true_label_series.empty:
                fold_results['sam_id'].append(sam_id)
                fold_results['true_label'].append(sam_true_label_series.iloc[0])
                fold_results['predicted_label'].append(predictions.get(sam_id, 0))
        
        # Save fold results
        fold_results_df = pd.DataFrame({
            'sam_id': fold_results['sam_id'],
            'true_label': fold_results['true_label'], 
            'predicted_label': fold_results['predicted_label']
        })
        
        # Save detailed fold metadata
        fold_metadata = {
            'train_sam_ids': fold_results['train_sam_ids'],
            'eval_sam_ids': fold_results['eval_sam_ids'],
            'f1_score': fold_results['f1_score'],
            'n_train_sams': len(fold_results['train_sam_ids']),
            'n_eval_sams': len(fold_results['eval_sam_ids']),
            'run_name': fold_results['run_name']
        }
        
        fold_output_dir = os.path.join(base_processed_data_dir_path, run_name_for_trial)
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # Save fold predictions and metadata
        fold_results_df.to_csv(os.path.join(fold_output_dir, 'fold_predictions.csv'), index=False)
        
        import json
        with open(os.path.join(fold_output_dir, 'fold_metadata.json'), 'w') as f:
            json.dump(fold_metadata, f, indent=2, default=str)
        
        # ===== NEW: Save full corrected data for evaluation SAMs as parquet =====
        print("Saving full corrected data for evaluation SAMs...")
        
        # Extract evaluation data (holdout set) with all corrections applied
        eval_data_corrected = data[data['SAM'].isin(eval_sam_ids_for_rf)].copy()
        
        if not eval_data_corrected.empty:
            # Add fold information and predictions to the data
            eval_data_corrected['fold_id'] = run_name_for_trial
            eval_data_corrected['is_holdout_set'] = True
            eval_data_corrected['cv_fold_f1_score'] = f1_score_fold
            
            # Add RF predictions for each SAM
            sam_predictions_map = {}
            for sam_id in eval_sam_ids_for_rf:
                sam_predictions_map[sam_id] = predictions.get(sam_id, 0)
            eval_data_corrected['rf_prediction'] = eval_data_corrected['SAM'].map(sam_predictions_map)
            
            # Add true labels for each SAM
            sam_labels_map = {}
            for sam_id in eval_sam_ids_for_rf:
                sam_label_series = true_labels_eval_sams_df[true_labels_eval_sams_df['identifier'] == sam_id]['label']
                if not sam_label_series.empty:
                    sam_labels_map[sam_id] = sam_label_series.iloc[0]
                else:
                    sam_labels_map[sam_id] = -1  # Unknown label
            eval_data_corrected['true_label'] = eval_data_corrected['SAM'].map(sam_labels_map)
            
            # Add confusion matrix category for each SAM
            eval_data_corrected['confusion_category'] = 'Unknown'
            for sam_id in eval_sam_ids_for_rf:
                true_label = sam_labels_map.get(sam_id, -1)
                predicted_label = sam_predictions_map.get(sam_id, 0)
                
                if true_label != -1:  # Valid label exists
                    if true_label == 1 and predicted_label == 1:
                        category = 'TP'  # True Positive
                    elif true_label == 1 and predicted_label == 0:
                        category = 'FN'  # False Negative
                    elif true_label == 0 and predicted_label == 1:
                        category = 'FP'  # False Positive
                    elif true_label == 0 and predicted_label == 0:
                        category = 'TN'  # True Negative
                    else:
                        category = 'Unknown'
                    
                    eval_data_corrected.loc[eval_data_corrected['SAM'] == sam_id, 'confusion_category'] = category
            
            # Calculate additional derived features for analysis
            if 'xco2_swath-BC' in eval_data_corrected.columns and 'xco2' in eval_data_corrected.columns:
                eval_data_corrected['correction_applied'] = (eval_data_corrected['xco2_swath-BC'] != eval_data_corrected['xco2'])
                eval_data_corrected['xco2_difference'] = eval_data_corrected['xco2_swath-BC'] - eval_data_corrected['xco2']
                eval_data_corrected['abs_correction_magnitude'] = np.abs(eval_data_corrected['xco2_difference'])
            
            # Save as parquet for efficient analysis
            parquet_path = os.path.join(fold_output_dir, 'eval_data_corrected.parquet')
            eval_data_corrected.to_parquet(parquet_path, index=False)
            print(f"Saved evaluation data (holdout set) to: {parquet_path}")
            print(f"  - {len(eval_data_corrected)} soundings from {eval_data_corrected['SAM'].nunique()} SAMs")
            print(f"  - Columns: {list(eval_data_corrected.columns)}")
            
            # Also save SAM-level features used for RF prediction
            if eval_sam_ids_for_rf:
                print("Extracting and saving SAM-level features for evaluation SAMs...")
                
                # We already have the features for all SAMs, just need to filter and save
                eval_sam_features = X_all_sams_features.loc[X_all_sams_features.index.isin(eval_sam_ids_for_rf)].copy()
                
                if not eval_sam_features.empty:
                    # Add labels and predictions
                    eval_sam_features['true_label'] = eval_sam_features.index.map(sam_labels_map)
                    eval_sam_features['rf_prediction'] = eval_sam_features.index.map(sam_predictions_map)
                    eval_sam_features['fold_id'] = run_name_for_trial
                    eval_sam_features['f1_score'] = f1_score_fold
                    
                    # Save SAM features as parquet
                    sam_features_path = os.path.join(fold_output_dir, 'sam_features.parquet')
                    eval_sam_features.to_parquet(sam_features_path)
                    print(f"Saved SAM-level features to: {sam_features_path}")
                    print(f"  - {len(eval_sam_features)} SAMs with {len(eval_sam_features.columns)} features")
        
        print(f"Saved fold results to: {fold_output_dir}")
    
    return f1_score_fold

# Global variable to track feature selection across all trials
TRIAL_FEATURE_TRACKING = []

# Main Optuna objective function
def objective(trial, full_data_df, all_labels_df, base_paths_config):
    """
    Optuna objective function for the new one-step approach with fold-wise feature selection and RF parameters tuning
    Tracks feature selection across all trials for comprehensive analysis.
    """
    config = {}
    
    # Feature selection configuration
    feature_method = 'iterative'
    feature_config = {
        'method': feature_method, 
        'include_std': trial.suggest_categorical('feat_include_std', [False])
    }
    if feature_method == 'iterative':
        feature_config['iterative_max'] = trial.suggest_int('feat_iter_max', 3, 8)  # Increased range due to jump features
    config['feature_config'] = feature_config

    # RF configuration
    # Ensure class weights always sum to 1 by making one dependent on the other
    class_weight_zero = trial.suggest_float('rf_cw_0', 0.02, 0.2, step=0.01)
    class_weight_one = 1.0 - class_weight_zero  # Ensures they sum to 1
    
    config['rf_config_params'] = {
        'max_depth': trial.suggest_int('rf_max_depth', 3, 10),  # Increased range
        'min_weight_fraction_leaf': trial.suggest_float('rf_min_weight', 0.01, 0.05, step=0.01),
        'class_weight_zero': class_weight_zero,
        'class_weight_one': class_weight_one
    }
    
    # Threshold configuration
    threshold_method = trial.suggest_categorical('rf_thresh_method', ['tune', 'fixed'])
    rf_threshold_config = {'method': threshold_method}
    if threshold_method == 'fixed':
        rf_threshold_config['fixed_value'] = trial.suggest_float('rf_thresh_fixed_val', 0.3, 0.8, step=0.05)
    config['rf_threshold_config'] = rf_threshold_config

    config['save_fig_trial'] = base_paths_config.get('save_trial_figs', False)
    config['save_data_trial'] = base_paths_config.get('save_trial_data', False)

    # Cross-validation setup
    labeled_sam_ids = all_labels_df['identifier'].unique()
    labels_for_stratify_df = all_labels_df[all_labels_df['label'] != 2].drop_duplicates(subset=['identifier']).set_index('identifier')
    
    valid_labeled_sams_for_strat = labels_for_stratify_df.index.intersection(labeled_sam_ids)
    labels_for_stratify = labels_for_stratify_df.loc[valid_labeled_sams_for_strat, 'label']
    aligned_sam_ids = labels_for_stratify.index.values
    aligned_labels = labels_for_stratify.values

    n_cv_folds = base_paths_config.get('n_cv_folds', 3)
    if len(aligned_sam_ids) < n_cv_folds:
        print(f"Warning: Not enough unique labeled SAMs ({len(aligned_sam_ids)}) for {n_cv_folds}-fold CV. Pruning trial.")
        raise optuna.exceptions.TrialPruned()

    if len(np.unique(aligned_labels)) < 2:
        print(f"Warning: Only one class present in labels. Cannot stratify. Pruning trial.")
        raise optuna.exceptions.TrialPruned()
        
    # STEP 1: Perform feature selection using fold-wise approach for this trial
    print(f"Trial {trial.number}: Performing fold-wise feature selection...")
    
    # Pre-compute features for all SAMs once
    # Define the complete list of candidate features (same as non-Optuna section)
    base_features_list = ['s31', 's32', 'dof_co2', 'xco2_zlo_bias', 'zlo_wco2'
        , 'solar_zenith_angle', 'sensor_zenith_angle', 'co2_ratio', 'h2o_ratio'
        , 'color_slice_noise_ratio_sco2', 'h_continuum_sco2', 'dp_abp', 'psurf'
        , 't700', 'tcwv', 'tcwv_apriori', 'dpfrac', 'co2_grad_del', 'eof2_2_rel'
        , 'aod_dust', 'aod_bc', 'aod_oc', 'aod_seasalt', 'aod_sulfate', 'dws'
        , 'aod_strataer', 'aod_water', 'aod_ice', 'aod_total', 'dust_height'
        , 'ice_height', 'h2o_scale', 'deltaT', 'albedo_o2a', 'albedo_wco2', 'albedo_sco2'
        , 'albedo_slope_o2a', 'albedo_slope_wco2', 'albedo_slope_sco2'
        , 'solar_azimuth_angle', 'sensor_azimuth_angle', 'max_declocking_o2a'
        , 'glint_angle', 'airmass', 'altitude']
    
    # Calculate jump features for ALL SAMs in the loaded dataset
    all_sams_jump_features = extract_jump_features_for_all_sams(full_data_df, var='xco2')
    
    # Calculate traditional features for ALL SAMs
    all_sams_clean = full_data_df[base_features_list + ['SAM']].dropna(subset=base_features_list)
    all_sams_traditional_features = all_sams_clean.groupby('SAM')[base_features_list].mean()
    
    # Combine into a single feature matrix for ALL processed SAMs
    X_all_sams_features = all_sams_traditional_features.merge(
        all_sams_jump_features, left_index=True, right_index=True, how='left'
    ).fillna(0)
    
    # Get labels for the labeled feature set
    y_full_labeled = labels_for_stratify_df.loc[labels_for_stratify_df.index.intersection(X_all_sams_features.index), 'label']
    common_sams = X_all_sams_features.index.intersection(y_full_labeled.index)
    
    X_full_labeled = X_all_sams_features.loc[common_sams]
    y_full_labeled = y_full_labeled.loc[common_sams]
    
    # Perform fold-wise feature selection
    kf_feature_selection = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=trial.number + 42)
    fold_feature_lists = []
    
    for fold_idx, (train_indices, eval_indices) in enumerate(kf_feature_selection.split(aligned_sam_ids, aligned_labels)):
        train_sams_fold = aligned_sam_ids[train_indices]
        X_train_fold = X_full_labeled.loc[X_full_labeled.index.intersection(train_sams_fold)]
        y_train_fold = y_full_labeled.loc[y_full_labeled.index.intersection(train_sams_fold)]
        
        if len(X_train_fold) > 0 and len(y_train_fold) > 0:
            fold_selected_features = select_features_for_fold(X_train_fold, y_train_fold, config['feature_config'])
            fold_feature_lists.append(fold_selected_features)
    
    # Combine features from all folds by frequency
    if fold_feature_lists:
        max_features = config['feature_config'].get('iterative_max', 5)
        selected_features = combine_features_from_folds(fold_feature_lists, top_n=max_features)
    else:
        selected_features = list(X_full_labeled.columns)[:5]  # Fallback
    
    print(f"Trial {trial.number}: Selected features: {selected_features}")
    
    # Track feature selection for this trial
    trial_feature_info = {
        'trial_number': trial.number,
        'selected_features': selected_features,
        'fold_feature_lists': fold_feature_lists,
        'n_folds': len(fold_feature_lists),
        'config': config.copy()
    }
    TRIAL_FEATURE_TRACKING.append(trial_feature_info)
    
    # STEP 2: Evaluate using the selected features
    kf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=trial.number + 42)
    fold_f1_scores = []

    for fold_idx, (train_indices, eval_indices) in enumerate(kf.split(aligned_sam_ids, aligned_labels)):
        print(f"Optuna Trial {trial.number}, CV Fold {fold_idx + 1}")
        train_sams_this_fold = aligned_sam_ids[train_indices]
        eval_sams_this_fold = aligned_sam_ids[eval_indices]

        run_name_for_this_fold_trial = f"trial_{trial.number}_fold_{fold_idx + 1}"

        if not list(train_sams_this_fold) or not list(eval_sams_this_fold):
            print(f"Warning: Fold {fold_idx+1} resulted in empty train or eval SAM set. Skipping fold.")
            continue

        f1_fold = evaluate_single_config_fold(
            full_data_df_original=full_data_df,
            all_labels_df=all_labels_df,
            train_sam_ids_for_rf=list(train_sams_this_fold),
            eval_sam_ids_for_rf=list(eval_sams_this_fold),
            config=config,
            X_full_labeled=X_full_labeled,
            y_full_labeled=y_full_labeled,
            X_all_sams_features=X_all_sams_features,
            base_model_dir_path=base_paths_config['base_model_dir'],
            base_processed_data_dir_path=base_paths_config['base_processed_data_dir'],
            base_figure_dir_path=base_paths_config['base_figure_dir'],
            project_tmp_dir_path=base_paths_config['project_tmp_dir'],
            run_name_for_trial=run_name_for_this_fold_trial,
            preselected_features=selected_features
        )
        fold_f1_scores.append(f1_fold)
    
    if not fold_f1_scores:
        print(f"Warning: Trial {trial.number} had no valid folds. Pruning.")
        raise optuna.exceptions.TrialPruned()
        
    mean_f1_for_trial = np.mean(fold_f1_scores)
    print(f"Optuna Trial {trial.number} - Mean F1 Score across folds: {mean_f1_for_trial:.4f}")
    return mean_f1_for_trial

def analyze_feature_selection_across_trials(trial_tracking_data, save_path=None):
    """
    Analyze feature selection patterns across all Optuna trials.
    
    Args:
        trial_tracking_data: List of trial feature information
        save_path: Optional path to save detailed analysis results
    
    Returns:
        Dictionary with comprehensive feature selection analysis
    """
    from collections import Counter, defaultdict
    import json
    
    if not trial_tracking_data:
        print("No trial tracking data available.")
        return {}
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE FEATURE SELECTION ANALYSIS ACROSS ALL OPTUNA TRIALS")
    print(f"{'='*80}")
    print(f"Total trials analyzed: {len(trial_tracking_data)}")
    
    # 1. Overall feature frequency across all trials
    all_selected_features = []
    for trial_info in trial_tracking_data:
        all_selected_features.extend(trial_info['selected_features'])
    
    overall_feature_counts = Counter(all_selected_features)
    
    print(f"\n🎯 OVERALL FEATURE SELECTION FREQUENCY (across {len(trial_tracking_data)} trials):")
    print("=" * 60)
    for i, (feature, count) in enumerate(overall_feature_counts.most_common(), 1):
        percentage = (count / len(trial_tracking_data)) * 100
        print(f"{i:2d}. {feature:<25} | {count:3d}/{len(trial_tracking_data)} trials ({percentage:5.1f}%)")
    
    # 2. Feature co-occurrence analysis
    print(f"\n🔗 FEATURE CO-OCCURRENCE ANALYSIS:")
    print("=" * 40)
    feature_pairs = Counter()
    for trial_info in trial_tracking_data:
        features = trial_info['selected_features']
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                pair = tuple(sorted([features[i], features[j]]))
                feature_pairs[pair] += 1
    
    print("Top feature pairs that appear together:")
    for i, (pair, count) in enumerate(feature_pairs.most_common(10), 1):
        percentage = (count / len(trial_tracking_data)) * 100
        print(f"{i:2d}. {pair[0]} + {pair[1]} | {count} times ({percentage:.1f}%)")
    
    # 3. Feature selection by performance
    trial_performance = {}
    for trial_info in trial_tracking_data:
        trial_num = trial_info['trial_number']
        # We'll need to get the F1 scores from the trials - this is a placeholder
        trial_performance[trial_num] = {
            'features': trial_info['selected_features'],
            'feature_count': len(trial_info['selected_features'])
        }
    
    # 4. Fold-level feature analysis
    print(f"\n📊 FOLD-LEVEL FEATURE SELECTION PATTERNS:")
    print("=" * 45)
    fold_feature_counts = Counter()
    fold_total_count = 0
    
    for trial_info in trial_tracking_data:
        for fold_features in trial_info['fold_feature_lists']:
            fold_feature_counts.update(fold_features)
            fold_total_count += 1
    
    print(f"Features selected across all folds in all trials (total folds: {fold_total_count}):")
    for i, (feature, count) in enumerate(fold_feature_counts.most_common(15), 1):
        percentage = (count / fold_total_count) * 100
        print(f"{i:2d}. {feature:<25} | {count:4d}/{fold_total_count} folds ({percentage:5.1f}%)")
    
    # 5. Feature stability analysis
    print(f"\n📈 FEATURE STABILITY ANALYSIS:")
    print("=" * 35)
    top_features = [feat for feat, _ in overall_feature_counts.most_common(10)]
    
    print("Consistency of top features across trials:")
    for feature in top_features:
        appearances = overall_feature_counts[feature]
        stability = (appearances / len(trial_tracking_data)) * 100
        if stability >= 80:
            status = "🟢 HIGHLY STABLE"
        elif stability >= 50:
            status = "🟡 MODERATELY STABLE"
        elif stability >= 25:
            status = "🟠 SOMEWHAT STABLE"
        else:
            status = "🔴 UNSTABLE"
        
        print(f"  {feature:<25} | {appearances:3d}/{len(trial_tracking_data)} ({stability:5.1f}%) {status}")
    
    # Compile comprehensive results
    analysis_results = {
        'total_trials': len(trial_tracking_data),
        'overall_feature_frequencies': dict(overall_feature_counts),
        'top_10_features': [feat for feat, _ in overall_feature_counts.most_common(10)],
        'top_feature_pairs': [{'pair': list(pair), 'count': count} for pair, count in feature_pairs.most_common(10)],
        'fold_level_frequencies': dict(fold_feature_counts),
        'total_folds_analyzed': fold_total_count,
        'feature_stability_scores': {
            feat: (count / len(trial_tracking_data)) * 100 
            for feat, count in overall_feature_counts.items()
        }
    }
    
    # Save detailed results if path provided
    if save_path:
        detailed_results = {
            'analysis_summary': analysis_results,
            'detailed_trial_data': trial_tracking_data
        }
        
        with open(save_path, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        print(f"\n💾 Detailed analysis saved to: {save_path}")
    
    return analysis_results

if __name__ == '__main__':
    # Import centralized configuration
    from ..utils.config_paths import PathConfig
    
    mode_main = 'SAM'
    year_main = None
    run_optuna_study = False  # Enable Optuna to retune features, class weights, and threshold
    
    # Use centralized configuration
    config = PathConfig()
    run_name_optuna_study = config.EXPERIMENT_NAME
    
    labels_file_main_path = config.LABELS_FILE
    
    base_paths = {
        'base_model_dir': str(config.MODEL_BASE_DIR / run_name_optuna_study),  # Parent to allow subdirs per trial
        'base_processed_data_dir': str(config.PROCESSED_BASE_DIR / run_name_optuna_study),
        'base_figure_dir': str(config.FIGURES_DIR / run_name_optuna_study),
        'project_tmp_dir': str(config.project_root / 'tmp' / run_name_optuna_study),
        'n_cv_folds': 4,
        'save_trial_figs': True,
        'save_trial_data': True
    }
    
    print(f"Using centralized configuration:")
    print(f"  Experiment: {config.EXPERIMENT_NAME}")
    print(f"  Model Version: {config.MODEL_VERSION}")
    print(f"  Base paths: {base_paths}")
    
    for path in base_paths.values():
        if isinstance(path, str):
            os.makedirs(path, exist_ok=True)

    print("Loading main dataset...")
    full_dataset = pd.DataFrame()
    if year_main is None:
        for yr_idx in range(2019, 2025):
            print(f"Loading year {yr_idx} for mode {mode_main}")
            data_yr = load_data(yr_idx, mode_main, preload_IO=True, clean_IO=True, TCCON=False)
            full_dataset = pd.concat([full_dataset, data_yr], ignore_index=True)
    else:
        full_dataset = load_data(year_main, mode_main, preload_IO=True, clean_IO=True, TCCON=False)
    
    if full_dataset.empty:
        raise SystemExit("No data loaded. Script cannot proceed.")
    print(f"Total soundings loaded: {len(full_dataset)}")

    # # QF0-only experiment: keep only retrievals with xco2_quality_flag == 0
    # if 'xco2_quality_flag' in full_dataset.columns:
    #     full_dataset = full_dataset[full_dataset['xco2_quality_flag'] == 0].copy()
    #     print(f"QF0 filter applied. Remaining soundings: {len(full_dataset)}")
    # else:
    #     raise SystemExit("xco2_quality_flag missing; cannot run QF0-only experiment.")

    print("Loading labels...")
    all_labels = pd.read_csv(labels_file_main_path)
    all_labels = all_labels.drop_duplicates(subset=['identifier'])
    all_labels_cleaned = all_labels[all_labels['label'] != 2].copy()
    
    if all_labels_cleaned.empty:
        raise SystemExit("No valid labels found after cleaning. Script cannot proceed.")

    labeled_sams_available = all_labels_cleaned['identifier'].unique()
    full_dataset_for_processing = full_dataset[full_dataset['SAM'].isin(labeled_sams_available)].copy()
    
    if full_dataset_for_processing.empty:
        raise SystemExit("No data remains after filtering by available labels. Script cannot proceed.")
    print(f"Soundings for processing: {len(full_dataset_for_processing)}")

    final_labels_for_cv = all_labels_cleaned[all_labels_cleaned['identifier'].isin(full_dataset_for_processing['SAM'].unique())].copy()
    print(f"Number of unique SAMs with labels: {len(final_labels_for_cv['identifier'].unique())}")

    if run_optuna_study:
        study = optuna.create_study(direction='maximize', study_name=run_name_optuna_study)
        # Allow override via env var OCO3_OPTUNA_N_TRIALS; default to 50
        n_trials_optuna = int(os.getenv('OCO3_OPTUNA_N_TRIALS', '50'))
        pbar = tqdm(total=n_trials_optuna, desc="Optuna Optimization (One-Step)")
        
        def tqdm_callback(study, frozen_trial):
            pbar.update(1)
            if study.best_trial:
                pbar.set_postfix_str(f"Best F1: {study.best_value:.4f}, Last: {frozen_trial.value:.4f}")
            else:
                pbar.set_postfix_str(f"Last trial: {frozen_trial.value:.4f}")
        
        study.optimize(
            lambda trial: objective(trial, full_dataset_for_processing, final_labels_for_cv, base_paths),
            n_trials=n_trials_optuna,
            callbacks=[tqdm_callback]
        )
        pbar.close()
        
        print("Optuna study finished.")
        best_trial = study.best_trial
        print("Best trial:")
        print(f"  Value (Max F1): {best_trial.value:.4f}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        
        print('\nOptuna optimization complete!')
        print(f'Best F1 Score: {best_trial.value:.4f}')
        if best_trial.value > 0.4:
            print(f'SUCCESS: Beat the target F1 score of 0.4!')
        else:
            print(f'Target F1 score of 0.4 not yet reached. Consider tuning hyperparameters further.')
        
        # COMPREHENSIVE FEATURE SELECTION ANALYSIS
        print(f"\n{'='*100}")
        print("PERFORMING COMPREHENSIVE FEATURE SELECTION ANALYSIS")
        print(f"{'='*100}")
        
        # Analyze feature selection patterns across all trials
        feature_analysis_path = os.path.join(base_paths['base_processed_data_dir'], 'comprehensive_feature_analysis.json')
        feature_analysis_results = analyze_feature_selection_across_trials(
            TRIAL_FEATURE_TRACKING, 
            save_path=feature_analysis_path
        )
        
        # Extract the most consistently selected features
        if feature_analysis_results:
            print(f"\n🏆 FINAL RECOMMENDATIONS:")
            print("=" * 30)
            top_features = feature_analysis_results['top_10_features'][:5]  # Top 5 most frequent
            print(f"Most consistently selected features across {len(TRIAL_FEATURE_TRACKING)} trials:")
            for i, feature in enumerate(top_features, 1):
                frequency = feature_analysis_results['overall_feature_frequencies'][feature]
                percentage = feature_analysis_results['feature_stability_scores'][feature]
                print(f"  {i}. {feature} (selected in {frequency}/{len(TRIAL_FEATURE_TRACKING)} trials, {percentage:.1f}%)")
            
            # Save the recommended feature set
            recommended_config = {
                'recommended_features': top_features,
                'feature_selection_source': 'optuna_comprehensive_analysis',
                'total_trials_analyzed': len(TRIAL_FEATURE_TRACKING),
                'best_trial_f1': best_trial.value,
                'best_trial_params': best_trial.params,
                'analysis_timestamp': str(pd.Timestamp.now()),
                'experiment_name': run_name_optuna_study
            }
            
            recommended_config_path = os.path.join(base_paths['base_processed_data_dir'], 'recommended_feature_config.json')
            with open(recommended_config_path, 'w') as f:
                import json
                json.dump(recommended_config, f, indent=2, default=str)
            
            print(f"\n💾 Recommended configuration saved to: {recommended_config_path}")
            print(f"📊 Detailed analysis saved to: {feature_analysis_path}")
            
        print(f"\n{'='*100}")
        print("OPTUNA OPTIMIZATION WITH COMPREHENSIVE FEATURE ANALYSIS COMPLETE!")
        print(f"{'='*100}")
    
    else:
        # Run with best configuration from Optuna study
        print("Running final evaluation with best configuration from Optuna study...")
        
        # Best parameters from Optuna run (F1: 0.7)
        best_config = {
            'selected_features': [
                'max_relative_jump',
                'h_continuum_sco2', 
                'dws',
                'max_declocking_o2a',
                'aod_sulfate'
            ],
            'rf_config_params': {
                'max_depth': 7,
                'min_weight_fraction_leaf': 0.03,
                'class_weight_zero': 0.14,  # rf_cw_0: 0.14
                'class_weight_one': 0.86   # 0.86
            },
            'rf_threshold_config': {
                'method': 'fixed', # 'fixed', 'tune'
                'fixed_value': 0.6  # rf_thresh_fixed_val: 0.6
            },
            'save_fig_trial': True,
            'save_data_trial': True
        }
        
        print("Best configuration:")
        for key, value in best_config.items():
            print(f"  {key}: {value}")
        
        # STEP 1: Create feature matrices ONCE for all data
        print("\n=== PRE-COMPUTING FEATURE MATRICES FOR ALL SAMS ===")
        
        # Define the complete list of candidate features
        base_features_list = ['s31', 's32', 'dof_co2', 'xco2_zlo_bias', 'zlo_wco2'
        , 'solar_zenith_angle', 'sensor_zenith_angle', 'co2_ratio', 'h2o_ratio'
        , 'color_slice_noise_ratio_sco2', 'h_continuum_sco2', 'dp_abp', 'psurf'
        , 't700', 'tcwv', 'tcwv_apriori', 'dpfrac', 'co2_grad_del', 'eof2_2_rel'
        , 'aod_dust', 'aod_bc', 'aod_oc', 'aod_seasalt', 'aod_sulfate', 'dws'
        , 'aod_strataer', 'aod_water', 'aod_ice', 'aod_total', 'dust_height'
        , 'ice_height', 'h2o_scale', 'deltaT', 'albedo_o2a', 'albedo_wco2', 'albedo_sco2'
        , 'albedo_slope_o2a', 'albedo_slope_wco2', 'albedo_slope_sco2'
        , 'solar_azimuth_angle', 'sensor_azimuth_angle', 'max_declocking_o2a'
        , 'glint_angle', 'airmass', 'altitude']
        
        # Calculate jump features for ALL SAMs in the loaded dataset
        all_sams_jump_features = extract_jump_features_for_all_sams(full_dataset_for_processing, var='xco2')
        
        # Calculate traditional features for ALL SAMs
        all_sams_clean = full_dataset_for_processing[base_features_list + ['SAM']].dropna(subset=base_features_list)
        all_sams_traditional_features = all_sams_clean.groupby('SAM')[base_features_list].mean()
        
        # Combine into a single feature matrix for ALL processed SAMs
        X_all_sams_features = all_sams_traditional_features.merge(
            all_sams_jump_features, left_index=True, right_index=True, how='left'
        ).fillna(0)
        print(f"Computed feature matrix for {len(X_all_sams_features)} SAMs.")
        
        # Get labels for the labeled feature set
        y_full_labeled = final_labels_for_cv.set_index('identifier')['label']
        common_sams = X_all_sams_features.index.intersection(y_full_labeled.index)
        
        # Create the final feature matrix and labels for labeled data
        X_full_labeled = X_all_sams_features.loc[common_sams]
        y_full_labeled = y_full_labeled.loc[common_sams]
        print(f"Found {len(X_full_labeled)} labeled SAMs with complete features for training/evaluation.")
        
        # Define arrays for cross-validation splitting from the labeled feature matrix
        aligned_sam_ids = y_full_labeled.index.values
        aligned_labels = y_full_labeled.values
        
        # STEP 2: Use hard-coded features from comprehensive 50-trial analysis
        print("\n=== USING HARD-CODED FEATURES FROM COMPREHENSIVE ANALYSIS ===")
        print("Using the top 5 most consistently selected features from 50-trial optimization...")
        
        # Features were selected based on their frequency across all trials:
        # max_relative_jump: 100% (50/50 trials)
        # h_continuum_sco2: 82% (41/50 trials) 
        # dws: 56% (28/50 trials)
        # max_declocking_o2a: 50% (25/50 trials)
        # aod_sulfate: 46% (23/50 trials)
        selected_features = best_config['selected_features']
        
        print(f"\nFINAL COMBINED FEATURES: {selected_features}")
        print("These features will be used consistently across all CV folds.\n")

        n_cv_folds = base_paths.get('n_cv_folds', 4)
        print(f"Running {n_cv_folds}-fold cross-validation with consistent features...")
        
        kf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
        fold_f1_scores = []

        for fold_idx, (train_indices, eval_indices) in enumerate(kf.split(aligned_sam_ids, aligned_labels)):
            print(f"\n=== FINAL EVALUATION - Fold {fold_idx + 1}/{n_cv_folds} ===")
            train_sams_this_fold = aligned_sam_ids[train_indices]
            eval_sams_this_fold = aligned_sam_ids[eval_indices]

            run_name_for_this_fold = f"final_best_config_fold_{fold_idx + 1}"

            f1_fold = evaluate_single_config_fold(
                full_data_df_original=full_dataset_for_processing,
                all_labels_df=final_labels_for_cv,
                train_sam_ids_for_rf=list(train_sams_this_fold),
                eval_sam_ids_for_rf=list(eval_sams_this_fold),
                config=best_config,
                X_full_labeled=X_full_labeled,
                y_full_labeled=y_full_labeled,
                X_all_sams_features=X_all_sams_features,
                base_model_dir_path=base_paths['base_model_dir'],
                base_processed_data_dir_path=base_paths['base_processed_data_dir'],
                base_figure_dir_path=base_paths['base_figure_dir'],
                project_tmp_dir_path=base_paths['project_tmp_dir'],
                run_name_for_trial=run_name_for_this_fold,
                preselected_features=selected_features
            )
            fold_f1_scores.append(f1_fold)
            print(f"Fold {fold_idx + 1} F1 Score: {f1_fold:.4f}")
        
        # Calculate and display final results
        mean_f1_final = np.mean(fold_f1_scores)
        std_f1_final = np.std(fold_f1_scores)
        
        print(f"\n{'='*60}")
        print("FINAL EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Configuration: Best from Optuna study (50 trials)")
        print(f"Mean F1 Score: {mean_f1_final:.4f} ± {std_f1_final:.4f}")
        print(f"Fold F1 Scores: {[f'{f1:.4f}' for f1 in fold_f1_scores]}")
        print(f"{'='*60}")
        
        # Train a final model on all data for deployment
        print("\nTraining final model on all available data...")
        final_run_name = "final_model_all_data"
        
        all_train_sams = list(aligned_sam_ids)  # Use all labeled SAMs for final training
        
        final_model_f1 = evaluate_single_config_fold(
            full_data_df_original=full_dataset_for_processing,
            all_labels_df=final_labels_for_cv,
            train_sam_ids_for_rf=all_train_sams,
            eval_sam_ids_for_rf=all_train_sams,  # Self-evaluation for final model
            config=best_config,
            X_full_labeled=X_full_labeled,
            y_full_labeled=y_full_labeled,
            X_all_sams_features=X_all_sams_features,
            base_model_dir_path=base_paths['base_model_dir'],
            base_processed_data_dir_path=base_paths['base_processed_data_dir'],
            base_figure_dir_path=base_paths['base_figure_dir'],
            project_tmp_dir_path=base_paths['project_tmp_dir'],
            run_name_for_trial=final_run_name,
            preselected_features=selected_features
        )
        
        print(f"Final model training F1 (self-evaluation): {final_model_f1:.4f}")
        print(f"Final model saved as: {final_run_name}")
        
        # Save aggregated CV results for analysis
        cv_summary = {
            'mean_f1_score': mean_f1_final,
            'std_f1_score': std_f1_final,
            'fold_f1_scores': fold_f1_scores,
            'n_folds': n_cv_folds,
            'selected_features': selected_features,
            'best_config': best_config,
            'final_model_f1': final_model_f1,
            'total_labeled_sams': len(aligned_sam_ids),
            'run_name': run_name_optuna_study
        }
        
        cv_summary_path = os.path.join(base_paths['base_processed_data_dir'], 'cv_summary.json')
        import json
        with open(cv_summary_path, 'w') as f:
            json.dump(cv_summary, f, indent=2, default=str)
        
        print(f"CV summary saved to: {cv_summary_path}")
        print("Final evaluation complete!")