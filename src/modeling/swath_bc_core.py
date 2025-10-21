import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
import shap
from tqdm import tqdm
import joblib

# Placeholder for base_features_list if it becomes globally configurable
# For now, it's defined within check_drivers

def iterative_feature_selection(X, y, candidate_features, max_features=6, cv=5, random_state=42):
    """
    Iteratively selects features based on the increase in recall.

    In each iteration, the function tests every remaining candidate by adding it to the
    currently selected set and then training a RandomForestClassifier. The feature that gives
    the highest average recall (using cross-validation) is added to the selected features.
    The loop stops when no candidate improves recall further or when max_features have been selected.

    Parameters:
        X (pandas.DataFrame): DataFrame containing candidate features.
        y (pandas.Series): Target labels.
        candidate_features (list): List of candidate feature names.
        max_features (int): Maximum number of features to select.
        cv (int): Number of cross-validation folds.
        random_state (int): Random seed for reproducibility.

    Returns:
        list: The ordered list of selected features.
    """

    selected_features = []
    remaining_features = candidate_features.copy()
    best_score = 0  # baseline recall to beat

    while len(selected_features) < max_features and remaining_features:
        best_feature = None
        best_feature_score = best_score

        # Try adding each remaining feature and measure cross-validated recall.
        for feat in remaining_features:
            current_features = selected_features + [feat]
            X_sub = X[current_features]
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
            clf = RandomForestClassifier(
                random_state=random_state,
                n_estimators=50,
                max_depth=4,
                min_weight_fraction_leaf=0.05,
                n_jobs=-1,
                class_weight={0: 0.05, 1: 0.95}
            )
            scores = cross_val_score(clf, X_sub, y, cv=skf, scoring=make_scorer(f1_score))
            avg_score = np.mean(scores)
            if avg_score > best_feature_score:
                best_feature_score = avg_score
                best_feature = feat

        # If a feature provided an improvement, add it to the selected set.
        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            best_score = best_feature_score
            print(f"Added feature: {best_feature} with F1: {best_feature_score:.3f}")
        else:
            print("No remaining feature improved the recall score further. Stopping selection.")
            break

    return selected_features

def check_drivers(data, # Full dataset for the current run/fold
                  labels_df, # DataFrame of all labels (identifier, label)
                  available_sams_for_rf_training, # List of SAM IDs to be used for RF training & internal validation
                  feature_config, # Dict: {'method': 'iterative'/'list', 'iterative_max': N, 'list_features': [...], 'include_std': True/False}
                  rf_config_params, # Dict: {'max_depth': M, 'min_weight_fraction_leaf': L, 'class_weight_zero': Z, 'class_weight_one': O} (n_estimators is fixed)
                  rf_threshold_config, # Dict: {'method': 'fixed'/'tune', 'fixed_value': V}
                  only_corrected_flag, # Boolean
                  model_artifacts_output_dir, # Path to save model
                  figure_output_dir_check_drivers, # Path to save figures
                  retrain=False, # If true, retrain model, else load
                  save_fig=False, 
                  project_tmp_dir=None,
                  train_on_all_available_data_after_tuning=False): # New parameter

    model_name = 'rf_model_classifier'

    if 'swath_bias' not in data.columns: 
        data['swath_bias'] = (data['xco2_swath-BC'] + data['mean_adjust'] - data['xco2'])

    if retrain:
        if only_corrected_flag:
            print("check_drivers: Training on 'only_corrected' SAMs.")
            data_for_rf_training = data[data['swath_bias'] != 0].copy() 
        else:
            print("check_drivers: Training on all SAMs (not 'only_corrected').")
            data_for_rf_training = data.copy()

    os.makedirs(model_artifacts_output_dir, exist_ok=True)
    if save_fig:
        os.makedirs(figure_output_dir_check_drivers, exist_ok=True)
    
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
    
    #base_features_list = ['h_continuum_sco2', 'aod_total', 'albedo_o2a', 'albedo_wco2', 'albedo_sco2', 'dpfrac', 'co2_grad_del', 'sensor_zenith_angle', 'solar_zenith_angle']
    
    current_features_to_use = [] 

    if retrain:
        sams_with_labels = labels_df['identifier'].unique()
        available_sams_set = set(available_sams_for_rf_training)
        
        data_train_candidate_pool = data_for_rf_training[
            data_for_rf_training['SAM'].isin(available_sams_set) &
            data_for_rf_training['SAM'].isin(sams_with_labels)
        ]

        relevant_labels_df = labels_df[labels_df['identifier'].isin(data_train_candidate_pool['SAM'].unique())].copy()
        relevant_labels_df = relevant_labels_df[relevant_labels_df['label'] != 2]
        data_train_candidate_pool = data_train_candidate_pool[data_train_candidate_pool['SAM'].isin(relevant_labels_df['identifier'].unique())]

        if data_train_candidate_pool.empty or relevant_labels_df.empty:
            print("Warning: No usable labeled SAMs for training RF in check_drivers. Skipping RF training.")
            return {sam: 0 for sam in data['SAM'].unique()}

        rf_train_sams_unique = data_train_candidate_pool['SAM'].unique()
        
        SAMs_rf_internal_train = []
        SAMs_rf_internal_test = []

        if len(rf_train_sams_unique) < 2 : 
             SAMs_rf_internal_train = rf_train_sams_unique
             SAMs_rf_internal_test = np.array([]) 
        else:
            sam_labels_for_split = relevant_labels_df.drop_duplicates(subset=['identifier']).set_index('identifier')
            sam_labels_for_split = sam_labels_for_split.loc[rf_train_sams_unique] 

            if len(sam_labels_for_split['label'].unique()) > 1 : 
                SAMs_rf_internal_train, SAMs_rf_internal_test = train_test_split(
                    rf_train_sams_unique, 
                    test_size=0.3 if len(rf_train_sams_unique) >= 7 else 0.15, 
                    random_state=42, 
                    stratify=sam_labels_for_split['label'].values
                )
            else: 
                 SAMs_rf_internal_train, SAMs_rf_internal_test = train_test_split(
                    rf_train_sams_unique, 
                    test_size=0.3 if len(rf_train_sams_unique) >= 7 else 0.15,
                    random_state=42
                )
        
        rf_data_full_feat_eng = data_train_candidate_pool[
            base_features_list + ['swath_bias', 'sounding_id', 'SAM', 'latitude', 'longitude']
        ].dropna(subset=base_features_list) 

        X_rf_aggregated_all_candidate = rf_data_full_feat_eng.groupby('SAM')[base_features_list].mean()

        if feature_config.get('include_std', False):
            X_rf_std_all_candidate = rf_data_full_feat_eng.groupby('SAM')[base_features_list].std().add_suffix('_std')
            X_rf_aggregated_all_candidate = X_rf_aggregated_all_candidate.merge(X_rf_std_all_candidate, left_index=True, right_index=True, how='left').fillna(0)
            current_features_for_selection = base_features_list + [f'{f}_std' for f in base_features_list]
        else:
            current_features_for_selection = base_features_list

        y_rf_aggregated_all_candidate = relevant_labels_df.set_index('identifier')['label'].loc[X_rf_aggregated_all_candidate.index]
        
        common_sams_all_candidate = X_rf_aggregated_all_candidate.index.intersection(y_rf_aggregated_all_candidate.index)
        X_rf_aggregated = X_rf_aggregated_all_candidate.loc[common_sams_all_candidate] # This is X for the full pool for this call
        y_rf_aggregated = y_rf_aggregated_all_candidate.loc[common_sams_all_candidate] # This is y for the full pool

        X_rf_train_internal = X_rf_aggregated[X_rf_aggregated.index.isin(SAMs_rf_internal_train)]
        y_rf_train_internal = y_rf_aggregated[y_rf_aggregated.index.isin(SAMs_rf_internal_train)]
        
        X_rf_test_internal = X_rf_aggregated[X_rf_aggregated.index.isin(SAMs_rf_internal_test)]
        y_rf_test_internal = y_rf_aggregated[y_rf_aggregated.index.isin(SAMs_rf_internal_test)]

        if feature_config['method'] == 'iterative':
            print(f"check_drivers: Using iterative feature selection, max_features={feature_config['iterative_max']}")
            selected_features_for_model = iterative_feature_selection(
                X_rf_train_internal, y_rf_train_internal, 
                candidate_features=current_features_for_selection, 
                max_features=feature_config['iterative_max'], 
                cv=3, 
                random_state=11
            )
            if project_tmp_dir and selected_features_for_model:
                selected_features_file = os.path.join(project_tmp_dir, 'selected_features_for_run.txt')
                os.makedirs(project_tmp_dir, exist_ok=True)
                with open(selected_features_file, 'w') as f:
                    for feature in selected_features_for_model:
                        f.write(f"{feature}\n")
            current_features_to_use = selected_features_for_model if selected_features_for_model else feature_config.get('list_features_fallback', base_features_list[:5])
        
        elif feature_config['method'] == 'list':
            # If 'list_features' is provided in config, use it, otherwise fallback
            current_features_to_use = feature_config.get('list_features', feature_config.get('list_features_fallback', base_features_list[:5]))
            print(f"check_drivers: Using predefined list of features: {current_features_to_use}")
        
        else: # Fallback for unknown method
             print(f"check_drivers: Warning - unknown feature selection method '{feature_config['method']}'. Using fallback: {base_features_list[:5]}")
             current_features_to_use = base_features_list[:5]


        if not current_features_to_use: 
             print("Warning: No features selected/provided, falling back to first 5 base features.")
             current_features_to_use = base_features_list[:5]
        
        print(f"check_drivers: Features selected for RF model: {current_features_to_use}")

        X_rf_train_final_for_tuning = X_rf_train_internal[current_features_to_use]
        
        if not X_rf_test_internal.empty:
             X_rf_test_final_for_tuning = X_rf_test_internal[current_features_to_use]
        else:
             X_rf_test_final_for_tuning = pd.DataFrame(columns=current_features_to_use)


        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=43,
            n_jobs=-1,
            max_depth=rf_config_params['max_depth'],
            class_weight={0: rf_config_params['class_weight_zero'], 1: rf_config_params['class_weight_one']},
            min_weight_fraction_leaf=rf_config_params['min_weight_fraction_leaf'],
            bootstrap=True
        )
        # Fit on the internal training set for tuning purposes
        model.fit(X_rf_train_final_for_tuning, y_rf_train_internal)

        current_rf_threshold = 0.5 # Default
        if rf_threshold_config['method'] == 'tune':
            if not X_rf_test_final_for_tuning.empty and not y_rf_test_internal.empty:
                print("check_drivers: Tuning RF prediction threshold.")
                probs_test = model.predict_proba(X_rf_test_final_for_tuning)[:, 1]
                best_f1 = -1
                tuned_threshold = 0.5 
                for th_candidate in np.linspace(0.1, 0.9, 17): 
                    y_pred_candidate = (probs_test >= th_candidate).astype(int)
                    f1_candidate = f1_score(y_rf_test_internal, y_pred_candidate)
                    if f1_candidate > best_f1:
                        best_f1 = f1_candidate
                        tuned_threshold = th_candidate
                current_rf_threshold = tuned_threshold
                print(f"check_drivers: Tuned RF threshold to: {current_rf_threshold:.2f} (achieved F1: {best_f1:.2f} on internal test set)")

                if save_fig: 
                    thresholds_plot = np.linspace(0.05, 0.95, 50)
                    f1_scores_plot = [f1_score(y_rf_test_internal, (probs_test >= th).astype(int)) for th in thresholds_plot]
                    plt.figure(figsize=(8, 5))
                    plt.plot(thresholds_plot, f1_scores_plot, label='F1 Score vs. Threshold')
                    plt.axvline(current_rf_threshold, color='r', linestyle='--', label=f'Selected Threshold: {current_rf_threshold:.2f}')
                    plt.xlabel('Threshold')
                    plt.ylabel('F1 Score (on internal test set)')
                    plt.title('RF Threshold Tuning (check_drivers internal)')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(os.path.join(figure_output_dir_check_drivers, 'rf_threshold_tuning_internal.png'))
                    plt.close()
            else:
                print("check_drivers: Warning - cannot tune threshold due to empty internal test set. Using fixed value from config or 0.5.")
                current_rf_threshold = rf_threshold_config.get('fixed_value', 0.5)
        else: # 'fixed'
            current_rf_threshold = rf_threshold_config['fixed_value']
            print(f"check_drivers: Using fixed RF threshold: {current_rf_threshold}")
        
        # Potentially re-train on all available data for the final model artifact
        if train_on_all_available_data_after_tuning:
            print(f"check_drivers: Re-training final model on all {len(X_rf_aggregated)} available SAMs using selected features and tuned threshold.")
            # Use all data that was part of the candidate pool for this check_drivers call
            X_rf_train_final_production = X_rf_aggregated[current_features_to_use]
            y_rf_train_final_production = y_rf_aggregated
            
            model.fit(X_rf_train_final_production, y_rf_train_final_production)
            print("check_drivers: Final model re-fit complete.")


        model.feature_names_in_ = current_features_to_use 
        model.rf_prediction_threshold_ = current_rf_threshold 
        joblib.dump(model, os.path.join(model_artifacts_output_dir, f'{model_name}.joblib'))

        if save_fig and not X_rf_test_final_for_tuning.empty and len(current_features_to_use) > 1: 
            print("check_drivers: Generating SHAP plots...")
            # SHAP explainer should ideally be trained on the same data the model for SHAP analysis was trained on.
            # If train_on_all_available_data_after_tuning=True, the `model` object is now trained on all data.
            # For consistency of SHAP values with the tuned model (if not re-trained on all), use X_rf_train_final_for_tuning.
            # However, if re-trained, then X_rf_aggregated[current_features_to_use] is more appropriate.
            # Let's use X_rf_train_final_for_tuning for explainer creation to match the data used for probability predictions for threshold tuning.
            explainer = shap.TreeExplainer(model, X_rf_train_final_for_tuning if not train_on_all_available_data_after_tuning else X_rf_aggregated[current_features_to_use])
            shap_values_rf_test = explainer.shap_values(X_rf_test_final_for_tuning) # Calculate for internal test set
            
            if isinstance(shap_values_rf_test, list): 
                shap_values_for_plot = shap_values_rf_test[1] 
            else: 
                shap_values_for_plot = shap_values_rf_test

            shap_plot_feature_names_np = np.array(current_features_to_use)
            X_display = X_rf_test_final_for_tuning[list(current_features_to_use)].copy()

            if len(current_features_to_use) > 1:
                plt.figure() 
                shap.summary_plot(shap_values_for_plot, X_display, feature_names=shap_plot_feature_names_np, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(figure_output_dir_check_drivers, 'shap_summary.png'))
                plt.close()
            elif len(current_features_to_use) == 1:
                 print("Skipping beeswarm SHAP summary plot as there is only one feature. Bar plot will be generated.")

            plt.figure()
            if len(current_features_to_use) == 1:
                shap.summary_plot(shap_values_for_plot, feature_names=shap_plot_feature_names_np, plot_type="bar", show=False)
            else:
                shap.summary_plot(shap_values_for_plot, X_display, plot_type="bar", feature_names=shap_plot_feature_names_np, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(figure_output_dir_check_drivers, 'shap_summary_bar.png'))
            plt.close()
        elif save_fig and (X_rf_test_final_for_tuning.empty or len(current_features_to_use) == 0):
            print("Skipping SHAP plots: internal test set is empty or no features were selected.")


    else: # Not retraining, load existing model
        print('check_drivers: Loading pre-trained RF model')
        model_path = os.path.join(model_artifacts_output_dir, f'{model_name}.joblib')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            # Ensure feature_names_in_ is a list for consistency, might be loaded as array from older models
            current_features_to_use = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else []
            current_rf_threshold = model.rf_prediction_threshold_ if hasattr(model, 'rf_prediction_threshold_') else 0.5
            print(f"check_drivers: Loaded model. Features: {current_features_to_use}, Threshold: {current_rf_threshold}")
        else:
            print(f"check_drivers: Error - Model file not found at {model_path}. Cannot make predictions.")
            return {sam: 0 for sam in data['SAM'].unique()} 

    all_sams_data_feat_eng = data[
        base_features_list + ['SAM'] 
    ].copy() 
    all_sams_data_feat_eng.dropna(subset=base_features_list, inplace=True) 

    X_all_sams_aggregated = all_sams_data_feat_eng.groupby('SAM')[base_features_list].mean()
    # Check if std features were used or might be needed by loaded model
    std_features_in_model = any('_std' in f for f in current_features_to_use)
    if feature_config.get('include_std', False) or std_features_in_model: 
        X_all_sams_std = all_sams_data_feat_eng.groupby('SAM')[base_features_list].std().add_suffix('_std')
        X_all_sams_aggregated = X_all_sams_aggregated.merge(X_all_sams_std, left_index=True, right_index=True, how='left').fillna(0)

    for feat in current_features_to_use:
        if feat not in X_all_sams_aggregated.columns:
            X_all_sams_aggregated[feat] = 0 
            print(f"Warning: Feature {feat} missing in aggregated data for prediction, filled with 0.")
            
    X_all_sams_final = X_all_sams_aggregated[current_features_to_use] if current_features_to_use else pd.DataFrame(index=X_all_sams_aggregated.index)
    
    swath_bias_pred_dict = {}
    if not X_all_sams_final.empty and current_features_to_use: # Ensure there are features to predict with
        probas_all_sams = model.predict_proba(X_all_sams_final)[:, 1]
        y_pred_all_sams = (probas_all_sams >= current_rf_threshold).astype(int)
        swath_bias_pred_dict = dict(zip(X_all_sams_final.index, y_pred_all_sams))
    elif X_all_sams_final.empty and current_features_to_use:
        print("Warning: X_all_sams_final is empty but features were expected. No predictions made.")
    elif not current_features_to_use:
        print("Warning: No features in the loaded/trained model. Predicting all as 0.")
        # Predictions will default to 0 due to empty swath_bias_pred_dict

    final_predictions_for_all_input_sams = {sam: swath_bias_pred_dict.get(sam, 0) for sam in data['SAM'].unique()}
    return final_predictions_for_all_input_sams


def adjust_swath_diffs_by_proximity(data_sam, swath_diffs, lat_col='latitude', lon_col='longitude', distance_threshold=0.1, mean_distance_threshold=0.5):
    import numpy as np
    from scipy.spatial.distance import cdist

    swaths = sorted(data_sam['swath'].unique())
    for i in range(1, len(swaths)):
        swath_prev = swaths[i - 1]
        swath_curr = swaths[i]

        coords_prev = data_sam.loc[data_sam['swath'] == swath_prev, [lat_col, lon_col]].values.astype(float)
        coords_curr = data_sam.loc[data_sam['swath'] == swath_curr, [lat_col, lon_col]].values.astype(float)

        if coords_prev.size == 0 or coords_curr.size == 0:
            continue

        distances = cdist(coords_prev, coords_curr)
        min_distance = np.nanmin(distances)
        mean_distance = cdist(np.mean(coords_prev, axis=0).reshape(1, -1), np.mean(coords_curr, axis=0).reshape(1, -1))

        if min_distance > distance_threshold or mean_distance > mean_distance_threshold:
            swath_diffs.loc[swath_curr] = np.nan
    return swath_diffs

def determine_significant_jumps(swath_diffs, scene_values, adaptive_threshold_value=None, absolute_threshold_value=None):
    if absolute_threshold_value is not None:
        # Use a fixed, absolute threshold if provided
        effective_threshold = absolute_threshold_value
    elif adaptive_threshold_value is not None:
        # Use the adaptive threshold scaled by scene standard deviation
        scene_std = np.nanstd(scene_values)
        if scene_std == 0 or np.isnan(scene_std):
            # Fallback for scenes with no variability; use the adaptive value directly as a guess
            effective_threshold = adaptive_threshold_value 
        else:
            effective_threshold = adaptive_threshold_value * scene_std
    else:
        raise ValueError("Either adaptive_threshold_value or absolute_threshold_value must be provided.")
    
    jump_biases = np.where(swath_diffs.abs() > effective_threshold, swath_diffs, 0)
    return jump_biases

def correct_swath_bias(data, var, swath_grouping_threshold_angle, 
                       jump_significance_threshold_value=0.6,
                       jump_significance_threshold_abs=None,
                       min_soundings_for_median=50,
                       log_stats=None):
    print('Applying swath bias correction')
    sams = data['SAM'].unique()
    
    # Initialize local logging counters
    sams_passed_swath_size = 0
    sams_passed_proximity = 0
    sams_with_significant_jumps = 0
    
    # Always create the corrected column as 'xco2_swath-BC' 
    # regardless of input var name for consistency
    corrected_var = 'xco2_swath-BC'
    if corrected_var not in data.columns:
        data[corrected_var] = data[var].copy()
    
    if var not in data.columns: 
        raise ValueError(f"Variable {var} to correct not found in DataFrame.")
        
    counter = 0
    data_by_SAM = {sam: df for sam, df in data.groupby('SAM')}

    for sam in tqdm(sams):
        data_sam = data_by_SAM[sam].copy()
        # Make sure we have the corrected column in the SAM data
        if corrected_var not in data_sam.columns:
            data_sam[corrected_var] = data_sam[var].copy()
            
        data_sam.loc[:, 'swath'] = (data_sam['pma_elevation_angle'].diff().abs() > swath_grouping_threshold_angle).cumsum()
        
        # Use the corrected variable for calculating medians and applying corrections
        swath_medians = data_sam.groupby('swath')[corrected_var].median()
        swath_indices = swath_medians.index.values
        swath_counts = data_sam.groupby('swath')[corrected_var].count()
        
        # Log SAMs that pass swath size filtering
        passed_swath_size = not (swath_counts < min_soundings_for_median).all()
        if passed_swath_size:
            sams_passed_swath_size += 1
        
        swath_medians[swath_counts < min_soundings_for_median] = np.nan  
        swath_diffs = swath_medians.diff()
        
        # Check if any diffs remain before proximity filtering
        had_diffs_before_proximity = not swath_diffs.isnull().all()
        swath_diffs = adjust_swath_diffs_by_proximity(data_sam, swath_diffs)
        
        # Log SAMs that pass proximity filtering
        passed_proximity = not swath_diffs.isnull().all() and had_diffs_before_proximity
        if passed_proximity:
            sams_passed_proximity += 1

        if swath_diffs.isnull().all():
            continue

        jump_biases = determine_significant_jumps(
            swath_diffs, 
            data_sam[corrected_var], 
            adaptive_threshold_value=jump_significance_threshold_value,
            absolute_threshold_value=jump_significance_threshold_abs
        )

        # Log SAMs with significant jumps
        has_significant_jumps = not np.all(jump_biases == 0)
        if has_significant_jumps:
            sams_with_significant_jumps += 1

        if np.all(jump_biases == 0):
            continue

        cumulative_correction = np.cumsum(np.nan_to_num(jump_biases))
        original_var_mean_before_correction = data_sam[corrected_var].mean()
        
        # Apply swath-by-swath corrections to the corrected variable
        for i, swath_idx_val in enumerate(swath_indices):
            data_sam.loc[data_sam['swath'] == swath_idx_val, corrected_var] -= cumulative_correction[i]

        mean_diff_after_local_correction = data_sam[corrected_var].mean() - original_var_mean_before_correction
        data_sam.loc[:, corrected_var] -= mean_diff_after_local_correction 
        
        # Update the main dataframe with corrected values
        data.loc[data['SAM'] == sam, corrected_var] = data_sam[corrected_var]
        counter += 1

    print(f"Corrected swath bias for {counter} SAMs out of {len(sams)} using threshold_angle={swath_grouping_threshold_angle}, jump_thresh(adaptive)={jump_significance_threshold_value}, jump_thresh(abs)={jump_significance_threshold_abs}")
    
    # Log detailed filtering statistics
    print(f"  - SAMs passed swath size filtering: {sams_passed_swath_size}/{len(sams)} ({sams_passed_swath_size/len(sams)*100:.1f}%)")
    print(f"  - SAMs passed proximity checks: {sams_passed_proximity}/{len(sams)} ({sams_passed_proximity/len(sams)*100:.1f}%)")
    print(f"  - SAMs with significant jumps: {sams_with_significant_jumps}/{len(sams)} ({sams_with_significant_jumps/len(sams)*100:.1f}%)")
    
    # Pass statistics back to global logging if provided
    if log_stats is not None:
        log_stats['after_swath_size_filtering'] = log_stats.get('after_swath_size_filtering', 0) + sams_passed_swath_size
        log_stats['after_proximity_checks'] = log_stats.get('after_proximity_checks', 0) + sams_passed_proximity
        log_stats['sams_with_significant_jumps'] = log_stats.get('sams_with_significant_jumps', 0) + sams_with_significant_jumps
    
    return data