#!/usr/bin/env python3
"""
Apply Swath Bias Correction using Random Forest model
This script applies the swath bias correction to OCO-3 Lite files.
"""




import os
import glob
import sys
import numpy as np
import pandas as pd
import netCDF4 as nc
import joblib
from tqdm import tqdm
import shutil # For copying files
from typing import Dict, Any

# Assuming this script is in src/processing/
# Adjust paths to import from other src subdirectories
from ..utils.main_util import read_oco_netcdf_to_df, NC_FILL_VALUE_FLOAT, NC_FILL_VALUE_INT # Import new function and fill values
from ..modeling.swath_bc_core import correct_swath_bias
# --- Centralized Configuration ---
from ..utils.config_paths import PathConfig

# Initialize path configuration
config = PathConfig()

# --- Debug Configuration ---
DEBUG = False  # Set to True to process only limited files for testing


def update_netcdf_with_bc(original_nc_path, output_dir, df_results_for_nc, processed_any_sams):
    """
    Copies the original NetCDF file and appends new bias correction variables.
    df_results_for_nc must contain 'sounding_id' and the new data columns:
    'xco2_swath_bc', 'swath_bias_corrected'.
    It must be sorted by 'sounding_id' and have one row per original sounding.
    """
    base_name = os.path.basename(original_nc_path)
    output_nc_filename = base_name.replace(".nc4", config.OUTPUT_FILE_SUFFIX)
    if ".nc4" not in base_name: # Handles if original is just .nc
        output_nc_filename = base_name.replace(".nc", config.OUTPUT_FILE_SUFFIX.replace('.nc4', '.nc'))

    output_nc_full_path = os.path.join(output_dir, output_nc_filename)

    print(f"Creating/updating NetCDF file: {output_nc_full_path}")
    
    shutil.copy2(original_nc_path, output_nc_full_path) # copy2 preserves metadata

    with nc.Dataset(output_nc_full_path, 'a') as rootgrp:
        # Ensure sounding_id dimension exists (should be 'sounding_id' or 'Sounding')
        # OCO L2 Lite files use 'Sounding' as the dimension name for sounding_id referenced variables
        dim_name = None
        if 'Sounding' in rootgrp.dimensions:
            dim_name = 'Sounding'
        elif 'sounding_id' in rootgrp.dimensions: # Fallback, less common for L2 Lite
            dim_name = 'sounding_id'
        else:
            print(f"Error: Could not find 'Sounding' or 'sounding_id' dimension in {output_nc_full_path}.")
            return

        num_soundings = len(rootgrp.dimensions[dim_name])
        if num_soundings != len(df_results_for_nc):
             print(f"Error: Mismatch in number of soundings. NC: {num_soundings}, DF: {len(df_results_for_nc)}. Skipping update for {output_nc_full_path}")
             return

        # Create new variables
        # 1. xco2_swath_bc (bias-corrected XCO2)
        if 'xco2_swath_bc' not in rootgrp.variables:
            bc_var = rootgrp.createVariable('xco2_swath_bc', 'f4', (dim_name,), fill_value=NC_FILL_VALUE_FLOAT)
            bc_var.units = 'ppm'
            bc_var.long_name = 'XCO2 with Swath Bias Correction'
            # Try to copy attributes from original xco2, then override/add
            if 'xco2' in rootgrp.variables:
                original_xco2_var = rootgrp.variables['xco2']
                for attr_name in original_xco2_var.ncattrs():
                    if attr_name != '_FillValue': # We set our own
                        setattr(bc_var, attr_name, getattr(original_xco2_var, attr_name))
            bc_var.comment = (f'XCO2 with swath bias correction applied using Random Forest model from {config.EXPERIMENT_NAME}. '
                              'Algorithm based on Mauceri et al. (in prep). Original xco2 is preserved.')
            bc_var.ancillary_variables = 'swath_bias_corrected'
        else:
            bc_var = rootgrp.variables['xco2_swath_bc']
        bc_var[:] = df_results_for_nc['xco2_swath_bc'].fillna(NC_FILL_VALUE_FLOAT).values

        # 2. swath_bias_corrected
        if 'swath_bias_corrected' not in rootgrp.variables:
            corrected_var = rootgrp.createVariable('swath_bias_corrected', 'i1', (dim_name,), fill_value=np.int8(NC_FILL_VALUE_INT if NC_FILL_VALUE_INT <= 127 else -1)) # byte needs careful fill
            corrected_var.long_name = 'Swath Bias Correction Flag'
            corrected_var.flag_values = np.array([0, 1], dtype=np.int8)
            corrected_var.flag_meanings = 'no_change correction_applied'
            corrected_var.valid_range = np.array([0, 1], dtype=np.int8)
            corrected_var.comment = 'Flag indicating whether swath bias correction was applied: 0=no change, 1=correction applied.'
        else:
            corrected_var = rootgrp.variables['swath_bias_corrected']
        corrected_var[:] = df_results_for_nc['swath_bias_corrected'].fillna(0).astype(np.int8).values

        # Update global attributes
        history = getattr(rootgrp, 'history', '')
        new_history_entry = (
            f"\n{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}: Processed with swath bias correction "
            f"(model: {config.EXPERIMENT_NAME}, script: apply_swath_bc_RF.py). "
            f"{'Bias correction applied to SAM data.' if processed_any_sams else 'No SAM data processed for bias correction.'}"
        )
        rootgrp.history = history + new_history_entry
        rootgrp.swath_bias_correction_model_version = config.MODEL_VERSION
        rootgrp.swath_bias_correction_training_run = config.EXPERIMENT_NAME
        rootgrp.swath_bias_correction_model_folder = "final_model_all_data"
        rootgrp.swath_bias_correction_method_reference = "Mauceri et al. (in prep, manuscript detailing OCO-3 swath bias correction)"
        rootgrp.swath_bias_correction_approach = "Random Forest with reordered pipeline (RF decision first, targeted corrections)"
        # Add DOIs if available and fixed for the model/method
        # rootgrp.swath_bias_correction_model_doi = "..."
        # rootgrp.swath_bias_correction_method_doi = "..."

    print(f"Successfully updated: {output_nc_full_path}")

def print_sam_processing_summary(sam_stats: Dict[str, Any]):
    """Print a summary of SAM processing statistics."""
    print("\n" + "="*50)
    print("SAM PROCESSING PIPELINE STATISTICS")
    print("="*50)
    print(f"Total SAMs processed: {sam_stats['total_sams_processed']:,}")
    print(f"After RF classification: {sam_stats['after_rf_classification']:,} ({sam_stats['after_rf_classification']/sam_stats['total_sams_processed']*100:.1f}% selected)")
    
    # Detailed filtering statistics
    if 'after_swath_size_filtering' in sam_stats and sam_stats['after_rf_classification'] > 0:
        print(f"After swath size filtering: {sam_stats['after_swath_size_filtering']:,} ({sam_stats['after_swath_size_filtering']/sam_stats['after_rf_classification']*100:.1f}% of selected)")
    
    if 'after_proximity_checks' in sam_stats and sam_stats['after_rf_classification'] > 0:
        print(f"After proximity checks: {sam_stats['after_proximity_checks']:,} ({sam_stats['after_proximity_checks']/sam_stats['after_rf_classification']*100:.1f}% of selected)")
    
    if 'sams_with_significant_jumps' in sam_stats and sam_stats['after_rf_classification'] > 0:
        print(f"SAMs with significant jumps: {sam_stats['sams_with_significant_jumps']:,} ({sam_stats['sams_with_significant_jumps']/sam_stats['after_rf_classification']*100:.1f}% of selected)")
    
    if sam_stats['enhancement_proxy_attempted'] > 0:
        print(f"Enhancement proxy attempted: {sam_stats['enhancement_proxy_attempted']:,}")
        print(f"Enhancement proxy successful: {sam_stats['enhancement_proxy_successful']:,} ({sam_stats['enhancement_proxy_successful']/sam_stats['enhancement_proxy_attempted']*100:.1f}% of attempted)")
    
    print(f"Final corrected SAMs: {sam_stats['final_corrected_sams']:,} ({sam_stats['final_corrected_sams']/sam_stats['total_sams_processed']*100:.1f}% of total)")
    
    print(f"\nMAJOR BOTTLENECKS:")
    bottlenecks = sam_stats['major_bottlenecks']
    print(f"- RF Classification: {bottlenecks['rf_classification_filtered_out']:,} SAMs filtered out ({bottlenecks['rf_classification_percent_filtered']:.1f}%)")
    
    # Additional bottleneck analysis
    if 'after_swath_size_filtering' in sam_stats and sam_stats['after_rf_classification'] > 0:
        swath_size_filtered = sam_stats['after_rf_classification'] - sam_stats['after_swath_size_filtering'] 
        print(f"- Swath size requirement: {swath_size_filtered:,} SAMs filtered out ({swath_size_filtered/sam_stats['after_rf_classification']*100:.1f}% of selected)")
    
    if 'after_proximity_checks' in sam_stats and 'after_swath_size_filtering' in sam_stats:
        proximity_filtered = sam_stats['after_swath_size_filtering'] - sam_stats['after_proximity_checks']
        print(f"- Proximity checks: {proximity_filtered:,} SAMs filtered out ({proximity_filtered/sam_stats['after_rf_classification']*100:.1f}% of selected)")
    
    print(f"- Final correction rate: {bottlenecks['final_correction_rate']:.1f}% of all SAMs")
    print("="*50)

def main():
    # Print configuration summary
    print("="*60)
    print("PROCESSING SWATH BIAS CORRECTION")
    print("="*50)
    config.print_config_summary()
    print()
    
    # Initialize global SAM processing statistics
    global_sam_stats = {
        'total_sams_processed': 0,
        'after_rf_classification': 0,
        'after_swath_size_filtering': 0,
        'after_proximity_checks': 0,
        'enhancement_proxy_attempted': 0,
        'enhancement_proxy_successful': 0,
        'final_corrected_sams': 0,
        'major_bottlenecks': {}
    }
    
    # Define input and output directories using config
    lite_file_base_dir = config.LITE_FILES_DIR
    output_base_dir = config.OUTPUT_FULL_DIR
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Temporary directory for check_drivers artifacts (if any, though not expected for retrain=False)
    project_tmp_dir_for_cd = os.path.join(config.project_root, 'tmp', f"{config.EXPERIMENT_NAME}_apply_mode")
    os.makedirs(project_tmp_dir_for_cd, exist_ok=True)

    # Load the trained RF model using config
    rf_model_path = config.get_model_path()
    print(f"Loading RF model from: {rf_model_path}")
    print(f"Expected model training run: {config.EXPERIMENT_NAME}")
    print(f"Expected model folder: final_model_all_data")
    
    if not os.path.exists(rf_model_path):
        print(f"Error: RF Model not found at {rf_model_path}")
        print(f"Make sure you have run the model training with Swath_BC_v3.py first.")
        print(f"Expected directory structure:")
        print(f"  {config.MODEL_EXPERIMENT_DIR}")
        return
        
    rf_model = joblib.load(rf_model_path)
    print(f"Successfully loaded RF model")
    
    actual_model_features = list(getattr(rf_model, 'feature_names_in_', []))
    if not actual_model_features:
        print("Error: Loaded RF model does not have feature_names_in_. Cannot proceed.")
        return
        
    # Check for the threshold attribute explicitly
    if hasattr(rf_model, 'rf_prediction_threshold_'):
        loaded_rf_threshold = rf_model.rf_prediction_threshold_
        print(f"Using RF prediction threshold: {loaded_rf_threshold}")
    else:
        print("Error: Loaded RF model does not have 'rf_prediction_threshold_'. Cannot proceed.")
        print("This attribute should have been saved with the model during training.")
        sys.exit(1)

    print(f"Model expects {len(actual_model_features)} features: {actual_model_features}")


    lite_files_pattern = config.get_lite_files_pattern()


    lite_files_to_process = sorted(glob.glob(lite_files_pattern))

    if DEBUG:
        lite_files_to_process = lite_files_to_process[:10]  # Process only 10 files for testing
        print(f"DEBUG MODE: Processing only {len(lite_files_to_process)} files for testing.")
    else:
        print(f"Found {len(lite_files_to_process)} Lite files to process.")


    for nc_file_path in tqdm(lite_files_to_process, desc="Processing Lite files"):
        print(f"\n--- Processing file: {os.path.basename(nc_file_path)} ---")

        df_lite_full = read_oco_netcdf_to_df(nc_file_path)
        if df_lite_full.empty:
            print(f"No data loaded or error reading {nc_file_path}, skipping.")
            continue
        
        # Prepare the final DataFrame structure for NetCDF output, initialized for all soundings
        df_results_for_nc = df_lite_full[['sounding_id', 'xco2']].copy()
        df_results_for_nc.rename(columns={'xco2': 'xco2_swath_bc'}, inplace=True) # Initially, corrected is same as original
        df_results_for_nc['swath_bias_corrected'] = 0 # Default: no change (0)
        
        processed_any_sams_in_file = False

        # Filter for SAM operation mode (model was trained on SAM data)
        # operation_mode: 0=Nadir, 1=Glint, 2=Target, 3=Transition, 4=SAM
        if 'operation_mode' not in df_lite_full.columns:
            print(f"Warning: 'operation_mode' not in {nc_file_path}. Cannot filter for SAMs. Skipping BC for this file.")
            update_netcdf_with_bc(nc_file_path, output_base_dir, df_results_for_nc, processed_any_sams_in_file)
            continue
            
        df_sam_data = df_lite_full[df_lite_full['operation_mode'] == 4].copy()
        
        # Log total SAMs from this file
        if not df_sam_data.empty:
            df_sam_data['orbit_str'] = df_sam_data['orbit'].astype(int).astype(str)
            df_sam_data['SAM'] = df_sam_data['target_id'].fillna('none') + '_' + df_sam_data['orbit_str']
            unique_sams_in_file = df_sam_data['SAM'].nunique()
            global_sam_stats['total_sams_processed'] += unique_sams_in_file

        if df_sam_data.empty:
            print(f"No SAM data (operation_mode == 4) in {os.path.basename(nc_file_path)}.")
        else:
            # Create 'SAM' identifier
            if 'orbit' not in df_sam_data.columns or 'target_id' not in df_sam_data.columns:
                print(f"Warning: 'orbit' or 'target_id' missing in SAM data for {os.path.basename(nc_file_path)}. Cannot create SAM ID.")
            else:
                df_sam_data['orbit_str'] = df_sam_data['orbit'].astype(int).astype(str)
                df_sam_data['SAM'] = df_sam_data['target_id'].fillna('none') + '_' + df_sam_data['orbit_str']
                # Ensure 'none' target_ids result in SAMs like 'none_orbit'
                df_sam_data.loc[df_sam_data['target_id'].fillna('none') == 'none', 'SAM'] = 'none_' + df_sam_data['orbit_str']
                df_sam_data['SAM'] = df_sam_data['SAM'].astype(str)

                # --- Apply Bias Correction ---
                # Target for swath correction is 'xco2', results stored in 'xco2_swath-BC' then 'xco2_swath_bc'
                df_sam_data['xco2_swath-BC'] = df_sam_data['xco2'].copy()

                if 'pma_elevation_angle' not in df_sam_data.columns:
                    print(f"Critical: 'pma_elevation_angle' missing in SAM data for {os.path.basename(nc_file_path)}. Skipping Swath BC.")
                else:
                    
                    print("Applying RF model to identify SAMs needing correction...")
                    sams_for_rf = [s for s in df_sam_data['SAM'].unique().tolist() if s and 'none' not in s.lower()]
                    
                    if not sams_for_rf:
                        print("No valid SAMs for RF model input.")
                        df_final_corrected_sams = df_sam_data.copy()
                        df_final_corrected_sams['xco2_swath-BC'] = df_final_corrected_sams['xco2'].copy()
                    else:
                        # Build feature matrix for the exact features the model expects
                        from ..modeling.Swath_BC_v3 import extract_jump_features_for_all_sams
                        
                        print(f"Model expects these {len(actual_model_features)} features: {actual_model_features}")
                        
                        # Identify which features are jump features vs traditional features
                        jump_features = [f for f in actual_model_features if 'jump' in f.lower()]
                        traditional_features = [f for f in actual_model_features if f not in jump_features]
                        
                        print(f"Jump features to calculate: {jump_features}")
                        print(f"Traditional features to extract: {traditional_features}")
                        
                        # Calculate jump features if needed
                        if jump_features:
                            jump_features_df = extract_jump_features_for_all_sams(df_sam_data, var='xco2')
                            # Check that ALL required jump features are available
                            missing_jump_features = [f for f in jump_features if f not in jump_features_df.columns]
                            if missing_jump_features:
                                print(f"FATAL ERROR: Required jump features missing from calculation: {missing_jump_features}")
                                print(f"Model expects: {jump_features}")
                                print(f"Available from calculation: {list(jump_features_df.columns)}")
                                print("Cannot proceed with bias correction. This indicates a code error.")
                                sys.exit(1)
                            X_jump = jump_features_df[jump_features]
                        else:
                            X_jump = pd.DataFrame(index=df_sam_data['SAM'].unique())
                        
                        # Extract traditional features directly from data
                        if traditional_features:
                            missing_traditional_features = [f for f in traditional_features if f not in df_sam_data.columns]
                            if missing_traditional_features:
                                print(f"FATAL ERROR: Required traditional features missing from input data: {missing_traditional_features}")
                                print(f"Model expects: {traditional_features}")
                                print(f"Available in data: {[f for f in traditional_features if f in df_sam_data.columns]}")
                                print("Cannot proceed with bias correction. Check input data completeness.")
                                sys.exit(1)
                            
                            traditional_data = df_sam_data[traditional_features + ['SAM']].dropna(subset=traditional_features)
                            if traditional_data.empty:
                                print("FATAL ERROR: No data remaining after dropping NaN traditional features.")
                                print("This indicates data quality issues that prevent safe bias correction.")
                                sys.exit(1)
                            
                            X_traditional = traditional_data.groupby('SAM')[traditional_features].mean()
                        else:
                            X_traditional = pd.DataFrame(index=df_sam_data['SAM'].unique())
                        
                        # Combine all features
                        X_combined = X_traditional.merge(X_jump, left_index=True, right_index=True, how='left')
                        
                        # Verify all required features are present in the combined matrix
                        missing_final_features = [f for f in actual_model_features if f not in X_combined.columns]
                        if missing_final_features:
                            print(f"FATAL ERROR: Required features missing from final feature matrix: {missing_final_features}")
                            print(f"Model expects: {actual_model_features}")
                            print(f"Available in combined matrix: {list(X_combined.columns)}")
                            print("This indicates a bug in feature extraction logic.")
                            sys.exit(1)
                        
                        # Check for any remaining NaN values that would cause prediction failure
                        X_final = X_combined[actual_model_features]
                        if X_final.isnull().any().any():
                            print("FATAL ERROR: NaN values detected in final feature matrix.")
                            print("Features with NaN values:")
                            for col in X_final.columns:
                                if X_final[col].isnull().any():
                                    print(f"  {col}: {X_final[col].isnull().sum()} NaN values")
                            print("Cannot proceed with model prediction.")
                            sys.exit(1)
                        probas = rf_model.predict_proba(X_final)[:, 1]
                        y_pred = (probas >= loaded_rf_threshold).astype(int)
                        
                        swath_bias_pred_dict = dict(zip(X_final.index, y_pred))
                        
                        # Apply corrections ONLY to SAMs that RF identified as needing correction
                        sams_needing_correction = [sam for sam, pred in swath_bias_pred_dict.items() if pred]
                        print(f"RF identified {len(sams_needing_correction)} SAMs for correction out of {len(sams_for_rf)} total SAMs")
                        
                        # Log RF classification results
                        global_sam_stats['after_rf_classification'] += len(sams_needing_correction)
                        
                        # Initialize result dataframe
                        df_final_corrected_sams = df_sam_data.copy()
                        df_final_corrected_sams['xco2_swath-BC'] = df_final_corrected_sams['xco2'].copy()
                        
                        if sams_needing_correction:
                            print("Applying swath bias correction to identified SAMs...")
                            # Apply corrections only to identified SAMs
                            data_subset = df_final_corrected_sams[df_final_corrected_sams['SAM'].isin(sams_needing_correction)].copy()
                            data_subset = correct_swath_bias(
                                data=data_subset,
                                var='xco2',  # var the correction is applied to.
                                swath_grouping_threshold_angle=1.0,
                                jump_significance_threshold_value=0.6,
                                min_soundings_for_median=50,
                                log_stats=global_sam_stats
                            )
                            
                            # Update main dataframe with corrections - ensure proper index alignment
                            # Use .update() to avoid index alignment issues or explicitly set by sounding_id
                            df_final_corrected_sams = df_final_corrected_sams.set_index('sounding_id')
                            data_subset_indexed = data_subset.set_index('sounding_id')
                            df_final_corrected_sams.loc[data_subset_indexed.index, 'xco2_swath-BC'] = data_subset_indexed['xco2_swath-BC']
                            df_final_corrected_sams = df_final_corrected_sams.reset_index()
                            
                            # Log actual corrections applied (count SAMs that were actually corrected)
                            # Create sam_mask after index operations to avoid alignment issues
                            sam_mask = df_final_corrected_sams['SAM'].isin(sams_needing_correction)
                            actually_corrected = df_final_corrected_sams.loc[sam_mask]
                            orig_xco2 = actually_corrected['xco2']
                            corrected_xco2 = actually_corrected['xco2_swath-BC']
                            sams_with_changes = actually_corrected[orig_xco2 != corrected_xco2]['SAM'].nunique()
                            global_sam_stats['final_corrected_sams'] += sams_with_changes
                        else:
                            print("No SAMs identified for correction by RF.")
                    
                    processed_any_sams_in_file = True
                    
                    # Merge results from processed SAMs back into the main result DataFrame
                    update_cols = ['xco2_swath-BC']
                    
                    df_update_payload = df_final_corrected_sams[['sounding_id'] + update_cols].copy()
                    df_update_payload.rename(columns={'xco2_swath-BC': 'xco2_swath_bc'}, inplace=True)

                    # Update df_results_for_nc based on df_update_payload (SAM data)
                    # Set index to sounding_id for faster update
                    df_results_for_nc = df_results_for_nc.set_index('sounding_id')
                    df_update_payload = df_update_payload.set_index('sounding_id')
                    
                    for col_to_update in df_update_payload.columns:
                         df_results_for_nc.loc[df_update_payload.index, col_to_update] = df_update_payload[col_to_update]
                    
                    df_results_for_nc = df_results_for_nc.reset_index()


                    # Add 'swath_bias_corrected' flag 0: no change, 1: correction applied. 
                    df_results_for_nc.loc[df_results_for_nc['sounding_id'].isin(df_update_payload.index), 'swath_bias_corrected'] = 0 # default: no change
                    # If xco2_swath_bc is different from original xco2 (from df_lite_full)
                    # temp merge original xco2 to compare
                    orig_xco2_map = df_lite_full.set_index('sounding_id')['xco2']
                    df_results_for_nc = df_results_for_nc.set_index('sounding_id')
                    df_results_for_nc['xco2_original_temp'] = orig_xco2_map
                    
                    corrected_mask = (df_results_for_nc['xco2_swath_bc'] != df_results_for_nc['xco2_original_temp']) & df_results_for_nc.index.isin(df_update_payload.index)
                    df_results_for_nc.loc[corrected_mask, 'swath_bias_corrected'] = 1 # correction applied

                    df_results_for_nc = df_results_for_nc.reset_index().drop(columns=['xco2_original_temp'])


        # Ensure all required columns for NetCDF are present and filled
        final_nc_cols = ['sounding_id', 'xco2_swath_bc', 'swath_bias_corrected']
        for col in final_nc_cols:
            if col not in df_results_for_nc.columns:
                 # This indicates a logic error if a primary output column is missing
                 print(f"FATAL ERROR: Column {col} missing before NetCDF write for {os.path.basename(nc_file_path)}")
                 # Create with default if absolutely necessary, but should not happen
                 if col == 'xco2_swath_bc': df_results_for_nc[col] = df_lite_full['xco2']
                 elif col == 'swath_bias_corrected': df_results_for_nc[col] = 0
        
        # Sort by sounding_id one last time before writing
        df_results_for_nc = df_results_for_nc.sort_values('sounding_id').reset_index(drop=True)
        
        update_netcdf_with_bc(nc_file_path, output_base_dir, df_results_for_nc, processed_any_sams_in_file)

    # Calculate major bottlenecks and print summary
    if global_sam_stats['total_sams_processed'] > 0:
        global_sam_stats['major_bottlenecks'] = {
            'rf_classification_filtered_out': global_sam_stats['total_sams_processed'] - global_sam_stats['after_rf_classification'],
            'rf_classification_percent_filtered': ((global_sam_stats['total_sams_processed'] - global_sam_stats['after_rf_classification']) / global_sam_stats['total_sams_processed']) * 100,
            'final_correction_rate': (global_sam_stats['final_corrected_sams'] / global_sam_stats['total_sams_processed']) * 100
        }
    
    print_sam_processing_summary(global_sam_stats)
    print("\n--- All files processed ---")

if __name__ == '__main__':
    main() 