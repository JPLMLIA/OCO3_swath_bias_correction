#!/usr/bin/env python3
"""
Visualize Swath Bias Correction Results
Script to create a plot of each SAM before and after correction. 
For the whole dataset this runs for 12h and makes 20k plots!
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import netCDF4 as nc
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools
import argparse

from ..utils.main_util import plot_SAM, read_oco_netcdf_to_df
from src.utils.config_paths import PathConfig

# --- Configuration ---
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# --- Debug Configuration ---
DEBUG = False  # Set to True to process only 100th of the files for testing

# --- Optional SAM ID Filter ---
# If this list is not empty, only SAMs with IDs in this list will be processed and plotted.
# Example: TARGET_SAM_IDS_TO_PLOT = ['fossil0006_21831', 'anotherSAM_12345']
TARGET_SAM_IDS_TO_PLOT = []#['fossil0005_20285']#['fossil0006_21831'] # Empty means process all SAMs

def filter_and_prepare_sam_data(df):
    """
    Filter for SAM data and create SAM identifiers.
    """
    if 'operation_mode' not in df.columns:
        print("Warning: 'operation_mode' not found. Cannot filter for SAMs.")
        return pd.DataFrame()
    
    # Filter for SAM operation mode (4=SAM)
    sam_data = df[df['operation_mode'] == 4].copy()
    
    if sam_data.empty:
        print("No SAM data found.")
        return sam_data
    
    # Create SAM identifier
    if 'orbit' in sam_data.columns and 'target_id' in sam_data.columns:
        sam_data['orbit_str'] = sam_data['orbit'].astype(int).astype(str)
        sam_data['SAM'] = sam_data['target_id'].fillna('none') + '_' + sam_data['orbit_str']
        sam_data['SAM'] = sam_data['SAM'].astype(str)
    else:
        print("Warning: Cannot create SAM identifier - missing orbit or target_id")
        return pd.DataFrame()
    
    return sam_data


def plot_sam_worker(sam_data_tuple, config, sam_plots_dir, qf0_only=False, qf_col='xco2_quality_flag'):
    """
    Worker function for parallel SAM plotting.
    
    Args:
        sam_data_tuple: Tuple of (sam_id, sam_df)
        config: PathConfig object
        sam_plots_dir: Directory to save plots
    
    Returns:
        int: 1 if SAM was plotted successfully, 0 otherwise
    """
    sam_id, sam_df = sam_data_tuple
    
    try:
        # Optionally restrict to QF==0 retrievals
        df_for_plot = sam_df
        if qf0_only and (qf_col in sam_df.columns):
            df_for_plot = sam_df[sam_df[qf_col] == 0]
        
        # Skip SAMs with too few soundings (after optional filtering)
        if len(df_for_plot) < 50:
            return 0
        
        # Calculate colorbar limits to be consistent between plots
        xco2_values = df_for_plot['xco2'].dropna()
        if len(xco2_values) == 0:
            return 0 # Skip if no valid xco2 data for colorbar
            
        vmin = np.round(np.nanpercentile(xco2_values, 10), 0)
        vmax = vmin + 5
        
        title_add_orig = 'Original XCO2'

        # Plot original XCO2
        plot_SAM(df_for_plot, 'xco2', vmin=vmin, vmax=vmax, 
                 save_fig=True, name=config.PROCESSING_VERSION, path=sam_plots_dir, qf=None,
                 title_addition=title_add_orig)
        
        # Plot bias-corrected XCO2 (only if different from original)
        if 'xco2_swath_bc' in df_for_plot.columns and not np.allclose(df_for_plot['xco2'].fillna(0), df_for_plot['xco2_swath_bc'].fillna(0), equal_nan=True):
            title_add_bc = 'Swath Bias Corrected XCO2'

            plot_SAM(df_for_plot, 'xco2_swath_bc', vmin=vmin, vmax=vmax,
                     save_fig=True, name=config.PROCESSING_VERSION, path=sam_plots_dir, qf=None,
                     title_addition=title_add_bc)
        
        return 1
    
    except Exception as e:
        print(f"Error plotting SAM {sam_id}: {e}")
        return 0


def main():
    """Main function"""
    print("Visualizing Swath Bias Correction Results...")
    print("=" * 50)
    
    # CLI options
    parser = argparse.ArgumentParser(description='Visualize Swath Bias Correction Results')
    parser.add_argument('--qf0', action='store_true', help='Only plot and compute metrics for retrievals with xco2_quality_flag == 0')
    args = parser.parse_args()

    # Initialize config
    config = PathConfig()
    
    # Use config for paths
    processed_data_dir = config.OUTPUT_FULL_DIR
    output_dir = config.FIGURES_DIR / "swath_bc_results"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output directories
    sub_folder_name = 'sam_plots_qf0' if args.qf0 else 'sam_plots'
    sam_plots_dir = os.path.join(output_dir, sub_folder_name)
    os.makedirs(sam_plots_dir, exist_ok=True)
    
    # Find processed files
    processed_files = sorted(glob.glob(os.path.join(processed_data_dir, '*_SwathBC.nc4')))
    
    if not processed_files:
        print(f"No processed files found in {processed_data_dir}")
        return
    
    if DEBUG:
        processed_files = processed_files[::100]
        print(f"DEBUG MODE: Processing only {len(processed_files)} files for testing.")
    else:
        print(f"Found {len(processed_files)} processed files.")

    if TARGET_SAM_IDS_TO_PLOT:
        print(f"TARGETING specific SAM IDs for plotting: {TARGET_SAM_IDS_TO_PLOT}")
    
    # Process each file
    plotted_sam_count = 0
    
    # Collect all SAM data for parallel processing
    all_sam_data = []

    for nc_file in tqdm(processed_files, desc="Processing files"):
        # Define variables to read from the processed NetCDF files
        required_vars_for_processed_file = [
            'sounding_id', 'latitude', 'longitude', 'operation_mode', 'orbit', 'target_id',
            'xco2', 
            'xco2_swath_bc',       
            'swath_bias_corrected',
            'vertex_latitude', 'vertex_longitude', 
            'windspeed_u_met', 'windspeed_v_met',
            'xco2_quality_flag'
        ]
        
        # Read the processed NetCDF file using the new utility function
        df = read_oco_netcdf_to_df(nc_file, variables_to_read=required_vars_for_processed_file)

        if df.empty:
            continue
        
        # Filter and prepare SAM data
        sam_data = filter_and_prepare_sam_data(df)
        if sam_data.empty:
            continue
        
        # Group by SAM and collect data for parallel processing
        sam_groups = sam_data.groupby('SAM')
        
        for sam_id, sam_df in sam_groups:
            if len(sam_df) < 50:  # Skip SAMs with too few soundings
                continue

            # Optional SAM ID filtering
            if TARGET_SAM_IDS_TO_PLOT and sam_id not in TARGET_SAM_IDS_TO_PLOT:
                continue # Skip this SAM if it's not in our target list
            
            # Add to collection for parallel processing
            all_sam_data.append((sam_id, sam_df.copy()))
    
    # Parallel processing of all SAMs
    if all_sam_data:
        print(f"\nProcessing {len(all_sam_data)} SAMs in parallel...")
        
        # Determine number of processes to use
        n_processes = min(cpu_count()//2, len(all_sam_data))
        if DEBUG:
            n_processes = min(2, n_processes)  # Limit processes in debug mode
        
        print(f"Using {n_processes} processes")
        
        # Create partial function with fixed arguments
        plot_func = functools.partial(plot_sam_worker, 
                                    config=config, 
                                    sam_plots_dir=sam_plots_dir,
                                    qf0_only=args.qf0)
        
        # Process SAMs in parallel
        with Pool(processes=n_processes) as pool:
            results = list(tqdm(pool.imap(plot_func, all_sam_data), 
                              total=len(all_sam_data), 
                              desc="Plotting SAMs"))
        
        plotted_sam_count = sum(results)

    print(f"\nProcessed and plotted {plotted_sam_count} SAMs.")
    
    if plotted_sam_count == 0:
        if TARGET_SAM_IDS_TO_PLOT:
            print(f"No targeted SAMs ({TARGET_SAM_IDS_TO_PLOT}) were found for visualization.")
        else:
            print("No SAMs were found or plotted.")
        return
    
    print(f"\nVisualization complete!")
    print(f"SAM plots saved to: {sam_plots_dir}")


if __name__ == '__main__':
    main() 