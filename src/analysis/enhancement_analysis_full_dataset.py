#!/usr/bin/env python3
"""
Enhancement Analysis for Full OCO-3 Corrected Dataset

This script analyzes the impact of swath bias correction on emission enhancement proxies
across the complete OCO-3 corrected SAM dataset (not just the labeled subset).

Unlike evaluation_analysis_plots.py which focuses on labeled data for model evaluation,
this script processes ALL corrected SAMs to understand the real-world impact of 
the bias correction on emission monitoring applications.
"""


import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation, theilslopes
from typing import Tuple, List, Optional
import glob
from tqdm import tqdm
from pathlib import Path

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.utils.main_util import read_oco_netcdf_to_df, SAM_enhancement, get_target_data
from src.utils.config_paths import PathConfig

def load_corrected_sams_data(config, max_files=None):
    """
    Load data for all SAMs that had corrections applied from the full processed dataset.
    """
    print("Loading full corrected dataset...")
    
    processed_files = sorted(glob.glob(os.path.join(config.OUTPUT_FULL_DIR, '*_SwathBC.nc4')))
    
    if not processed_files:
        print(f"No processed files found in {config.OUTPUT_FULL_DIR}")
        print("Please run the bias correction processing first: python -m src.processing.apply_swath_bc_RF")
        return pd.DataFrame()
    
    if max_files:
        processed_files = processed_files[:max_files]
        print(f"DEBUG: Processing only {len(processed_files)} files")
    
    print(f"Scanning {len(processed_files)} files for corrected SAMs...")
    
    all_corrected_data = []
    
    for nc_file in tqdm(processed_files, desc="Loading corrected SAMs"):
        try:
            required_vars = [
                'sounding_id', 'latitude', 'longitude', 'operation_mode', 'orbit', 'target_id',
                'xco2', 'xco2_swath_bc', 'swath_bias_corrected',
                'vertex_latitude', 'vertex_longitude', 
                'windspeed_u_met', 'windspeed_v_met',
                'xco2_quality_flag'
            ]
            
            df = read_oco_netcdf_to_df(nc_file, variables_to_read=required_vars)
            
            if df.empty:
                continue
            
            # Filter for SAM data only (operation_mode == 4)
            sam_data = df[df['operation_mode'] == 4].copy()
            if sam_data.empty:
                continue
            
            # Create SAM identifier
            sam_data['orbit_str'] = sam_data['orbit'].astype(int).astype(str)
            sam_data['SAM'] = sam_data['target_id'].fillna('none') + '_' + sam_data['orbit_str']
            
            # Filter out 'none' target SAMs (same as processing script)
            sam_data = sam_data[~sam_data['SAM'].str.contains('none', case=False, na=False)]
            if sam_data.empty:
                continue
            
            # Only keep SAMs that actually had corrections applied
            corrected_sam_data = sam_data[sam_data['swath_bias_corrected'] == 1]
            
            if not corrected_sam_data.empty:
                all_corrected_data.append(corrected_sam_data)
                
        except Exception as e:
            print(f"Warning: Error processing {os.path.basename(nc_file)}: {e}")
            continue
    
    if all_corrected_data:
        all_data = pd.concat(all_corrected_data, ignore_index=True)
        print(f"Loaded {len(all_data)} soundings from {all_data['SAM'].nunique()} corrected SAMs")
        return all_data
    else:
        print("No corrected SAMs found in the dataset!")
        return pd.DataFrame()

def calculate_enhancements_for_sams(data, qf_column=None, wind_speed_threshold=None):
    """
    Calculate emission enhancement proxies for all corrected SAMs.
    """
    print("Calculating enhancement proxies...")
    
    enhancements = []
    
    # Group by SAM and calculate enhancements
    for sam_id, sam_df in tqdm(data.groupby('SAM'), desc="Processing SAMs"):
        if len(sam_df) < 100:  # Skip SAMs with too few soundings for reliable enhancement calculation
            continue
        
        try:
            # Calculate enhancement for original and bias-corrected data
            qf_arg = qf_column if qf_column in sam_df.columns else None
            enhancement_orig = SAM_enhancement(sam_df, 'xco2', qf=qf_arg, custom_SAM=True)
            enhancement_bc = SAM_enhancement(sam_df, 'xco2_swath_bc', qf=qf_arg, custom_SAM=True)
            
            # Skip if either calculation failed
            if np.isnan(enhancement_orig) or np.isnan(enhancement_bc):
                continue
                
            # Calculate correction statistics
            correction_magnitude = np.abs(sam_df['xco2_swath_bc'] - sam_df['xco2']).mean()
            max_correction = np.abs(sam_df['xco2_swath_bc'] - sam_df['xco2']).max()
            # Calculate mean wind speed per SAM (m/s)
            wind_speed_mean = np.sqrt(sam_df['windspeed_u_met']**2 + sam_df['windspeed_v_met']**2).mean()

            # Skip if wind speed is below threshold
            if wind_speed_threshold is not None and wind_speed_mean < wind_speed_threshold:
                continue
            
            enhancements.append({
                'SAM': sam_id,
                'target_id': sam_df['target_id'].iloc[0],
                'latitude': sam_df['latitude'].mean(),
                'longitude': sam_df['longitude'].mean(),
                'n_soundings': len(sam_df),
                'enhancement_original': enhancement_orig,
                'enhancement_bc': enhancement_bc,
                'enhancement_change': enhancement_bc - enhancement_orig,
                'enhancement_abs_change': abs(enhancement_bc - enhancement_orig),
                'correction_magnitude_mean': correction_magnitude,
                'correction_magnitude_max': max_correction,
                'wind_speed_mean': wind_speed_mean
            })
            
        except Exception as e:
            print(f"Warning: Could not calculate enhancement for SAM {sam_id}: {e}")
            continue
    
    if enhancements:
        enhancements_df = pd.DataFrame(enhancements)
        print(f"Successfully calculated enhancements for {len(enhancements_df)} corrected SAMs")
        return enhancements_df
    else:
        print("No enhancement calculations could be completed!")
        return pd.DataFrame()

def analyze_enhancement_impacts(enhancements_df, output_dir):
    """
    Analyze and visualize the impact of bias correction on enhancement proxies.
    """
    print("Analyzing enhancement impacts...")
    
    # Remove extreme outliers (top/bottom 1% to avoid skewing statistics)
    for col in ['enhancement_original', 'enhancement_bc']:
        q01 = enhancements_df[col].quantile(0.01)
        q99 = enhancements_df[col].quantile(0.99)
        enhancements_df = enhancements_df[(enhancements_df[col] >= q01) & (enhancements_df[col] <= q99)]
    
    if len(enhancements_df) < 5:
        print("Not enough valid enhancements after outlier removal.")
        return
    
    print(f"Analyzing {len(enhancements_df)} corrected SAMs after outlier removal...")
    
    # Calculate key statistics
    stats = {
        'n_corrected_sams': len(enhancements_df),
        'mean_original': enhancements_df['enhancement_original'].mean(),
        'std_original': enhancements_df['enhancement_original'].std(),
        'mean_bc': enhancements_df['enhancement_bc'].mean(),
        'std_bc': enhancements_df['enhancement_bc'].std(),
        'correlation': enhancements_df['enhancement_original'].corr(enhancements_df['enhancement_bc']),
        'mean_abs_change': enhancements_df['enhancement_abs_change'].mean(),
        'median_abs_change': enhancements_df['enhancement_abs_change'].median()
    }
    
    # Calculate regression statistics
    m, b = np.polyfit(enhancements_df['enhancement_original'], enhancements_df['enhancement_bc'], 1)
    r_squared = stats['correlation'] ** 2
    
    stats['regression_slope'] = m
    stats['regression_intercept'] = b
    stats['r_squared'] = r_squared
    
    # Print key results
    print(f"\n" + "="*60)
    print("ENHANCEMENT ANALYSIS RESULTS - CORRECTED SAMs ONLY")
    print("="*60)
    print(f"Number of corrected SAMs analyzed: {stats['n_corrected_sams']}")
    print(f"Original enhancement - Mean: {stats['mean_original']:.3f}, Std: {stats['std_original']:.3f} ppm m/s")
    print(f"Bias-corrected enhancement - Mean: {stats['mean_bc']:.3f}, Std: {stats['std_bc']:.3f} ppm m/s")
    print(f"Correlation: {stats['correlation']:.3f}")
    print(f"Regression: y = {m:.3f}x + {b:.3f} (R² = {r_squared:.3f})")
    print(f"Mean absolute enhancement change: {stats['mean_abs_change']:.3f} ppm m/s")
    print(f"Median absolute enhancement change: {stats['median_abs_change']:.3f} ppm m/s")
    
    # Create scatter plot
    plt.figure(figsize=(4.5, 4.5)) 
    
    # Plot corrected SAMs with smaller dots
    plt.scatter(enhancements_df['enhancement_original'], enhancements_df['enhancement_bc'],
               alpha=0.5, s=10, color='red', label=f'Corrected SAMs (n={len(enhancements_df)})', marker='.')
    
    # Add 1:1 line
    plot_range = [min(enhancements_df['enhancement_original'].min(), enhancements_df['enhancement_bc'].min()),
                  max(enhancements_df['enhancement_original'].max(), enhancements_df['enhancement_bc'].max())]
    plt.plot(plot_range, plot_range, 'k--', alpha=0.5, linewidth=1, label='1:1 Line')
    
    # Add regression line
    x_line = np.linspace(plot_range[0], plot_range[1], 100)
    y_line = m * x_line + b
    plt.plot(x_line, y_line, 'b-', alpha=0.7, linewidth=2, 
             label=f'Fit: y = {m:.2f}x + {b:.2f} (R² = {r_squared:.2f})')
    
    plt.xlabel('Emission Proxy Original [ppm m/s]')
    plt.ylabel('Emission Proxy Bias Corrected [ppm m/s]')
    # Removed title as requested
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'enhancement_full_dataset_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create histogram of enhancement changes
    plt.figure(figsize=(8, 6))
    plt.hist(enhancements_df['enhancement_change'], bins=40, alpha=0.7, color='purple', 
             label=f'Mean = {enhancements_df["enhancement_change"].mean():.3f} ppm m/s')
    plt.axvline(enhancements_df['enhancement_change'].mean(), color='black', linestyle='--', 
                label=f'Mean = {enhancements_df["enhancement_change"].mean():.3f} ppm m/s')
    plt.axvline(0, color='red', linestyle='-', alpha=0.5, label='No Change')
    plt.xlabel('Emission Proxy Change (Corrected - Original) [ppm m/s]')
    plt.ylabel('Count')
    plt.title('Distribution of Emission Proxy Changes\n(Full Corrected Dataset)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'enhancement_change_histogram_full_dataset.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed statistics
    stats_df = pd.DataFrame([stats])
    
    # Add documentation
    documentation = [
        "# Enhancement Analysis Statistics - Full Corrected Dataset",
        "# This analysis includes ALL corrected SAMs from the complete OCO-3 data record",
        "# (not just the labeled subset used for model training/evaluation)",
        "#",
        "# n_corrected_sams: Total number of corrected SAMs with valid enhancement calculations",
        "# mean_original, std_original: Statistics for original enhancement proxy [ppm m/s]",
        "# mean_bc, std_bc: Statistics for bias-corrected enhancement proxy [ppm m/s]", 
        "# correlation: Correlation between original and corrected enhancement values",
        "# regression_slope, regression_intercept: Linear fit parameters",
        "# r_squared: R² value for the linear fit",
        "# mean_abs_change: Mean absolute change in enhancement proxy [ppm m/s]",
        "# median_abs_change: Median absolute change in enhancement proxy [ppm m/s]",
        ""
    ]
    
    # Write documentation and data
    csv_path = os.path.join(output_dir, 'enhancement_full_dataset_stats.csv')
    with open(csv_path, 'w') as f:
        for line in documentation:
            f.write(line + '\n')
        stats_df.to_csv(f, index=False)
    
    # Save detailed SAM-level results
    enhancements_df.to_csv(os.path.join(output_dir, 'enhancement_full_dataset_sam_details.csv'), index=False)
    
    print(f"\nResults saved to:")
    print(f"  Statistics: {csv_path}")
    print(f"  SAM details: enhancement_full_dataset_sam_details.csv")
    print(f"  Plots: enhancement_full_dataset_analysis.png, enhancement_change_histogram_full_dataset.png")
    
    return stats

def analyze_enhancement_impacts_positive_wind(enhancements_df, output_dir):
    """
    Analyze and visualize the impact of bias correction on enhancement proxies
    for cases with enhancement > 0 and mean wind speed > 2 m/s.
    Also adjusts plots to force axes to start at 0.
    """
    print("Analyzing enhancement impacts for enhancement>0 and wind>2 m/s...")

    # Ensure required columns exist
    required_cols = ['enhancement_original', 'enhancement_bc', 'enhancement_change', 'enhancement_abs_change', 'wind_speed_mean']
    for col in required_cols:
        if col not in enhancements_df.columns:
            print(f"Missing required column: {col}. Cannot perform filtered analysis.")
            return

    # Apply filters
    df = enhancements_df[(enhancements_df['enhancement_original'] > 0) &
                         (enhancements_df['enhancement_bc'] > 0) &
                         (enhancements_df['wind_speed_mean'] > 2)].copy()

    # df = enhancements_df[(enhancements_df['wind_speed_mean'] > 2)].copy()

    if len(df) < 5:
        print("Not enough valid enhancements after filtering.")
        return

    # Remove extreme outliers (top/bottom 1% to avoid skewing statistics)
    for col in ['enhancement_original', 'enhancement_bc']:
        q01 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        df = df[(df[col] >= q01) & (df[col] <= q99)]

    if len(df) < 5:
        print("Not enough valid enhancements after outlier removal.")
        return

    print(f"Analyzing {len(df)} corrected SAMs after filtering and outlier removal...")

    # Calculate key statistics
    stats = {
        'n_corrected_sams': len(df),
        'mean_original': df['enhancement_original'].mean(),
        'std_original': df['enhancement_original'].std(),
        'mean_bc': df['enhancement_bc'].mean(),
        'std_bc': df['enhancement_bc'].std(),
        'correlation': df['enhancement_original'].corr(df['enhancement_bc']),
        'mean_abs_change': df['enhancement_abs_change'].mean(),
        'median_abs_change': df['enhancement_abs_change'].median(),
        'wind_speed_mean_mean': df['wind_speed_mean'].mean()
    }

    # Calculate regression statistics
    m, b = np.polyfit(df['enhancement_original'], df['enhancement_bc'], 1)
    r_squared = stats['correlation'] ** 2

    stats['regression_slope'] = m
    stats['regression_intercept'] = b
    stats['r_squared'] = r_squared

    # Print key results
    print(f"\n" + "="*60)
    print("ENHANCEMENT ANALYSIS RESULTS - ENH>0 & WIND>2 m/s")
    print("="*60)
    print(f"Number of corrected SAMs analyzed: {stats['n_corrected_sams']}")
    print(f"Original enhancement - Mean: {stats['mean_original']:.3f}, Std: {stats['std_original']:.3f} ppm m/s")
    print(f"Bias-corrected enhancement - Mean: {stats['mean_bc']:.3f}, Std: {stats['std_bc']:.3f} ppm m/s")
    print(f"Correlation: {stats['correlation']:.3f}")
    print(f"Regression: y = {m:.3f}x + {b:.3f} (R² = {r_squared:.3f})")
    print(f"Mean absolute enhancement change: {stats['mean_abs_change']:.3f} ppm m/s")
    print(f"Median absolute enhancement change: {stats['median_abs_change']:.3f} ppm m/s")
    print(f"Mean wind speed (m/s): {stats['wind_speed_mean_mean']:.3f}")

    # Create scatter plot with axes starting at 0
    plt.figure(figsize=(4.5, 4.5))

    plt.scatter(df['enhancement_original'], df['enhancement_bc'],
               alpha=0.5, s=10, color='red', label=f'Filtered SAMs (n={len(df)})', marker='.')

    # Set axes starting at 0 with symmetric max across both axes
    max_val = max(df['enhancement_original'].max(), df['enhancement_bc'].max())
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)

    # Add 1:1 line and regression line within the same 0-based range
    x_line = np.linspace(0, max_val, 100)
    plt.plot(x_line, x_line, 'k--', alpha=0.5, linewidth=1, label='1:1 Line')
    y_line = m * x_line + b
    plt.plot(x_line, y_line, 'b-', alpha=0.7, linewidth=2,
             label=f'Fit: y = {m:.2f}x + {b:.2f} (R² = {r_squared:.2f})')

    plt.xlabel('Emission Proxy Original [ppm m/s]')
    plt.ylabel('Emission Proxy Bias Corrected [ppm m/s]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig(os.path.join(output_dir, 'enhancement_filtered_analysis_enhgt0_windgt2.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create histogram of enhancement changes
    plt.figure(figsize=(8, 6))
    plt.hist(df['enhancement_change'], bins=40, alpha=0.7, color='purple',
             label=f'Mean = {df["enhancement_change"].mean():.3f} ppm m/s')
    plt.axvline(df['enhancement_change'].mean(), color='black', linestyle='--',
                label=f'Mean = {df["enhancement_change"].mean():.3f} ppm m/s')
    plt.axvline(0, color='red', linestyle='-', alpha=0.5, label='No Change')
    plt.xlabel('Emission Proxy Change (Corrected - Original) [ppm m/s]')
    plt.ylabel('Count')
    plt.title('Distribution of Emission Proxy Changes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'enhancement_change_histogram_enhgt0_windgt2.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save detailed statistics
    stats_df = pd.DataFrame([stats])
    documentation = [
        "# Enhancement Analysis Statistics - Filtered (Enhancement>0 & Wind>2 m/s)",
        "# This analysis includes only corrected SAMs with positive enhancements and sufficient wind.",
        "# n_corrected_sams: Total number of filtered SAMs with valid enhancement calculations",
        "# mean_original, std_original: Statistics for original enhancement proxy [ppm m/s]",
        "# mean_bc, std_bc: Statistics for bias-corrected enhancement proxy [ppm m/s]",
        "# correlation: Correlation between original and corrected enhancement values",
        "# regression_slope, regression_intercept: Linear fit parameters",
        "# r_squared: R² value for the linear fit",
        "# mean_abs_change: Mean absolute change in enhancement proxy [ppm m/s]",
        "# median_abs_change: Median absolute change in enhancement proxy [ppm m/s]",
        "# wind_speed_mean_mean: Mean of mean wind speeds across SAMs [m/s]",
        ""
    ]

    csv_path = os.path.join(output_dir, 'enhancement_filtered_stats_enhgt0_windgt2.csv')
    with open(csv_path, 'w') as f:
        for line in documentation:
            f.write(line + '\n')
        stats_df.to_csv(f, index=False)

    # Save detailed SAM-level results for the filtered set
    df.to_csv(os.path.join(output_dir, 'enhancement_filtered_sam_details_enhgt0_windgt2.csv'), index=False)

    print(f"\nResults saved to:")
    print(f"  Statistics: {csv_path}")
    print(f"  SAM details: enhancement_filtered_sam_details_enhgt0_windgt2.csv")
    print(f"  Plots: enhancement_filtered_analysis_enhgt0_windgt2.png, enhancement_change_histogram_enhgt0_windgt2.png")

    return stats

def analyze_enhancement_impacts_positive_wind_qf0(enhancements_df, output_dir):
    """
    Same as analyze_enhancement_impacts_positive_wind but assumes enhancements were
    computed using only retrievals with xco2_quality_flag == 0.
    Filters to enhancement > 0 and wind > 2 m/s and starts axes at 0.
    """
    print("Analyzing enhancement impacts for enhancement>0 and wind>2 m/s with QF=0...")

    required_cols = ['enhancement_original', 'enhancement_bc', 'enhancement_change', 'enhancement_abs_change', 'wind_speed_mean']
    for col in required_cols:
        if col not in enhancements_df.columns:
            print(f"Missing required column: {col}. Cannot perform filtered analysis.")
            return

    df = enhancements_df[(enhancements_df['enhancement_original'] > 0) &
                         (enhancements_df['enhancement_bc'] > 0) &
                         (enhancements_df['wind_speed_mean'] > 2)].copy()

    if len(df) < 5:
        print("Not enough valid enhancements after filtering.")
        return

    for col in ['enhancement_original', 'enhancement_bc']:
        q01 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        df = df[(df[col] >= q01) & (df[col] <= q99)]

    if len(df) < 5:
        print("Not enough valid enhancements after outlier removal.")
        return

    print(f"Analyzing {len(df)} corrected SAMs after filtering and outlier removal...")

    stats = {
        'n_corrected_sams': len(df),
        'mean_original': df['enhancement_original'].mean(),
        'std_original': df['enhancement_original'].std(),
        'mean_bc': df['enhancement_bc'].mean(),
        'std_bc': df['enhancement_bc'].std(),
        'correlation': df['enhancement_original'].corr(df['enhancement_bc']),
        'mean_abs_change': df['enhancement_abs_change'].mean(),
        'median_abs_change': df['enhancement_abs_change'].median(),
        'wind_speed_mean_mean': df['wind_speed_mean'].mean()
    }

    m, b = np.polyfit(df['enhancement_original'], df['enhancement_bc'], 1)
    r_squared = stats['correlation'] ** 2

    stats['regression_slope'] = m
    stats['regression_intercept'] = b
    stats['r_squared'] = r_squared

    print(f"\n" + "="*60)
    print("ENHANCEMENT ANALYSIS RESULTS - ENH>0 & WIND>2 m/s (QF=0)")
    print("="*60)
    print(f"Number of corrected SAMs analyzed: {stats['n_corrected_sams']}")
    print(f"Original enhancement - Mean: {stats['mean_original']:.3f}, Std: {stats['std_original']:.3f} ppm m/s")
    print(f"Bias-corrected enhancement - Mean: {stats['mean_bc']:.3f}, Std: {stats['std_bc']:.3f} ppm m/s")
    print(f"Correlation: {stats['correlation']:.3f}")
    print(f"Regression: y = {m:.3f}x + {b:.3f} (R² = {r_squared:.3f})")
    print(f"Mean absolute enhancement change: {stats['mean_abs_change']:.3f} ppm m/s")
    print(f"Median absolute enhancement change: {stats['median_abs_change']:.3f} ppm m/s")
    print(f"Mean wind speed (m/s): {stats['wind_speed_mean_mean']:.3f}")

    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(df['enhancement_original'], df['enhancement_bc'],
               alpha=0.5, s=10, color='red', label=f'Filtered SAMs (n={len(df)})', marker='.')
    max_val = max(df['enhancement_original'].max(), df['enhancement_bc'].max())
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    x_line = np.linspace(0, max_val, 100)
    plt.plot(x_line, x_line, 'k--', alpha=0.5, linewidth=1, label='1:1 Line')
    y_line = m * x_line + b
    plt.plot(x_line, y_line, 'b-', alpha=0.7, linewidth=2,
             label=f'Fit: y = {m:.2f}x + {b:.2f} (R² = {r_squared:.2f})')
    plt.xlabel('Emission Proxy Original [ppm m/s]')
    plt.ylabel('Emission Proxy Bias Corrected [ppm m/s]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'enhancement_filtered_analysis_enhgt0_windgt2_qf0.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(df['enhancement_change'], bins=40, alpha=0.7, color='purple',
             label=f'Mean = {df["enhancement_change"].mean():.3f} ppm m/s')
    plt.axvline(df['enhancement_change'].mean(), color='black', linestyle='--',
                label=f'Mean = {df["enhancement_change"].mean():.3f} ppm m/s')
    plt.axvline(0, color='red', linestyle='-', alpha=0.5, label='No Change')
    plt.xlabel('Emission Proxy Change (Corrected - Original) [ppm m/s]')
    plt.ylabel('Count')
    plt.title('Distribution of Emission Proxy Changes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'enhancement_change_histogram_enhgt0_windgt2_qf0.png'), dpi=300, bbox_inches='tight')
    plt.close()

    stats_df = pd.DataFrame([stats])
    documentation = [
        "# Enhancement Analysis Statistics - Filtered (Enhancement>0 & Wind>2 m/s, QF=0)",
        "# Enhancements computed using only retrievals with xco2_quality_flag == 0.",
        "# n_corrected_sams: Total number of filtered SAMs with valid enhancement calculations",
        "# mean_original, std_original: Statistics for original enhancement proxy [ppm m/s]",
        "# mean_bc, std_bc: Statistics for bias-corrected enhancement proxy [ppm m/s]",
        "# correlation: Correlation between original and corrected enhancement values",
        "# regression_slope, regression_intercept: Linear fit parameters",
        "# r_squared: R² value for the linear fit",
        "# mean_abs_change: Mean absolute change in enhancement proxy [ppm m/s]",
        "# median_abs_change: Median absolute change in enhancement proxy [ppm m/s]",
        "# wind_speed_mean_mean: Mean of mean wind speeds across SAMs [m/s]",
        ""
    ]
    csv_path = os.path.join(output_dir, 'enhancement_filtered_stats_enhgt0_windgt2_qf0.csv')
    with open(csv_path, 'w') as f:
        for line in documentation:
            f.write(line + '\n')
        stats_df.to_csv(f, index=False)
    df.to_csv(os.path.join(output_dir, 'enhancement_filtered_sam_details_enhgt0_windgt2_qf0.csv'), index=False)

    print(f"\nResults saved to:")
    print(f"  Statistics: {csv_path}")
    print(f"  SAM details: enhancement_filtered_sam_details_enhgt0_windgt2_qf0.csv")
    print(f"  Plots: enhancement_filtered_analysis_enhgt0_windgt2_qf0.png, enhancement_change_histogram_enhgt0_windgt2_qf0.png")

    return stats

def analyze_enhancement_impacts_wind_qf0(enhancements_df, output_dir):
    """
    Same as analyze_enhancement_impacts_positive_wind but assumes enhancements were
    computed using only retrievals with xco2_quality_flag == 0.
    Filters to wind > 2 m/s and starts axes at 0.
    """
    print("Analyzing enhancement impacts for wind>2 m/s with QF=0...")

    required_cols = ['enhancement_original', 'enhancement_bc', 'enhancement_change', 'enhancement_abs_change', 'wind_speed_mean']
    for col in required_cols:
        if col not in enhancements_df.columns:
            print(f"Missing required column: {col}. Cannot perform filtered analysis.")
            return

    df = enhancements_df[(enhancements_df['wind_speed_mean'] > 2)].copy()

    if len(df) < 5:
        print("Not enough valid enhancements after filtering.")
        return

    for col in ['enhancement_original', 'enhancement_bc']:
        q01 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        df = df[(df[col] >= q01) & (df[col] <= q99)]

    if len(df) < 5:
        print("Not enough valid enhancements after outlier removal.")
        return

    print(f"Analyzing {len(df)} corrected SAMs after filtering and outlier removal...")

    stats = {
        'n_corrected_sams': len(df),
        'mean_original': df['enhancement_original'].mean(),
        'std_original': df['enhancement_original'].std(),
        'mean_bc': df['enhancement_bc'].mean(),
        'std_bc': df['enhancement_bc'].std(),
        'correlation': df['enhancement_original'].corr(df['enhancement_bc']),
        'mean_abs_change': df['enhancement_abs_change'].mean(),
        'median_abs_change': df['enhancement_abs_change'].median(),
        'wind_speed_mean_mean': df['wind_speed_mean'].mean()
    }

    m, b = np.polyfit(df['enhancement_original'], df['enhancement_bc'], 1)
    r_squared = stats['correlation'] ** 2

    stats['regression_slope'] = m
    stats['regression_intercept'] = b
    stats['r_squared'] = r_squared

    print(f"\n" + "="*60)
    print("ENHANCEMENT ANALYSIS RESULTS - WIND>2 m/s (QF=0)")
    print("="*60)
    print(f"Number of corrected SAMs analyzed: {stats['n_corrected_sams']}")
    print(f"Original enhancement - Mean: {stats['mean_original']:.3f}, Std: {stats['std_original']:.3f} ppm m/s")
    print(f"Bias-corrected enhancement - Mean: {stats['mean_bc']:.3f}, Std: {stats['std_bc']:.3f} ppm m/s")
    print(f"Correlation: {stats['correlation']:.3f}")
    print(f"Regression: y = {m:.3f}x + {b:.3f} (R² = {r_squared:.3f})")
    print(f"Mean absolute enhancement change: {stats['mean_abs_change']:.3f} ppm m/s")
    print(f"Median absolute enhancement change: {stats['median_abs_change']:.3f} ppm m/s")
    print(f"Mean wind speed (m/s): {stats['wind_speed_mean_mean']:.3f}")

    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(df['enhancement_original'], df['enhancement_bc'],
               alpha=0.5, s=10, color='red', label=f'Filtered SAMs (n={len(df)})', marker='.')
    max_val = max(df['enhancement_original'].max(), df['enhancement_bc'].max())
    min_val = min(df['enhancement_original'].min(), df['enhancement_bc'].min())
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    x_line = np.linspace(min_val, max_val, 100)
    plt.plot(x_line, x_line, 'k--', alpha=0.5, linewidth=1, label='1:1 Line')
    y_line = m * x_line + b
    plt.plot(x_line, y_line, 'b-', alpha=0.7, linewidth=2,
             label=f'Fit: y = {m:.2f}x + {b:.2f} (R² = {r_squared:.2f})')
    plt.xlabel('Emission Proxy Original [ppm m/s]')
    plt.ylabel('Emission Proxy Bias Corrected [ppm m/s]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'enhancement_filtered_analysis_windgt2_qf0.png'), dpi=300, bbox_inches='tight')    
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(df['enhancement_change'], bins=40, alpha=0.7, color='purple',
             label=f'Mean = {df["enhancement_change"].mean():.3f} ppm m/s')
    plt.axvline(df['enhancement_change'].mean(), color='black', linestyle='--',
                label=f'Mean = {df["enhancement_change"].mean():.3f} ppm m/s')
    plt.axvline(0, color='red', linestyle='-', alpha=0.5, label='No Change')
    plt.xlabel('Emission Proxy Change (Corrected - Original) [ppm m/s]')
    plt.ylabel('Count')
    plt.title('Distribution of Emission Proxy Changes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'enhancement_change_histogram_windgt2_qf0.png'), dpi=300, bbox_inches='tight')
    plt.close()

    stats_df = pd.DataFrame([stats])
    documentation = [
        "# Enhancement Analysis Statistics - Filtered (Wind>2 m/s, QF=0)",
        "# Enhancements computed using only retrievals with xco2_quality_flag == 0.",
        "# n_corrected_sams: Total number of filtered SAMs with valid enhancement calculations",
        "# mean_original, std_original: Statistics for original enhancement proxy [ppm m/s]",
        "# mean_bc, std_bc: Statistics for bias-corrected enhancement proxy [ppm m/s]",
        "# correlation: Correlation between original and corrected enhancement values",
        "# regression_slope, regression_intercept: Linear fit parameters",
        "# r_squared: R² value for the linear fit",
        "# mean_abs_change: Mean absolute change in enhancement proxy [ppm m/s]",
        "# median_abs_change: Median absolute change in enhancement proxy [ppm m/s]",
        "# wind_speed_mean_mean: Mean of mean wind speeds across SAMs [m/s]",
        ""
    ]
    csv_path = os.path.join(output_dir, 'enhancement_filtered_stats_windgt2_qf0.csv')
    with open(csv_path, 'w') as f:
        for line in documentation:
            f.write(line + '\n')
        stats_df.to_csv(f, index=False)
    df.to_csv(os.path.join(output_dir, 'enhancement_filtered_sam_details_windgt2_qf0.csv'), index=False)

    print(f"\nResults saved to:")
    print(f"  Statistics: {csv_path}")
    print(f"  SAM details: enhancement_filtered_sam_details_windgt2_qf0.csv")
    print(f"  Plots: enhancement_filtered_analysis_windgt2_qf0.png, enhancement_change_histogram_windgt2_qf0.png")

    return stats


    ep_o = enhancements_df['enhancement_original']
    ep_c = enhancements_df['enhancement_bc']
    med_o = float(np.nanmedian(ep_o))
    med_c = float(np.nanmedian(ep_c))
    iqr_o = np.nanpercentile(ep_o, [25, 75])
    iqr_c = np.nanpercentile(ep_c, [25, 75])
    r2 = float(ep_o.corr(ep_c)**2)
    med_abs_d = float(np.nanmedian(np.abs(ep_c - ep_o)))
    frac_abs_gt1 = float((np.abs(ep_c - ep_o) > 1.0).mean())

    # Directionality for U>=2
    df_w = enhancements_df[enhancements_df['wind_speed_mean'] >= 2.0]
    n_w = len(df_w)
    frac_dir_u2 = float((df_w['enhancement_bc'] > 0).mean()) if n_w else np.nan
    ci_u2 = _wilson_ci(int((df_w['enhancement_bc'] > 0).sum()), n_w) if n_w else (np.nan, np.nan)

    # Positive-only bias check
    mean_pos = float(ep_o[ep_o > 0].mean()) if (ep_o > 0).any() else np.nan
    median_full = float(np.nanmedian(ep_o))
    pos_bias_excess_pct = float(100.0 * (mean_pos - median_full) / abs(median_full)) if np.isfinite(mean_pos) and median_full != 0 else np.nan

    row = {
        'EP_median_orig': med_o,
        'EP_median_corr': med_c,
        'EP_IQR_orig_low': float(iqr_o[0]), 'EP_IQR_orig_high': float(iqr_o[1]),
        'EP_IQR_corr_low': float(iqr_c[0]), 'EP_IQR_corr_high': float(iqr_c[1]),
        'TheilSen_slope': (robust_fit or {}).get('slope', np.nan),
        'TheilSen_slope_CI_low': (robust_fit or {}).get('slope_ci', (np.nan, np.nan))[0],
        'TheilSen_slope_CI_high': (robust_fit or {}).get('slope_ci', (np.nan, np.nan))[1],
        'R2_report': r2,
        'median_abs_delta_EP': med_abs_d,
        'fraction_abs_delta_gt1': frac_abs_gt1,
        'directionality_fraction_Uge2': frac_dir_u2,
        'directionality_fraction_Uge2_CI_low': ci_u2[0],
        'directionality_fraction_Uge2_CI_high': ci_u2[1],
        'positive_only_bias_excess_percent': pos_bias_excess_pct
    }
    pd.DataFrame([row]).to_csv(os.path.join(output_dir, 'minimal_text_stats.csv'), index=False)

def main():
    """Main function to run the full dataset enhancement analysis."""
    
    print("Enhancement Analysis for Full OCO-3 Corrected Dataset")
    print("=" * 60)

    DEBUG = False
    if DEBUG: print("DEBUG MODE: Running with limited number of files")
    
    # Initialize configuration
    config = PathConfig()
    
    # Create output directory
    output_dir = config.FIGURES_DIR / "enhancement_analysis_full"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load corrected SAMs data (set max_files=10 for testing, None for full dataset)
    max_files = 1000 if DEBUG else None
    corrected_data = load_corrected_sams_data(config, max_files=max_files)
    
    if corrected_data.empty:
        print("No corrected SAMs found. Exiting.")
        return
    
    # Calculate enhancement proxies (no QF filter)
    enhancements_df = calculate_enhancements_for_sams(corrected_data, wind_speed_threshold=2.0, qf_column='xco2_quality_flag')
    
    if enhancements_df.empty:
        print("No enhancement calculations could be completed. Exiting.")
        return
    
    # Analyze and visualize impacts (full set)
    analyze_enhancement_impacts(enhancements_df, str(output_dir))

    # Analyze and visualize impacts for wind>2 m/s
    # analyze_enhancement_impacts_positive_wind(enhancements_df, str(output_dir))

    # Analyze and visualize impacts for wind>2 m/s with QF=0
    analyze_enhancement_impacts_wind_qf0(enhancements_df, str(output_dir))

    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")

if __name__ == '__main__':
    main() 