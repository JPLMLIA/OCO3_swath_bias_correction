#!/usr/bin/env python3
"""
Generate Final Paper Statistics for OCO-3 Swath Bias Correction

This script processes the full OCO-3 data record to quantify the impact of the swath
bias correction. It uses the `swath_bias_corrected` flag in the Lite files as the
ground truth for identifying corrected scenes.

CORRECTED: This version properly filters for operation_mode == 4 (SAM mode) to match
the processing script and provide accurate statistics.

The script has two primary outputs:
1.  A Parquet file (`global_sam_summary.parquet`) containing every unique SAM,
    its correction status, and location. This is used for generating the global map.
2.  A CSV file (`full_dataset_enhancement_impact_stats.csv`) with detailed
    statistics on how the correction impacted the emission enhancement proxy for the
    subset of SAMs that were actually corrected.
3.  A CSV file (`corrected_sams_category_breakdown.csv`) with category distribution analysis.

Usage:
    python -m src.analysis.generate_paper_stats --full-dataset [--category-analysis]
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor

# --- Robust Path Setup ---
# Add the project root to the path to allow absolute imports from 'src'
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from this project
from src.utils.main_util import read_oco_netcdf_to_df, SAM_enhancement
from src.utils.config_paths import PathConfig


DEBUG_FILE_LIMIT = 10

def categorize_sam(sam_name):
    """Create mapping of SAM categories based on SAM names."""
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

def get_sam_info(file_path):
    """Reads a single NetCDF file and returns a DataFrame with SAM info (CORRECTED)."""
    try:
        # DO NOT filter columns. The SAM ID needs to be generated from other columns.
        df = read_oco_netcdf_to_df(file_path)
        if df is None or df.empty:
            return None
        
        # *** CRITICAL FIX: Filter for SAM operation mode only ***
        if 'operation_mode' not in df.columns:
            print(f"Warning: No operation_mode in {os.path.basename(file_path)}")
            return None
        
        # Filter for SAM operation mode (4=SAM) - same as processing script
        sam_data = df[df['operation_mode'] == 4]
        if sam_data.empty:
            return None
        
        # Also filter out 'none' target_ids like the processing script does
        sam_data = sam_data[~sam_data['SAM'].str.contains('none', case=False, na=False)]
        if sam_data.empty:
            return None
        
        # Get the correction status for each SAM (True if any sounding is flagged)
        sam_info = sam_data.groupby('SAM').agg(
            latitude=('latitude', 'mean'),
            longitude=('longitude', 'mean'),
            is_corrected=('swath_bias_corrected', 'max') # max will be 1 if any sounding is corrected
        ).reset_index()
        sam_info['is_corrected'] = sam_info['is_corrected'].astype(bool)
        return sam_info
    except BaseException as e:
        # This will catch low-level errors from the NetCDF library too
        print(f"Warning: Could not process file {os.path.basename(file_path)}: {e}", file=sys.stderr)
        return None

def perform_category_analysis(all_sams_df, output_dir):
    """Perform detailed category analysis on SAM data."""
    print(f"\n" + "="*60)
    print("CORRECTED SAMS CATEGORY DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Add category mapping
    all_sams_df['category'] = all_sams_df['SAM'].apply(categorize_sam)
    
    total_sams = len(all_sams_df)
    corrected_sams = all_sams_df[all_sams_df['is_corrected']]
    total_corrected = len(corrected_sams)
    
    print(f"Total unique TRUE SAMs in analysis: {total_sams:,}")
    print(f"Total corrected SAMs: {total_corrected:,} ({total_corrected/total_sams*100:.1f}%)")
    
    print(f"\nDistribution of ALL TRUE SAMs by category:")
    all_category_counts = all_sams_df['category'].value_counts().sort_index()
    for category, count in all_category_counts.items():
        print(f"  {category}: {count:,} ({count/total_sams*100:.1f}%)")
    
    print(f"\nDistribution of CORRECTED SAMs by category:")
    corrected_category_counts = corrected_sams['category'].value_counts().sort_index()
    for category, count in corrected_category_counts.items():
        percent_of_corrected = count/total_corrected*100
        category_total = all_category_counts.get(category, 0)
        correction_rate = count/category_total*100 if category_total > 0 else 0
        print(f"  {category}: {count:,} ({percent_of_corrected:.1f}% of corrected, {correction_rate:.1f}% correction rate)")
    
    # Create summary table
    summary_stats = []
    for category in all_category_counts.index:
        total_in_category = all_category_counts[category]
        corrected_in_category = corrected_category_counts.get(category, 0)
        correction_rate = corrected_in_category/total_in_category*100 if total_in_category > 0 else 0
        
        summary_stats.append({
            'category': category,
            'total_sams': total_in_category,
            'corrected_sams': corrected_in_category,
            'correction_rate_percent': correction_rate,
            'percent_of_all_corrected': corrected_in_category/total_corrected*100 if total_corrected > 0 else 0
        })
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Save results
    output_file = output_dir / "corrected_sams_category_breakdown.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"\nDetailed breakdown saved to: {output_file}")
    
    # Generate text for paper
    print(f"\n" + "="*60)
    print("TEXT FOR PAPER:")
    print("="*60)
    
    # Find top categories by corrected SAMs
    top_categories = corrected_category_counts.head(3)
    
    text_parts = []
    for i, (category, count) in enumerate(top_categories.items()):
        percent = count/total_corrected*100
        if i == 0:
            text_parts.append(f"{category} targets ({count:,} SAMs, {percent:.1f}%)")
        elif i == len(top_categories) - 1:
            text_parts.append(f"and {category} targets ({count:,} SAMs, {percent:.1f}%)")
        else:
            text_parts.append(f"{category} targets ({count:,} SAMs, {percent:.1f}%)")
    
    suggested_text = f"The majority of corrections occur over {', '.join(text_parts)}."
    print(f"Suggested text: {suggested_text}")

def analyze_full_dataset(full_data_dir_path, output_dir_path, debug=False, category_analysis=False):
    """
    Analyzes the full, final corrected dataset from the NetCDF files (CORRECTED).
    """
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("ANALYZING FULL CORRECTED DATASET (TRUE SAMs ONLY)")
    print(f"Reading from: {full_data_dir_path}")
    print("="*60)

    # Check if the directory exists (it might not if processing hasn't been run)
    if not os.path.exists(full_data_dir_path):
        print(f"Error: Directory {full_data_dir_path} does not exist.")
        print("This suggests the bias correction processing hasn't been run yet.")
        print("Please run: python -m src.processing.apply_swath_bc_RF")
        return

    nc_files = sorted(list(Path(full_data_dir_path).glob('*.nc4')))
    if not nc_files:
        print(f"Error: No NetCDF files found in {full_data_dir_path}", file=sys.stderr)
        print("This suggests the bias correction processing hasn't been run yet.")
        return

    if debug:
        print(f"--- DEBUG MODE: Processing only {DEBUG_FILE_LIMIT} files. ---")
        nc_files = nc_files[:DEBUG_FILE_LIMIT]

    # --- Pass 1: Get info for all SAMs (ID, location, correction status) ---
    print(f"\n--- Pass 1: Gathering info for TRUE SAMs from {len(nc_files)} files ---")
    all_sam_info_list = []
    # Use a ThreadPool for I/O bound tasks
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(get_sam_info, nc_files), total=len(nc_files), desc="Scanning SAMs"))

    for result in results:
        if result is not None:
            all_sam_info_list.append(result)

    if not all_sam_info_list:
        print("Error: Could not read SAM info from any files.", file=sys.stderr)
        sys.exit(1)

    # Consolidate and find unique SAMs
    all_sams_df = pd.concat(all_sam_info_list).drop_duplicates(subset=['SAM']).reset_index(drop=True)
    
    total_sam_count = len(all_sams_df)
    corrected_sams_df = all_sams_df[all_sams_df['is_corrected']].copy()
    corrected_sam_count = len(corrected_sams_df)
    
    print(f"\n*** CORRECTED STATISTICS (TRUE SAMs ONLY) ***")
    print(f"Found {total_sam_count:,} total unique TRUE SAMs (operation_mode=4, no 'none').")
    print(f"Found {corrected_sam_count:,} corrected SAMs.")
    if total_sam_count > 0:
        correction_rate = (corrected_sam_count / total_sam_count) * 100
        print(f"Correction Rate: {correction_rate:.1f}% ({corrected_sam_count} / {total_sam_count} SAMs)")
        print(f"\nThis should match the ~19k SAMs processed by apply_swath_bc_RF.py")

    # Save the summary for the plotting script
    summary_path = output_dir / "global_sam_summary.parquet"
    all_sams_df.to_parquet(summary_path)
    print(f"\nSAM summary saved to: {summary_path}")
    
    # Perform category analysis if requested
    if category_analysis:
        perform_category_analysis(all_sams_df, output_dir)

    corrected_sams_set = set(corrected_sams_df['SAM'])
    if not corrected_sams_set:
        print("\nNo corrected SAMs found. Skipping enhancement analysis.")
        return

    # --- Pass 2: Load full data ONLY for corrected SAMs and calculate enhancement ---
    print("\n--- Pass 2: Loading full data for corrected SAMs to analyze impact ---")
    
    def load_full_data_for_corrected_sams(file_path):
        try:
            # Must read all columns to ensure we can filter by SAM
            full_df = read_oco_netcdf_to_df(file_path)
            if full_df is None or full_df.empty:
                return None

            # Filter for SAM mode and corrected SAMs
            if 'operation_mode' in full_df.columns:
                full_df = full_df[full_df['operation_mode'] == 4]
            
            # Filter for corrected SAMs only
            return full_df[full_df['SAM'].isin(corrected_sams_set)]
        except BaseException as e:
            print(f"Warning: Could not load full data from {os.path.basename(file_path)}: {e}", file=sys.stderr)
            return None

    corrected_data_list = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(load_full_data_for_corrected_sams, nc_files), total=len(nc_files), desc="Loading corrected SAMs"))

    for result in results:
        if result is not None:
            corrected_data_list.append(result)

    if not corrected_data_list:
        print("Error: Failed to load detailed data for any corrected SAMs.", file=sys.stderr)
        return

    # Use specific columns for duplicate detection to avoid unhashable numpy arrays
    corrected_data = pd.concat(corrected_data_list)
    # Drop duplicates based on key identifying columns only
    corrected_data = corrected_data.drop_duplicates(subset=['sounding_id']).reset_index(drop=True)
    
    print(f"\nAnalysis starting on {len(corrected_data):,} soundings from {corrected_data['SAM'].nunique():,} corrected SAMs.")
    
    # --- Final Analysis and Output Generation ---
    # Rename columns for consistency with plotting functions
    if 'xco2' in corrected_data.columns and 'xco2_swath_bc' in corrected_data.columns:
        corrected_data.rename(columns={'xco2': 'xco2_original', 'xco2_swath_bc': 'xco2'}, inplace=True)
    else:
        print("Error: Required columns 'xco2' or 'xco2_swath_bc' not found.", file=sys.stderr)
        sys.exit(1)
        
    def calculate_enhancement_impact(sam_df):
        # Pass copies to SAM_enhancement to avoid modifying the original df
        enhancement_orig = SAM_enhancement(sam_df.rename(columns={'xco2_original': 'xco2'}), 'xco2', qf=None, custom_SAM=True)
        enhancement_corr = SAM_enhancement(sam_df, 'xco2', qf=None, custom_SAM=True)
        
        if enhancement_orig is None or enhancement_corr is None:
            return None
            
        impact = enhancement_corr - enhancement_orig
        mean_corr_mag = (sam_df['xco2'] - sam_df['xco2_original']).abs().mean()
        
        return pd.Series({
            'enhancement_original': enhancement_orig,
            'enhancement_corrected': enhancement_corr,
            'enhancement_impact': impact,
            'enhancement_abs_impact': abs(impact),
            'correction_magnitude_mean': mean_corr_mag
        })

    print("\n--- Quantitative Impact on Corrected SAMs ---")
    enhancement_stats = corrected_data.groupby('SAM').apply(calculate_enhancement_impact).dropna()
    print(enhancement_stats.describe())

    stats_path = output_dir / "full_dataset_enhancement_impact_stats.csv"
    enhancement_stats.to_csv(stats_path)
    print(f"\nEnhancement impact statistics saved to {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate final paper statistics for OCO-3 Swath BC (CORRECTED).")
    parser.add_argument('--full-dataset', action='store_true',
                        help="Analyze the full, final corrected dataset.")
    parser.add_argument('--category-analysis', action='store_true',
                        help="Include detailed category distribution analysis.")
    parser.add_argument('--debug', action='store_true',
                        help="Run in debug mode on a small subset of files.")
    
    args = parser.parse_args()

    paths = PathConfig()
    
    if args.full_dataset:
        # Use the configured output directory
        config = PathConfig()
        full_data_dir = config.OUTPUT_FULL_DIR
        output_dir = os.path.join(paths.FIGURES_DIR, "paper_final_stats")
        analyze_full_dataset(full_data_dir, output_dir, debug=args.debug, category_analysis=args.category_analysis)
    else:
        print("This script is intended for the full dataset analysis.")
        print("Please run with the --full-dataset flag.")

if __name__ == "__main__":
    main() 