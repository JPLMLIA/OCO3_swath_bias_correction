#!/usr/bin/env python3
"""
Create Case Study Figure for Paper Section 5.5
Similar to confusion matrix examples but focused on specific case studies
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import glob

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.utils.main_util import plot_SAM, read_oco_netcdf_to_df, get_foreground_background_indices, get_target_data
from src.utils.config_paths import PathConfig


CASE_STUDIES = [
    {
        'sam_id': 'fossil0006_9386',
        'name': 'Shanghai, China',
        'file': 'oco3_LtCO2_201230_B11072Ar_240913233633s_SwathBC.nc4'
    },
    {
        'sam_id': 'fossil0012_11092',
        'name': 'Wuxi, China',
        'file': 'oco3_LtCO2_210419_B11072Ar_240915204225s_SwathBC.nc4'
    },
    {
        'sam_id': 'fossil0183_29888',
        'name': 'Colstrip Power Plant, USA',
        'file': 'oco3_LtCO2_240814_B11072Ar_241126004843s_SwathBC.nc4'
    }
]

def load_case_study_sam(sam_id, filename, config):
    """Load data for a specific case study SAM."""
    
    full_path = os.path.join(config.OUTPUT_FULL_DIR, filename)
    if not os.path.exists(full_path):
        print(f"  ✗ File not found: {full_path}")
        return None
    
    # Load required variables
    required_vars = [
        'sounding_id', 'latitude', 'longitude', 'vertex_latitude', 'vertex_longitude',
        'operation_mode', 'orbit', 'target_id', 'xco2', 'xco2_swath_bc', 'swath_bias_corrected',
        'windspeed_u_met', 'windspeed_v_met', 'xco2_quality_flag'
    ]
    
    print(f"  Loading {filename} for {sam_id}...")
    df = read_oco_netcdf_to_df(full_path, variables_to_read=required_vars)
    
    if df.empty:
        print(f"  ✗ No data loaded")
        return None
    
    # Filter for SAM data and create SAM identifier
    sam_data = df[df['operation_mode'] == 4].copy()
    if sam_data.empty:
        print(f"  ✗ No SAM data found")
        return None
    
    sam_data['orbit_str'] = sam_data['orbit'].astype(int).astype(str)
    sam_data['SAM'] = sam_data['target_id'].fillna('none') + '_' + sam_data['orbit_str']
    
    # Extract our specific SAM
    target_sam_data = sam_data[sam_data['SAM'] == sam_id].copy()

        # Filter to QF=0 only
    if 'xco2_quality_flag' not in target_sam_data.columns:
        print("  ✗ Missing xco2_quality_flag; cannot apply QF=0 filter")
        return None
    target_sam_data = target_sam_data[target_sam_data['xco2_quality_flag'] == 0].copy()
    if target_sam_data.empty:
        print(f"  ✗ After QF=0 filter, no data remains for {sam_id}")
        return None
    
    if target_sam_data.empty:
        print(f"  ✗ SAM {sam_id} not found in file")
        return None
    
    print(f"  ✓ Loaded {len(target_sam_data)} soundings for {sam_id}")
    return target_sam_data

def create_individual_case_study_plots(sam_data, case_info, output_dir):
    """Create individual plots for a case study SAM."""
    
    sam_id = case_info['sam_id']
    case_name = case_info['name']
    
    # Create individual plots directory
    plots_dir = os.path.join(output_dir, 'individual_case_studies')
    os.makedirs(plots_dir, exist_ok=True)
    
    if len(sam_data) < 10:
        print(f"  ✗ Insufficient data for {sam_id}")
        return []
    
    # Calculate colorbar limits for consistency
    xco2_values = sam_data['xco2'].dropna()
    if len(xco2_values) == 0:
        print(f"  ✗ No valid XCO2 data for {sam_id}")
        return []
    
    vmin = np.round(np.nanpercentile(xco2_values, 5), 1)
    vmax = np.round(np.nanpercentile(xco2_values, 95), 1)
    
    # Check if correction was applied
    correction_applied = sam_data['swath_bias_corrected'].max() > 0
    
    # Create difference column
    if correction_applied and 'xco2_swath_bc' in sam_data.columns:
        sam_data['xco2_difference'] = sam_data['xco2_swath_bc'] - sam_data['xco2']
        
        # Calculate colorbar limits for difference plot
        max_diff = np.nanmax(np.abs(sam_data['xco2_difference']))
        diff_vmin, diff_vmax = -max_diff, max_diff
        if max_diff < 0.1:
            diff_vmin, diff_vmax = -0.5, 0.5
    else:
        sam_data['xco2_difference'] = np.zeros(len(sam_data))
        diff_vmin, diff_vmax = -0.5, 0.5
    
    plot_paths = []
    
    # Calculate foreground/background for borders on original plot
    foreground_data = None
    background_data = None
    try:
        # Get wind direction and speed
        wind_dir = np.mean(np.arctan2(sam_data['windspeed_v_met'], sam_data['windspeed_u_met']))
        if wind_dir < 0:
            wind_dir += 2*np.pi
        wind_speed = np.mean(np.sqrt(sam_data['windspeed_v_met']**2 + sam_data['windspeed_u_met']**2))
        
        # Get target coordinates
        target_id = sam_data['target_id'].iloc[0]
        target_lat, target_lon = get_target_data(target_id)
        
        # Get foreground and background indices
        foreground_data, background_data = get_foreground_background_indices(
            sam_data, target_lat, target_lon, wind_dir, wind_speed, custom_SAM=True
        )
    except Exception as e:
        print(f"  Warning: Could not calculate foreground/background for {sam_id}: {e}")
        foreground_data = None
        background_data = None
    
    # 1. Original XCO2 plot
    title_orig = f'{case_name}\nOriginal XCO₂'
    plot_name_orig = f'{sam_id}_original'
    
    plot_SAM(sam_data, 'xco2', vmin=vmin, vmax=vmax,
             save_fig=True, name=plot_name_orig, path=plots_dir,
             title_addition=title_orig, simplified_title=True)
    
    orig_path = os.path.join(plots_dir, f'{sam_id}_xco2_{plot_name_orig}.png')
    if os.path.exists(orig_path):
        plot_paths.append(('original', orig_path))
    
    # 2. Corrected XCO2 plot
    if 'xco2_swath_bc' in sam_data.columns:
        title_corr = f'{case_name}\nBias Corrected XCO₂'
        plot_name_corr = f'{sam_id}_corrected'
        
        plot_SAM(sam_data, 'xco2_swath_bc', vmin=vmin, vmax=vmax,
                 save_fig=True, name=plot_name_corr, path=plots_dir,
                 title_addition=title_corr, simplified_title=True)
        
        corr_path = os.path.join(plots_dir, f'{sam_id}_xco2_swath_bc_{plot_name_corr}.png')
        if os.path.exists(corr_path):
            plot_paths.append(('corrected', corr_path))
    
    # 3. Difference plot (always create, even if zeros)
    # Get correction statistics
    mean_correction = np.nanmean(np.abs(sam_data['xco2_difference']))
    max_correction = np.nanmax(np.abs(sam_data['xco2_difference']))
    
    if correction_applied:
        title_diff = f'{case_name}\nApplied Correction\nMean: {mean_correction:.3f} ppm, Max: {max_correction:.3f} ppm'
    else:
        title_diff = f'{case_name}\nApplied Correction\nNo correction applied (all zeros)'
    
    plot_name_diff = f'{sam_id}_difference'
    
    plot_SAM(sam_data, 'xco2_difference', vmin=diff_vmin, vmax=diff_vmax,
             save_fig=True, name=plot_name_diff, path=plots_dir,
             title_addition=title_diff, simplified_title=True)
    
    diff_path = os.path.join(plots_dir, f'{sam_id}_xco2_difference_{plot_name_diff}.png')
    if os.path.exists(diff_path):
        plot_paths.append(('difference', diff_path))
    
    print(f"  ✓ Created {len(plot_paths)} plots for {sam_id}")
    return plot_paths

def create_combined_case_study_figure(case_study_plots, output_dir):
    """Create the combined case study figure for the paper."""
    
    print("Creating combined case study figure...")
    
    # Organize plots by case study
    n_cases = len(case_study_plots)
    n_cols = 3  # Original, Corrected, Difference
    
    # Create figure with appropriate size
    fig, axes = plt.subplots(n_cases, n_cols, figsize=(15, 5*n_cases))
    
    if n_cases == 1:
        axes = axes.reshape(1, -1)
    
    col_titles = ['Original XCO₂', 'Bias Corrected XCO₂', 'Applied Correction']
    
    # Set column titles
    for j, col_title in enumerate(col_titles):
        axes[0, j].set_title(col_title, fontsize=14, fontweight='bold', pad=20)
    
    for i, (case_info, plot_paths) in enumerate(case_study_plots):
        case_name = case_info['name']
        
        # Organize plots by type
        plots_by_type = {plot_type: path for plot_type, path in plot_paths}
        
        for j, plot_type in enumerate(['original', 'corrected', 'difference']):
            ax = axes[i, j]
            
            if plot_type in plots_by_type and os.path.exists(plots_by_type[plot_type]):
                # Load and display the image
                img = plt.imread(plots_by_type[plot_type])
                ax.imshow(img)
                
                # Case study labels removed - will be added as A/B/C in Photoshop
                pass
            else:
                # Plot not available
                ax.text(0.5, 0.5, f'Plot not available\n{plot_type.title()}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, color='gray')
            
            ax.axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, top=0.95, hspace=0.1, wspace=0.05)
    
    # Main figure title removed - will be handled in paper caption
    
    # Save the combined figure
    combined_path = os.path.join(output_dir, 'case_study_figure.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Combined case study figure saved to: {combined_path}")
    return combined_path

def main():
    """Main function to create case study plots."""
    
    print("Creating Case Study Figure ...")
    print("=" * 50)
    
    # Initialize configuration
    config = PathConfig()
    
    # Create output directory
    output_dir = config.FIGURES_DIR / "case_studies"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    case_study_plots = []
    
    # Process each case study
    for case_info in CASE_STUDIES:
        sam_id = case_info['sam_id']
        case_name = case_info['name']
        filename = case_info['file']
        
        print(f"\nProcessing case study: {case_name} ({sam_id})")
        
        # Load SAM data
        sam_data = load_case_study_sam(sam_id, filename, config)
        
        if sam_data is not None:
            # Create individual plots
            plot_paths = create_individual_case_study_plots(sam_data, case_info, str(output_dir))
            
            if plot_paths:
                case_study_plots.append((case_info, plot_paths))
            else:
                print(f"  ✗ Failed to create plots for {sam_id}")
        else:
            print(f"  ✗ Could not load data for {sam_id}")
    
    # Create combined figure if we have case studies
    if case_study_plots:
        combined_path = create_combined_case_study_figure(case_study_plots, str(output_dir))
        
        print(f"\n✓ Case study figure creation complete!")
        print(f"Output directory: {output_dir}")
        print(f"Combined figure: {combined_path}")
        
        # Print summary
        print(f"\nCase Study Figure includes {len(case_study_plots)} case studies:")
        for case_info, plot_paths in case_study_plots:
            print(f"  - {case_info['name']}")
            print(f"    Plots created: {[plot_type for plot_type, path in plot_paths]}")
        
    else:
        print("✗ No case studies could be processed.")

if __name__ == '__main__':
    main() 