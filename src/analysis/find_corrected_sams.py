#!/usr/bin/env python3
"""
Find SAMs that actually had swath bias corrections applied
to identify better examples for case studies
"""

import os
import sys
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.utils.main_util import read_oco_netcdf_to_df
from src.utils.config_paths import PathConfig

def find_corrected_sams(config, max_files=None):
    """Find SAMs that actually had corrections applied."""
    
    processed_files = sorted(glob.glob(os.path.join(config.OUTPUT_FULL_DIR, '*_SwathBC.nc4')))
    
    if not processed_files:
        print(f"No processed files found in {config.OUTPUT_FULL_DIR}")
        return pd.DataFrame()
    
    if max_files:
        processed_files = processed_files[:max_files]
        print(f"DEBUG: Processing only {len(processed_files)} files")
    
    print(f"Searching {len(processed_files)} files for corrected SAMs...")
    
    corrected_sams = []
    
    for nc_file in tqdm(processed_files, desc="Scanning files"):
        try:
            required_vars = [
                'sounding_id', 'latitude', 'longitude', 'operation_mode', 'orbit', 'target_id',
                'xco2', 'xco2_swath_bc', 'swath_bias_corrected'
            ]
            
            # Try to include RF features if available
            optional_vars = ['max_relative_jump', 'h_continuum_sco2', 'dws', 'max_declocking_o2a', 'aod_sulfate']
            
            df = read_oco_netcdf_to_df(nc_file, variables_to_read=required_vars)
            
            if df.empty:
                continue
            
            # Filter for SAM data
            sam_data = df[df['operation_mode'] == 4].copy()
            if sam_data.empty:
                continue
            
            # Create SAM identifier
            sam_data['orbit_str'] = sam_data['orbit'].astype(int).astype(str)
            sam_data['SAM'] = sam_data['target_id'].fillna('none') + '_' + sam_data['orbit_str']
            
            # Filter out 'none' target SAMs
            sam_data = sam_data[~sam_data['SAM'].str.contains('none', case=False, na=False)]
            if sam_data.empty:
                continue
            
            # Find SAMs with corrections applied
            corrected_sam_data = sam_data[sam_data['swath_bias_corrected'] == 1]
            
            if not corrected_sam_data.empty:
                # Get summary info for each corrected SAM
                sam_summaries = corrected_sam_data.groupby('SAM').agg({
                    'latitude': 'mean',
                    'longitude': 'mean',
                    'target_id': 'first',
                    'xco2': 'mean',
                    'xco2_swath_bc': 'mean',
                    'swath_bias_corrected': 'max'
                }).reset_index()
                
                # Calculate correction magnitude
                sam_summaries['correction_magnitude'] = np.abs(sam_summaries['xco2_swath_bc'] - sam_summaries['xco2'])
                sam_summaries['n_soundings'] = corrected_sam_data.groupby('SAM').size().values
                sam_summaries['file'] = os.path.basename(nc_file)
                
                corrected_sams.append(sam_summaries)
                
        except Exception as e:
            print(f"Error processing {os.path.basename(nc_file)}: {e}")
            continue
    
    if corrected_sams:
        all_corrected = pd.concat(corrected_sams, ignore_index=True)
        return all_corrected
    else:
        return pd.DataFrame()

def categorize_target(target_id):
    """Categorize targets by type."""
    target_lower = str(target_id).lower()
    if 'fossil' in target_lower:
        return 'Fossil'
    elif 'volcano' in target_lower:
        return 'Volcano'
    elif 'sif' in target_lower:
        return 'SIF'
    elif 'texmex' in target_lower:
        return 'TexMex'
    elif 'ecostress' in target_lower:
        return 'ECOSTRESS'
    elif 'desert' in target_lower:
        return 'Desert'
    else:
        return 'Other'

def main():
    """Main function to find and analyze corrected SAMs."""
    
    print("Finding SAMs with applied corrections...")
    print("=" * 50)
    
    # Initialize configuration
    config = PathConfig()
    
    # Find corrected SAMs (use max_files=20 for quick testing, None for full scan)
    corrected_sams = find_corrected_sams(config, max_files=None)
    
    if corrected_sams.empty:
        print("No SAMs with corrections applied found!")
        return
    
    print(f"\nFound {len(corrected_sams)} SAMs with corrections applied!")
    
    # Add target category
    corrected_sams['category'] = corrected_sams['target_id'].apply(categorize_target)
    
    # Sort by correction magnitude
    corrected_sams = corrected_sams.sort_values('correction_magnitude', ascending=False)
    
    # Print summary statistics
    print(f"\nCorrection Summary:")
    print(f"Mean correction magnitude: {corrected_sams['correction_magnitude'].mean():.3f} ppm")
    print(f"Max correction magnitude: {corrected_sams['correction_magnitude'].max():.3f} ppm")
    print(f"Min correction magnitude: {corrected_sams['correction_magnitude'].min():.3f} ppm")
    
    print(f"\nBy category:")
    category_summary = corrected_sams.groupby('category').agg({
        'SAM': 'count',
        'correction_magnitude': ['mean', 'max']
    }).round(3)
    print(category_summary)
    
    # Show top examples by correction magnitude
    print(f"\nTop 10 SAMs by correction magnitude:")
    top_examples = corrected_sams.head(10)[['SAM', 'target_id', 'category', 'correction_magnitude', 'n_soundings']]
    print(top_examples.to_string(index=False))
    
    # Find good fossil fuel examples (power plants, etc.)
    fossil_examples = corrected_sams[corrected_sams['category'] == 'Fossil'].head(5)
    if not fossil_examples.empty:
        print(f"\nTop 5 Fossil Fuel SAMs for case studies:")
        fossil_display = fossil_examples[['SAM', 'target_id', 'correction_magnitude', 'n_soundings']]
        print(fossil_display.to_string(index=False))
        
        # Save these for the case study script
        best_examples = {
            'high_correction_fossil': fossil_examples.iloc[0]['SAM'],
            'medium_correction_fossil': fossil_examples.iloc[1]['SAM'] if len(fossil_examples) > 1 else fossil_examples.iloc[0]['SAM'],
            'low_correction_fossil': fossil_examples.iloc[-1]['SAM']
        }
        
        print(f"\nBest examples for case study:")
        for key, sam_id in best_examples.items():
            row = fossil_examples[fossil_examples['SAM'] == sam_id].iloc[0]
            print(f"  {key}: {sam_id} (magnitude: {row['correction_magnitude']:.3f} ppm, {row['n_soundings']} soundings)")
    
    # Also check if any Belchatow SAMs were corrected
    belchatow_corrected = corrected_sams[corrected_sams['target_id'] == 'fossil0193']
    if not belchatow_corrected.empty:
        print(f"\nBelchatow SAMs with corrections:")
        print(belchatow_corrected[['SAM', 'correction_magnitude', 'n_soundings']].to_string(index=False))
    else:
        print(f"\nNo Belchatow SAMs (fossil0193) had corrections applied.")
    
    # Save results
    output_file = config.FIGURES_DIR / "corrected_sams_analysis.csv"
    corrected_sams.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main() 