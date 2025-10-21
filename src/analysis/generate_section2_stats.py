import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Ensure the utils module can be found
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from ..utils.main_util import load_data, plot_map

# Configuration
PAPER_FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'figures', 'paper_figures')
os.makedirs(PAPER_FIGURES_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(PAPER_FIGURES_DIR, "section2_dataset_stats_log.txt")

# --- Helper for logging ---
def log_and_print(message, log_file_handle):
    print(message)
    log_file_handle.write(str(message) + "\\n")

def main():
    with open(LOG_FILE_PATH, "w") as log_file:
        log_and_print("Starting Section 2 dataset statistics generation.", log_file)

        mode = 'SAM'
        all_data = pd.DataFrame()
        log_and_print("Loading data for years 2019-2024...", log_file)
        # The paper mentions data up to Feb 2025. We load available yearly files.
        for year in range(2019, 2025): # Up to 2024 inclusive
            try:
                log_and_print(f"Loading data for {year} using mode='{mode}'...", log_file)
                # Using default load_data parameters which implies preload_IO=True, clean_IO=True
                # This uses pre-processed parquet files.
                data_y = load_data(year=year, mode=mode)
                if data_y is not None and not data_y.empty:
                    all_data = pd.concat([all_data, data_y], axis=0, ignore_index=True)
                    log_and_print(f"Loaded {len(data_y)} soundings for {year}. Total soundings so far: {len(all_data)}", log_file)
                else:
                    log_and_print(f"No data loaded or empty dataframe for {year}.", log_file)
            except Exception as e:
                log_and_print(f"Could not load data for year {year}: {e}", log_file)
                import traceback
                log_and_print(traceback.format_exc(), log_file)


        if all_data.empty:
            log_and_print("No data loaded across all years. Exiting.", log_file)
            return

        log_and_print(f"Total soundings loaded after concatenation: {len(all_data)}", log_file)

        # Ensure 'SAM' identifier exists. load_data with mode='SAM' should provide it.
        if 'SAM' not in all_data.columns:
            log_and_print("Error: 'SAM' column not found in loaded data. Cannot proceed with SAM-specific stats.", log_file)
            # Attempt to create if essential columns exist (as a fallback, though load_data should handle this)
            if 'orbit' in all_data.columns and 'target_id' in all_data.columns:
                log_and_print("Attempting to create 'SAM' identifier.", log_file)
                all_data['orbit_str'] = all_data['orbit'].astype(int).astype(str)
                # Ensure target_id is string before concatenation, handle potential NaNs
                all_data['SAM'] = all_data['target_id'].astype(str).fillna('unknown') + '_' + all_data['orbit_str']
            else:
                log_and_print("Essential columns ('orbit', 'target_id') for SAM ID creation are missing. Exiting.", log_file)
                return
        
        # 1. Total number of SAMs and soundings
        total_sams = all_data['SAM'].nunique()
        total_soundings = len(all_data)
        log_and_print(f"Total unique SAMs: {total_sams}", log_file)
        log_and_print(f"Total soundings in these SAMs: {total_soundings}", log_file)

        # 2. SAMs over water vs. land
        if 'land_water_indicator' in all_data.columns:
            log_and_print("Calculating land/water distribution of SAMs based on mode of land_water_indicator.", log_file)
            # Define a helper to robustly get mode
            def get_mode(series):
                modes = series.mode()
                return modes[0] if not modes.empty else -1 # Return -1 if mode is empty (e.g. all NaN)

            sam_lwi_mode = all_data.groupby('SAM')['land_water_indicator'].apply(get_mode)
            
            sams_land = (sam_lwi_mode == 0).sum()
            sams_water = (sam_lwi_mode == 1).sum()
            sams_inland_water = (sam_lwi_mode == 2).sum()
            sams_mixed = (sam_lwi_mode == 3).sum()
            sams_unknown_lwi = (sam_lwi_mode == -1).sum()
            
            log_and_print(f"Number of SAMs primarily over land (LWI mode 0): {sams_land}", log_file)
            log_and_print(f"Number of SAMs primarily over water (LWI mode 1): {sams_water}", log_file)
            log_and_print(f"Number of SAMs primarily over inland water (LWI mode 2): {sams_inland_water}", log_file)
            log_and_print(f"Number of SAMs primarily over mixed (LWI mode 3): {sams_mixed}", log_file)
            if sams_unknown_lwi > 0:
                log_and_print(f"Number of SAMs with unknown/indeterminate LWI mode: {sams_unknown_lwi}", log_file)
        else:
            log_and_print("Warning: 'land_water_indicator' column not found. Cannot determine land/water SAMs.", log_file)

        # 3. Histogram of soundings per SAM
        log_and_print("Generating histogram of soundings per SAM...", log_file)
        soundings_per_sam = all_data.groupby('SAM').size()
        plt.figure(figsize=(10, 6))
        plt.hist(soundings_per_sam, bins=np.logspace(np.log10(10),np.log10(soundings_per_sam.max()),50), edgecolor='black') # Log bins
        plt.gca().set_xscale("log")
        plt.title('Distribution of Soundings per SAM')
        plt.xlabel('Number of Soundings in SAM (Log Scale)')
        plt.ylabel('Number of SAMs')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        hist_path = os.path.join(PAPER_FIGURES_DIR, 'soundings_per_sam_histogram.png')
        plt.savefig(hist_path, dpi=300)
        plt.close()
        log_and_print(f"Histogram of soundings per SAM saved to: {hist_path}", log_file)

        # 4. Global map of SAM locations
        if 'latitude' in all_data.columns and 'longitude' in all_data.columns:
            log_and_print("Generating global map of SAM locations...", log_file)
            data_for_map_loc = all_data[['latitude', 'longitude']].copy()
            # plot_map with aggregate='count' will use the length of data in bins.
            # The 'vars' needs a valid column name, but its values don't matter for 'count'.
            data_for_map_loc['sam_density_placeholder'] = 1 
            
            plot_map_title = 'Global Distribution of SAM Observations'
            plot_map_var_name_for_file = 'sam_density' # Keep it simple for filename
            plot_map_output_name_suffix = '_Global_Distribution'

            # Call plot_map. It saves file as path/var_name + name + .png
            # Example: PAPER_FIGURES_DIR/sam_density_placeholder_Global_Distribution.png
            plot_map(data_for_map_loc, vars=['sam_density_placeholder'], aggregate='count', 
                     name=plot_map_output_name_suffix, # This becomes part of the title and filename
                     path=PAPER_FIGURES_DIR, save_fig=True, 
                     pos_neg_IO=False, # For counts, non-negative scale
                     set_nan=False, # Show all cells with data
                     min=1) # Sensible min for count colorbar

            generated_path = os.path.join(PAPER_FIGURES_DIR, f'sam_density_placeholder{plot_map_output_name_suffix}.png')
            target_path = os.path.join(PAPER_FIGURES_DIR, 'sam_locations_map.png')
            
            try:
                if os.path.exists(generated_path):
                    os.rename(generated_path, target_path)
                    log_and_print(f"Global map of SAM locations saved to: {target_path}", log_file)
                else:
                    log_and_print(f"Failed to find generated SAM location map at {generated_path} to rename.", log_file)
            except OSError as e:
                log_and_print(f"Error renaming SAM location map: {e}", log_file)
        else:
            log_and_print("Warning: 'latitude' or 'longitude' columns not found. Cannot generate SAM location map.", log_file)

        # 5. Global map of h_continuum_sco2
        if 'h_continuum_sco2' in all_data.columns:
            if all_data['h_continuum_sco2'].isnull().all():
                log_and_print("Warning: 'h_continuum_sco2' column is all NaN. Cannot generate its global map.", log_file)
            else:
                log_and_print("Generating global map of h_continuum_sco2...", log_file)
                
                plot_map_var_h_continuum = 'h_continuum_sco2'
                plot_map_name_suffix_h_continuum = '_Global_Mean_Distribution'
                
                plot_map(all_data, vars=[plot_map_var_h_continuum], aggregate='mean', 
                         name=plot_map_name_suffix_h_continuum, 
                         path=PAPER_FIGURES_DIR, save_fig=True, 
                         pos_neg_IO=False) # Assuming h_continuum_sco2 is typically positive

                generated_path_h = os.path.join(PAPER_FIGURES_DIR, f'{plot_map_var_h_continuum}{plot_map_name_suffix_h_continuum}.png')
                target_path_h = os.path.join(PAPER_FIGURES_DIR, 'h_continuum_sco2_map.png')

                try:
                    if os.path.exists(generated_path_h):
                        os.rename(generated_path_h, target_path_h)
                        log_and_print(f"Global map of h_continuum_sco2 saved to: {target_path_h}", log_file)
                    else:
                        log_and_print(f"Failed to find generated h_continuum_sco2 map at {generated_path_h} to rename.", log_file)
                except OSError as e:
                    log_and_print(f"Error renaming h_continuum_sco2 map: {e}", log_file)
        else:
            log_and_print("Warning: 'h_continuum_sco2' column not found. Cannot generate its global map.", log_file)

        log_and_print(f"Finished Section 2 dataset statistics generation. Log file: {LOG_FILE_PATH}", log_file)

if __name__ == '__main__':
    main() 