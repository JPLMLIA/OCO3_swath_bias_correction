#!/usr/bin/env python3
"""
Script to generate a three-panel figure showing examples of each swath bias class. Used for the paper

This script creates a figure with three SAM examples side by side:
- Class 0: No swath bias
- Class 1: Clear swath bias  
- Class 2: Uncertain swath bias

The script uses the plot_SAM function from main_util.py to generate individual
high-quality SAM plots, then combines them into a single three-panel figure.
The figure is intended for inclusion in the paper.

Usage:
    python src/analysis/swath_bias_examples_figure.py

Requirements:
    - OCO3_bias conda environment activated
    - Access to the preloaded SAM data files
    - Internet connection for downloading map features (first run only)

Output:
    - PNG and PDF versions saved to config.FIGURES_DIR/swath_bias_examples/
    - High-resolution figures suitable for publication
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Add the parent directory to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.main_util import load_data, plot_SAM
from src.utils.config_paths import PathConfig
from PIL import Image


def create_swath_bias_examples_figure():
    """
    Create a three-panel figure showing examples of each swath bias class.
    Uses the plot_SAM function from main_util.py to generate individual SAM plots,
    then combines them into a single three-panel figure.
    """
    
    # Initialize config
    config = PathConfig()
    
    # Define representative SAM IDs for each class
    # These were selected from the labeled dataset as good examples
    example_sam_ids = {
        0: 'fossil0001_19513',        # Class 0: No swath bias
        1: 'fossil0008_10293',        # Class 1: Clear swath bias
        2: 'fossil0003_13217'         # Class 2: Uncertain
    }
    
    class_names = {
        0: 'Class 0: No Swath Bias',
        1: 'Class 1: Swath Bias', 
        2: 'Class 2: Uncertain'
    }
    
    # Load the SAM data  
    print("Loading SAM data...")
    full_dataset = pd.DataFrame()
    
    # Load data for multiple years to ensure we get the SAMs we need
    for year in range(2019, 2025):
        try:
            print(f"Loading year {year}...")
            data_year = load_data(year, 'SAM', preload_IO=True, clean_IO=True, TCCON=False)
            if not data_year.empty:
                full_dataset = pd.concat([full_dataset, data_year], ignore_index=True)
                # Check if we found all our target SAMs
                current_sams = set(full_dataset['SAM'].unique())
                target_sams = set(example_sam_ids.values())
                if target_sams.issubset(current_sams):
                    print(f"Found all target SAMs by year {year}")
                    break
        except Exception as e:
            print(f"Could not load data for year {year}: {e}")
            continue
    
    if full_dataset.empty:
        raise ValueError("No data could be loaded")
    
    print(f"Total soundings loaded: {len(full_dataset)}")
    print(f"Unique SAMs available: {len(full_dataset['SAM'].unique())}")
    
    # Check which example SAMs are available
    available_sams = full_dataset['SAM'].unique()
    for class_id, sam_id in example_sam_ids.items():
        if sam_id not in available_sams:
            print(f"Warning: SAM {sam_id} for class {class_id} not found in data")
            # Find an alternative from the same class
            try:
                labels_path = config.LABELS_DIR / 'Swath_Bias_labels.csv'
                labels_df = pd.read_csv(labels_path)
                alternatives = labels_df[labels_df['label'] == class_id]['identifier'].values
                for alt in alternatives:
                    if alt in available_sams:
                        print(f"Using alternative SAM {alt} for class {class_id}")
                        example_sam_ids[class_id] = alt
                        break
            except Exception as e:
                print(f"Could not load labels file: {e}")
                print(f"Keeping original SAM {sam_id} even though not found")
    
    # Define consistent color scale across all panels
    all_xco2_data = []
    for class_id, sam_id in example_sam_ids.items():
        sam_data = full_dataset[full_dataset['SAM'] == sam_id]
        if not sam_data.empty and 'xco2' in sam_data.columns:
            all_xco2_data.extend(sam_data['xco2'].dropna().values)


    
    # Create temporary directory for individual SAM plots using config
    temp_dir = config.project_root / 'tmp' / 'individual_sam_plots'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate individual SAM plots using plot_SAM function
    temp_image_paths = []
    
    for class_id, sam_id in example_sam_ids.items():
        print(f"Creating plot for {class_names[class_id]} ({sam_id})...")
        
        # Get data for this SAM
        sam_data = full_dataset[full_dataset['SAM'] == sam_id]
        
        if sam_data.empty:
            print(f"Warning: No data found for SAM {sam_id}")
            continue

        # Use plot_SAM function from main_util.py
        temp_filename = f"sam_{class_id}_{sam_id}"
        
        # Calculate percentiles for the xco2 data
        xco2_data = sam_data['xco2'].dropna()
        if len(xco2_data) > 0:
            vmin_val = np.round(np.nanpercentile(xco2_data, 10), 0)
            vmax_val = vmin_val + 5
        else:
            vmin_val = 400  # Default values if no data
            vmax_val = 405
            
        plot_SAM(
            data=sam_data, 
            var='xco2',
            vmin=vmin_val,
            vmax=vmax_val,
            save_fig=True, 
            name=temp_filename,
            path=str(temp_dir) + '/',
            qf=None,
            title_addition=f'\n{class_names[class_id]}'
        )
        
        # Store the path to the generated image
        temp_image_path = temp_dir / f"{sam_id}_xco2_{temp_filename}.png"
        temp_image_paths.append(temp_image_path)
    
    # Combine the individual plots into a three-panel figure
    if len(temp_image_paths) == 3:
        # Load the individual images
        images = []
        for img_path in temp_image_paths:
            if os.path.exists(img_path):
                img = Image.open(img_path)
                images.append(img)
            else:
                print(f"Warning: Could not find image {img_path}")
        
        if len(images) == 3:
            # Create the combined figure with tighter spacing
            fig, axes = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={'wspace': 0.0, 'hspace': 0.0})
            
            for i, (img, class_id) in enumerate(zip(images, example_sam_ids.keys())):
                axes[i].imshow(img)
                axes[i].axis('off')
                axes[i].set_title(class_names[class_id], fontsize=14, weight='bold', pad=2)
            
            plt.subplots_adjust(wspace=0.0, hspace=0.0, left=0.01, right=0.99, top=0.95, bottom=0.02)
            
            # Save the combined figure
            output_dir = config.FIGURES_DIR / "swath_bias_examples"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / 'swath_bias_examples_figure.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Combined figure saved to: {output_path}")
            
            # Also save as PDF for paper
            output_path_pdf = output_dir / 'swath_bias_examples_figure.pdf'
            plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
            print(f"Combined figure also saved as PDF: {output_path_pdf}")
            
            # Clean up temporary files
            for img_path in temp_image_paths:
                if os.path.exists(img_path):
                    os.remove(img_path)
            
            # Display the figure
            plt.show()
            
            return fig
        else:
            print("Error: Could not load all individual SAM images")
            return None
    else:
        print("Error: Could not generate all individual SAM plots")
        return None


if __name__ == '__main__':
    # Create the tmp directory if it doesn't exist  
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    os.makedirs(os.path.join(project_root, 'tmp'), exist_ok=True)
    
    # Create the figure
    try:
        fig = create_swath_bias_examples_figure()
        print("Successfully created swath bias examples figure!")
    except Exception as e:
        print(f"Error creating figure: {e}")
        import traceback
        traceback.print_exc() 