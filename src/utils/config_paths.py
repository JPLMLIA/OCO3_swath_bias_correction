#!/usr/bin/env python3
"""
BACKUP: Original Path Configuration for OCO-3 Swath Bias Correction
This is a backup of the original config with hardcoded local paths.
Keep this file for local development/analysis but don't use in public version.

This file contains all path configurations, version strings, and directory structures
used throughout the project. Update this file to run new experiments or change
data locations without modifying individual scripts.

Usage:
    from src.utils.config_paths_original_backup import PathConfig
    config = PathConfig()
    model_path = config.get_model_path()
"""

import os
from pathlib import Path


class PathConfig:
    """Centralized configuration for all paths and versions in the project"""
    
    def __init__(self, project_root=None):
        # Project root directory
        if project_root is None:
            # Auto-detect project root (assumes config is in src/utils/)
            self.project_root = Path(__file__).resolve().parent.parent.parent
        else:
            self.project_root = Path(project_root)
            
        # =================================================================
        # EXPERIMENT CONFIGURATION - UPDATE THESE FOR NEW EXPERIMENTS
        # =================================================================
        
        # Model version and experiment name
        self.MODEL_VERSION = "v4.0"
        self.EXPERIMENT_NAME = "Swath_BC_" + self.MODEL_VERSION + "_th_06"
        # Optional experiment suffix via environment variable to separate runs (e.g., QF0)
        suffix = os.getenv('OCO3_EXP_SUFFIX', '').strip()
        if suffix:
            self.EXPERIMENT_NAME = f"{self.EXPERIMENT_NAME}_{suffix}"
        
        # Production processing version
        self.PROCESSING_VERSION = self.MODEL_VERSION
        
        # =================================================================
        # DATA INPUT PATHS - UPDATE THESE FOR DIFFERENT DATA LOCATIONS
        # =================================================================
        
        # OCO-3 Lite files (input data) - ORIGINAL HARDCODED PATHS
        self.LITE_FILES_DIR = "/Volumes/OCO/LiteFiles/B11_OCO3"
        self.LITE_FILES_PATTERN = "oco3_LtCO2_*B11072Ar*.nc4"
        
        # Labels and reference data
        self.LABELS_FILE = self.project_root / "data" / "labels" / "Swath_Bias_labels.csv"
        self.TARGET_DATA_FILE = self.project_root / "tmp" / "clasp_report.csv"
        
        # =================================================================
        # PROJECT OUTPUT PATHS - AUTOMATICALLY GENERATED FROM ABOVE
        # =================================================================
        
        # Model artifacts
        self.MODEL_BASE_DIR = self.project_root / "data" / "models"
        self.MODEL_EXPERIMENT_DIR = self.MODEL_BASE_DIR / self.EXPERIMENT_NAME
        self.MODEL_FINAL_DIR = self.MODEL_EXPERIMENT_DIR / "final_model_all_data"
        
        # Processed data (CV results, predictions, etc.)
        self.PROCESSED_BASE_DIR = self.project_root / "data" / "processed"
        self.PROCESSED_EXPERIMENT_DIR = self.PROCESSED_BASE_DIR / self.EXPERIMENT_NAME
        self.PROCESSED_FINAL_DIR = self.PROCESSED_EXPERIMENT_DIR / "final_model_all_data"
        
        # Results and figures
        self.RESULTS_DIR = self.project_root / "results"
        self.FIGURES_DIR = self.RESULTS_DIR / "figures"
        
        # Intermediate data
        self.INTERMEDIATE_DIR = self.project_root / "data" / "intermediate"
        
        # Output processed files - ORIGINAL HARDCODED PATHS
        self.OUTPUT_BASE_DIR = "/Volumes/OCO/LiteFiles/B11_OCO3_SwathBiasCorrected"
        self.OUTPUT_VERSION_DIR = f"Lite_w_SwathBC_{self.PROCESSING_VERSION}"
        self.OUTPUT_FULL_DIR = os.path.join(self.OUTPUT_BASE_DIR, self.OUTPUT_VERSION_DIR)
        
        # =================================================================
        # MODEL AND FILE NAMES
        # =================================================================
        
        # Model file names
        self.RF_MODEL_FILENAME = "rf_model_classifier_with_jumps.joblib"
        self.MODEL_METADATA_FILENAME = "model_metadata.json"
        
        # Output file suffixes
        self.OUTPUT_FILE_SUFFIX = "_SwathBC.nc4"
        
        # =================================================================
        # ANALYSIS-SPECIFIC PATHS
        # =================================================================
        
        # SHAP analysis output
        self.SHAP_OUTPUT_DIR = self.FIGURES_DIR / "rf_shap_analysis_paper"
        
        # Bias plots output
        self.BIAS_PLOTS_DIR = self.FIGURES_DIR / "improved_bias_plots"
        
        # Evaluation analysis output
        self.EVALUATION_PLOTS_DIR = self.FIGURES_DIR / "evaluation_analysis"
        
        # Swath bias examples output
        self.BIAS_EXAMPLES_DIR = self.FIGURES_DIR / "swath_bias_examples"
        
        # Results visualization output
        self.RESULTS_VIZ_DIR = self.FIGURES_DIR / "swath_bc_visualization"

    # =================================================================
    # CONVENIENCE METHODS
    # =================================================================
    
    def get_model_path(self):
        """Get the full path to the trained RF model"""
        return self.MODEL_FINAL_DIR / self.RF_MODEL_FILENAME
    
    def get_model_metadata_path(self):
        """Get the path to model metadata"""
        return self.MODEL_FINAL_DIR / self.MODEL_METADATA_FILENAME
    
    def get_fold_predictions_path(self):
        """Get the path to fold predictions"""
        return self.PROCESSED_FINAL_DIR / "fold_predictions.csv"
    
    def get_lite_files_pattern(self):
        """Get the full pattern for finding Lite files"""
        return os.path.join(self.LITE_FILES_DIR, self.LITE_FILES_PATTERN)
    
    def get_output_file_path(self, input_filename):
        """Get the output path for a processed file"""
        base_name = os.path.basename(input_filename).replace('.nc4', '')
        output_filename = f"{base_name}{self.OUTPUT_FILE_SUFFIX}"
        return os.path.join(self.OUTPUT_FULL_DIR, output_filename)
    
    def ensure_output_dirs(self):
        """Create all output directories if they don't exist"""
        dirs_to_create = [
            self.MODEL_EXPERIMENT_DIR,
            self.MODEL_FINAL_DIR,
            self.PROCESSED_EXPERIMENT_DIR,
            self.PROCESSED_FINAL_DIR,
            self.RESULTS_DIR,
            self.FIGURES_DIR,
            self.INTERMEDIATE_DIR,
            self.SHAP_OUTPUT_DIR,
            self.BIAS_PLOTS_DIR,
            self.EVALUATION_PLOTS_DIR,
            self.BIAS_EXAMPLES_DIR,
            self.RESULTS_VIZ_DIR
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Also create the external output directory
        os.makedirs(self.OUTPUT_FULL_DIR, exist_ok=True)
    
    def get_experiment_info(self):
        """Get a summary of the current experiment configuration"""
        return {
            'model_version': self.MODEL_VERSION,
            'experiment_name': self.EXPERIMENT_NAME,
            'processing_version': self.PROCESSING_VERSION,
            'model_path': str(self.get_model_path()),
            'processed_data_path': str(self.PROCESSED_FINAL_DIR),
            'output_directory': self.OUTPUT_FULL_DIR,
            'input_data_pattern': self.get_lite_files_pattern()
        }
    
    def print_config_summary(self):
        """Print a summary of the current configuration"""
        print("="*60)
        print("OCO-3 SWATH BIAS CORRECTION - PATH CONFIGURATION")
        print("="*60)
        print(f"Model Version: {self.MODEL_VERSION}")
        print(f"Experiment: {self.EXPERIMENT_NAME}")
        print(f"Processing Version: {self.PROCESSING_VERSION}")
        print(f"Project Root: {self.project_root}")
        print(f"Input Data: {self.get_lite_files_pattern()}")
        print(f"Output Directory: {self.OUTPUT_FULL_DIR}")
        print(f"Model Path: {self.get_model_path()}")
        print(f"Results Directory: {self.FIGURES_DIR}")
        print("="*60) 