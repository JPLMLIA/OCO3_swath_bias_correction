#!/usr/bin/env python3
"""
Centralized Path Configuration for OCO-3 Swath Bias Correction

This file contains all path configurations, version strings, and directory structures
used throughout the project. Update this file to run new experiments or change
data locations without modifying individual scripts.

For public use, configure your paths in the User configuration section below.
Defaults: project_root/data/input (Lite files) and project_root/data/output (processed files).

Usage:
    from src.utils.config_paths import PathConfig
    config = PathConfig()
    model_path = config.get_model_path()
"""

import os
from pathlib import Path


# =============================================================
# User configuration
# -------------------------------------------------------------
# Set these to absolute paths to override defaults.
# Leave as None to use project-root defaults:
#   - Lite files:   <project_root>/data/input
#   - Output base:  <project_root>/data/output
USER_DATA_DIR = None       # e.g., "/path/to/your/oco3/lite/files"
USER_OUTPUT_DIR = None     # e.g., "/path/to/output/directory"

# Optional: load local overrides if present (not tracked in git)
try:
    from . import config_local as _config_local  # type: ignore
    USER_DATA_DIR = getattr(_config_local, 'USER_DATA_DIR', USER_DATA_DIR)
    USER_OUTPUT_DIR = getattr(_config_local, 'USER_OUTPUT_DIR', USER_OUTPUT_DIR)
except Exception:
    _config_local = None


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
        self.EXPERIMENT_NAME = "Swath_BC_" + self.MODEL_VERSION + "_06"
        
        # Production processing version
        self.PROCESSING_VERSION = self.MODEL_VERSION
        
        # =================================================================
        # DATA INPUT PATHS - UPDATE THESE FOR DIFFERENT DATA LOCATIONS
        # =================================================================
        
        # OCO-3 Lite files (input data) - Use user setting or project default
        self.LITE_FILES_DIR = str(USER_DATA_DIR or (self.project_root / "data" / "input"))
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
        
        # Output processed files - Use user setting or project default
        self.OUTPUT_BASE_DIR = str(USER_OUTPUT_DIR or (self.project_root / "data" / "output"))
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

    def validate_paths(self):
        """Validate that required paths exist and warn about missing ones"""
        issues = []
        
        # Check input data directory
        if not os.path.exists(self.LITE_FILES_DIR):
            issues.append(f"Input data directory not found: {self.LITE_FILES_DIR}")
            
        # Check if labels file exists
        if not self.LABELS_FILE.exists():
            issues.append(f"Labels file not found: {self.LABELS_FILE}")
            
        return issues


# =================================================================
# CONVENIENCE FUNCTIONS FOR BACKWARD COMPATIBILITY
# =================================================================

def get_default_config():
    """Get the default path configuration"""
    return PathConfig()

def get_model_path():
    """Quick access to model path"""
    return get_default_config().get_model_path()

def get_processed_data_dir():
    """Quick access to processed data directory"""
    return get_default_config().PROCESSED_FINAL_DIR

def get_output_dir():
    """Quick access to output directory"""
    return get_default_config().OUTPUT_FULL_DIR


# =================================================================
# MIGRATION HELPER
# =================================================================

def update_experiment_config(model_version=None, experiment_name=None, processing_version=None):
    """
    Helper function to easily update experiment configuration
    
    Example:
        update_experiment_config(
            model_version="v4.0",
            experiment_name="Swath_BC_v4.0_NewFeatures",
            processing_version="v4.0"
        )
    """
    config = PathConfig()
    
    if model_version:
        config.MODEL_VERSION = model_version
    if experiment_name:
        config.EXPERIMENT_NAME = experiment_name
    if processing_version:
        config.PROCESSING_VERSION = processing_version
    
    # Regenerate dependent paths
    config.__init__(config.project_root)
    
    return config


if __name__ == "__main__":
    # Demo the configuration
    config = PathConfig()
    config.print_config_summary()
    
    print("\nCreating output directories...")
    config.ensure_output_dirs()
    print("✅ All directories created successfully!") 