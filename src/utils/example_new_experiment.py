#!/usr/bin/env python3
"""
Example: How to Set Up a New Experiment Configuration

This script demonstrates how easy it is to run a new experiment with different
version numbers, model names, or data locations using the centralized configuration.

To start a new experiment, you only need to:
1. Update the configuration values in config_paths.py
2. All scripts will automatically use the new paths
"""

from config_paths import PathConfig, update_experiment_config

def demonstrate_current_config():
    """Show the current configuration"""
    print("="*60)
    print("CURRENT EXPERIMENT CONFIGURATION")
    print("="*60)
    
    config = PathConfig()
    config.print_config_summary()
    
    return config

def demonstrate_new_experiment():
    """Demonstrate setting up a new experiment"""
    print("\n" + "="*60)
    print("EXAMPLE: NEW EXPERIMENT CONFIGURATION")
    print("="*60)
    
    # Example: Set up a new experiment for version 4.0
    new_config = update_experiment_config(
        model_version="v4.0",
        experiment_name="Swath_BC_v4.0_NewFeatures",
        processing_version="v4.0"
    )
    
    new_config.print_config_summary()
    
    print("\nWith this new configuration:")
    print(f"- Model will be saved to: {new_config.get_model_path()}")
    print(f"- Processed data will go to: {new_config.PROCESSED_FINAL_DIR}")
    print(f"- Output files will be in: {new_config.OUTPUT_FULL_DIR}")
    print(f"- Results figures will be in: {new_config.FIGURES_DIR}")
    
    return new_config

def show_easy_migration_steps():
    """Show how easy it is to migrate to a new experiment"""
    print("\n" + "="*60)
    print("MIGRATION STEPS FOR NEW EXPERIMENT")
    print("="*60)
    print("""
To start a new experiment:

1. Edit src/utils/config_paths.py:
   - Change MODEL_VERSION = "v4.0"
   - Change EXPERIMENT_NAME = "Swath_BC_v4.0_NewFeatures"
   - Change PROCESSING_VERSION = "v4.0"

2. Run your experiment:
   python -m src.modeling.Swath_BC_v3  # Will use new paths automatically
   
3. Process data:
   python -m src.processing.apply_swath_bc_RF  # Will use new paths automatically
   
4. Analyze results:
   python -m src.analysis.run_comprehensive_analysis --core  # Will use new paths automatically

That's it! All scripts automatically use the new configuration.

OLD WAY (bad):
- Update paths in 15+ different scripts
- Risk missing some hardcoded paths
- Inconsistent naming across scripts
- Hard to track what experiment you're running

NEW WAY (good):
- Update 3 lines in 1 file
- All scripts automatically consistent
- Clear experiment tracking
- Easy to switch between experiments
""")

if __name__ == "__main__":
    # Show current configuration
    current_config = demonstrate_current_config()
    
    # Show how to set up a new experiment
    new_config = demonstrate_new_experiment()
    
    # Show migration steps
    show_easy_migration_steps()
    
    print("\n" + "="*60)
    print("KEY BENEFITS OF CENTRALIZED CONFIGURATION")
    print("="*60)
    print("""
✅ Single source of truth for all paths
✅ Easy experiment versioning and tracking
✅ Automatic consistency across all scripts
✅ Quick switching between configurations
✅ Reduced errors from hardcoded paths
✅ Better reproducibility and documentation
✅ Easy collaboration (team uses same config)
✅ Automatic directory creation
✅ Clear experiment metadata tracking
""") 