#!/usr/bin/env python3
"""
Comprehensive Analysis Runner for OCO-3 Swath Bias Correction

This script runs the complete analysis pipeline after swath bias correction processing.
It can run all analysis scripts or selective subsets based on user preferences.

The only script that is not run by this script is visualize_swath_bc_results.py, which is run manually.
This scripts makes a fiure for each SAM and takes 12h to run!

Prerequisites:
    - Activate the OCO3_bias conda environment first: conda activate OCO3_bias
    - Ensure trained model exists at: data/models/Swath_BC_v3.0_FinalBest/
    - Ensure processed data exists at: data/processed/Swath_BC_v3.0_FinalBest/

Usage:
    conda activate OCO3_bias
    python -m src.analysis.run_comprehensive_analysis --all
    python -m src.analysis.run_comprehensive_analysis --shap --plots --comparisons
    python -m src.analysis.run_comprehensive_analysis --help
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import centralized configuration
from src.utils.config_paths import PathConfig

class AnalysisRunner:
    """Manages running comprehensive analysis scripts"""
    
    def __init__(self, python_env_path=None):
        self.project_root = PROJECT_ROOT
        # Use the specified Python environment from the rules
        if python_env_path is None:
            python_env_path = "/Users/smauceri/opt/anaconda3/envs/OCO3_bias/bin/python"
        self.python_env = python_env_path
        
        # Use centralized configuration for paths
        self.config = PathConfig()
        self.model_dir = self.config.MODEL_EXPERIMENT_DIR
        self.processed_dir = self.config.PROCESSED_EXPERIMENT_DIR
        
        # Define available analysis scripts
        self.analysis_scripts = {
            'shap': {
                'script': 'src.analysis.run_rf_shap_analysis',
                'description': 'SHAP analysis for Random Forest model interpretability',
                'estimated_time': '5-10 minutes'
            },
            'bias_plots': {
                'script': 'src.analysis.generate_improved_bias_plots', 
                'description': 'Generate improved bias rate plots with better statistics',
                'estimated_time': '2-5 minutes'
            },
            'shap_core': {
                'script': 'src.analysis.rf_shap_analysis',
                'description': 'Core SHAP analysis implementation (advanced options)',
                'estimated_time': '10-20 minutes'
            },
            'evaluation_plots': {
                'script': 'src.analysis.evaluation_analysis_plots',
                'description': 'Comprehensive evaluation analysis with model performance metrics',
                'estimated_time': '5-15 minutes'
            },
            'bias_examples': {
                'script': 'src.analysis.swath_bias_examples_figure',
                'description': 'Generate publication-quality swath bias examples figure',
                'estimated_time': '3-8 minutes'
            },
        }
        
        # Define analysis groups
        self.analysis_groups = {
            'core': ['shap', 'bias_plots', 'evaluation_plots'],
            'plots': ['bias_plots', 'simple_bias_plots', 'regenerate_plots', 'bias_examples'],
            'validation': ['shap_core'],
            'publication': ['bias_examples', 'evaluation_plots', 'shap'],
            'all': list(self.analysis_scripts.keys())
        }

    def run_script(self, script_module, script_name):
        """Run a single analysis script"""
        print(f"\n{'='*60}")
        print(f"Running: {script_name}")
        print(f"Module: {script_module}")
        print(f"Expected time: {self.analysis_scripts[script_name]['estimated_time']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Check if this script needs special parameters
            if script_name == 'shap':
                # For SHAP wrapper, add parameters to make it work with reduced samples for speed
                cmd = [self.python_env, '-m', script_module]
            elif script_name == 'shap_core':
                # For core SHAP, add parameters
                cmd = [self.python_env, '-m', script_module]
            else:
                # Standard module execution
                cmd = [self.python_env, '-m', script_module]
            
            print(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, 
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ SUCCESS: {script_name} completed in {elapsed_time:.1f}s")
                if result.stdout.strip():
                    print("Output:")
                    print(result.stdout)
            else:
                print(f"❌ FAILED: {script_name} failed after {elapsed_time:.1f}s")
                print("Error output:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ TIMEOUT: {script_name} exceeded 30 minutes")
            return False
        except Exception as e:
            print(f"💥 ERROR: {script_name} crashed with exception: {e}")
            return False
            
        return True

    def check_data_requirements(self):
        """Check if required data directories exist"""
        missing = []
        
        if not self.model_dir.exists():
            missing.append(f"Model directory: {self.model_dir}")
        if not self.processed_dir.exists():
            missing.append(f"Processed data directory: {self.processed_dir}")
            
        if missing:
            print("⚠️  Missing required data directories:")
            for item in missing:
                print(f"   {item}")
            print(f"\nCurrent experiment configuration:")
            print(f"   Experiment: {self.config.EXPERIMENT_NAME}")
            print(f"   Model Version: {self.config.MODEL_VERSION}")
            print(f"   Expected model: {self.config.get_model_path()}")
            print(f"\nPlease run the training pipeline first:")
            print(f"   python -m src.modeling.Swath_BC_v3")
            print(f"\nOr update the experiment configuration in src/utils/config_paths.py")
            return False
        return True

    def run_analysis_group(self, group_name, skip_failed=True):
        """Run a group of analysis scripts"""
        if group_name not in self.analysis_groups:
            print(f"Unknown analysis group: {group_name}")
            print(f"Available groups: {list(self.analysis_groups.keys())}")
            return False
            
        # Check data requirements
        if not self.check_data_requirements():
            return False
            
        scripts_to_run = self.analysis_groups[group_name]
        total_scripts = len(scripts_to_run)
        successful = 0
        failed = []
        
        print(f"\n🚀 Starting analysis group: {group_name}")
        print(f"📋 Scripts to run: {total_scripts}")
        print(f"🐍 Python environment: {self.python_env}")
        print(f"🔬 Experiment: {self.config.EXPERIMENT_NAME}")
        print(f"📁 Model data: {self.model_dir}")
        print(f"📁 Processed data: {self.processed_dir}")
        print(f"📊 Results output: {self.config.FIGURES_DIR}")
        
        for i, script_name in enumerate(scripts_to_run, 1):
            print(f"\n📊 Progress: {i}/{total_scripts}")
            
            script_info = self.analysis_scripts[script_name]
            success = self.run_script(script_info['script'], script_name)
            
            if success:
                successful += 1
                print(f"✅ {script_name}: SUCCESS")
            else:
                failed.append(script_name)
                print(f"❌ {script_name}: FAILED")
                
                if not skip_failed:
                    print("Stopping execution due to failure (--no-skip-failed)")
                    break
        
        # Summary
        print(f"\n{'='*60}")
        print(f"📈 ANALYSIS GROUP SUMMARY: {group_name}")
        print(f"{'='*60}")
        print(f"✅ Successful: {successful}/{total_scripts}")
        print(f"❌ Failed: {len(failed)}/{total_scripts}")
        
        if failed:
            print(f"Failed scripts: {', '.join(failed)}")
        
        success_rate = successful / total_scripts * 100
        print(f"📊 Success rate: {success_rate:.1f}%")
        
        return len(failed) == 0

    def list_available_analyses(self):
        """List all available analysis scripts and groups"""
        print("📋 Available Analysis Scripts:")
        print("="*50)
        
        for name, info in self.analysis_scripts.items():
            print(f"🔹 {name}")
            print(f"   Description: {info['description']}")
            print(f"   Est. time: {info['estimated_time']}")
            print()
        
        print("📋 Available Analysis Groups:")
        print("="*50)
        
        for group_name, scripts in self.analysis_groups.items():
            print(f"🔸 {group_name}: {', '.join(scripts)}")
        print()

def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive analysis for OCO-3 swath bias correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                    # Run all analysis scripts
  %(prog)s --core                   # Run core analysis (SHAP, plots, evaluation)
  %(prog)s --plots                  # Run all plotting scripts
  %(prog)s --publication            # Generate publication-ready figures
  %(prog)s --section54              # Run Section 5.4 global impact analysis
  %(prog)s --validation             # Run validation scripts
  %(prog)s --shap --bias-examples    # Run specific scripts
  %(prog)s --global-impact          # Run global impact analysis only
  %(prog)s --list                   # List available analyses
        """
    )
    
    # Group options
    parser.add_argument('--all', action='store_true',
                       help='Run all available analysis scripts')
    parser.add_argument('--core', action='store_true',
                       help='Run core analysis (SHAP, plots, evaluation)')
    parser.add_argument('--plots', action='store_true', 
                       help='Run all plotting scripts')
    parser.add_argument('--validation', action='store_true',
                       help='Run validation scripts (RF test, core SHAP)')
    parser.add_argument('--publication', action='store_true',
                        help='Generate publication-ready figures')
    parser.add_argument('--section54', action='store_true',
                        help='Run Section 5.4 global impact analysis')
    
    # Individual script options
    parser.add_argument('--shap', action='store_true',
                       help='Run SHAP analysis')
    parser.add_argument('--bias-plots', action='store_true',
                       help='Generate improved bias plots')
    parser.add_argument('--simple-bias-plots', action='store_true',
                       help='Generate simple bias plots')
    parser.add_argument('--regenerate-plots', action='store_true',
                       help='Regenerate bias plots')
    parser.add_argument('--shap-core', action='store_true',
                       help='Run core SHAP analysis implementation')
    parser.add_argument('--evaluation-plots', action='store_true',
                       help='Generate comprehensive evaluation plots')
    parser.add_argument('--bias-examples', action='store_true',
                       help='Generate publication bias examples figure')
    parser.add_argument('--global-impact', action='store_true',
                       help='Run global impact analysis for Section 5.4')
    
    # Control options
    parser.add_argument('--list', action='store_true',
                       help='List available analysis scripts and exit')
    parser.add_argument('--no-skip-failed', action='store_true',
                       help='Stop execution if any script fails (default: continue)')
    parser.add_argument('--python-env', type=str,
                       help='Path to Python executable (default: uses current environment)')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = AnalysisRunner(python_env_path=args.python_env)
    
    # Handle list option
    if args.list:
        runner.list_available_analyses()
        return
    
    # Determine what to run
    skip_failed = not args.no_skip_failed
    
    # Check for group options
    groups_to_run = []
    if args.all:
        groups_to_run.append('all')
    if args.core:
        groups_to_run.append('core')
    if args.plots:
        groups_to_run.append('plots')
    if args.validation:
        groups_to_run.append('validation')
    if args.publication:
        groups_to_run.append('publication')
    if args.section54:
        groups_to_run.append('section54')
    
    # Check for individual script options
    individual_scripts = []
    script_mapping = {
        'shap': args.shap,
        'bias_plots': args.bias_plots,
        'simple_bias_plots': args.simple_bias_plots,
        'regenerate_plots': args.regenerate_plots,
        'shap_core': args.shap_core,
        'evaluation_plots': args.evaluation_plots,
        'bias_examples': args.bias_examples,
        'global_impact': getattr(args, 'global_impact', False),
    }
    
    for script_name, should_run in script_mapping.items():
        if should_run:
            individual_scripts.append(script_name)
    
    # Execute analysis
    if groups_to_run:
        # Run groups
        all_success = True
        for group in groups_to_run:
            print(f"\n🎯 Running analysis group: {group}")
            success = runner.run_analysis_group(group, skip_failed)
            all_success = all_success and success
            
        sys.exit(0 if all_success else 1)
        
    elif individual_scripts:
        # Run individual scripts
        successful = 0
        for script_name in individual_scripts:
            script_info = runner.analysis_scripts[script_name]
            success = runner.run_script(script_info['script'], script_name)
            if success:
                successful += 1
            elif not skip_failed:
                break
        
        print(f"\n📊 Individual Scripts Summary: {successful}/{len(individual_scripts)} successful")
        sys.exit(0 if successful == len(individual_scripts) else 1)
        
    else:
        # No options specified
        print("❗ No analysis specified. Use --help to see options.")
        print("💡 Quick start: python run_comprehensive_analysis.py --core")
        parser.print_help()

if __name__ == "__main__":
    main() 