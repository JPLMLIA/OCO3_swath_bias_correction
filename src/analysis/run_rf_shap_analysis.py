#!/usr/bin/env python3
"""
Run comprehensive RF SHAP analysis for OCO-3 Swath Bias Correction
Enhanced version for publication-ready analysis.
"""

import subprocess
import sys
import argparse
from pathlib import Path

# Add config import
from src.utils.config_paths import PathConfig

def run_shap_analysis(max_samples=500, output_suffix=""):
    """Run SHAP analysis with specified parameters"""
    
    # Initialize config
    config = PathConfig()
    
    # Determine output directory using config
    output_dir = config.FIGURES_DIR / f"rf_shap_analysis{output_suffix}"
    
    cmd = [
        sys.executable, "-m", "src.analysis.rf_shap_analysis",
        "--max_samples", str(max_samples),
        "--output_dir", str(output_dir),
        "--save_figs"
    ]
    
    print(f"Running SHAP analysis with {max_samples} samples...")
    print(f"Output directory: {output_dir}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Analysis completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running analysis: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RF SHAP Analysis')
    parser.add_argument('--n_samples', type=int, default=500, 
                        help='Number of samples for SHAP analysis')
    parser.add_argument('--save_figs', action='store_true', 
                        help='Save figures')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (defaults to config path with _paper suffix)')
    
    args = parser.parse_args()
    
    print("OCO-3 Random Forest SHAP Analysis Runner")
    print("=" * 50)
    
    # Initialize config and determine output suffix
    config = PathConfig()
    
    if args.output_dir is None:
        output_suffix = "_paper"
    else:
        output_suffix = "_" + args.output_dir.split("_")[-1] if "_" in args.output_dir else "_paper"
    
    # Run main analysis for paper
    success = run_shap_analysis(max_samples=args.n_samples, output_suffix=output_suffix)
    
    if success:
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE!")
        print("=" * 50)
        output_base = config.FIGURES_DIR / f"rf_shap_analysis{output_suffix}"
        print("Key outputs for paper:")
        print(f"1. SHAP summary plots: {output_base}/shap_summary/")
        print(f"2. Feature dependence plots: {output_base}/shap_dependence/")
        print(f"3. Feature importance comparison: {output_base}/feature_comparison/")
        print(f"4. h_continuum_sco2 analysis: {output_base}/decision_analysis/")
        print(f"5. Paper summary: {output_base}/paper_summary.md")
        print("\nRecommended figures for paper:")
        print("- shap_summary_beeswarm.png (main SHAP summary)")
        print("- shap_dependence_max_relative_jump.png (most important feature)")
        print("- shap_dependence_h_continuum_sco2.png (threshold feature)")
        print("- feature_importance_comparison.png (SHAP vs RF importance)")
        print("- h_continuum_sco2_analysis.png (threshold analysis)")
    else:
        print("Analysis failed. Please check the error messages above.")
        sys.exit(1) 