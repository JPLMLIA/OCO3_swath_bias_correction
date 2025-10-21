#!/usr/bin/env python3
"""
Literature Re-analysis Template for Section 5.5

This template script helps re-analyze specific SAMs that were featured in 
published studies to show how swath bias correction would have affected their results.

Usage:
    python -m src.analysis.literature_reanalysis_template --sam-id fossil0123_45678
    python -m src.analysis.literature_reanalysis_template --study bell2023
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Import existing utility functions
from ..utils.main_util import plot_SAM, SAM_enhancement
from ..utils.config_paths import PathConfig
from .evaluation_analysis_plots import load_evaluation_data, create_sam_category_mapping

class LiteratureReanalyzer:
    """Re-analyzes SAMs from published studies with bias correction applied."""
    
    def __init__(self):
        self.config = PathConfig()
        self.output_dir = self.config.FIGURES_DIR / "literature_reanalysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define known problematic SAMs from literature
        self.literature_examples = {
            'bell2023': {
                'description': 'Bell et al. (2023) strong bias examples',
                'sams': [
                    # Add specific SAM IDs mentioned in Bell et al. 2023
                    # Example format: 'desert0001_12345'
                ],
                'expected_bias': 'High across-swath bias (~3 ppm)',
                'reference': 'Bell et al. (2023)'
            },
            'kiel2021': {
                'description': 'Kiel et al. (2021) Los Angeles case study',
                'sams': [
                    # Add LA SAM IDs from Kiel et al. 2021 if available
                ],
                'expected_bias': 'Moderate bias affecting urban enhancement',
                'reference': 'Kiel et al. (2021)'
            },
            'roten2023': {
                'description': 'Roten et al. (2023) emission quantification examples',
                'sams': [
                    # Add SAM IDs from Roten et al. 2023 if available
                ],
                'expected_bias': 'Bias affecting emission rate estimates',
                'reference': 'Roten et al. (2023)'
            }
        }
    
    def analyze_single_sam(self, sam_id, study_context=None):
        """
        Analyze a single SAM before and after bias correction.
        
        Args:
            sam_id: SAM identifier (e.g., 'fossil0123_45678')
            study_context: Optional context about which study featured this SAM
        """
        print(f"Analyzing SAM: {sam_id}")
        if study_context:
            print(f"Study context: {study_context}")
        
        # Load data
        try:
            data, _ = load_evaluation_data(str(self.config.PROCESSED_EXPERIMENT_DIR))
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
        
        # Filter for specific SAM
        sam_data = data[data['SAM'] == sam_id]
        
        if sam_data.empty:
            print(f"SAM {sam_id} not found in dataset")
            return None
        
        print(f"Found {len(sam_data)} soundings for SAM {sam_id}")
        
        # Calculate key metrics
        results = self._calculate_sam_metrics(sam_data, sam_id)
        
        # Create visualizations
        self._create_sam_visualizations(sam_data, sam_id, study_context)
        
        # Save results
        self._save_sam_results(results, sam_id, study_context)
        
        return results
    
    def _calculate_sam_metrics(self, sam_data, sam_id):
        """Calculate before/after metrics for a SAM."""
        
        results = {
            'sam_id': sam_id,
            'n_soundings': len(sam_data),
            'corrected': False,
            'enhancement_original': np.nan,
            'enhancement_corrected': np.nan,
            'enhancement_change': np.nan,
            'mean_correction': np.nan,
            'max_correction': np.nan,
            'std_correction': np.nan
        }
        
        # Check if correction was applied
        if 'xco2_swath-BC' in sam_data.columns and 'xco2' in sam_data.columns:
            correction_applied = (np.abs(sam_data['xco2_swath-BC'] - sam_data['xco2']) > 1e-6).any()
            results['corrected'] = correction_applied
            
            if correction_applied:
                # Calculate correction statistics
                correction_magnitude = sam_data['xco2_swath-BC'] - sam_data['xco2']
                results['mean_correction'] = correction_magnitude.mean()
                results['max_correction'] = correction_magnitude.abs().max()
                results['std_correction'] = correction_magnitude.std()
        
        # Calculate enhancement metrics
        try:
            enhancement_orig = SAM_enhancement(sam_data, 'xco2', qf=None, custom_SAM=True)
            results['enhancement_original'] = enhancement_orig
            
            if 'xco2_swath-BC' in sam_data.columns:
                enhancement_corr = SAM_enhancement(sam_data, 'xco2_swath-BC', qf=None, custom_SAM=True)
                results['enhancement_corrected'] = enhancement_corr
                
                if not (np.isnan(enhancement_orig) or np.isnan(enhancement_corr)):
                    results['enhancement_change'] = enhancement_corr - enhancement_orig
                    
        except Exception as e:
            print(f"Warning: Could not calculate enhancement for {sam_id}: {e}")
        
        return results
    
    def _create_sam_visualizations(self, sam_data, sam_id, study_context):
        """Create before/after visualizations for the SAM."""
        
        # Create SAM-specific output directory
        sam_dir = self.output_dir / f"sam_{sam_id}"
        sam_dir.mkdir(exist_ok=True)
        
        # Calculate colorbar range
        xco2_values = sam_data['xco2'].dropna()
        if len(xco2_values) == 0:
            return
            
        vmin = np.percentile(xco2_values, 10)
        vmax = np.percentile(xco2_values, 90)
        
        # Original XCO2 plot
        context_str = f" ({study_context})" if study_context else ""
        
        plot_SAM(sam_data, 'xco2',
                vmin=vmin, vmax=vmax,
                save_fig=True,
                name='_original',
                path=str(sam_dir),
                title_addition=f'Literature Re-analysis{context_str}\nOriginal XCO₂ - SAM: {sam_id}',
                simplified_title=True)
        
        # Bias-corrected XCO2 plot (if available)
        if 'xco2_swath-BC' in sam_data.columns:
            plot_SAM(sam_data, 'xco2_swath-BC',
                    vmin=vmin, vmax=vmax,
                    save_fig=True,
                    name='_corrected',
                    path=str(sam_dir),
                    title_addition=f'Literature Re-analysis{context_str}\nBias Corrected XCO₂ - SAM: {sam_id}',
                    simplified_title=True)
            
            # Difference plot
            sam_data_diff = sam_data.copy()
            sam_data_diff['correction_applied'] = sam_data_diff['xco2_swath-BC'] - sam_data_diff['xco2']
            
            # Only create difference plot if there's meaningful variation
            if sam_data_diff['correction_applied'].std() > 1e-6:
                plot_SAM(sam_data_diff, 'correction_applied',
                        vmin=-1, vmax=1,
                        save_fig=True,
                        name='_difference',
                        path=str(sam_dir),
                        title_addition=f'Literature Re-analysis{context_str}\nCorrection Applied - SAM: {sam_id}',
                        simplified_title=True)
        
        print(f"Visualizations saved to: {sam_dir}")
    
    def _save_sam_results(self, results, sam_id, study_context):
        """Save quantitative results for the SAM."""
        
        # Add study context to results
        if study_context:
            results['study_context'] = study_context
        
        # Convert to DataFrame and save
        results_df = pd.DataFrame([results])
        
        output_file = self.output_dir / f"results_{sam_id}.csv"
        results_df.to_csv(output_file, index=False)
        
        # Print summary
        print(f"\nResults for {sam_id}:")
        print(f"  Soundings: {results['n_soundings']}")
        print(f"  Correction applied: {results['corrected']}")
        
        if results['corrected']:
            print(f"  Mean correction: {results['mean_correction']:.3f} ppm")
            print(f"  Max correction: {results['max_correction']:.3f} ppm")
        
        if not np.isnan(results['enhancement_original']):
            print(f"  Original enhancement: {results['enhancement_original']:.3f} ppm m/s")
            
        if not np.isnan(results['enhancement_corrected']):
            print(f"  Corrected enhancement: {results['enhancement_corrected']:.3f} ppm m/s")
            
        if not np.isnan(results['enhancement_change']):
            print(f"  Enhancement change: {results['enhancement_change']:.3f} ppm m/s")
            change_percent = (results['enhancement_change'] / results['enhancement_original'] * 100) if results['enhancement_original'] != 0 else 0
            print(f"  Relative change: {change_percent:.1f}%")
        
        print(f"  Results saved to: {output_file}")
    
    def analyze_study_examples(self, study_key):
        """Analyze all examples from a specific study."""
        
        if study_key not in self.literature_examples:
            print(f"Unknown study key: {study_key}")
            print(f"Available studies: {list(self.literature_examples.keys())}")
            return
        
        study_info = self.literature_examples[study_key]
        print(f"Analyzing examples from: {study_info['reference']}")
        print(f"Description: {study_info['description']}")
        print(f"Expected bias: {study_info['expected_bias']}")
        
        if not study_info['sams']:
            print("No SAM IDs defined for this study yet.")
            print("Please add specific SAM IDs to the literature_examples dictionary.")
            return
        
        all_results = []
        
        for sam_id in study_info['sams']:
            print(f"\n{'='*60}")
            results = self.analyze_single_sam(sam_id, study_info['reference'])
            if results:
                all_results.append(results)
        
        # Save combined results
        if all_results:
            combined_df = pd.DataFrame(all_results)
            output_file = self.output_dir / f"study_results_{study_key}.csv"
            combined_df.to_csv(output_file, index=False)
            print(f"\nCombined results for {study_key} saved to: {output_file}")
    
    def list_available_studies(self):
        """List available study examples."""
        print("Available study examples for re-analysis:")
        print("="*50)
        
        for key, info in self.literature_examples.items():
            print(f"Study key: {key}")
            print(f"  Reference: {info['reference']}")
            print(f"  Description: {info['description']}")
            print(f"  SAMs defined: {len(info['sams'])}")
            print()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Literature Re-analysis for Section 5.5")
    parser.add_argument('--sam-id', type=str,
                       help='Specific SAM ID to analyze (e.g., fossil0123_45678)')
    parser.add_argument('--study', type=str,
                       help='Study key to analyze all examples from that study')
    parser.add_argument('--list-studies', action='store_true',
                       help='List available study examples')
    parser.add_argument('--context', type=str,
                       help='Additional context about the SAM being analyzed')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = LiteratureReanalyzer()
    
    if args.list_studies:
        analyzer.list_available_studies()
        return
    
    if args.sam_id:
        print("Literature Re-analysis: Individual SAM")
        print("=" * 50)
        analyzer.analyze_single_sam(args.sam_id, args.context)
        
    elif args.study:
        print("Literature Re-analysis: Study Examples")
        print("=" * 50)
        analyzer.analyze_study_examples(args.study)
        
    else:
        print("Please specify either --sam-id or --study")
        print("Use --list-studies to see available study examples")
        parser.print_help()


if __name__ == "__main__":
    main() 