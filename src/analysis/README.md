# Analysis Scripts for OCO-3 Swath Bias Correction

This directory contains analysis and visualization scripts for the OCO-3 swath bias correction project.

## Centralized Configuration System

**All analysis scripts now use a centralized path configuration system** (`src.utils.config_paths.PathConfig`) that eliminates hardcoded paths and version strings. This system:

- **Centralizes all path management** in a single configuration class
- **Eliminates hardcoded version strings** scattered across scripts
- **Enables easy experiment switching** by changing only 3 lines in `config_paths.py`
- **Provides consistent paths** across all analysis, processing, and modeling scripts
- **Supports multiple output formats** (models, processed data, figures, results)

### Key Configuration Parameters
```python
MODEL_VERSION = "v3.0"
EXPERIMENT_NAME = "Swath_BC_v3.0_FinalBest"  
PROCESSING_VERSION = "v3.0"
```

### Example Usage in Scripts
```python
from src.utils.config_paths import PathConfig

config = PathConfig()
model_path = config.get_model_path()
output_dir = config.OUTPUT_FIGURES_DIR / "my_analysis"
data_dir = config.PROCESSED_DATA_DIR
```

### Starting a New Experiment
To start a new experiment, edit only these 3 lines in `src/utils/config_paths.py`:
```python
MODEL_VERSION = "v4.0"
EXPERIMENT_NAME = "Swath_BC_v4.0_NewFeatures"
PROCESSING_VERSION = "v4.0" 
```
All scripts automatically use the new configuration without individual modifications.

## Available Scripts

### Core Analysis and Pipeline Control

1. **`run_comprehensive_analysis.py`** - Central runner for all analysis scripts
   - Groups: `--core`, `--plots`, `--validation`, `--publication`, `--all`
   - Individual script selection available
   - Error handling and logging
   - **Path**: Uses centralized config for model and data directories
   - **Usage**: `python -m src.analysis.run_comprehensive_analysis --core`

### Model Analysis and SHAP

2. **`rf_shap_analysis.py`** (80KB) - Comprehensive SHAP analysis implementation
   - SHAP value calculation and visualization
   - Feature importance analysis
   - Decision boundary analysis
   - 2D feature interaction analysis
   - **Paths**: Model and output directories from config
   - **Usage**: `python -m src.analysis.rf_shap_analysis`

3. **`run_rf_shap_analysis.py`** - SHAP analysis runner
   - Wrapper for comprehensive SHAP analysis
   - Publication-ready output management
   - **Paths**: Output directory from config with suffix support
   - **Usage**: `python -m src.analysis.run_rf_shap_analysis`

### Evaluation and Performance Analysis

4. **`evaluation_analysis_plots.py`** (59KB) - Comprehensive evaluation visualizations
   - Confusion matrix analysis
   - Performance metrics by SAM category
   - Spatial and temporal pattern analysis
   - Correction magnitude analysis
   - **Paths**: Processed data and figures directories from config
   - **Usage**: `python -m src.analysis.evaluation_analysis_plots`

### Feature Selection and Model Optimization

5. **`feature_selection_analyzer.py`** (25KB) - Comprehensive feature selection analysis
   - Analyzes feature selection results from machine learning experiments
   - Feature stability analysis across multiple trials
   - Feature combination and co-occurrence patterns
   - Fold-level feature selection patterns
   - Generates recommendations with confidence levels
   - **Usage**: `python -m src.analysis.feature_selection_analyzer --input_file results.json`

6. **`extract_recommended_features.py`** - Feature recommendation extractor
   - Extracts recommended features from analysis results
   - Multiple output formats (list, python, json, summary, config)
   - Confidence level filtering
   - **Usage**: `python -m src.analysis.extract_recommended_features --input_file analysis.json --format summary`

7. **`classifier_threshold_analysis.py`** - Classifier feature response analysis
   - Creates plots showing relationship between feature values and prediction probability
   - Uses real data points for analysis
   - Decision threshold visualization
   - **Usage**: `python -m src.analysis.classifier_threshold_analysis`

### Results Visualization and Publication Figures

8. **`visualize_swath_bc_results.py`** - Results visualization
   - SAM-by-SAM plotting of corrected vs original data
   - Statistical summaries
   - **Paths**: Output data directory from config
   - **Usage**: Processes files from `config.OUTPUT_FULL_DIR`

9. **`swath_bias_examples_figure.py`** - Publication figure generation
   - Three-panel figure showing bias classes
   - Representative SAM examples for paper
   - High-resolution PNG and PDF output
   - **Paths**: Labels and output directories from config
   - **Usage**: `python -m src.analysis.swath_bias_examples_figure`

10. **`generate_improved_bias_plots.py`** - Enhanced bias visualizations
    - Improved statistical plotting
    - Better sample size handling
    - **Paths**: Model and results directories from config
    - **Usage**: `python -m src.analysis.generate_improved_bias_plots`

### Paper Statistics and Publication Support

11. **`generate_paper_stats.py`** (8.8KB) - Final paper statistics generation
    - Processes full OCO-3 data record to quantify swath bias correction impact
    - Uses `swath_bias_corrected` flag in Lite files as ground truth
    - Generates global SAM summary for mapping
    - Calculates enhancement impact statistics for corrected SAMs
    - **Output**: `global_sam_summary.parquet` and `full_dataset_enhancement_impact_stats.csv`
    - **Usage**: `python -m src.analysis.generate_paper_stats --full-dataset`

12. **`literature_reanalysis_template.py`** (12KB) - Literature study re-analysis
    - Template for re-analyzing SAMs featured in published studies
    - Shows how swath bias correction would have affected published results
    - Before/after comparison visualizations
    - Supports specific studies (Bell et al. 2023, Kiel et al. 2021, Roten et al. 2023)
    - **Usage**: `python -m src.analysis.literature_reanalysis_template --sam-id fossil0123_45678`

## Path Migration Benefits

### Before (Scattered Hardcoded Paths)
- `"data/models/Swath_BC_v3.0_FinalBest"` in multiple files
- `"results/figures/"` hardcoded throughout scripts  
- Version strings (`"v3.0"`) scattered across codebase
- Manual path updates needed for new experiments

### After (Centralized Configuration)
- All paths defined once in `config_paths.py`
- Automatic path generation based on experiment config
- Version-agnostic script implementation
- Easy experiment switching with 3-line config change

### Migration Impact
✅ **All 12 analysis scripts updated** to use centralized configuration  
✅ **Zero hardcoded paths** remaining in analysis directory  
✅ **Future-proof design** for new experiments and versions  
✅ **Consistent path resolution** across entire analysis pipeline

## Detailed Script Documentation

### Evaluation Analysis Plots (`evaluation_analysis_plots.py`)

This script generates a comprehensive suite of plots for evaluating the performance of the `Swath_BC_v3.py` model. It is designed to work with the new fold-based data structure.

**Key Features**:
- **Data Loading**: Reads from the fold structure saved by `Swath_BC_v3.py`, loading individual fold results from `final_best_config_fold_X/` directories.
- **Cross-Validation Integration**: Loads results from all CV folds automatically, combines fold predictions into a unified analysis, and derives confusion matrix categories (TP/TN/FP/FN).
- **Comprehensive Plotting**: Generates a wide range of plots for performance metrics, geographic analysis, temporal patterns, and feature analysis.
- **Compatibility**: Handles different column naming conventions and can work with older data formats if available.

**Expected Data Structure**:
```
data/processed/Swath_BC_v3.0_FinalBest/
├── cv_summary.json
├── final_best_config_fold_1/
│   ├── fold_predictions.csv
│   └── fold_metadata.json
└── ... (more folds)
```

**Available Plots**:
You can generate all plots or specify individual ones. Key plots include:
- `dataset_statistics`
- `confusion_matrix_heatmap`
- `errors_vs_predictor`
- `performance_by_sam_category`
- `spatial_error_distribution`
- `correction_magnitude_distributions`
- `f1_score_by_sam_category_standalone`
- `enhancement_analysis`

**Output**:
- All plots are saved in `results/figures/evaluation_analysis/`

### Swath Bias Examples Figure (`swath_bias_examples_figure.py`)

This script generates Figure X for the paper, showing examples of each swath bias classification class.

**Purpose**: Creates a publication-quality figure with three panels showing:
1. **Class 0 (No Swath Bias)**: SAM with no visually apparent swath-level discontinuities
2. **Class 1 (Swath Bias)**: SAM with obvious and systematic XCO₂ differences between adjacent swaths  
3. **Class 2 (Uncertain)**: SAM where the presence of swath bias is ambiguous

**Features**:
- Uses the `plot_SAM` function from `main_util.py` for consistency
- OCO-3 XCO₂ retrievals as colored footprints overlaid on maps
- Target locations marked with red stars
- Consistent color scale across all panels
- Clean layout suitable for paper inclusion

**Representative SAMs**:
- **Class 0**: `fossil0001_19513` - Clean fossil fuel target with no apparent swath bias
- **Class 1**: `fossil0008_10293` - Fossil fuel target with clear swath bias artifacts  
- **Class 2**: `fossil0003_13217` - Fossil fuel target with uncertain/ambiguous swath patterns

**Output**:
- PNG: `results/figures/swath_bias_examples/swath_bias_examples_figure.png` (300 DPI)
- PDF: `results/figures/swath_bias_examples/swath_bias_examples_figure.pdf` (vector format)

### Feature Selection Analysis (`feature_selection_analyzer.py`)

This script provides comprehensive analysis of feature selection results from machine learning experiments.

**Key Features**:
- **Stability Analysis**: Calculates how frequently features are selected across multiple trials
- **Combination Analysis**: Identifies common feature pairs and sets
- **Fold-level Patterns**: Analyzes consistency within cross-validation folds
- **Confidence Scoring**: Assigns confidence levels (HIGH/MEDIUM/LOW) to recommendations
- **Visualization**: Creates frequency plots and stability charts

**Input Format**: Expects JSON files with `analysis_summary` and `detailed_trial_data` keys

**Output**: 
- Detailed analysis reports
- Feature recommendation lists with confidence scores
- Visualization plots for publication

### Generate Paper Stats (`generate_paper_stats.py`)

This script replaces the older `global_impact_analysis.py` and provides final statistics for the paper.

**Key Functions**:
- **Global Analysis**: Processes the complete OCO-3 data record
- **Correction Quantification**: Uses `swath_bias_corrected` flag as ground truth
- **Geographic Summary**: Creates global SAM summary for mapping applications
- **Enhancement Impact**: Calculates detailed statistics on emission enhancement proxy changes

**Two Primary Outputs**:
1. `global_sam_summary.parquet` - Every unique SAM with correction status and location
2. `full_dataset_enhancement_impact_stats.csv` - Detailed impact statistics for corrected SAMs

**Requirements**:
- Access to final corrected NetCDF files
- Threading support for parallel processing
- Sufficient memory for large dataset processing

## Usage

Most scripts can be run directly from the project root directory:

```bash
# From the code/ directory
python -m src.analysis.run_rf_shap_analysis
python -m src.analysis.rf_shap_analysis --help
python -m src.analysis.generate_improved_bias_plots
python -m src.analysis.evaluation_analysis_plots
python -m src.analysis.swath_bias_examples_figure
python -m src.analysis.generate_paper_stats --full-dataset
python -m src.analysis.feature_selection_analyzer --input_file results.json
python -m src.analysis.extract_recommended_features --input_file analysis.json --format summary
python -m src.analysis.classifier_threshold_analysis
python -m src.analysis.literature_reanalysis_template --sam-id fossil0123_45678
# etc.
```

Or use the comprehensive analysis runner:

```bash
python -m src.analysis.run_comprehensive_analysis --core
python -m src.analysis.run_comprehensive_analysis --shap --bias-plots
```

## Script Categories

### Core Analysis & SHAP
- **`rf_shap_analysis.py`** - Comprehensive SHAP analysis with detailed feature importance
- **`run_rf_shap_analysis.py`** - Runner script for SHAP analysis
- **`evaluation_analysis_plots.py`** - Model performance evaluation and visualization

### Feature Selection & Model Optimization
- **`feature_selection_analyzer.py`** - Comprehensive feature selection analysis
- **`extract_recommended_features.py`** - Feature recommendation extraction
- **`classifier_threshold_analysis.py`** - Feature response analysis

### Bias Analysis & Visualization
- **`generate_improved_bias_plots.py`** - Enhanced bias rate plotting
- **`swath_bias_examples_figure.py`** - Publication-quality bias examples
- **`visualize_swath_bc_results.py`** - SAM-specific results visualization

### Publication & Paper Support
- **`generate_paper_stats.py`** - Final paper statistics and global impact analysis
- **`literature_reanalysis_template.py`** - Re-analysis of published study examples

### Pipeline Control
- **`run_comprehensive_analysis.py`** - Central runner for coordinated analysis

## Requirements

All scripts require:
- **Fold Results**: Output from `Swath_BC_v3.py` in `data/processed/Swath_BC_v3.0_FinalBest/`
- **SAM Data**: Raw SAM data for multiple years (2019-2024)
- **Labels**: `data/labels/Swath_Bias_labels.csv`
- **Model Files**: Trained Random Forest models from the modeling pipeline

## Note

These scripts are primarily for research and development. For production bias correction, use the main pipeline in `src/modeling/Swath_BC_v3.py`. 