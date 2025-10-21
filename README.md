# OCO-3 Swath Bias Correction

This repository contains code for detecting and correcting swath-dependent biases in OCO-3 Snapshot Area Map (SAM) observations using a Random Forest-based approach.


### Installation
```bash
git clone https://github.com/your-username/oco3-swath-bias-correction.git
cd oco3-swath-bias-correction
conda create -n oco3_bias python=3.9
conda activate oco3_bias
pip install -r requirements.txt
```

### Environment Setup
Set the following environment variables (or let the system use defaults):
```bash
# Required: Directory containing your OCO-3 Lite files
export OCO3_DATA_DIR="/path/to/your/oco3/lite/files"

# Optional: Output directory for processed files (defaults to ./data/output)
export OCO3_OUTPUT_DIR="/path/to/output/directory"
```


## 📁 Repository Structure

```
code/
├── src/                    # Source code
│   ├── analysis/          # Research, analysis, and visualization scripts
│   │   └── run_comprehensive_analysis.py # Central analysis runner
│   ├── data_preparation/  # Data loading and preprocessing
│   ├── evaluation_analysis/ # Model evaluation and metrics
│   ├── modeling/          # Core bias correction models
│   │   └── Swath_BC_v3.py # Main RF pipeline
│   ├── processing/        # Data processing utilities
│   │   └── apply_swath_bc_RF.py # Production processing
│   ├── tools/            # Supporting tools and utilities
│   └── utils/            # Core utility functions and configuration
│       ├── config_paths.py # Centralized path configuration
│       └── main_util.py   # Main utility functions

├── data/                  # Data storage (contents excluded from git)
│   ├── intermediate/     # Intermediate analysis data
│   ├── labels/          # Manual bias labels (1,723 SAMs) - included
│   ├── models/          # Trained model artifacts
│   └── processed/       # Cross-validation results
├── results/              # Analysis outputs (structure preserved)
│   └── figures/         # Generated plots and figures
├── docs/                # Documentation
│   └── DATA_REQUIREMENTS.md # Detailed data setup guide
├── tmp/                 # Temporary files (excluded from git)

# Configuration and setup files
├── config_example.py    # Configuration examples for different setups
├── setup.sh            # Automated environment setup script
├── use_original_config.py    # Switch to original config (development)
├── restore_public_config.py # Switch to public config (sharing)
├── requirements.txt    # Python dependencies
├── LICENSE            # MIT license
├── Paper.md          # Research paper draft
└── README.md         # This file
```

## Workflow and Usage

The project follows a three-phase workflow:

```
1. Model Training & Optimization    →    2. Data Processing    →    3. Analysis & Visualization
   (Development/Research)                 (Production)               (Results & Insights)
```

---

### 🔬 Phase 1: Model Training & Optimization

**Purpose**: Develop and optimize the Random Forest model for swath bias detection.

**Script**: `src/modeling/Swath_BC_v3.py`

**What it does**:
- Uses 1,279 labeled SAMs for training.
- Performs hyperparameter optimization using Optuna.
- Employs 4-fold cross-validation for robust evaluation.
- **NEW**: Implements a reordered pipeline (RF decision first, then targeted corrections).
- Saves the optimized model, CV results, and performance metrics.

**Usage**:
```bash
python -m src.modeling.Swath_BC_v3
```
**Output**: Trained model in `data/models/` with an F1-score of ~0.67.

---

### 🏭 Phase 2: Data Processing (Production)

**Purpose**: Apply the trained swath bias correction model to operational OCO-3 Lite files.

**Script**: `src/processing/apply_swath_bc_RF.py`

**What it does**:
- Loads the final trained RF model from Phase 1.
- Processes OCO-3 Lite files (`*.nc4`) in batch.
- Efficiently applies corrections only to SAMs identified by the RF model (~15% of total).
- Creates new bias-corrected NetCDF files with added variables:
  - `xco2_swath_bc`: The bias-corrected XCO₂ value.
  - `swath_bias_corrected`: A flag indicating if a correction was applied (0 for no, 1 for yes).

**Usage**:
```bash
python -m src.processing.apply_swath_bc_RF
```
**Input**: OCO-3 Lite files (`oco3_LtCO2_*B11072Ar*.nc4`) from your data directory
**Output**: Corrected files in your configured output directory

---

### 📊 Phase 3: Analysis & Visualization

**Purpose**: Generate comprehensive analysis and visualizations of the model's performance and the correction results.

**Script**: `run_comprehensive_analysis.py` (a new, unified runner script)

**What it does**:
- Provides a single, powerful command-line interface to run all analysis and visualization scripts.
- Offers pre-defined groups of scripts for common tasks (`--core`, `--plots`, `--publication`, `--validation`).
- Allows for running individual scripts.
- Tracks progress, estimates run times, and handles errors gracefully.

**Usage**:
```bash
# Run the core analysis suite (recommended for a standard check)
python -m src.analysis.run_comprehensive_analysis --core

# Generate all figures for the paper
python -m src.analysis.run_comprehensive_analysis --publication

# See all available analysis options
python -m src.analysis.run_comprehensive_analysis --list
```

**Available Analysis Groups**:
- **`--core`**: Essential analysis (SHAP, bias plots, evaluation plots).
- **`--plots`**: All visualization scripts.
- **`--validation`**: In-depth model validation (RF test, core SHAP).
- **`--publication`**: Scripts to generate publication-ready figures.
- **`--all`**: The complete analysis suite (9 scripts, ~30-45 minutes).


## ⚙️ Centralized Configuration System

All paths and experiment settings are now centralized in `src/utils/config_paths.py` for easy experiment management.

### Quick Experiment Setup
To start a new experiment, edit **only 3 lines** in `src/utils/config_paths.py`:
```python
# In src/utils/config_paths.py
MODEL_VERSION = "v4.0"                           # Your new version
EXPERIMENT_NAME = "Swath_BC_v4.0_NewFeatures"    # Your experiment name  
PROCESSING_VERSION = "v4.0"                      # Processing version
```

All scripts automatically use the new configuration:
```bash
python -m src.modeling.Swath_BC_v3                        # Train model
python -m src.processing.apply_swath_bc_RF                # Process data
python -m src.analysis.run_comprehensive_analysis --core  # Analyze results
```


### Example Usage:
```bash
# Set your data paths
export OCO3_DATA_DIR="/your/path/to/oco3/lite/files"
export OCO3_OUTPUT_DIR="/your/path/to/output"

# Run the complete pipeline
python -m src.modeling.Swath_BC_v3                        # Train model
python -m src.processing.apply_swath_bc_RF                # Process data
python -m src.analysis.run_comprehensive_analysis --core  # Analyze results
```

### Directory Structure Created:
```
your-project/
├── data/
│   ├── models/Swath_BC_v4.0_FinalHardcoded/
│   ├── processed/Swath_BC_v4.0_FinalHardcoded/ 
│   └── output/Lite_w_SwathBC_v4.0/
└── results/figures/
```

## 📊 Data Requirements

### OCO-3 Level-2 Lite Files
This code requires OCO-3 Level-2 Lite files (Build B11 or later). 

**Download OCO-3 Data:**
- **Official Source**: [NASA Goddard Earth Sciences Data and Information Services Center (GES DISC)](https://disc.gsfc.nasa.gov/datasets?keywords=oco3)
- **Required Product**: OCO-3 Level-2 geolocated XCO₂ retrievals (Lite files)
- **File Pattern**: `oco3_LtCO2_*B11072Ar*.nc4`
- **Coverage**: Snapshot Area Map (SAM) observations (operation_mode = 4)

**Minimum Dataset for Testing:**
- A few OCO-3 Lite files containing SAM observations
- The included labels file (`data/labels/Swath_Bias_labels.csv`) for model training

### Labeled Training Data
The repository includes manually labeled SAM data (`data/labels/Swath_Bias_labels.csv`) containing 1,723 expert-classified scenes for model training.

## 📖 Documentation
- **`src/analysis/README.md`**: Detailed documentation for all analysis scripts.
- **`src/utils/config_paths.py`**: Centralized path configuration (⭐ **KEY FILE** ⭐).
- **`Paper.md`**: Research paper draft with comprehensive results.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{mauceri2025oco3swath,
  title={Machine Learning Detection and Correction of Swath-Dependent Biases in OCO-3 Snapshot Area Map Observations},
  author={Mauceri, S. and others},
  journal={Under Review},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Disclaimer:** This bias correction dataset does not replace the official OCO-3 product and should be used for research applications only.

