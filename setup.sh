#!/bin/bash

# OCO-3 Swath Bias Correction - Setup Script
# This script helps you set up the environment and verify your installation

set -e  # Exit on any error

echo "🚀 OCO-3 Swath Bias Correction - Setup Script"
echo "=============================================="
echo

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    case $color in
        "green") echo -e "\033[32m✅ $message\033[0m" ;;
        "red") echo -e "\033[31m❌ $message\033[0m" ;;
        "yellow") echo -e "\033[33m⚠️  $message\033[0m" ;;
        "blue") echo -e "\033[34mℹ️  $message\033[0m" ;;
    esac
}

# Check Python version
echo "🔍 Checking Python installation..."
if command -v python3 &> /dev/null; then
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ $(echo "$python_version >= 3.8" | bc -l) ]]; then
        print_status "green" "Python $python_version found"
    else
        print_status "red" "Python 3.8+ required, found $python_version"
        exit 1
    fi
else
    print_status "red" "Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Check if conda is available
echo
echo "🔍 Checking conda installation..."
if command -v conda &> /dev/null; then
    print_status "green" "Conda found"
    CONDA_AVAILABLE=true
else
    print_status "yellow" "Conda not found. Will use pip for installation"
    CONDA_AVAILABLE=false
fi

# Create environment
echo
echo "🏗️  Setting up Python environment..."
if $CONDA_AVAILABLE; then
    echo "Creating conda environment 'oco3_bias' from environment.yml..."
    if conda env list | grep -q "oco3_bias"; then
        if [[ "$FORCE_RECREATE" == "1" ]]; then
            print_status "yellow" "Environment 'oco3_bias' exists; recreating due to FORCE_RECREATE=1"
            conda env remove -n oco3_bias -y
            conda env create -f environment.yml
        else
            print_status "yellow" "Environment 'oco3_bias' already exists (skipping creation). Set FORCE_RECREATE=1 to recreate."
        fi
    else
        conda env create -f environment.yml
    fi

    print_status "blue" "Activating environment..."
    eval "$(conda shell.bash hook)"
    conda activate oco3_bias
    print_status "green" "Dependencies installed via conda-forge (environment.yml)"
else
    # Check if in virtual environment
    if [[ -z "$VIRTUAL_ENV" ]]; then
        print_status "yellow" "Not in a virtual environment. Consider creating one:"
        echo "  python3 -m venv oco3_bias"
        echo "  source oco3_bias/bin/activate"
    fi

    echo
    echo "📦 Installing dependencies via pip..."
    print_status "yellow" "Note: cartopy, pyproj, netcdf4, and h5py require system GEOS, PROJ, and HDF5 libraries."
    print_status "yellow" "If pip install fails, install conda and re-run this script (recommended)."
    pip install -r requirements.txt
fi

# Verify installation
echo
echo "🧪 Verifying installation..."
if python3 -c "from src.utils.config_paths import PathConfig; print('Installation OK')" 2>/dev/null; then
    print_status "green" "Package imports working"
else
    print_status "red" "Installation verification failed"
    exit 1
fi

# Check configuration
echo
echo "⚙️  Checking configuration..."
python3 -c "from src.utils.config_paths import PathConfig; PathConfig().print_config_summary()"

# Create necessary directories
echo
echo "📁 Creating directory structure..."
python3 -c "from src.utils.config_paths import PathConfig; PathConfig().ensure_output_dirs()"
print_status "green" "Directory structure created"

# Path configuration guidance
echo
echo "🔧 Path Configuration:"
echo "Set paths in src/utils/config_paths.py under the 'User configuration' section."
echo "Defaults: data/input for Lite files and data/output for processed files."

# Next steps
echo
echo "🎯 Next Steps:"
echo "1. Download OCO-3 data from NASA GES DISC:"
echo "   https://disc.gsfc.nasa.gov/datasets?keywords=oco3"
echo
echo "2. Configure paths in src/utils/config_paths.py (User configuration)."
echo "   Optionally set USER_DATA_DIR and USER_OUTPUT_DIR; otherwise defaults are used."
echo
echo "3. Test with a small dataset:"
echo "   python -m src.modeling.Swath_BC_v3"
echo
echo "4. Process your data:"
echo "   python -m src.processing.apply_swath_bc_RF"
echo
echo "5. Run analysis:"
echo "   python -m src.analysis.run_comprehensive_analysis --core"

echo
print_status "green" "Setup complete! 🎉"

if $CONDA_AVAILABLE; then
    echo
    print_status "blue" "Remember to activate your environment before use:"
    echo "conda activate oco3_bias"
fi 