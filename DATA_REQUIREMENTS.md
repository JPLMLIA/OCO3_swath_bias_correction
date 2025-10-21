# Data Requirements for OCO-3 Swath Bias Correction

## Overview

This software requires OCO-3 Level-2 Lite files to detect and correct swath-dependent biases in Snapshot Area Map (SAM) observations. This document describes the data requirements, where to obtain the data, and how to structure it for use with this software.

## Required Data

### 1. OCO-3 Level-2 Lite Files

**Product:** OCO-3 Level-2 geolocated XCO₂ retrievals (Lite files)

**Source:** [NASA Goddard Earth Sciences Data and Information Services Center (GES DISC)](https://disc.gsfc.nasa.gov/datasets?keywords=oco3)

**Specific Dataset:** OCO-3 Level-2 geolocated XCO₂ retrievals
- **Collection:** Version 11 (Build B11)
- **File Format:** NetCDF4 (`.nc4`)
- **File Pattern:** `oco3_LtCO2_YYMMDD_B11072Ar_YYMMDDHHMMSS.nc4`

### 2. Data Filtering Requirements

**Operation Mode:** The software specifically processes SAM observations:
- Filter: `operation_mode == 4` (SAM mode)
- Other modes (Nadir, Glint, Target, Transition) are not used

**Quality Requirements:**
- Standard OCO-3 quality flags apply
- Minimum 500 soundings per SAM for bias detection
- Geographic coverage: Global (all target types supported)

## Data Access and Download

### Option 1: Direct Download from GES DISC

1. **Register:** Create account at [NASA Earthdata](https://urs.earthdata.nasa.gov/)
2. **Browse:** Visit [OCO-3 data portal](https://disc.gsfc.nasa.gov/datasets?keywords=oco3)
3. **Search:** Filter by date range and geographic region
4. **Download:** Use direct download or bulk download tools


## Directory Structure

### Recommended Organization

```
your_data_directory/
├── oco3_lite_files/
│   ├── 2019/
│   │   ├── oco3_LtCO2_190801_B11072Ar_*.nc4
│   │   ├── oco3_LtCO2_190802_B11072Ar_*.nc4
│   │   └── ...
│   ├── 2020/
│   ├── 2021/
│   ├── 2022/
│   ├── 2023/
│   └── 2024/
└── processed_output/
    └── (created automatically)
```

### Environment Variable Setup

```bash
# Set the path to your OCO-3 data
export OCO3_DATA_DIR="/path/to/your/oco3_lite_files"

# Optional: Set output directory (defaults to ./data/output)
export OCO3_OUTPUT_DIR="/path/to/output/directory"
```

## Data Volume Estimates

### Storage Requirements

| Dataset Size | Files | Storage | Use Case |
|-------------|--------|---------|----------|
| **Minimal Test** | ~10 files | ~500 MB | Algorithm testing |
| **Single Month** | ~100 files | ~5 GB | Small-scale analysis |
| **Single Year** | ~1,200 files | ~60 GB | Annual analysis |
| **Full Dataset** | ~6,000 files | ~300 GB | Complete research |

### Processing Requirements

- **RAM:** 16 GB recommended
- **CPU:** Multi-core recommended for parallel processing
- **Disk:** Additional ~50 GB for models, intermediate files, and outputs

## Data Structure and Variables

### Required Variables

The software expects the following variables in OCO-3 Lite files:

#### Core Variables
- `xco2`: Retrieved XCO₂ [ppm]
- `xco2_raw`: Raw (uncorrected) XCO₂ [ppm]
- `latitude`, `longitude`: Geolocation [degrees]
- `operation_mode`: Observation mode (4 = SAM)
- `target_id`: Target identifier for SAMs
- `orbit`: Orbit number

#### Diagnostic Variables
- `xco2_bias_correction`: Bias correction applied
- `surface_pressure_bias_correction`: Surface pressure bias correction
- `h_continuum_sco2`: Strong CO₂ continuum radiance level correction
- `max_declocking_o2a`: O₂-A band declocking correction
- `aod_*`: Aerosol optical depth retrievals

#### Quality and Metadata
- Various quality flags and retrieval diagnostics
- Viewing geometry variables
- Surface and atmospheric state variables

### Data Validation

Before processing, verify your data:

```python
# Example validation script
import netCDF4 as nc
import numpy as np

def validate_oco3_file(filepath):
    """Basic validation of OCO-3 Lite file"""
    with nc.Dataset(filepath, 'r') as ds:
        # Check required variables exist
        required_vars = ['xco2', 'latitude', 'longitude', 'operation_mode']
        missing_vars = [var for var in required_vars if var not in ds.variables]
        
        if missing_vars:
            print(f"Missing variables: {missing_vars}")
            return False
            
        # Check for SAM data
        operation_mode = ds.variables['operation_mode'][:]
        n_sam = np.sum(operation_mode == 4)
        
        print(f"File: {filepath}")
        print(f"Total soundings: {len(operation_mode)}")
        print(f"SAM soundings: {n_sam}")
        
        return n_sam > 0

# Use it
validate_oco3_file("your_file.nc4")
```

## Troubleshooting

### Common Issues

1. **No SAM data found**
   - Check `operation_mode` variable
   - Verify file contains target observations (not just nadir/glint)

2. **Missing variables**
   - Ensure using Build B11 or later
   - Some variables may have slightly different names in different builds

3. **File access errors**
   - Verify Earthdata credentials
   - Check file permissions and download completion

4. **Memory issues**
   - Process data in smaller chunks
   - Use the `Save_RAM` option in data loading functions

### Support

For software issues:
- Check the main repository README and documentation
- Review example configurations and setup scripts 