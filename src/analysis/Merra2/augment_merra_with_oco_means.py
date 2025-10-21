#!/usr/bin/env python3
"""
Augment MERRA-2 collocation CSV with per-SAM means of selected OCO-3 variables.

Inputs:
  - Collocation CSV with at least column 'id' (e.g., data/intermediate/aod_merra2_collocation.csv)
  - Processed OCO-3 NetCDF outputs (from apply_swath_bc_RF) in OUTPUT_FULL_DIR

Requested variables to average per SAM:
  Retrieval: albedo_o2a, aod_bc, aod_dust (also accept 'ado_dust'), aod_ice, aod_oc, aod_seasalt, aod_strataer, aod_sulfate, aod_total, aod_water
  Ungrouped: solar_zenith_angle, sensor_zenith_angle
  Sounding: sensor_azimuth_angle, solar_azimuth_angle

Usage:
  python -m src.analysis.augment_merra_with_oco_means \
    --input-csv data/intermediate/aod_merra2_collocation.csv \
    --output-csv data/intermediate/aod_merra2_collocation_with_oco_means.csv
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

# Ensure project root on path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config_paths import PathConfig
from src.utils.main_util import read_oco_netcdf_to_df


DEFAULT_VARS: List[str] = [
    # Retrieval group
    'albedo_o2a', 'aod_bc', 'aod_dust', 'aod_ice', 'aod_oc', 'aod_seasalt',
    'aod_strataer', 'aod_sulfate', 'aod_total', 'aod_water',
    # Ungrouped
    'solar_zenith_angle', 'sensor_zenith_angle',
    # Sounding group
    'sensor_azimuth_angle', 'solar_azimuth_angle'
]

# Alternate spellings to consider (map alt->canonical)
ALT_NAMES = {
    'ado_dust': 'aod_dust'
}


def find_available_vars(df: pd.DataFrame, desired: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Return available variable names in df; include alt-name mappings.
    Returns (present, mappings) where mappings are (alt_name, canonical).
    """
    present = [v for v in desired if v in df.columns]
    mappings: List[Tuple[str, str]] = []
    for alt, canon in ALT_NAMES.items():
        if alt in df.columns and canon not in present:
            mappings.append((alt, canon))
            present.append(alt)
    return present, mappings


def compute_sam_means(config: PathConfig, desired_vars: List[str], max_files: int = None) -> pd.DataFrame:
    output_dir = Path(config.OUTPUT_FULL_DIR)
    if not output_dir.exists():
        raise FileNotFoundError(f"Processed outputs not found: {output_dir}")

    nc_files = sorted(output_dir.glob('*.nc4'))
    if max_files is not None:
        nc_files = nc_files[:max_files]
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files in {output_dir}")

    # Always include identifiers
    base_cols = ['operation_mode', 'orbit', 'target_id']
    # We'll request all desired vars; missing columns will be dropped after read
    req_cols = base_cols + desired_vars + list(ALT_NAMES.keys())

    all_rows = []
    for nc in tqdm(nc_files, desc='Scanning OCO NetCDF outputs'):
        try:
            df = read_oco_netcdf_to_df(str(nc), variables_to_read=req_cols)
            if df is None or df.empty:
                continue
            if 'operation_mode' not in df.columns:
                continue
            df = df[df['operation_mode'] == 4].copy()
            if df.empty:
                continue
            # Build SAM id and drop 'none'
            df['orbit_str'] = df['orbit'].astype(int).astype(str)
            df['SAM'] = df['target_id'].fillna('none') + '_' + df['orbit_str']
            df = df[~df['SAM'].str.contains('none', case=False, na=False)]
            if df.empty:
                continue

            # Determine available vars in this file
            present, mappings = find_available_vars(df, desired_vars)
            # Apply alt-name mapping to canonical columns
            for alt, canon in mappings:
                if alt in df.columns and canon not in df.columns:
                    df[canon] = df[alt]
            use_cols = [c for c in desired_vars if c in df.columns]
            if not use_cols:
                continue

            needed = ['SAM'] + use_cols
            all_rows.append(df[needed])
        except BaseException:
            continue

    if not all_rows:
        return pd.DataFrame(columns=['SAM'] + desired_vars)

    big = pd.concat(all_rows, axis=0, ignore_index=True)
    # Compute means per SAM
    sam_means = big.groupby('SAM')[desired_vars].mean().reset_index()
    sam_means.rename(columns={'SAM': 'id'}, inplace=True)
    return sam_means


def main():
    parser = argparse.ArgumentParser(description='Augment MERRA-2 collocation with OCO per-SAM means')
    parser.add_argument('--input-csv', type=str, default=None, help='Path to MERRA-2 collocation CSV')
    parser.add_argument('--output-csv', type=str, default=None, help='Path for merged output CSV')
    parser.add_argument('--max-files', type=int, default=None, help='Limit number of processed NetCDF files to scan')
    parser.add_argument('--vars', nargs='*', default=DEFAULT_VARS, help='Variables to average per SAM')
    args = parser.parse_args()

    config = PathConfig()
    inter_dir = Path(config.INTERMEDIATE_DIR)
    inter_dir.mkdir(parents=True, exist_ok=True)

    input_csv = Path(args.input_csv) if args.input_csv else (inter_dir / 'aod_merra2_collocation.csv')
    if not input_csv.exists():
        raise FileNotFoundError(f"Input collocation CSV not found: {input_csv}")

    merra_df = pd.read_csv(input_csv)
    if 'id' not in merra_df.columns:
        raise ValueError("Input collocation CSV must contain column 'id'")

    print('Computing per-SAM means from processed OCO outputs...')
    sam_means = compute_sam_means(config, args.vars, max_files=args.max_files)
    if sam_means.empty:
        print('Warning: No OCO per-SAM means computed; writing original CSV unchanged.')
        out_path = Path(args.output_csv) if args.output_csv else (inter_dir / 'aod_merra2_collocation_with_oco_means.csv')
        merra_df.to_csv(out_path, index=False)
        print(f'Wrote {len(merra_df)} rows to {out_path}')
        return 0

    merged = merra_df.merge(sam_means, on='id', how='left')
    out_path = Path(args.output_csv) if args.output_csv else (inter_dir / 'aod_merra2_collocation_with_oco_means.csv')
    merged.to_csv(out_path, index=False)
    print(f'Wrote merged CSV with OCO means to {out_path} (rows={len(merged)})')
    return 0


if __name__ == '__main__':
    sys.exit(main())


