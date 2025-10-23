#!/usr/bin/env python3
"""
Export SuperLite NetCDF files from bias-corrected OCO-3 Lite files by
preserving only a minimal, configurable subset of variables.

Defaults:
- Input dir: PathConfig().OUTPUT_FULL_DIR (bias-corrected Lite files)
- Output dir: <INPUT_DIR>/SuperLite
- Pattern: oco3_LtCO2_*_SwathBC.nc4
- Keep variables (configurable via --vars):
    - sounding_id
    - latitude
    - longitude
    - time
    - Sounding/operation_mode
    - Sounding/orbit
    - Sounding/target_id
    - xco2
    - xco2_quality_flag
    - xco2_swath_bc
    - swath_bias_corrected

Notes:
- Group variables (e.g., "Sounding/target_id") are flattened on write (→ "target_id").
- All global attributes are preserved. Variable attributes are copied when available.
- Numeric variables are written with compression; vlen strings are written without compression.
"""

import os
import glob
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import netCDF4 as nc
from tqdm import tqdm

from ..utils.config_paths import PathConfig


# -----------------------------
# Default variable keep list
# -----------------------------
DEFAULT_KEEP_VARIABLES: List[str] = [
    'sounding_id',
    'latitude',
    'longitude',
    'time',
    'Sounding/operation_mode',
    'Sounding/orbit',
    'Sounding/target_id',
    'xco2',
    'xco2_quality_flag',
    'xco2_swath_bc',
    'swath_bias_corrected',
]


def _read_var(ds: nc.Dataset, var_path: str) -> Tuple[str, Any, Dict[str, Any]]:
    """
    Read a variable (possibly in a group like "Sounding/var") from a dataset.
    Returns a tuple of (flattened_name, data_array, attributes_dict).
    Raises KeyError if the variable is not found.
    """
    if '/' in var_path:
        group_name, var_name = var_path.split('/', 1)
        if group_name not in ds.groups or var_name not in ds.groups[group_name].variables:
            raise KeyError(f"Variable {var_path} not found")
        var_obj = ds.groups[group_name].variables[var_name]
    else:
        if var_path not in ds.variables:
            raise KeyError(f"Variable {var_path} not found")
        var_obj = ds.variables[var_path]

    # Read data and attributes
    data = var_obj[:]
    attrs: Dict[str, Any] = {attr: getattr(var_obj, attr) for attr in var_obj.ncattrs()}
    flat_name = var_path.split('/')[-1]
    return flat_name, data, attrs


def read_netcdf_variables(file_path: str, variables: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Open a netCDF file and read only the specified variables.
    Returns a mapping: flat_var_name -> { 'data': np.ndarray or array-like, 'attrs': dict }
    Missing variables are skipped with a warning.
    """
    out: Dict[str, Dict[str, Any]] = {}
    with nc.Dataset(file_path, 'r') as ds:
        for v in variables:
            try:
                flat_name, data, attrs = _read_var(ds, v)
                out[flat_name] = {'data': np.array(data), 'attrs': attrs}
            except KeyError:
                print(f"Warning: Variable {v} not found in file {file_path}.")
                continue
    return out


def _infer_num_records(ds: nc.Dataset, kept_data: Dict[str, Dict[str, Any]]) -> int:
    """
    Infer number of soundings to define the primary dimension.
    Prefer the source 'sounding_id' dimension if present; otherwise derive
    from the first 1D kept variable.
    """
    # Prefer source dimension if available
    if 'sounding_id' in ds.dimensions:
        return len(ds.dimensions['sounding_id'])
    if 'Sounding' in ds.dimensions:
        return len(ds.dimensions['Sounding'])
    # Fallback: derive from kept 1D variable
    for meta in kept_data.values():
        data = meta['data']
        if hasattr(data, 'ndim') and data.ndim >= 1:
            return int(data.shape[0])
    raise RuntimeError("Unable to infer number of records for primary dimension")


def _create_or_get_output_dir(input_dir: str, output_dir: str = None) -> str:
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    superlite_dir = os.path.join(input_dir, 'SuperLite')
    os.makedirs(superlite_dir, exist_ok=True)
    return superlite_dir


def write_super_lite_netcdf(source_file: str, output_dir: str, kept: Dict[str, Dict[str, Any]]):
    """
    Create a new NetCDF file and write the kept variables, copying global and
    variable attributes when available. Group names are flattened.
    """
    base_name = os.path.basename(source_file)
    if base_name.endswith('.nc4'):
        out_name = base_name[:-4] + '_SuperLite.nc4'
    elif base_name.endswith('.nc'):
        out_name = base_name[:-3] + '_SuperLite.nc'
    else:
        out_name = base_name + '_SuperLite.nc4'

    out_path = os.path.join(output_dir, out_name)

    with nc.Dataset(source_file, 'r') as src:
        num_records = _infer_num_records(src, kept)

        # Cache global attributes
        global_attrs = {attr: getattr(src, attr) for attr in src.ncattrs()}

        # Create output dataset
        ds_out = nc.Dataset(out_path, 'w', format='NETCDF4')

        # Copy global attributes
        for attr, value in global_attrs.items():
            try:
                ds_out.setncattr(attr, value)
            except Exception:
                # Some attributes might not serialize cleanly; skip them
                pass

        # Append provenance to history
        from datetime import datetime
        history_prev = getattr(src, 'history', '')
        keep_list_preview = ','.join(sorted(list(kept.keys())))
        history_new = (
            f"\n{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}: Exported SuperLite file "
            f"(variables: {keep_list_preview}) using export_super_lite_files.py"
        )
        try:
            ds_out.history = (history_prev or '') + history_new
        except Exception:
            pass

        # Primary dimension
        ds_out.createDimension('sounding_id', num_records)

        # Helper to decide compression params
        def compression_kwargs(dtype_obj):
            # vlen strings cannot be compressed
            if dtype_obj is str:
                return {}
            try:
                np.dtype(dtype_obj)
            except Exception:
                return {}
            return dict(zlib=True, complevel=4, shuffle=True)

        # Helper to detect string-like arrays
        def is_string_like_array(arr) -> bool:
            try:
                dt = getattr(arr, 'dtype', None)
                if dt is None:
                    return False
                if dt.kind in {'U', 'S'}:
                    return True
                if dt.kind == 'O':
                    # Probe first non-null element
                    try:
                        it = np.nditer(arr, flags=['refs_ok'])
                        for x in it:
                            val = x.item()
                            if val is None:
                                continue
                            if isinstance(val, (str, bytes)):
                                return True
                            # found a non-string object
                            return False
                        return False
                    except Exception:
                        return False
                return False
            except Exception:
                return False

        # Write each kept variable
        for flat_name, meta in kept.items():
            data = meta['data']
            attrs = meta.get('attrs', {})

            # Determine dtype and dimensions
            dims = ['sounding_id']
            var_dtype = getattr(data, 'dtype', None)

            # Handle string variables as vlen strings
            is_string = is_string_like_array(data)
            if is_string:
                vtype = str  # vlen string
            else:
                vtype = var_dtype if var_dtype is not None else np.array(data).dtype

            # Additional dims if needed
            if hasattr(data, 'ndim') and data.ndim > 1:
                for i in range(1, data.ndim):
                    dim_name = f"{flat_name}_dim_{i}"
                    if dim_name not in ds_out.dimensions:
                        ds_out.createDimension(dim_name, data.shape[i])
                    dims.append(dim_name)

            # Respect source _FillValue when possible
            fill_value = attrs.get('_FillValue', None)

            # Create variable
            try:
                if is_string:
                    var_out = ds_out.createVariable(flat_name, vtype, tuple(dims))
                else:
                    kwargs = compression_kwargs(vtype)
                    if fill_value is not None:
                        var_out = ds_out.createVariable(flat_name, vtype, tuple(dims), fill_value=fill_value, **kwargs)
                    else:
                        var_out = ds_out.createVariable(flat_name, vtype, tuple(dims), **kwargs)
            except Exception:
                # Fallback without compression
                if is_string:
                    var_out = ds_out.createVariable(flat_name, vtype, tuple(dims))
                else:
                    if fill_value is not None:
                        var_out = ds_out.createVariable(flat_name, vtype, tuple(dims), fill_value=fill_value)
                    else:
                        var_out = ds_out.createVariable(flat_name, vtype, tuple(dims))

            # Assign data
            try:
                if is_string:
                    # Ensure python str objects
                    if isinstance(data, np.ndarray):
                        # Convert bytes to str where needed
                        coerced = np.empty(data.shape, dtype=object)
                        it = np.nditer(data, flags=['refs_ok', 'multi_index'])
                        for x in it:
                            val = x.item()
                            if isinstance(val, bytes):
                                coerced[it.multi_index] = val.decode('utf-8', errors='ignore')
                            elif val is None:
                                coerced[it.multi_index] = ''
                            else:
                                coerced[it.multi_index] = str(val)
                        var_out[:] = coerced
                    else:
                        # Fall back to list of strings
                        var_out[:] = np.array([str(x) if not isinstance(x, bytes) else x.decode('utf-8', errors='ignore') for x in data], dtype=object)
                else:
                    var_out[:] = data
            except Exception:
                # Attempt to coerce strings if needed
                if is_string:
                    var_out[:] = np.array(data, dtype=object)
                else:
                    raise

            # Copy attributes (excluding _FillValue which is set on createVariable)
            for attr, value in attrs.items():
                if attr == '_FillValue':
                    continue
                try:
                    var_out.setncattr(attr, value)
                except Exception:
                    # Skip attributes that cannot be serialized
                    pass

        ds_out.close()
    
    return out_path


def process_files(input_dir: str, output_dir: str, pattern: str, keep_vars: List[str], overwrite: bool = False, limit: int = None):
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if limit is not None and limit > 0:
        files = files[:limit]

    if not files:
        print(f"No files matched pattern in {input_dir}: {pattern}")
        return

    out_dir = _create_or_get_output_dir(input_dir, output_dir)

    for src_file in tqdm(files, desc="Exporting SuperLite files"):
        base_name = os.path.basename(src_file)
        if base_name.endswith('.nc4'):
            out_name = base_name[:-4] + '_SuperLite.nc4'
        elif base_name.endswith('.nc'):
            out_name = base_name[:-3] + '_SuperLite.nc'
        else:
            out_name = base_name + '_SuperLite.nc4'
        out_path = os.path.join(out_dir, out_name)

        if os.path.exists(out_path) and not overwrite:
            # Skip existing unless overwrite is specified
            continue

        kept = read_netcdf_variables(src_file, keep_vars)
        if 'sounding_id' not in kept:
            # Ensure sounding_id exists for safe merges later; attempt to read regardless of user list
            with nc.Dataset(src_file, 'r') as src_ds:
                if 'sounding_id' in src_ds.variables:
                    sid_obj = src_ds.variables['sounding_id']
                    kept['sounding_id'] = {
                        'data': np.array(sid_obj[:]),
                        'attrs': {attr: getattr(sid_obj, attr) for attr in sid_obj.ncattrs()}
                    }
                else:
                    print(f"Warning: 'sounding_id' not found in {src_file} and not kept. Output will lack sounding_id.")

        try:
            written_path = write_super_lite_netcdf(src_file, out_dir, kept)
            try:
                size_mb = os.path.getsize(written_path) / 1024 / 1024
                print(f"Wrote: {written_path} ({size_mb:.2f} MB)")
            except Exception:
                print(f"Wrote: {written_path}")
        except Exception as e:
            print(f"Error exporting {src_file}: {e}")


def parse_args() -> argparse.Namespace:
    config = PathConfig()
    default_input_dir = config.OUTPUT_FULL_DIR  # Bias-corrected Lite files
    default_pattern = 'oco3_LtCO2_*_SwathBC.nc4'

    parser = argparse.ArgumentParser(description='Export SuperLite NetCDF files with a minimal set of variables.')
    parser.add_argument('--input-dir', type=str, default=default_input_dir, help='Directory containing input Lite files (bias-corrected).')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to write SuperLite files (default: <input_dir>/SuperLite).')
    parser.add_argument('--pattern', type=str, default=default_pattern, help='Glob pattern for input files.')
    parser.add_argument('--vars', type=str, default=None, help='Comma-separated list of variables to keep (group paths allowed).')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing SuperLite files if present.')
    parser.add_argument('--limit', type=int, default=None, help='Process only first N files (for testing).')
    return parser.parse_args()


def main():
    args = parse_args()

    keep_vars = DEFAULT_KEEP_VARIABLES
    if args.vars:
        keep_vars = [v.strip() for v in args.vars.split(',') if v.strip()]

    print("=" * 60)
    print("EXPORT SUPERLITE FILES")
    print("=" * 60)
    print(f"Input dir: {args.input_dir}")
    print(f"Output dir: {args.output_dir or os.path.join(args.input_dir, 'SuperLite')}")
    print(f"Pattern: {args.pattern}")
    print(f"Keep variables ({len(keep_vars)}): {keep_vars}")
    print(f"Overwrite: {args.overwrite}")
    if args.limit:
        print(f"Limit: {args.limit}")
    print()

    process_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        keep_vars=keep_vars,
        overwrite=args.overwrite,
        limit=args.limit,
    )


if __name__ == '__main__':
    main()


