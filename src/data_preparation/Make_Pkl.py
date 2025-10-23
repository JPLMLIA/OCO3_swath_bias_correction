# makes pkl files out of Lite files and removes unnecessary variables, cleans up naming
# 03/2022 Steffen Mauceri

import numpy as np
import pandas as pd
import glob
import netCDF4 as nc
from numba import int64, int32
from tqdm import tqdm
from ..utils.config_paths import PathConfig

def get_all_headers_with_dims(f):
    headers = []
    dims = []
    # Get variables in the root group
    for var_name in f.variables.keys():
        var = f.variables[var_name]
        headers.append(var_name)
        dims.append(var.ndim)
    # Get variables in subgroups
    groups = list(f.groups.keys())
    for g in groups:
        for var_name in f[g].variables.keys():
            full_var_name = g + '/' + var_name
            var = f[g].variables[var_name]
            headers.append(full_var_name)
            dims.append(var.ndim)
    return headers, dims




# years = [ 2021, 2022, 2023, 2020, 2019] #
def read_Lite(year):

    fill_value = 999999


    counts=0
    print(year)
    data_dict = {}
    # get LiteFile data via centralized config
    import os
    config = PathConfig()
    data_dir = config.LITE_FILES_DIR
    # Build a year-specific pattern from the configured pattern
    year_prefix = f"oco3_LtCO2_{year-2000:02d}*"
    pattern = config.LITE_FILES_PATTERN.replace('oco3_LtCO2_*', year_prefix)
    Lite_files = glob.glob(os.path.join(data_dir, pattern))

    if not Lite_files:
        print(f"No Lite files found in {data_dir} for year {year} with pattern {pattern}")
        return pd.DataFrame()

    # Get Lite vars
    l_ds = nc.Dataset(Lite_files[0])
    l_vars, l_dims = get_all_headers_with_dims(l_ds)
    # Remove vars we don't need
    vars_to_remove = [
        'bands', 'footprints', 'levels', 'vertices', 'Retrieval/surface_type','Retrieval/iterations','Retrieval/dp', 'Retrieval/dp_o2a',
        'Retrieval/dp_sco2', 'file_index', 'frames',
        'Meteorology/psurf_apriori_o2a', 'Meteorology/psurf_apriori_sco2', 'Meteorology/psurf_apriori_wco2',
         'date', 'source_files', 'pressure_levels', 'Sounding/polarization_angle','Sounding/att_data_source',
        'Retrieval/diverging_steps',
        'Preprocessors/co2_ratio_offset_per_footprint', 'Preprocessors/h2o_ratio_offset_per_footprint',
        'Retrieval/SigmaB', 'xco2_qf_simple_bitflag', 'xco2_qf_bitflag', 'Sounding/l1b_type']

    # add additional vars to remove that would be needed for TCCON AK adjustment
    vars_to_remove += ['pressure_weight', 'xco2_averaging_kernel']

    l_vars = [e for e in l_vars if e not in vars_to_remove]

    # remove vars that start with L1b
    l_vars = [e for e in l_vars if not e.startswith('L1b')]

    # Separate variables by dimension
    l_vars_1d = []
    l_vars_2d = []
    l_vars_nd = []
    for v in l_vars:
        var = l_ds[v]
        ndim = var.ndim
        if ndim == 1:
            l_vars_1d.append(v)
        elif ndim == 2:
            l_vars_2d.append(v)
        else:
            l_vars_nd.append(v)

    # Initialize data_dict keys
    for v in l_vars_1d + l_vars_2d + l_vars_nd:
        data_dict[v] = []

    # Read in data
    for l in tqdm(Lite_files):
        try:

            l_ds = nc.Dataset(l)

            # Read 1D variables
            for v in l_vars_1d:
                val = l_ds[v][:]
                val = np.where(val == fill_value, np.nan, val)
                data_dict[v].append(val)

            # Read 2D variables
            for v in l_vars_2d:
                val = l_ds[v][:]  # shape: (num_soundings, dim2)
                val = np.where(val == fill_value, np.nan, val)
                data_dict[v].append(val)

            # Handle variables with ndim > 2
            for v in l_vars_nd:
                val = l_ds[v][:]  # shape: (num_soundings, dim2, dim3, ...)
                val = np.where(val == fill_value, np.nan, val)
                data_dict[v].append(val)

            counts += 1
        except:
            print('Error reading file: ' + l)
            continue

    # Concatenate data from all files
    for v in data_dict.keys():
        data_dict[v] = np.concatenate(data_dict[v], axis=0)

    # Create DataFrame
    data_all_df = pd.DataFrame()

    # Add 1D variables to DataFrame
    for v in l_vars_1d:
        data_all_df[v] = data_dict[v]

    # Add 2D variables to DataFrame (each entry is an array)
    for v in l_vars_2d:
        print(v)
        data_all_df[v] = list(data_dict[v])

    # Handle variables with ndim > 2 if needed (each entry is an array)
    for v in l_vars_nd:
        data_all_df[v] = list(data_dict[v])

    del data_dict


    # Clean up features
    Features = data_all_df.columns
    feature_dict = {}
    for f in Features:
        features_clean = f.split('/')[-1]
        if f.split('/')[0] != f.split('/')[-1]:
            feature_dict[f] = features_clean
    data_all_df.rename(columns=feature_dict, inplace=True)



    # Note: TCCON AK adjustment can be added here if needed
    # data_all_df['xco2_averaging_kernel'] = data_all_df['xco2_averaging_kernel'] * data_all_df['pressure_weight'] # un-normalize xco2_ak
    # drop pressure_weight
    # data_all_df.drop(columns=['pressure_weight'], inplace=True)



    # Cast some variables to different types to save RAM
    data_all_df['sounding_id'] = data_all_df['sounding_id'].astype('int64')
    data_all_df['land_water_indicator'] = data_all_df['land_water_indicator'].astype('int8')
    data_all_df['footprint'] = data_all_df['footprint'].astype('int8')
    data_all_df['orbit'] = data_all_df['orbit'].astype('int32')
    data_all_df['operation_mode'] = data_all_df['operation_mode'].astype('int8')
    data_all_df['land_fraction'] = data_all_df['land_fraction'].astype('float32')
    data_all_df['snow_flag'] = data_all_df['snow_flag'].astype('int8')
    data_all_df['xco2_quality_flag'] = data_all_df['xco2_quality_flag'].astype('int8')

    # sort data by sounding_id
    data_all_df = data_all_df.sort_values('sounding_id')

    # defragment data frame
    data_save = data_all_df.copy()
    del data_all_df

    # print('saving ' + str(year))
    # data_save.to_pkl('/Volumes/OCO/Pkl_OCO3/LiteB11_' +str(year)+ '.pkl')

    # clean up RAM
    return data_save




# print('>>> Done')
