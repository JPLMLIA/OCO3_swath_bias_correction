# makes preload files for Swath Bias Correction.
# does not contain any models, TCCON data, or pre-filtering.


from .Make_Pkl import read_Lite
# from .PreFiltering import prefilter_data
from ..utils.main_util import remove_missing_values


mode = 'TG' # 'SAM', 'SAM_fossil', 'SAM_TG', 'TG'
years = range(2019, 2024)

for year in years:
    # load data
    data = read_Lite(year)

    data.drop(columns=['windspeed', 'windspeed_apriori'], inplace=True)

    if mode == 'SAM':
        # Operation Mode: 0=Nadir, 1=Glint, 2=Target, 3=Transition, 4=SAM"
        print('removing non-SAM data')
        data = data.loc[(data['operation_mode'] == 4), :]
    elif mode == 'SAM_fossil':
        print('removing non fossil SAMs')
        data = data.loc[(data['operation_mode'] == 4), :]
        # remove SAM data that is not fossil fuel
        data = data[data['SAM'].str.contains('fossil')]
    elif mode == 'SAM_TG':
        print('removing non-SAM no-Target data')
        data = data.loc[((data['operation_mode'] == 4) | (data['operation_mode'] == 2)), :]
    elif mode == 'TG':
        print('removing non-TG data')
        data = data.loc[(data['operation_mode'] == 2), :]
    else:
        ValueError ('mode not recognized')

    # make SAMs
    orbit = data['orbit'].astype(int)
    data['SAM'] = data['target_id'] + '_' + orbit.astype(str)
    # set SAMs to None when target_id is 'none'
    data.loc[data['target_id'] == 'none', 'SAM'] = None
    # Cast SAM to string
    data['SAM'] = data['SAM'].astype(str)

    # remove missing data
    data = remove_missing_values(data)
    print(f'# of samples after removing missing data: {data.shape[0]}')


    # save as parquet - use environment variable or default
    import os
    preload_dir = os.getenv('OCO3_PRELOAD_DIR', './data/preload')
    os.makedirs(preload_dir, exist_ok=True)
    path = os.path.join(preload_dir, f'PreLoad_oco3_B11_V3_{mode}_{year}.parquet')
    data.to_parquet(path)

print('Done >>>')