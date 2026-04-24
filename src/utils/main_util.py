# util functions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix
# import forestci as fci
from tqdm import tqdm
import collections
import cartopy.crs as ccrs
from joblib import Parallel, delayed

# SAM plotting
import time
import os
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from pyproj import Proj, transform
import calendar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from GESDISC_API_Subsetting import subset
import urllib
from PIL import Image
from cartopy.mpl.geoaxes import GeoAxes

import matplotlib
matplotlib.use('Agg')

import netCDF4 as nc # Added import
from src.data_preparation.Make_Pkl import get_all_headers_with_dims # Added import
from scipy.stats import linregress
import cartopy.feature as cfeature
from .config_paths import PathConfig
import re


# --- Define NetCDF Fill Values ---
NC_FILL_VALUE_FLOAT = -999999.0
NC_FILL_VALUE_INT = -9999

# --- Default list of variables to remove for raw OCO-3 Lite file processing ---
# THIS LIST IS NOW AT THE MODULE LEVEL
DEFAULT_RAW_LITE_VARS_TO_REMOVE = [
    'bands', 'footprints', 'levels', 'vertices', 'co2_profile_apriori', 'Retrieval/surface_type', 'Retrieval/iterations',
    'Retrieval/dp', 'Retrieval/dp_o2a', 'Retrieval/dp_sco2', 'file_index', 'frames',
    'Meteorology/psurf_apriori_o2a', 'Meteorology/psurf_apriori_sco2', 'Meteorology/psurf_apriori_wco2',
    'date', 'source_files', 'pressure_levels', 'Sounding/polarization_angle', 'Sounding/att_data_source',
    'Retrieval/diverging_steps',
    'Preprocessors/co2_ratio_offset_per_footprint', 'Preprocessors/h2o_ratio_offset_per_footprint',
    'Retrieval/SigmaB', 'xco2_qf_simple_bitflag', 'xco2_qf_bitflag', 'Sounding/l1b_type',
    'pressure_weight', 'xco2_averaging_kernel', 
    'L1b/land_fraction', 'L1b/latitude', 'L1b/longitude', 'L1b/orbit', 'L1b/selection_flag_sel', 'L1b/operation_mode', 'L1b/sounding_l1b_quality_flag', 'L1b/time'
]


def remove_missing_values(data):
    print('removing missing values')
    # remove samples with missing values = -999999
    data = data[~data.isin([-999999]).any(axis=1)]
    data = data[~data.isin([np.inf]).any(axis=1)]
    data = data[data['xco2_raw'] > 0]
    data = data[data['xco2'] > 0]

    # make an expception for TCCON and Model data and cld data
    vars = ['CT_2022+NRT2023-1', 'LoFI_m2ccv1bsim', 'MACC_v21r1', 'UnivEd_v5.2', 'xco2tccon','SAM',
            'tccon_name','tccon_dist','cld_dist']

    for v in vars:
        if v in data:
            data[v] = data[v].replace(np.nan, 0)

    # remove rows with nans
    data.dropna(inplace=True)

    #change TCCON and Models back to nan
    for v in vars:
        if v in data:
            data[v] = data[v].replace(0, np.nan)

    return data



def load_data(year, mode, min_SA_size=20, verbose_IO=False, preload_IO = True, clean_IO=True,
              TCCON=False, balanced=False, Save_RAM=False, remove_inland_water=True):
    # load soundings from pkl file
    max_n = int(0.5 * 10 ** 7)
    custom_ID = False

    print('loading data from '+ str(year) + '...')
    if preload_IO & clean_IO:
        # max_n = 10**7 # to not run out of RAM
        # Use environment variable or default for preload directory
        preload_dir = os.getenv('OCO3_PRELOAD_DIR', './data/preload')
        path = os.path.join(preload_dir, f'PreLoad_oco3_B11_V3_{mode}_{year}.parquet')

        data = pd.read_parquet(path)

        if Save_RAM:
            #drop every 2nd sample to save RAM
            data = data.iloc[::2]

        if TCCON: # remove soundings without TCCON match ups
            data = data[data['xco2tccon'] > 0]

    elif preload_IO:
        # Use environment variable or default for preload directory  
        preload_dir = os.getenv('OCO3_PRELOAD_DIR', './data/preload')
        path = os.path.join(preload_dir, f'PreLoad_oco3_B11_V3_{mode}_{year}.parquet')

        data = pd.read_parquet(path)

        if Save_RAM:
            #drop every 2nd sample to save RAM
            data = data.iloc[::2]

        if TCCON: # remove soundings without TCCON match ups
            data = data[data['xco2tccon'] > 0]

    else:
        data = load_data_raw_year(year, mode)

    if remove_inland_water: # remove inland water
        # only keep data where this condition is not met: (data['land_water_indicator'] == 1) & (data['altitude'] != 0)
        data = data.loc[~((data['land_water_indicator'] == 1) & (data['altitude'] != 0)), :]

    # remove sea or land data
    # land_water_indicator: 0: land; 1: water; 2: inland water; 3: mixed.
    # operation_mode: Nadir(0), Glint(1), Target(2), or Transition(3)
    if mode == 'LndNDGL':
        print('removing ocean')
        data = data.loc[(data['land_water_indicator'] == 0), :]
        data = data.loc[(data['operation_mode'] != 3), :]
        data.drop(columns=['windspeed', 'windspeed_apriori'], errors='ignore', inplace=True)

    elif mode == 'LndND':
        print('removing ocean')
        data = data.loc[(data['land_water_indicator'] == 0), :]
        data = data.loc[data['operation_mode'] == 0, :]
        data.drop(columns=['windspeed', 'windspeed_apriori'], errors='ignore', inplace=True)

    elif mode == 'Lnd':
        print('removing ocean')
        data = data.loc[(data['land_water_indicator'] == 0), :]
        data = data.loc[(data['operation_mode'] != 3), :]
        data.drop(columns=['windspeed', 'windspeed_apriori'], errors='ignore', inplace=True)

    elif mode == 'LndGL':
        print('removing ocean')
        data = data.loc[(data['land_water_indicator'] == 0), :]
        data = data.loc[data['operation_mode'] == 1, :]
        data.drop(columns=['windspeed', 'windspeed_apriori'], errors='ignore', inplace=True)

    elif mode == 'SeaGL':
        print('removing land')
        data = data.loc[(data['land_water_indicator'] == 1), :]
        # data = data.loc[data['operation_mode'] == 1, :]

    elif mode == 'all':
        print('removing nothing')
        data.drop(columns=['windspeed', 'windspeed_apriori'], errors='ignore', inplace=True)

    elif mode == 'SAM':
        # Operation Mode: 0=Nadir, 1=Glint, 2=Target, 3=Transition, 4=SAM"
        print('removing non-SAM data')
        data = data.loc[(data['operation_mode'] == 4), :]
        data.drop(columns=['windspeed', 'windspeed_apriori'], errors='ignore', inplace=True)

    elif mode == 'SAM_fossil':
        print('removing non fossil SAMs')
        data = data.loc[(data['operation_mode'] == 4), :]
        # remove SAM data that is not fossil fuel
        data = data[data['SAM'].str.contains('fossil')]

        data.drop(columns=['windspeed', 'windspeed_apriori'], errors='ignore', inplace=True)

    elif mode == 'SAM_TG':
        print('removing non-SAM no-Target data')
        data = data.loc[((data['operation_mode'] == 4) | (data['operation_mode'] == 2)), :]
        data.drop(columns=['windspeed', 'windspeed_apriori'], errors='ignore', inplace=True)

    else:
        print('mode needs to be [LndNDGL, LndND, LndGL, SeaGL, all] ')

    if custom_ID:
        # Optional: remove soundings that are not in custom_ID
        # load txt file with sounding id's with pandas
        # custom_ID = pd.read_csv('/path/to/custom/ID_file.txt', header=0)
        # Note: Custom ID file loading would go here if needed
        # custom_ID = custom_ID['ysoundingid'].to_numpy()
        # # only keep data that is in the custom_ID list
        # data = data[data['sounding_id'].isin(custom_ID)]
        print("Warning: custom_ID functionality is disabled in public version")
        pass

    if clean_IO: # clean data
        # remove missing values
        data = remove_missing_values(data)

        # remove coasts
        #data = data.loc[data['coast'] == 0, :]

        # remove outliers
        # data = data.loc[data['xco2_MLquality_flag'] == 0, :]

    if len(data) > max_n:
        # make sure we get all soundings with TCCON match ups
        data_t = data.loc[data['xco2tccon'] > 0]


        if balanced:
            # weight SAs by geographic density
            weights = balance_sounding_loc(data)

            # subsample SA's from data until we have at least max_n samples - len(data_t)
            SAs = data['SA'].unique()
            previous_SAs = []
            data_sampled = []
            n_samples = 0
            data_grouped = data.groupby('SA')
            print('sampling SAs')
            while n_samples < max_n - len(data_t):
                # sample a SA with weights
                SA_i = np.random.choice(SAs, size=1, replace=False, p=weights)
                # make sure we don't sample the same SA twice
                while SA_i in previous_SAs:
                    SA_i = np.random.choice(SAs, size=1, replace=False, p=weights)
                previous_SAs.append(SA_i)
                # get the data for the SA
                data_SA = data_grouped.get_group(SA_i[0])
                # data_SA = data.loc[data['SA'] == SA_i[0]]
                n_samples += len(data_SA)
                # append to data_sampled
                data_sampled.append(data_SA)
            # concat data_sampled to dataFrame
            data_SA = pd.concat(data_sampled)
        else:
            # use this if we don't want to weight SAs
            data_SA = data.sample(max_n - len(data_t), random_state=1)

        # concat data_SA and data_t
        data = pd.concat([data_SA, data_t])

    return data



def scatter_density(x, y, x_name, y_name, title, dir, save_IO=False):
    '''
    makes a scatter plot and color codes where most of the data is

    :param x: x-value
    :param y: y-value
    :param x_name: name on x-axis
    :param y_name: name on y-axis
    :param title: title
    :param dir: save location
    :param save_IO: save plot?
    :return: -
    '''
    print('making scatter density plot ... ')
    # need to reduce number of samples to keep processing time reasonable.
    # Reduce if processing time too long or run out of RAM
    max_n = 50000
    if len(x) > max_n:
        subsample = int(len(x) / max_n)
        x = x[::subsample]
        y = y[::subsample]
    try:
        r, _ = stats.pearsonr(x, y) # get R
    except:
        print('could not calculate r, set to nan')
        r = np.nan
    xy = np.vstack([x, y])
    if np.mean(x) == np.mean(y):
        z = np.arange(len(x))
    else:
        z = stats.gaussian_kde(xy)(xy)# calculate density
    # sort points by density
    idx = z.argsort()
    d_feature = x[idx]
    d_target = y[idx]
    z = z[idx]
    # plot everything
    plt.scatter(d_feature, d_target, c=z, s=2, label='R = ' + str(np.round(r, 2)))
    plt.legend()
    plt.xlim(np.percentile(d_feature, 1), np.percentile(d_feature, 99))
    plt.ylim(np.percentile(d_target, 1), np.percentile(d_target, 99))
    plt.ylabel(y_name)
    plt.xlabel(x_name)
    #plt.ylim([-2, 2])
    plt.title(title)
    plt.tight_layout()

    if save_IO:
        # save to file
        plt.savefig(dir + title + x_name + y_name + '.png')
    else:
        plt.show()
    plt.close()


def scatter_hist(x_var, y_var, x_name, y_name, title, dir, save_IO=False, bias_IO=True):
    # plot all errors vs state variables
    nbins = 50
    epsilon = 1e-10

    # bin error by variable
    bin_mean = np.zeros(nbins) * np.nan
    bin_5 = np.zeros(nbins) * np.nan
    bin_95 = np.zeros(nbins) * np.nan
    bin_edges = np.linspace(np.percentile(x_var,1), np.percentile(x_var,99), nbins+1)
    # bin_edges = np.linspace(np.min(x_var), np.max(x_var), nbins + 1)
    #_, bin_edges = np.histogram(x_var, bins=nbins)
    gap = np.mean(np.diff(bin_edges))
    x = bin_edges[:-1] + gap / 2
    x[0] = bin_edges[0]
    x[-1] = bin_edges[-1]

    i = -1
    for bin in bin_edges[:-1]:
        i += 1
        t = y_var[(x_var >= bin) & (x_var < bin + gap)]
        if len(t) > 0:
            bin_mean[i] = np.mean(t)
            bin_5[i] = np.percentile(t, 5)
            bin_95[i] = np.percentile(t, 95)
    plt.figure(figsize=(4, 3))
    plt.scatter(x_var, y_var, s=1, color='gray', zorder=1)
    plt.fill_between(x, bin_5, bin_95, color='orange', zorder=2, alpha=0.5)
    plt.plot(x, bin_mean, color='red')
    plt.xlim(x[0], x[-1])

    if bias_IO:
        plt.ylim(-2, 2)
        #plt.ylim(-5, 5)
    else:
        plt.ylim(-5, 2)

    plt.grid()
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.tight_layout()
    if save_IO:
        plt.savefig(dir + title + x_name + y_name + '_hist.png', dpi=300)
    else:
        plt.show()
    plt.close()

def get_RMSE(data, ignore_nan=False):
    if ignore_nan:
        RMSE = np.nanmean(data ** 2) ** 0.5
    else:
        RMSE = np.mean(data ** 2) ** 0.5

    return RMSE


def get_variability_reduction(data, var_tp, name, path, save_fig=False, qf=None):
    # calculate variability in SA before and after bias correction
    SA = list(pd.unique(data['SA']))
    data_SA = data['SA'].to_numpy()
    idx = data.index
    SA_xco2raw_RMSE = []
    SA_xco2_RMSE = []
    SA_xco2corr_RMSE = []

    assert len(SA) > 0, 'No data to calculate variability'
    assert data_SA[0] <= data_SA[-1], 'SA needs to be sorted'
    print('get idx for SAs')
    result = collections.defaultdict(list)
    for i in tqdm(range(len(data_SA))):
        SA_i = data_SA[i]
        result[SA_i].append(i)

    for i in tqdm(range(len(SA))):
        a = SA[i]
        idx_SA = result[a]
        if len(idx_SA) > 2:
            SA_xco2raw_RMSE.append((data['xco2raw_SA_bias'].iloc[idx_SA] ** 2).mean() ** 0.5)
            SA_xco2_RMSE.append((data['xco2_SA_bias'].iloc[idx_SA] ** 2).mean() ** 0.5)
            SA_xco2corr_RMSE.append((data['xco2raw_SA_bias-ML'].iloc[idx_SA] ** 2).mean() ** 0.5)


    # plot distribution
    plt.figure(figsize=(4, 3))
    bins = np.arange(0, np.percentile(SA_xco2_RMSE, 98), 0.1)
    l = plt.hist(SA_xco2raw_RMSE, bins=bins, label='OCO-2 raw', histtype='step', color='b')
    n = plt.hist(SA_xco2_RMSE, bins=bins, label='OCO-2 B11', histtype='step', color='r')
    m = plt.hist(SA_xco2corr_RMSE, bins=bins, label='OCO-2 corr.', histtype='step', color='k')

    #plt.title('Std for XCO2 before / after bias correction based on SAs')
    plt.title('Variability based on SA for QF=' + str(qf))
    plt.xlabel('Standard deviation [ppm]')
    plt.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(path + 'SA_var_' + name + '.png', dpi=300)
    else:
        plt.show()
    plt.close()

    #return SA_xco2_std_mean,SA_xco2corr_std_mean# SA_xco2_mean_mean, , SA_xco2corr_mean_mean


def custom_threshold_accuracy(Classifier_threshold):
    def score(estimator, X, y, ):
        # Predict probabilities for class 1
        probas = estimator.predict_proba(X)[:, 1]
        # Apply custom threshold
        y_pred = (probas >= Classifier_threshold).astype(int)
        # Calculate accuracy
        return accuracy_score(y, y_pred)
    return score


def get_importance(rf, X, y, name, dir, save_IO=False, Classifier_threshold=None):

    if Classifier_threshold is not None:
        # get feature importance
        result = permutation_importance(rf, X, y, n_repeats=5, scoring=custom_threshold_accuracy(Classifier_threshold))
    else:
        # get feature importance
        result = permutation_importance(rf, X, y, n_repeats=5)

    sorted_idx = result.importances_mean.argsort()

    # plot importances
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=X.columns[sorted_idx])
    ax.set_xlabel('Feature Importance')
    ax.set_xlim(0,0.5)
    fig.tight_layout()

    if save_IO:
        plt.savefig(dir + 'Imp_' + name + '.png', dpi=300)
    else:
        plt.show()
    plt.close('all')


def raster_data(Var, Lat, Lon, res=1, aggregate='mean', set_nan=True):
    """
    Function to rasterize data
    """
    world_map = np.zeros((180 // res, 360 // res))

    # Create a temporary DataFrame from the input arrays
    data = pd.DataFrame({
        'Var': Var,
        'latitude': Lat,
        'longitude': Lon
    })

    lat_bins = np.arange(-90, 91, res)
    lon_bins = np.arange(-180, 181, res)
    
    # Digitize the coordinates to find which bin they belong to
    data['lat_bin'] = np.digitize(data['latitude'], lat_bins)
    data['lon_bin'] = np.digitize(data['longitude'], lon_bins)
    
    if aggregate == 'sum':
        grouped = data.groupby(['lat_bin', 'lon_bin'])['Var'].sum()
    else: # Default to mean
        grouped = data.groupby(['lat_bin', 'lon_bin'])['Var'].mean()
        
    for (lat_i, lon_i), val in grouped.items():
        # Adjust indices to fit in the world_map array
        row_idx = len(lat_bins) - lat_i
        col_idx = lon_i - 1
        if 0 <= row_idx < world_map.shape[0] and 0 <= col_idx < world_map.shape[1]:
            world_map[row_idx, col_idx] = val

    if set_nan:
        world_map[world_map == 0] = np.nan

    return world_map


def Earth_Map_Raster(raster, MIN, MAX, var_name, Title, Save=False, Save_Name='None', res=1,colormap=plt.cm.coolwarm,
                    extend = 'both'):
    ''' makes beautiful plot of data organized in a matrix representing lat lon of the globe

    :param raster: gridded data
    :param MIN: min value for plot
    :param MAX: max value for plot
    :param var_name: Name of var to be plotted
    :param Title:
    :param Save: wether to save the plot or show it
    :param Save_Name:
    :param res: resolution of map in deg
    :param colormap: what colormap to use
    :param extend: add triangles to sides of colorbar ['neither', 'both']
    :return:
    '''
    # New version with Cartopy
    limits = [-180, 180, -90, 90]
    offset = res / 2 * np.array([0, 0, 2, 2])
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    im = ax.imshow(np.flipud(raster), interpolation='nearest', origin='lower',
                   extent=np.array(limits)+offset, cmap=colormap, vmin=MIN, vmax=MAX,
                   transform=ccrs.PlateCarree(), alpha=0.9)

    ax.set_title(Title, fontsize=15, pad=10)
    plt.colorbar(im, fraction=0.066, pad=0.08, extend=extend, location='bottom', label=var_name)

    if Save:
        plt.savefig(Save_Name + '.png', dpi=200)
        plt.close()
    else:
        plt.show()
    return fig


def plot_map(data, vars, save_fig=False, path='None', name='None', pos_neg_IO = True, max=None, min=None, aggregate='mean', cmap=None, set_nan = True):
    '''

    :param data: pd.DataFrame
    :param vars: list; vars we wish to plot
    :param save_fig: bool
    :param path: str; save path
    :param name: str; name for saving
    :param pos_neg_IO: bool; changes colorbar and min max of colorbar
    :param aggregate: str, ['mean', 'count'] calc mean; count number of soundings
    :return: image

    '''
    res = 2

    # make vars into a list in case we didn't pass a list
    if isinstance(vars,str):
        vars = [vars]

    for var in vars:
        raster = raster_data(data[var].to_numpy(), data['latitude'].to_numpy(), data['longitude'].to_numpy(), res=res, aggregate=aggregate, set_nan=set_nan)
        if pos_neg_IO:
            if max == None:
                MAX = np.abs(np.nanpercentile(raster, 95))
                MIN = np.abs(np.nanpercentile(raster, 5))
                MAXX = np.max([MAX, MIN])
                MIN = -MAXX
                MAX = MAXX
            else:
                MIN = min
                MAX = max
            if cmap == None:
                colormap = plt.cm.coolwarm
            else:
                colormap = cmap
            extend = 'both'
        else:
            if max == None:
                MAX = np.nanpercentile(raster, 95)
            else:
                MAX = max
            if min == None:
                MIN = np.nanpercentile(raster,5)
            else:
                MIN = min
            if cmap == None:
                colormap = plt.cm.get_cmap('OrRd')
            else:
                colormap = cmap
            extend = 'max'
        var_name = var
        Earth_Map_Raster(raster, MIN, MAX, var_name, name, res=res, Save=save_fig,
                         Save_Name=path + '/' + var_name + name, colormap=colormap, extend = extend)



def hex_plot(data,name,path,save_fig = False,var1 = ['h2o_ratio'],var2 = ['dpfrac'], bias = 'xco2raw_SA_bias'):
    for idx, var in enumerate(var1):
        x = data[var]
        y = data[var2[idx]]
        z = data[bias]


        f, ax = plt.subplots(figsize=(5, 4))
        plt.hexbin(x=x,y=y,C=z, cmap = "RdBu", alpha = 1, gridsize = 20)
        plt.axvline(x=0.75, linestyle='--', color='black', lw=2)  # Default threshold line
        plt.axvline(x=1.07, linestyle='--', color='black', lw=2)
        plt.axhline(y=-3.5, linestyle='--', color='black', lw=2)
        plt.axhline(y=3.0, linestyle='--', color='black', lw=2)

        plt.clim(-1,1)
        plt.colorbar(extend='both',  label='\u03B4' + 'XCO2 [ppm]')
        plt.xlabel(var)
        plt.ylabel(var2[idx])
        plt.tight_layout()
        if save_fig:
            plt.savefig(path+name+"_xco2diffML_hex_b11.png", dpi = 300)


def plot_decision_surface(M, data_test,y_test_c,save_fig = True, file_path = ''):
    feature_names = data_test.columns
    # Create the scatter plots
    # fig, axs = plt.subplots(npredictors+1, npredictors+1, figsize=(15, 15))
    # fig.subplots_adjust(wspace=.35)
    for f1 in range(feature_names+1):
        for f2 in range(feature_names+1): 
            if f1 == f2:
                continue;
            else:
                x1 = data_test.loc[:,f1]
                x2 = data_test.loc[:,f2]
                X = pd.concat([x1,x2], axis = 1)
                disp = DecisionBoundaryDisplay.from_estimator(
                     M, X, response_method="predict",
                     alpha=0.5, xlabel = feature_names[f1], ylabel = feature_names[f2]
                )
                disp.ax_.scatter(data_test[:, 0], data_test[:, 1], c=y_test_c, edgecolor="k")
                disp.save_fig(file_path, dpi = 200)


def dist(lat1, lat2, lon1, lon2):
    '''distance calculation between points given degree

    :param lat1:
    :param lat2:
    :param lon1:
    :param lon2:
    :return: distance in km
    '''

    # transforms deg to rad
    lat1 = lat1*np.pi / 180.
    lat2 = lat2 * np.pi / 180.
    lon1 = lon1 * np.pi / 180.
    lon2 = lon2 * np.pi / 180.
    #calculate distance in km between two points in spherical coordinates


    d = np.arccos(np.sin(lat1)*np.sin(lat2)+np.cos(lat1)*np.cos(lat2)*np.cos(lon1-lon2))*6371
    return d



def get_season(data):
    Date = data['sounding_id'].to_numpy()
    Month_List = []
    Year_List = []
    for d in Date:
        d = str(d)
        Year = int(d[0:4])
        Month = list(map(int, d[4:6]))
        if Month[0] == 0:
            Month = Month[1]
        else:
            Month = Month[1] + 10
        Month_List.append(Month)
        Year_List.append(Year)

    Months = np.stack(Month_List)
    Years = np.stack(Year_List)

    data.loc[(Months >= 3) & (Months <= 5), 'season'] = 'MAM'  # Mar,Apr,May
    data.loc[(Months >= 6) & (Months <= 8), 'season'] = 'JJA'  # Jun,Jul,Aug
    data.loc[(Months >= 9) & (Months <= 11), 'season'] = 'SON'  # Sept,Oct,Nov
    data.loc[(Months == 12) | (Months == 1) | (Months == 2), 'season'] = 'DJF'  # Dec, Jan, Feb

    data.loc[:,'Month'] = Months
    data.loc[:,'Year'] = Years

    return data


def plot_SAM(data, var,vmin=None ,vmax=None, save_fig=False, name='', path='', qf=None, title_addition = '', simplified_title=False, foreground_data=None, background_data=None, include_emission_proxy=False):
    ''' plot a OCO-3 SAM with Worldview Map in the background
    :param data: pd.DataFrame, data containting a single SAM
    :param var: str, variable to plot
    :param save_fig: bool, save figure
    :param path: str, path to save figure'''

    window_size = 2 # in degrees

    if var == 'rad_continuum_o2':
        var_cbar_string = 'Radiance [Ph s' + r'$^{-1}$' + ' m' + r'$^{-2}$' + ' sr' + r'$^{-1}$' + r'$\mu$' + 'm' + r'$^{-1}$' + ']'
        var_title_string = "O" + r'$_2$' + ' A-Band Radiance'
        var_cbar = plt.cm.Purples_r

    elif var == 'xco2':
        var_cbar_string = r'$X_{CO_2}$' + ' [ppm]'
        var_title_string = r'B11 $X_{CO_2}$'
        var_cbar = plt.cm.viridis

    elif var == 'xco2_raw':
        var_cbar_string = r'$X_{CO_2}$' + ' [ppm]'
        var_title_string = r'Raw $X_{CO_2}$'
        var_cbar = plt.cm.viridis

    elif var == 'xco2MLcorr':
        var_cbar_string = r'$X_{CO_2}$' + ' [ppm]'
        var_title_string = r'ML corr. $X_{CO_2}$'
        var_cbar = plt.cm.viridis

    elif 'xco2_diff' in var or 'correction' in var.lower() or 'difference' in var.lower():
        var_cbar_string = r'$\Delta X_{CO_2}$' + ' [ppm]'
        var_title_string = 'XCO₂ Correction Applied'
        var_cbar = plt.cm.RdBu_r

    else:
        var_string = var
        var_cbar_string = var
        var_title_string = var
        var_cbar = plt.cm.viridis


    vertices_lat = pd.DataFrame(data['vertex_latitude'].tolist(), index=data.index)
    vertices_lon = pd.DataFrame(data['vertex_longitude'].tolist(), index=data.index)
    vertices_lat.columns = [f"vertex_latitude{i}" for i in range(4)]
    vertices_lon.columns = [f"vertex_longitude{i}" for i in range(4)]
    data = pd.concat([data, vertices_lat, vertices_lon], axis=1)


    data_all = data.copy()


    # get target_lat and target_lon
    target_id = data['target_id'].iloc[0]
    try:
        target_lat, target_lon = get_target_data(target_id)
    except:
        print('No target data found for ' + target_id)
        return


    # Determine plotting box
    w = window_size/2

    N, S, E, W = np.round(target_lat + w, 1), \
        np.round(target_lat - w,1), \
        np.round(target_lon + w, 1), \
        np.round(target_lon - w,1)

    # And keep anything within the plotting box
    data = data[(data['latitude'] < N) & (data['latitude'] > S) & (data['longitude'] < E) & (data['longitude'] > W)]

    # check that we have data left
    if len(data) == 0:
        print('No data left to plot for ' + target_id)
        return

    # Get some datetime info from the sounding_id
    sounding_id = data['sounding_id'].iloc[0].astype(str)
    month_str = calendar.month_abbr[int(sounding_id[4:6].lstrip("0"))]
    day_str, year_str = sounding_id[6:8].lstrip("0"), sounding_id[:4].lstrip("0")
    hour_str, minute_str = sounding_id[8:10], sounding_id[10:12]

    # get average wind speed and direction of SAM
    wind_speed = np.mean(np.sqrt(data['windspeed_v_met']**2 + data['windspeed_u_met']**2))
    wind_dir = np.mean(np.arctan2(data['windspeed_v_met'], data['windspeed_u_met']))
    if wind_dir < 0:
        wind_dir = wind_dir + 2*np.pi

    # calculate flux proxy of SAM
    enhancement = SAM_enhancement(data, var, qf)

    # Cartopy
    fig = plt.figure(figsize=(20, 20))
    ax1 = plt.axes(projection=ccrs.epsg(3857))
    ax1.set_extent([W, E, S, N], ccrs.PlateCarree())


    # Grid
    gl = ax1.gridlines(draw_labels=True, color="0.75")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 24}
    gl.ylabel_style = {'size': 24}

    # Background map
    m1 = Proj("epsg:3857", preserve_units=True)
    W_3857, S_3857 = m1(W, S)
    E_3857, N_3857 = m1(E, N)
    xpixels = 2000
    ypixels = int((N_3857 - S_3857) / (E_3857 - W_3857) * xpixels)
    # check if we already have the image on disk
    if not os.path.exists('tmp/ESRI_' + str(W) + '_' + str(S) + '_' + str(E) + '_' + str(N) + '.png'):
        # if not, download it
        url = f'http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={W_3857},{S_3857},{E_3857},{N_3857}&bboxSR=3857&imageSR=3857&size={xpixels},{ypixels},&dpi=96&format=png32&transparent=true&f=image'
        try:
            ESRI = np.array(Image.open(urllib.request.urlopen(url)))
        except:  # Sometimes this fails randomly, so try again
            print('Error downloading ESRI image, will trying again in 20 seconds')
            time.sleep(20)
            ESRI = np.array(Image.open(urllib.request.urlopen(url)))
        # Save the image to disk
        plt.imsave('tmp/ESRI_' + str(W) + '_' + str(S) + '_' + str(E) + '_' + str(N) + '.png', ESRI)
    else:
        ESRI = plt.imread('tmp/ESRI_' + str(W) + '_' + str(S) + '_' + str(E) + '_' + str(N) + '.png')
    im1 = ax1.imshow(ESRI, extent=ax1.get_extent(), origin="upper")

    # Plot wind vectors
    arrow_length_factor = 0.02
    start_x, start_y = 0.08, 0.92
    end_x = start_x + arrow_length_factor * wind_speed * np.cos(wind_dir)
    end_y = start_y + arrow_length_factor * wind_speed * np.sin(wind_dir)

    ax1.annotate('', xy=(end_x, end_y), xycoords='axes fraction', xytext=(start_x, start_y), arrowprops=dict(arrowstyle="->", color='red', lw=2))
    # add wind direction and speed as txt
    # ax1.text(start_x - 0.02, 0.97, str(int(np.rad2deg(wind_dir))) + '$^\circ$ ' + str(int(wind_speed)) + ' m/s', transform=ax1.transAxes, color='red', fontsize=18)
    ax1.text(start_x - 0.02, start_y + 0.02, str(int(wind_speed)) + ' m/s', transform=ax1.transAxes, color='red', fontsize=18)



    # Plot footprints ############################################################
    # check if we have more than one quality flag
    if qf is not None:
        patches1 = []
        # plot QF 1 data
        data = data_all[data_all[qf] == 1]
        for j in range(len(data['vertex_longitude1'])):
            if (data['vertex_longitude0'].iloc[j] == 0.0) | (data['vertex_longitude1'].iloc[j] == 0.0) | (
                    data['vertex_longitude0'].iloc[j] == 0.0) | (data['vertex_longitude0'].iloc[j] == 0.0):
                print("Bad vertex...")
            else:
                patches1 += [Polygon([(data['vertex_longitude0'].iloc[j], data['vertex_latitude0'].iloc[j]),
                                     (data['vertex_longitude1'].iloc[j], data['vertex_latitude1'].iloc[j]),
                                     (data['vertex_longitude2'].iloc[j], data['vertex_latitude2'].iloc[j]),
                                     (data['vertex_longitude3'].iloc[j], data['vertex_latitude3'].iloc[j])])]
        p1 = PatchCollection(patches1, alpha=1, transform=ccrs.PlateCarree())
        p1.set_array(data[var])

    patches = []
    # plot QF 0 data
    if qf is not None:
        data = data_all[data_all[qf] == 0]
    else:
        data = data_all

    vlon0 = data['vertex_longitude0'].to_numpy()
    vlat0 = data['vertex_latitude0'].to_numpy()
    vlon1 = data['vertex_longitude1'].to_numpy()
    vlat1 = data['vertex_latitude1'].to_numpy()
    vlon2 = data['vertex_longitude2'].to_numpy()
    vlat2 = data['vertex_latitude2'].to_numpy()
    vlon3 = data['vertex_longitude3'].to_numpy()
    vlat3 = data['vertex_latitude3'].to_numpy()

    for j in range(len(vlon1)):
        if (vlon0[j] == 0.0) or (vlon1[j] == 0.0) or (vlon2[j] == 0.0) or (vlon3[j] == 0.0):
            print("Bad vertex...")
        else:
            patches.append(Polygon([(vlon0[j], vlat0[j]),
                                    (vlon1[j], vlat1[j]),
                                    (vlon2[j], vlat2[j]),
                                    (vlon3[j], vlat3[j])]))



    if qf is not None:
        p = PatchCollection(patches, alpha=1, transform=ccrs.PlateCarree(), edgecolor='black', linewidth=1)
    else:
        p = PatchCollection(patches, alpha=1, transform=ccrs.PlateCarree())
    
    p.set_array(data[var])


    if vmin is not None and vmax is not None:
        p.set_clim(vmin, vmax)
    else:
        p.set_clim(np.percentile(data_all[var], 10), np.percentile(data_all[var], 10) + 5)

    # p.set_lw(1.0)
    p.set_cmap(var_cbar)
    ax1.add_collection(p)
    if qf is not None:
        ax1.add_collection(p1)
    
    # Add foreground and background borders if provided
    if foreground_data is not None and len(foreground_data) > 0:
        # Process vertex data for foreground
        fg_vertices_lat = pd.DataFrame(foreground_data['vertex_latitude'].tolist(), index=foreground_data.index)
        fg_vertices_lon = pd.DataFrame(foreground_data['vertex_longitude'].tolist(), index=foreground_data.index)
        fg_vertices_lat.columns = [f"vertex_latitude{i}" for i in range(4)]
        fg_vertices_lon.columns = [f"vertex_longitude{i}" for i in range(4)]
        fg_data_with_vertices = pd.concat([foreground_data, fg_vertices_lat, fg_vertices_lon], axis=1)
        
        # Create patches for foreground borders
        fg_patches = []
        for j in range(len(fg_data_with_vertices)):
            if (fg_data_with_vertices['vertex_longitude0'].iloc[j] == 0.0) or (fg_data_with_vertices['vertex_longitude1'].iloc[j] == 0.0) or (fg_data_with_vertices['vertex_longitude2'].iloc[j] == 0.0) or (fg_data_with_vertices['vertex_longitude3'].iloc[j] == 0.0):
                continue
            else:
                fg_patches.append(Polygon([(fg_data_with_vertices['vertex_longitude0'].iloc[j], fg_data_with_vertices['vertex_latitude0'].iloc[j]),
                                         (fg_data_with_vertices['vertex_longitude1'].iloc[j], fg_data_with_vertices['vertex_latitude1'].iloc[j]),
                                         (fg_data_with_vertices['vertex_longitude2'].iloc[j], fg_data_with_vertices['vertex_latitude2'].iloc[j]),
                                         (fg_data_with_vertices['vertex_longitude3'].iloc[j], fg_data_with_vertices['vertex_latitude3'].iloc[j])]))
        
        if fg_patches:
            fg_border = PatchCollection(fg_patches, alpha=1, transform=ccrs.PlateCarree(), 
                                      edgecolor='white', linewidth=2, facecolor='none')
            ax1.add_collection(fg_border)
    
    if background_data is not None and len(background_data) > 0:
        # Process vertex data for background
        bg_vertices_lat = pd.DataFrame(background_data['vertex_latitude'].tolist(), index=background_data.index)
        bg_vertices_lon = pd.DataFrame(background_data['vertex_longitude'].tolist(), index=background_data.index)
        bg_vertices_lat.columns = [f"vertex_latitude{i}" for i in range(4)]
        bg_vertices_lon.columns = [f"vertex_longitude{i}" for i in range(4)]
        bg_data_with_vertices = pd.concat([background_data, bg_vertices_lat, bg_vertices_lon], axis=1)
        
        # Create patches for background borders
        bg_patches = []
        for j in range(len(bg_data_with_vertices)):
            if (bg_data_with_vertices['vertex_longitude0'].iloc[j] == 0.0) or (bg_data_with_vertices['vertex_longitude1'].iloc[j] == 0.0) or (bg_data_with_vertices['vertex_longitude2'].iloc[j] == 0.0) or (bg_data_with_vertices['vertex_longitude3'].iloc[j] == 0.0):
                continue
            else:
                bg_patches.append(Polygon([(bg_data_with_vertices['vertex_longitude0'].iloc[j], bg_data_with_vertices['vertex_latitude0'].iloc[j]),
                                         (bg_data_with_vertices['vertex_longitude1'].iloc[j], bg_data_with_vertices['vertex_latitude1'].iloc[j]),
                                         (bg_data_with_vertices['vertex_longitude2'].iloc[j], bg_data_with_vertices['vertex_latitude2'].iloc[j]),
                                         (bg_data_with_vertices['vertex_longitude3'].iloc[j], bg_data_with_vertices['vertex_latitude3'].iloc[j])]))
        
        if bg_patches:
            bg_border = PatchCollection(bg_patches, alpha=1, transform=ccrs.PlateCarree(), 
                                      edgecolor='black', linewidth=2, facecolor='none')
            ax1.add_collection(bg_border)


    # Colorbar
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size=0.4, pad=0.25, axes_class=plt.Axes)
    if var == 'aod_total':
        cbar = fig.colorbar(p, extend='max', cax=cax)
    else:
        cbar = fig.colorbar(p, extend='both', cax=cax)
    cbar.set_label(var_cbar_string, size=28, rotation=270, labelpad=35)
    cbar.ax.tick_params(labelsize=22)
    cbar.ax.yaxis.get_offset_text().set_fontsize(22)

    # Title
    if simplified_title:
        if include_emission_proxy:
            title = ax1.set_title(data_all['SAM'].iloc[0] + '\n' +
                     hour_str + ':' + minute_str + ' UTC ' + day_str + ' ' + month_str + ' ' + year_str + '\n' +
                                  'SAM flux proxy: ' + str(np.round(enhancement,2)) + ' ppm m/s', size=30, y=1.01)
        else:
            title = ax1.set_title(data_all['SAM'].iloc[0] + '\n' +
                     hour_str + ':' + minute_str + ' UTC ' + day_str + ' ' + month_str + ' ' + year_str, size=30, y=1.01)
    else:
        if include_emission_proxy:
            title = ax1.set_title('OCO-3 ' + var_title_string + '\n' + data_all['SAM'].iloc[0] + '\n' +
                     hour_str + ':' + minute_str + ' UTC ' + day_str + ' ' + month_str + ' ' + year_str + '\n' +
                                  title_addition, size=30, y=1.01)
            title.set_text(title.get_text() + '\n' + 'SAM flux proxy: ' + str(np.round(enhancement,2)) + ' ppm m/s')
        else:
            title = ax1.set_title('OCO-3 ' + var_title_string + '\n' + data_all['SAM'].iloc[0] + '\n' +
                     hour_str + ':' + minute_str + ' UTC ' + day_str + ' ' + month_str + ' ' + year_str + '\n' +
                                  title_addition, size=30, y=1.01)


    # Globe inset
    ax2 = inset_axes(ax1, width=2., height=2., loc="upper right", axes_class=GeoAxes,
                     axes_kwargs=dict(projection=ccrs.Orthographic(((E + W) / 2.), ((N + S) / 2.))))
    ax2.set_global()
    ax2.scatter(((W + E) / 2.), ((N + S) / 2.), c='r', s=100, marker='*', zorder=3,
                transform=ccrs.PlateCarree())
    ax2.stock_img()
    ax2.coastlines(color="0.25")

    # Mark the target
    ax1.scatter(target_lon, target_lat, c='r', marker="*", s=600, zorder=3,transform=ccrs.PlateCarree())

    if save_fig:
        save_path = os.path.join(path, data_all['SAM'].iloc[0] + "_" + var + "_" +name + '.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
    else:
        plt.show()

    # Close figure
    plt.close('all')
    import gc
    gc.collect()

def confusion_matrix_rates(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (tn + fp)
    fnr = fn / (tp + fn)
    return {'TPR': tpr, 'TNR': tnr, 'FPR': fpr, 'FNR': fnr}

def get_target_data(target_id):
    ############################################
    # Find the correct target and its corresponding indices using the SAM/TG list files
    clasp_data = pd.read_csv('tmp/clasp_report.csv', header=[0])

    # Find the target in the clasp data
    target_info = clasp_data[clasp_data['Target ID'] == target_id]
    # get the target lat and lon.
    target_lat_lon = target_info['Site Center WKT']
    #parse the string to get the lat and lon. example string: 'POINT(-115.6902 38.497)'
    target_lon = float(target_lat_lon.values[0].split(' ')[0].split('(')[1])
    target_lat = float(target_lat_lon.values[0].split(' ')[1].split(')')[0])

    return target_lat, target_lon


def pixel_to_pixel_std(data, var, qf=None):
    ''' calculate pixel to pixel standard deviation for a given variable over
    multiple or a single SAM
    data: pd.DataFrame, data containing variables
    var: str, variable to calculate variance for
    qf: str, quality flag to use to filter data
    '''
    print('calculating pixel to pixel standard deviation')

    # add column to data to store results
    data['px_to_px_std' + var] = np.nan

    if qf is not None:
        data = data[data[qf] == 0]

    # get unique SAMs
    sams = data['SAM'].unique()
    # subset data by SAM
    stds = []

    for sam in tqdm(sams):
        data_sam = data[data['SAM'] == sam]
        # check that SAM has at least 100 soundings
        if len(data_sam) < 100:
            continue
        # itterate over each sounding and find nearest neighbor

        diffs = []
        # randomly pick 100 soundings to calculate pixel to pixel std
        sounding_ids = np.random.choice(len(data_sam), 100, replace=False)
        for i in sounding_ids:
            # get the sounding
            sounding = data_sam.iloc[i]
            # get the nearest neighbor
            distances = cdist(sounding[['latitude','longitude']].values[None, :].astype(float), data_sam[['latitude','longitude']].to_numpy().astype(float))
            # remove the distance to itself
            nearest_neighbor_id = np.argmin(distances[distances != 0])
            nearest_neighbor = data_sam.iloc[nearest_neighbor_id]
            # calculate the difference
            diff = np.abs(nearest_neighbor[var] - sounding[var])
            # append to list
            diffs.append(diff)
        # calculate standard deviation
        std = np.sqrt(np.mean(np.array(diffs)**2))
        stds.append(std)

        # add to data
        data.loc[data['SAM'] == sam, 'px_to_px_std'  + var] = std

    # return the mean standard deviation
    return np.mean(stds)


def swath_bias(data, var, qf=None, return_all=False):
    ''' calculate swath bias for a given variable over multiple or a single SAM'''

    print('calculating swath bias')
    if qf is not None:
        data = data[data[qf] == 0]

    # get unique SAMs
    sams = data['SAM'].unique()
    # subset data by SAM
    stds = []

    if return_all:
        data['swath_bias'] = np.nan

    for sam in tqdm(sams):
        data_sam = data[data['SAM'] == sam]

        # itterate over each sounding and find nearest neighbor
        diffs = []

        # only keep soundings at the edges of the swath
        data_sam_1 = data_sam[data_sam['footprint'] == 1]
        data_sam_8 = data_sam[data_sam['footprint'] == 8]

        # check that SAM has at least 100 soundings
        if len(data_sam_1) < 10 or len(data_sam_8) < 10:
            continue


        # itterate over footprint 1 soundings and find nearest neighbor in footprint 8
        for i in range(len(data_sam_1)):
            # get the sounding
            sounding = data_sam_1.iloc[i]
            # get the nearest neighbor
            dist = cdist(sounding[['latitude','longitude']].values[None, :].astype(float), data_sam_8[['latitude','longitude']].astype(float))
            nearest_neighbor = data_sam_8.iloc[dist.argmin()]

            # check that diff is less than 4 footprint widths (~10km) [km to deg conversion = km/110]
            if np.min(dist) <= 10 / 110:
                # calculate the difference
                diff =  sounding[var] - nearest_neighbor[var]
                diffs.append(np.abs(diff))

                # This saves the bias for each sounding. Time consuming!
                if return_all:
                    data.loc[sounding.name, 'swath_bias'] = diff


        #check that we have enough soundings
        if len(diffs) < 10:
            continue
        # calculate standard deviation
        std = np.sqrt(np.mean(np.array(diffs)**2))
        stds.append(std)

    if return_all:
        return data
    else:
        # return the mean standard deviation
        return np.mean(stds)


def enough_pixel_kept(data, var, qf, name, threshold=0.2, save_fig=False, path=''):
    ''' calculate the percentage of pixels kept for a given variable over multiple or a single SAM
    data: pd.DataFrame, data containing variables
    var: str, variable to calculate variance for
    qf: str, quality flag to use to filter data
    threshold: float, threshold for percentage of pixels kept that is useful
    :return: float, percentage of pixels kept'''

    print('calculating percentage of pixels kept per SAM')

    if qf is None:
        print('No quality flag provided')
        return 0

    # get unique SAMs
    sams = data['SAM'].unique()
    # subset data by SAM
    frac_kepts = []
    for sam in tqdm(sams):
        data_sam = data[data['SAM'] == sam]
        # check that SAM has at least 100 soundings
        if len(data_sam) < 100:
            continue

        # calculate total number of pixels
        total = len(data_sam)
        # calculate number of pixels that pass quality filter
        passed = len(data_sam[data_sam[qf] == 0])
        # calculate fraction of pixels kept
        kept = passed / total
        frac_kepts.append(kept)

    # calculate percentile of SAMs that have more than threshold percentage of pixels kept
    frac_kepts = np.array(frac_kepts)
    frac_kept = len(frac_kepts[frac_kepts > threshold]) / len(frac_kepts) * 100

    # plot histogram of pc_kepts
    plt.figure(figsize=(5, 5))
    # make histogram between 0 and 100
    plt.hist(frac_kepts * 100, bins = 20, range=(0, 100))
    plt.xlabel('Percentage of pixels kept per SAM')
    plt.ylabel('Number of SAMs')
    plt.title('Percentage of pixels kept per SAM [' + var + ']')
    # add threshold line
    plt.axvline(x=threshold*100, color='r', linestyle='--', label='Threshold')
    plt.legend()
    plt.tight_layout()

    if save_fig:
        plt.savefig(path + name + 'Percentage_pixels_kept_' + var + '.png', dpi = 300)
    plt.close()


    # return the mean percentage of pixels kept
    return frac_kept


def get_foreground_background_indices(data_sam, target_lat, target_lon, wind_dir, wind_speed, custom_SAM=False):
    '''Extract foreground and background pixel indices for a single SAM.
    
    Args:
        data_sam: DataFrame containing data for a single SAM
        target_lat, target_lon: Target coordinates
        wind_dir, wind_speed: Wind direction and speed
        custom_SAM: Whether to print debug messages
        
    Returns:
        tuple: (foreground_data, background_data) or (None, None) if calculation fails
    '''
    data_sam_foreground = data_sam.copy()
    # calculate angle of each pixel to target
    angle = np.arctan2(data_sam_foreground['latitude'] - target_lat, data_sam_foreground['longitude'] - target_lon)
    # make all angles positive
    angle[angle < 0] = angle[angle < 0] + 2*np.pi
    # calculate angle between wind direction and pixel
    angle_diff = np.abs(wind_dir - angle)
    # remove pixels that are not within 45 degrees of wind direction
    data_sam_foreground = data_sam_foreground[angle_diff < np.pi/4]

    # calculate distance to target of each sounding
    distance = cdist(data_sam_foreground[['latitude','longitude']].astype(float), np.array([[target_lat, target_lon]]))

    # sort data by distance
    distance_id = np.squeeze(np.argsort(distance, axis=0))
    
    # make sure we have at least 10 soundings in the foreground that are closer than 0.2 deg to target
    if len(distance[distance < 0.2]) < 10:
        if custom_SAM:
            print('Not enough soundings in foreground for ' + data_sam['SAM'].iloc[0])
        return None, None
    
    # get foreground
    foreground = data_sam_foreground.iloc[distance_id[:int(len(distance_id) * 0.2)]]

    # make a subset of data_sam to calculate the background
    # remove any that are part of the foreground
    data_sam_background = data_sam.drop(foreground.index)
    # remove pixel that are within 45 degrees of wind direction from target
    # calculate angle of each pixel to target
    angle = np.arctan2(data_sam_background['latitude'] - target_lat, data_sam_background['longitude'] - target_lon)
    # make all angles positive
    angle[angle < 0] = angle[angle < 0] + 2*np.pi
    # calculate angle between wind direction and pixel
    angle_diff = np.minimum(np.abs(wind_dir - angle), 2*np.pi - np.abs(wind_dir - angle))
    # remove pixels that are within 45 degrees of wind direction
    data_sam_background = data_sam_background[angle_diff > np.pi/4]
    # make sure we still have some pixels left
    if len(data_sam_background) < 10:
        if custom_SAM:
            print('Not enough soundings in background for ' + data_sam['SAM'].iloc[0])
        return None, None
    
    # only keep 20% of pixels furthest from target
    distance = cdist(data_sam_background[['latitude','longitude']].astype(float), np.array([[target_lat, target_lon]]))
    distance_id = np.squeeze(np.argsort(distance, axis=0))
    background = data_sam_background.iloc[distance_id[-int(len(distance_id) * 0.2):]]
    if len(background) < 10:
        if custom_SAM:
            print('Not enough soundings in background for ' + data_sam['SAM'].iloc[0])
        return None, None
    
    return foreground, background


def SAM_enhancement(data, var, qf, name=None, save_fig=False, path='', custom_SAM = False):
    ''' calculate the difference in var from the center of the SAM to the edge of the SAM as a proxy for CO2 flux
    data: pd.DataFrame, data containing variables
    var: str, variable to calculate variance for
    qf: str, quality flag to use to filter data
    :return: float, average difference in var from center to edge of SAM [ppm]'''


    # define foreground as 20% of pixels closest to target of SAM
    # background is 20% of pixels furthest from target of SAM and opposite side as find direction

    # apply qf
    if qf is not None:
        data = data[data[qf] == 0]

    # get unique SAMs
    sams = data['SAM'].unique()
    # subset data by SAM
    enhancements = []

    # make sure we only run tqdm if we have more than 1 SAM
    sams_ = tqdm(sams) if len(sams) > 1 else sams

    for sam in sams_:
        data_sam = data[data['SAM'] == sam]
        # check that SAM has at least 100 soundings
        if len(data_sam) < 100:
            if custom_SAM:
                print('Not enough soundings for ' + sam)
            continue

        # get wind direction
        wind_dir = np.mean(np.arctan2(data_sam['windspeed_v_met'], data_sam['windspeed_u_met']))
        if wind_dir < 0: # make sure wind_dir is between 0 and 2pi
            wind_dir += 2*np.pi
        # get wind speed
        wind_speed = np.mean(np.sqrt(data_sam['windspeed_v_met']**2 + data_sam['windspeed_u_met']**2))

        # get target_lat and target_lon
        target_id = data_sam['target_id'].iloc[0]
        try:
            target_lat, target_lon = get_target_data(target_id)
        except:
            print('No target data found for ' + target_id)
            continue

        # Use helper function to get foreground and background
        foreground, background = get_foreground_background_indices(
            data_sam, target_lat, target_lon, wind_dir, wind_speed, custom_SAM
        )
        
        if foreground is None or background is None:
            continue

        # calculate enhancement in ppm
        enhancement = np.mean(foreground[var]) - np.mean(background[var])

        # calculate flux proxy in ppm m/s
        enhancement = enhancement * wind_speed

        # add to list for final summary statistics
        enhancements.append(enhancement)

    if len(enhancements) > 1:
        print('# of SAMs with calc enhancements: ' + str(len(enhancements)))
        # plot histogram of enhancements
        plt.figure(figsize=(5, 5))
        plt.hist(enhancements, bins = 20, range=(-10, 10))
        # vertical line at mean enhancement
        plt.axvline(x=np.mean(enhancements), color='r', linestyle='--', label='Mean ' + str(np.round(np.mean(enhancements), 2)) + ' ppm')
        plt.axvline(x=np.median(enhancements), color='b', linestyle='-',
                    label='Median ' + str(np.round(np.median(enhancements), 2)) + ' ppm')
        plt.axvline(x=0, color='k', linestyle='-')
        plt.legend()
        plt.xlabel('SAM enhancement [ppm m/s]')
        plt.ylabel('Number of SAMs')
        plt.title('SAM enhancement [' + var + ']')
        plt.tight_layout()
        if save_fig:
            plt.savefig(path + name + 'SAM_enhancement_' + var + '.png', dpi = 300)
        plt.close()
        # save enhancements to csv for later use
        np.savetxt(path + name + 'SAM_enhancements_' + var + '.csv', enhancements, delimiter=',')


    # return the mean enhancement
    enhancements = np.array(enhancements)
    # check if we have any non-nan enhancements

    # check if we have any enhancements that are not nan
    if len(enhancements) > 0:
        if  ~np.isnan(enhancements).all():
            return np.nanmean(enhancements)
        else:
            return np.nan
    else:
        return np.nan

def read_oco_netcdf_to_df(file_path, variables_to_read=None, raw_vars_to_remove_list=None):
    """
    Reads an OCO NetCDF file into a pandas DataFrame.
    Applies a default list of variables to remove (raw_vars_to_remove_list) before reading data,
    primarily for OCO-3 Lite files. This list can be overridden or disabled by passing an empty list.
    If variables_to_read is provided, it should contain the final desired (stripped) column names.

    Args:
        file_path (str): Path to the NetCDF file.
        variables_to_read (list, optional): List of final (stripped) variable names to keep in the DataFrame.
                                            If None, all discovered/processed variables are kept.
        raw_vars_to_remove_list (list, optional): List of variable names (full path or stripped)
                                                  to exclude before reading data.
                                                  Defaults to DEFAULT_RAW_LITE_VARS_TO_REMOVE.
                                                  Pass an empty list ([]) to disable default raw lite filtering.
                                                  The 'L1b/' prefix filter is also applied if this list is the
                                                  DEFAULT_RAW_LITE_VARS_TO_REMOVE constant.
    Returns:
        pd.DataFrame: DataFrame containing the read data, sorted by sounding_id.
                      Returns an empty DataFrame on error or if no variables are processed.
    """

    if raw_vars_to_remove_list is None:
        raw_vars_to_remove_list = DEFAULT_RAW_LITE_VARS_TO_REMOVE

    fill_value_from_make_pkl_temp = 999999
    data_dict_raw = {}
    
    try:
        with nc.Dataset(file_path, 'r') as nc_ds:
            all_discovered_paths_from_nc = []
            
            # First, discover all group variables
            for group_name, group in nc_ds.groups.items():
                for var_name in group.variables:
                    all_discovered_paths_from_nc.append(f"{group_name}/{var_name}")
            
            # Then, discover top-level variables - don't skip them even if similar names exist in groups
            for var_name in nc_ds.variables:
                # Only skip if the EXACT same path already exists (which shouldn't happen for top-level vars)
                if var_name not in all_discovered_paths_from_nc:
                    all_discovered_paths_from_nc.append(var_name)

            paths_to_actually_read = []
            active_remove_list = raw_vars_to_remove_list if raw_vars_to_remove_list else []

            for v_path in all_discovered_paths_from_nc:
                stripped_name = v_path.split('/')[-1]
                remove_this_var = False
                if v_path in active_remove_list or stripped_name in active_remove_list:
                    remove_this_var = True
                
                if not remove_this_var:
                    paths_to_actually_read.append(v_path)
            
            l_vars_to_initially_process_paths = paths_to_actually_read

            if not l_vars_to_initially_process_paths:
                print(f"Warning: No variables selected to process in {file_path} after pre-filtering.")
                return pd.DataFrame()

            for v_path in l_vars_to_initially_process_paths:
                if '/' in v_path:
                    group, var_name = v_path.split('/', 1)
                    if group in nc_ds.groups and var_name in nc_ds.groups[group].variables:
                         var_obj = nc_ds.groups[group].variables[var_name]
                    else: continue 
                else:
                    if v_path in nc_ds.variables:
                        var_obj = nc_ds.variables[v_path]
                    else: continue 
                val = var_obj[:]
                if hasattr(var_obj, '_FillValue'):
                    nc_fill = var_obj._FillValue
                    if np.issubdtype(val.dtype, np.integer):
                        val = np.where(val == nc_fill, fill_value_from_make_pkl_temp, val)
                    else:
                        val = np.where(val == nc_fill, np.nan, val)
                elif isinstance(val, np.ma.MaskedArray):
                    if np.issubdtype(val.dtype, np.integer):
                        val = np.where(val.mask, fill_value_from_make_pkl_temp, val.data)
                    else:
                        val = np.where(val.mask, np.nan, val.data)
                data_dict_raw[v_path] = val

            column_dict_intermediate = {}
            expected_length = None
            sounding_id_kpath_for_len_check = None
            for k_path_iter in data_dict_raw.keys():
                if k_path_iter.endswith('sounding_id'): 
                    sounding_id_kpath_for_len_check = k_path_iter
                    break
            if not sounding_id_kpath_for_len_check:
                for k_path_iter in data_dict_raw.keys():
                    if k_path_iter.split('/')[-1] == 'sounding_id':
                        sounding_id_kpath_for_len_check = k_path_iter
                        break
            
            if sounding_id_kpath_for_len_check:
                expected_length = len(data_dict_raw[sounding_id_kpath_for_len_check])
                s_id_clean_name = sounding_id_kpath_for_len_check.split('/')[-1]
                column_dict_intermediate[s_id_clean_name] = data_dict_raw[sounding_id_kpath_for_len_check]

            for v_path, values in data_dict_raw.items():
                feature_clean_name = v_path.split('/')[-1]
                if feature_clean_name == 'sounding_id' and feature_clean_name in column_dict_intermediate:
                    continue 

                if values.ndim == 1:
                    if expected_length is None:
                        expected_length = len(values)
                        if values.dtype == object and any(isinstance(item, (list, np.ndarray)) for item in values):
                            del column_dict_intermediate[feature_clean_name] # Should not be there yet
                            expected_length = None 
                        else:
                            column_dict_intermediate[feature_clean_name] = values
                    elif len(values) == expected_length:
                        if values.dtype == object and any(isinstance(item, (list, np.ndarray)) for item in values):
                            pass  # Skip ragged object arrays
                        else:
                            column_dict_intermediate[feature_clean_name] = values
                elif values.ndim > 1:
                    if feature_clean_name in ['vertex_latitude', 'vertex_longitude']:
                        if expected_length is None or values.shape[0] == expected_length:
                            column_dict_intermediate[feature_clean_name] = list(values)
                            if expected_length is None: 
                                expected_length = values.shape[0]
            
            if not column_dict_intermediate:
                print(f"Warning: column_dict_intermediate is empty for {file_path}.")
                return pd.DataFrame()
                
            df_intermediate = pd.DataFrame(column_dict_intermediate)
            del data_dict_raw, column_dict_intermediate 

            df_final = pd.DataFrame()
            if variables_to_read is not None:
                cols_to_keep = [col for col in variables_to_read if col in df_intermediate.columns]
                if 'sounding_id' not in cols_to_keep and 'sounding_id' in df_intermediate.columns:
                    cols_to_keep.append('sounding_id')
                elif 'sounding_id' not in df_intermediate.columns and 'sounding_id' in variables_to_read:
                     print(f"Warning: 'sounding_id' requested in variables_to_read but not found in df_intermediate for {file_path}.")                     

                if not cols_to_keep:
                    print(f"Warning: None of the requested variables_to_read found in df_intermediate for {file_path}. Returning empty DataFrame.")
                    return pd.DataFrame()
                df_final = df_intermediate[cols_to_keep].copy() 
            else:
                df_final = df_intermediate.copy()

            if df_final.empty:
                print(f"Warning: df_final is empty after variable selection for {file_path}.")
            
            if 'sounding_id' not in df_final.columns:
                print(f"Error: 'sounding_id' is not in the final DataFrame for {file_path}. Cannot process.")
                return pd.DataFrame()

            # --- Type Casting and Sorting (applied to df_final) ---
            common_casts = {
                'sounding_id': 'int64', 'land_water_indicator': 'int8', 'footprint': 'int8',
                'orbit': 'int32', 'operation_mode': 'int8', 'land_fraction': 'float32',
                'snow_flag': 'int8', 
                'xco2_quality_flag': 'int8',
                'swath_bias_corrected': 'int8'
            }
            for col, dtype in common_casts.items():
                if col in df_final.columns:
                    try:
                        if 'int' in dtype:
                            df_final[col] = df_final[col].replace(fill_value_from_make_pkl_temp, NC_FILL_VALUE_INT)
                            if df_final[col].isnull().any():
                                df_final[col] = df_final[col].fillna(NC_FILL_VALUE_INT)
                            df_final[col] = df_final[col].astype(dtype)
                        else:
                            df_final[col] = df_final[col].astype(dtype)
                    except Exception as e:
                        print(f"Warning: Could not cast column '{col}' to {dtype}. Error: {e}")

            for col in df_final.select_dtypes(include=np.integer).columns:
                if pd.api.types.is_numeric_dtype(df_final[col]):
                    if df_final[col].isin([fill_value_from_make_pkl_temp]).any():
                        df_final[col] = df_final[col].replace(fill_value_from_make_pkl_temp, NC_FILL_VALUE_INT)
            df_final = df_final.sort_values('sounding_id').reset_index(drop=True)
            
            # --- Auto-generate SAM column if missing (default behavior for Lite files) ---
            if 'SAM' not in df_final.columns and 'target_id' in df_final.columns and 'orbit' in df_final.columns:
                # Robustly handle target_id conversion to string
                if df_final['target_id'].dtype == 'object' and isinstance(df_final['target_id'].iloc[0], bytes):
                     df_final['target_id'] = df_final['target_id'].str.decode('utf-8').str.strip()
                else:
                    df_final['target_id'] = df_final['target_id'].astype(str).str.strip()
                
                df_final['orbit'] = df_final['orbit'].astype(str)
                df_final['SAM'] = df_final['target_id'] + '_' + df_final['orbit']
            
            return df_final

    except Exception as e:
        print(f"Error during read_oco_netcdf_to_df for {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def tg_overlap_agreement_metric(
    data,
    var='xco2',
    swath_grouping_threshold_angle=1.0,
    samples_per_swath=100,
    radius_deg=0.1,
    min_neighbors=3,
    min_soundings_for_swath=50,
    show_progress=False
):
    """
    Calculate Target-mode overlap agreement per scene (SAM identifier) by comparing
    soundings that lie on top of each other across different swaths.

    For each scene:
      - Create swath indices using PMA elevation change threshold
      - For each swath, randomly sample up to N soundings
      - For each sampled sounding, find all soundings in other swaths within radius_deg
      - Compute the standard deviation of XCO2 differences between the sampled sounding and neighbors
      - The scene metric is the mean of these per-sample std values

    Returns:
      pd.DataFrame with columns ["SAM", f"overlap_std_{var}", "num_samples_used"]
    """
    if 'pma_elevation_angle' not in data.columns:
        print("tg_overlap_agreement_metric: 'pma_elevation_angle' missing. Cannot compute swath groupings.")
        return pd.DataFrame()

    scenes = data['SAM'].unique()
    results = []

    scene_iter = tqdm(scenes) if show_progress else scenes
    for scene in scene_iter:
        df_scene = data[data['SAM'] == scene].copy()
        if len(df_scene) < 100:
            continue

        # Group into swaths
        df_scene.loc[:, 'swath'] = (df_scene['pma_elevation_angle'].diff().abs() > swath_grouping_threshold_angle).cumsum()
        # Filter swaths by minimum soundings
        swath_counts = df_scene.groupby('swath')[var].count()
        valid_swaths = swath_counts[swath_counts >= min_soundings_for_swath].index.tolist()
        if len(valid_swaths) < 2:
            continue

        df_scene = df_scene[df_scene['swath'].isin(valid_swaths)].copy()

        per_sample_std = []
        # For each swath, sample
        for swath_id, df_swath in df_scene.groupby('swath'):
            if len(df_swath) == 0:
                continue
            sample_n = min(samples_per_swath, len(df_swath))
            sampled = df_swath.sample(sample_n, replace=False, random_state=43)

            # Build other-swath pool
            df_others = df_scene[df_scene['swath'] != swath_id]
            if df_others.empty:
                continue

            other_latlon = df_others[['latitude', 'longitude']].to_numpy().astype(float)

            for _, row in sampled.iterrows():
                latlon = row[['latitude', 'longitude']].to_numpy().astype(float)[None, :]
                dists = cdist(latlon, other_latlon)
                # approx deg distance threshold; use direct deg threshold
                neighbor_mask = (dists.flatten() <= radius_deg)
                if not np.any(neighbor_mask):
                    continue
                neighbors = df_others.loc[neighbor_mask]
                if len(neighbors) < min_neighbors:
                    continue
                diffs = neighbors[var].to_numpy() - row[var]
                # Use standard deviation as requested
                std_here = np.nanstd(diffs)
                if np.isfinite(std_here):
                    per_sample_std.append(std_here)

        if len(per_sample_std) == 0:
            continue
        scene_metric = float(np.nanmean(per_sample_std))
        results.append({
            'SAM': scene,
            f'overlap_std_{var}': scene_metric,
            'num_samples_used': int(len(per_sample_std))
        })

    return pd.DataFrame(results)


def tg_overlap_before_after(
    data,
    original_var='xco2',
    corrected_var_candidates=(
        'xco2_swath_bc',  # NetCDF output name
        'xco2_swath-BC'   # In-memory processing name
    ),
    **metric_kwargs
):
    """
    Convenience function to compute overlap agreement before and after correction for TG scenes.
    Returns a DataFrame with columns: ["SAM", "overlap_std_before", "overlap_std_after", "improvement"].
    """
    corrected_var = None
    for cand in corrected_var_candidates:
        if cand in data.columns:
            corrected_var = cand
            break
    if corrected_var is None:
        print("tg_overlap_before_after: No corrected variable found. Expected one of: ", corrected_var_candidates)
        return pd.DataFrame()

    before_df = tg_overlap_agreement_metric(data, var=original_var, **metric_kwargs)
    after_df = tg_overlap_agreement_metric(data, var=corrected_var, **metric_kwargs)

    if before_df.empty or after_df.empty:
        return pd.DataFrame()

    before_df = before_df.rename(columns={f'overlap_std_{original_var}': 'overlap_std_before'})
    after_df = after_df.rename(columns={f'overlap_std_{corrected_var}': 'overlap_std_after'})
    merged = before_df[['SAM', 'overlap_std_before']].merge(after_df[['SAM', 'overlap_std_after']], on='SAM', how='inner')
    merged['improvement'] = merged['overlap_std_before'] - merged['overlap_std_after']
    return merged
