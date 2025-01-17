import numpy as np
#import ac3airborne
import pandas as pd
#import typhon as ty
import scipy.stats
import matplotlib.pyplot as plt
#from progressbar import ProgressBar
#from tensorflow import keras
from netCDF4 import Dataset
import datetime
#from glob import glob
import xarray as xr
#from math import radians, cos, sin, asin, sqrt, atan2


def normed_log10_pdf_v0(var,bin_nr=10,bin_start=-1,bin_end=4):
    """
    Function to calculate relative frequencies of occurence (pdf)
    for a specified variable in logarthmic space, for specified 
    range and bins.
    """
    
    bins = np.linspace(start=bin_start, stop=bin_end, num=bin_nr)
    
    # count zeros and exclude them from variable
    n_zeros = len(var[var==0.])
    var = var[var!=0.]
    
    n_absolute, bins = np.histogram(np.log10(var),bins=bins)
    
    n_total = np.sum(n_absolute)+n_zeros
    
    prop_zero = n_zeros/n_total
    
    n_normed = np.array([n_absolute[i]/n_total for i in range(len(n_absolute))])
    n_normed[n_normed==0]=np.nan
    
    return n_normed, prop_zero, bins

def normed_log10_pdf(var,bin_nr=10,bin_start=-1,bin_end=4,density=True):
    """
    Function to calculate relative frequencies of occurence (pdf)
    for a specified variable in logarthmic space, for specified 
    range and bins.
    """
    
    bins = np.linspace(start=bin_start, stop=bin_end, num=bin_nr)
    
    if density == True:
        n, bins = np.histogram(np.log10(var), bins=bins, density=True)
        n[n==0.]=np.nan
    if density == False:
        n, bins = np.histogram(np.log10(var), bins=bins, density=False)
    
    return n, bins
    

def integrate_hydm(hym_c,p,T,RH,z,z_max=None,eq_distant=True,rho_moist=False,axis=2):
    """
    Vertical integration of ICON hydrometeor contents and water vapor.
    
    Parameters
    ----------
    hym_c : array_like
        3D Array containing all 5 hydrometeor classes of ICON
        Array must contain hydrometeors in following order:
        cwc,cic,crc,csc,cgc
    p : array_like
        1D ICON pressure profile
    T : array_like
        1D ICON temperature profile
    RH : array_like
        1D ICON relative humidity profile
    z : array_like
        1D ICON height profile
    z_max : Float
        Maximum height until which shall be integrated.
        If None, integration over whole ICON profiles.
    axis : Int
        Axis of hym_c Array along which shall be integrated.
    rho_moist : bool
        True if wet density should be used for integration.
        False if dry density.
       

    Returns
    -------
    2D Arrays of ice & liquid water path (vertically integrated hydrometeor contents) 
    and integrated water vapor.
    
    """
    
    # add all frozen hydrometeor contents
    # cic + csc + cgc
    hym_frozen = hym_c[:,:,1] + hym_c[:,:,3] + hym_c[:,:,4]
    # add all liquid hydrometeor contents
    # cwc + crc
    hym_liquid = hym_c[:,:,0] + hym_c[:,:,2]
    
    # calculates the wet density by the virtual temperature
    if rho_moist==True:
        vmr = ty.physics.relative_humidity2vmr(RH, p, T)
        q = ty.physics.vmr2specific_humidity(vmr)
        Tv = T*(1+0.61*q)
        rho = ty.physics.density(p,Tv)
    # dry density
    else:
        rho = ty.physics.density(p,T)
    
    # height difference = integration length
    if eq_distant == True: # constant height diff
        dz = np.diff(z)[0]
    if eq_distant == False: # height diff vector
        dz = np.diff(z,prepend=0)
        
    # find height grid-index of maximum height, if specified
    if z_max is not None:
        z_diffs = np.absolute(z - z_max)
        z_max_index = np.argmin(z_diffs)+1
    if z_max is None:
        z_max_index = None
    
    if hym_c.shape[0]==1: # single profile
        # Vertically integrate frozen/liquid hydrometeors and water vapor
        fwp = np.nansum(hym_frozen.squeeze()[:z_max_index] * rho[:z_max_index] * dz, axis=axis)
        lwp = np.nansum(hym_liquid.squeeze()[:z_max_index] * rho[:z_max_index] * dz, axis=axis)
        iwv = np.nansum(q[:z_max_index] * rho[:z_max_index] * dz, axis=axis)
    if hym_c.shape[0]>1: # multiple profiles
        # Vertically integrate frozen/liquid hydrometeors and water vapor
        iwp = np.nansum(hym_c[:,:z_max_index,1] * rho[:,:z_max_index] * dz, axis=axis)
        swp = np.nansum(hym_c[:,:z_max_index,3] * rho[:,:z_max_index] * dz, axis=axis)
        gwp = np.nansum(hym_c[:,:z_max_index,4] * rho[:,:z_max_index] * dz, axis=axis)
        fwp = np.nansum(hym_frozen[:,:z_max_index] * rho[:,:z_max_index] * dz, axis=axis)
        
        cwp = np.nansum(hym_c[:,:z_max_index,0] * rho[:,:z_max_index] * dz, axis=axis)
        rwp = np.nansum(hym_c[:,:z_max_index,2] * rho[:,:z_max_index] * dz, axis=axis)
        lwp = np.nansum(hym_liquid[:,:z_max_index] * rho[:,:z_max_index] * dz, axis=axis)
        iwv = np.nansum(q[:,:z_max_index] * rho[:,:z_max_index] * dz, axis=axis)
    
    return np.array([iwp,swp,gwp,fwp]), np.array([cwp,rwp,lwp]), iwv
    

def correct_radardata(ds,FLIGHT_NR):
    """
    Some rough radar corrections to get altitude and time
    of unprocessed HAMP radar measurements.
    """

    flight = f'HALO-AC3_HALO_RF{FLIGHT_NR}'
    cat = ac3airborne.get_intake_catalog()

    gps_ins = cat['HALO-AC3']['HALO']['GPS_INS'][flight](
        user='maximilian.ringel',password='HALOac32022!').to_dask()

    gps_alt_interpol = gps_ins.alt.interp(time=ds.time)
    gps_alt_interpol_2d = np.tile(gps_alt_interpol,(len(ds.range),1))
    range_2d = np.tile(ds.range,(len(gps_alt_interpol),1)).T
    alt_2d = gps_alt_interpol_2d - range_2d# - 580.
    time_2d = np.tile(ds.time,(len(ds.range),1))

    return alt_2d, time_2d

def xr_keep(obj, varlist):
    """drop all data_vars exept the ones provided in `varlist` """
    drop_vars = [a for a in obj.data_vars if a not in varlist]

    return obj.drop(drop_vars)

def get_takeoff_landing_ts(FLIGHT_NR):
    """ Extract takeoff and landing time from flight meta data """
    flight = f'HALO-AC3_HALO_RF{FLIGHT_NR}'
    meta = ac3airborne.get_flight_segments()

    takeoff_ts = meta['HALO-AC3']['HALO'][flight]['takeoff']
    landing_ts = meta['HALO-AC3']['HALO'][flight]['landing']

    return takeoff_ts,landing_ts

def find_nn(lon_model, lat_model, lon_point, lat_point):
    """
    Caclulate the 2D distances in km between a given point coordinate and a 2D grid
    (both given in decimal degrees) on the sphere of Earth, using the harvesine function.
    """
    # convert lons and lats from decimal degrees to radians for calculation
    lon_model = np.deg2rad(lon_model)
    lat_model = np.deg2rad(lat_model)
    lon_point = np.deg2rad(lon_point)
    lat_point = np.deg2rad(lat_point)

    # create array of point lons, lats of model grid shape
    lon_point_arr = np.zeros(len(lon_model))
    lat_point_arr = np.zeros(len(lat_model))
    lon_point_arr[:] = lon_point
    lat_point_arr[:] = lat_point

    # calculate lon, lat distances between model and point
    dlon = lon_point - lon_model
    dlat = lat_point - lat_model 

    # calculate 2D distances between model and 
    # point on the sphere of earth using the harvesine function
    a = np.sin(dlat/2)**2 + np.cos(lat_model) * np.cos(lat_point) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    d = c * r

    # returning model grid index of closest grid box 
    # to given point coordinate and the corresponding distance
    nn_ind = np.unravel_index(np.argmin(d, axis=None), d.shape)
    d_min = d[nn_ind]

    return nn_ind, d_min

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def find_nearest_value(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def int_column(var,plevels):
    """Integrate a variable over a vertical column weighted by pressure level differences"""
    # create array with pressure level differences
    p_diff = np.diff(plevels)
    # insert pressure difference of lowest level for shap matching
    p_diff = np.insert(p_diff,0,25)
    # convert from hpa to pa
    p_diff = p_diff*100

    # integrate var over vertical column, weighted by the pressure differences
    int_var = np.array([np.sum((var[i,:]*1000)*p_diff)/(9.81*1000) for i in range(len(var[:,0]))])
    # set negative (non-pyhsical values) to NAN
    int_var[int_var<0] = np.nan

    return int_var

def adapt_icon_time_index(da,date):
    """
    The ICON time coordinates are given in ordinal format,
    this function changes the index into the typical format
    %YYYY-%MM-%DD %HH:%MM (so rounded to minutes). This is required for the
    precedent interpolation in time to minutely frequencies.
    da : xr.DataArray
        ICON-Variable as DataArray (xarray).
    date : str
        str with 8 letters specifying the date, format: YYYYMMDD
    flights : list
        list of given flight
    Returns
    -------
    da : xr.DataArray
        ICON-Variable as DataArray (xarray) with updated Time Coordinates.
    """
    #Create index, fixed for 10-minute data
    if "-" in date:
        start_date=date[0:4]+date[5:7]+date[8:10]
    else:
        start_date=date#years[flight[0]]+months[flight[0]]+days[flight[0]]
    new_time_index=pd.to_datetime(abs(int(start_date)-np.array(da.time)),
                                  unit="d",origin=start_date)
    new_time_index=new_time_index.round("min")
    #Assign this index to the DataArray
    da=da.assign_coords(time=new_time_index)
        
    return da
    
def preprocess_icon_files(ds):
    """
    Function to preprocess ICON files before merging them 
    into one single multifile with xr.open_mfdataset() by:
     - converting (numerical) ICON timestamps into datetime timestamps
     - droping all but the desired variables 
     - droping unnecessary height dimensions
    """
    ds = adapt_icon_time_index(ds,DATE)
    ds = xr_keep(ds, ['z_mc','pres','temp','rh',
                          'tot_qc_dia','qr','tqc_dia',
                          'tot_qi_dia','qs','qg','tqi_dia',
                         ])
    ds = ds.drop(['height_2','height_3','height_4'])
    return ds

def preprocess_icon_forcing_files(ds,date):
    """
    Function to preprocess ICON files before merging them 
    into one single multifile with xr.open_mfdataset() by:
     - converting (numerical) ICON timestamps into datetime timestamps
     - droping unnecessary dimensions
     - droping unnecessary variables
    """
    ds = adapt_icon_time_index(ds,date)
    ds = ds.drop_dims(
        ['ncells','vertices_2','bnds',
         'height','height_2','height_3','height_4',
         'depth','depth_2'])
    ds = ds.drop(
        ['gz0','t_ice','h_ice','alb_si','w_i','t_snow','w_snow',
         'rho_snow','h_snow','snowfrac_lc','freshsnow'])
    
    return ds

def unix_to_dt64_timestamps(unix_timestamps):
    """
    Function to convert numerical unix timestamps 
    to python datetime64 timestamps.
    """

    dt64_timestamps = np.array(
        [datetime.datetime.utcfromtimestamp(unix_timestamps[t]) 
         for t in range(len(unix_timestamps))]).astype(np.datetime64)
    
    return dt64_timestamps

def get_HAMP_freqs_of(select='all_2side'):
    """
    Function to return frequenices of specified HAMP channel(s).
    """
  
    HAMP_freqs = {'K_band':[22.24,23.04,23.84,25.44,26.24,27.84,31.40],
                  'V_band':[50.30,51.76,52.8,53.75,54.94,56.66,58.00],
                  'W_band':[90.00],
                  'F_band_1side':[118.75+1.4,118.75+2.3,118.75+4.2,118.75+8.5],
                  'F_band_2side':[118.75-8.5,118.75-4.2,118.75-2.3,118.75-1.4,
                                  118.75+1.4,118.75+2.3,118.75+4.2,118.75+8.5],
                  'G_band_1side':[183.31+0.6,183.31+1.5,183.31+2.5,183.31+3.5,183.31+5.0,183.31+7.5,183.31+12.5],
                  'G_band_2side':[183.31-12.5,183.31-7.5,183.31-5.0,183.31-3.5,183.31-2.5,183.31-1.5,183.31-0.6,
                                  183.31+0.6,183.31+1.5,183.31+2.5,183.31+3.5,183.31+5.0,183.31+7.5,183.31+12.5],
                  'NN_freqs':[22.24,23.04,23.84,25.44,26.24,27.84,31.40,
                              50.30,51.76,52.8,53.75,54.94,56.66,58.00,
                              90.00,
                              118.75+1.4,118.75+2.3,118.75+4.2,118.75+8.5,
                              183.31+0.6,183.31+2.5,183.31+3.5,183.31+5.0,183.31+7.5],
                 }
    
    
    if select == 'all_1side':
        freqs = np.concatenate((
            np.array(HAMP_freqs['K_band']),
            np.array(HAMP_freqs['V_band']),
            np.array(HAMP_freqs['W_band']),
            np.array(HAMP_freqs['F_band_1side']),
            np.array(HAMP_freqs['G_band_1side']),))
            
    elif select == 'all_2side':
        freqs = np.concatenate((
            np.array(HAMP_freqs['K_band']),
            np.array(HAMP_freqs['V_band']),
            np.array(HAMP_freqs['W_band']),
            np.array(HAMP_freqs['F_band_2side']),
            np.array(HAMP_freqs['G_band_2side']),))
   
    else:
        freqs = HAMP_freqs[select]
            
    return np.array(freqs)

def hamp_offset_correction(TBs,date):
    
    offset_corrections = xr.open_dataset(f'/home/u/u301238/master_thesis/TB_offset_corrections/HALO-AC3_HALO_HAMP_TB_offset_correction_{date}.nc')
    
    offset_corrections = offset_corrections.where(TBs_offsets.freq != 184.81, drop=True)
    bias = offset_corrections.bias.values.squeeze()
    slope = offset_corrections.slope.values.squeeze()
    
    TBs_corrected = [TBs[TB,freq]+(bias[freq]*slope[freq]) for TB in range(len(TBs)) for freq in range(24)]
    
    return TBs_corrected

def load_all_hamp_obs_v1(freqs=[],sfc_type='all'):
    
    """
    Function to load all hamp observations of the HALO-AC3 campaign.
    """
    
    # create list of all along-track ICON surface properties files 
    sfc_files = sorted(glob('/work/um0203/u301238/ICON/HALO-ICON_collocations/collocation_HALO-ICON_AC3_*_sfc_properties.nc'))

    # preprocess ICON surface files by dropping unnecessary dimensions
    def preprocess_sfc_data(ds):
        ds = ds.drop(['time_icon'])
        ds = ds.update({'time_hamp': ('time_hamp',unix_to_dt64_timestamps(ds.time_hamp))})
        return ds

    # read in all along-track ICON surface properties as one dataset
    sfc = xr.open_mfdataset(
        sfc_files,
        preprocess = preprocess_sfc_data,
        combine='nested',
        concat_dim='time_hamp')

    # extract land- and sea-ice-mask from surface files
    landmask = xr.DataArray(sfc.fr_land.values,dims=['time'])
    seaicemask = xr.DataArray(sfc.fr_seaice.values,dims=['time'])
    
    # create list of all hamp files
    # (delete files which belong to transfer flights and flights which do not have
    # corresponding surface files, due to missing ICON NWP data)
    hamp_files = sorted(glob('/work/bb1086/haloac3_unified_hamp/radiometer_*'))
    del hamp_files[0] # transfer flight
    del hamp_files[0] # RF02 (no ICON NWP data)
    del hamp_files[3] # RF05 (no ICON NWP data)
    del hamp_files[2] # RF06 (no ICON NWP data)
    del hamp_files[-1] # transfer flight
    
    # read in all hamp files as one dataset
    hamp = xr.open_mfdataset(
        hamp_files,
        combine='nested',
        concat_dim='time')

    # select specified frequenices (default = all / len(freqs) = 0) 
    # and timestamps equal to those of the pamtra simulations
    if len(freqs) != 0:
        hamp = hamp.sel(
            uniRadiometer_freq=xr.DataArray(freqs,dims=['uniRadiometer_freq']),
            time=sfc.time_hamp.values)
    else:
        hamp = hamp.sel(
            time=sfc.time_hamp.values)

    # select only measurements over desired surface type
    if sfc_type == 'ocean':
        hamp = hamp.where((landmask==0)&(seaicemask==0),drop=True)
    if sfc_type == 'seaice':
        hamp = hamp.where((landmask==0)&(seaicemask==1),drop=True)
    if sfc_type == 'land':
        hamp = hamp.where((landmask==1)&(seaicemask==0),drop=True)
    if sfc_type == 'all':
        hamp = hamp

    return hamp

def load_all_hamp_obs_v2(freqs='all_1side',sfc_type='all',alt_range=None,timestamps=None):
    """
    New function to load all HAMP observations of updated radiometer nc files
    (including surface mask in newer version).
    """
    
    
    hamp_files = sorted(glob('/home/u/u301238/master_thesis/HAMP_corr/radiometer_*_v1.6_TBcorr.nc'))
    del hamp_files[4] # RF06 (no ICON NWP data)
    
    #hamp_files = sorted(glob('/work/bb1086/haloac3_unified_hamp/radiometer_*_v1.6.nc'))
    #del hamp_files[0] # RF01 (transfer flight)
    #del hamp_files[4] # RF06 (no ICON NWP data)

    hamp = xr.open_mfdataset(
        hamp_files,
        combine='nested',
        concat_dim='time')
    
    if alt_range is not None:
        hamp = hamp.where((hamp.alt>=alt_range[0])&(hamp.alt<=alt_range[1]),drop=True)
        
    if timestamps is not None:
        hamp = hamp.sel(time=timestamps,drop=True)

    frequencies = get_HAMP_freqs_of(freqs)
    if (freqs == 'all_1side') or (freqs == 'G_band_1side'):
        frequencies = np.delete(frequencies,-1) # remove 195 GHz channel (was not working during HALO-AC3)
        
    hamp = hamp.sel(uniRadiometer_freq=xr.DataArray(frequencies,dims='uniRadiometer_freq'))
    
    if sfc_type == 'ocean':
        hamp = hamp.where(hamp.surface_mask == 0.,drop=True)
    if sfc_type == 'land':
        hamp = hamp.where(hamp.surface_mask < 0.,drop=True)
    if sfc_type == 'sea-ice':
        hamp = hamp.where(hamp.surface_mask > 0., drop=True)
    if sfc_type == 'all':
        hamp = hamp
        
    return hamp

def rolling_window(a, window):
    """
    Make an ndarray with a rolling window of the last dimension

    Parameters
    ----------
    a : array_like
       Array to add rolling window to
    window : int
       Size of rolling window

    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.
    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
          [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])
    Calculate rolling mean of last dimension:
    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
          [ 6.,  7.,  8.]])
    """
    if window < 1:
       raise ValueError("`window` must be at least 1.")
    if window > a.shape[-1]:
       raise ValueError("`window` is too long.")
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def extend_reduced_rolling(a, window):
    """Extend the result of a reduced rolling_window (e.g. .mean(axis-1) to the old length.
    window/2 'np.nan's are inserted before a and appended at the end of a.

    Parameters
    ----------
    a : array_like
       Array to be extended
    window : int
       Size of rolling window (== number of nans that are added)

    Returns
    -------
    Array
    """
    return np.concatenate([
        np.full(a.shape[:-1] + ((window-1)//2, ), np.nan),
        a,
        np.full(a.shape[:-1] + (window//2, ), np.nan),
    ], axis=-1)


def replace_outliers_with_CHmean(TB_array,lower_thrs):
    """
    Function to replace brightness temperature values below a lower threshold with average 
    brightness temperature (excluding values below threshold) of the corresponding channel
    """
    if len(TB_array.shape) == 2:
        
        if TB_array.shape[1] > TB_array.shape[0]:
            raise ValueError('Input array has wrong shape. Shape has to be [profile,channel].')
        
        for channel in range(TB_array.shape[1]):
            for profile in range(TB_array.shape[0]):

                TB = TB_array[profile,channel]

                if TB < lower_thrs:
                    TB_array[profile,channel] = np.nanmean(TB_array[:,channel][TB_array[:,channel]>lower_thrs])
    
    if len(TB_array.shape) == 1:
        for profile in range(TB_array.shape[0]):

            if TB_array[profile] < lower_thrs:
                TB_array[profile] = np.nanmean(TB_array[:][TB_array[:]>lower_thrs])
    
    return TB_array

def standardize_nn_training_data(TBs_train):
        
    TBs_centered = np.zeros(TBs_train.shape)
    mu_train = np.zeros(TBs_train.shape[1])
    sigma_train = np.zeros(TBs_train.shape[1])
    for channel in range(TBs_train.shape[1]):

        mu_train[channel] = np.nanmean(TBs_train[:,channel])
        sigma_train[channel] = np.nanstd(TBs_train[:,channel])   

        TBs_centered[:,channel] = (TBs_train[:,channel] - mu_train[channel])/sigma_train[channel]
        
    return TBs_centered, mu_train, sigma_train

def standardize_nn_input_data_v1(TBs, nn_level=12500,version=1):
    """
    Center / Standardize a given array for its second axis with mean 0 and std of 1.
    """
    
    TBs = np.asarray(TBs)
    # if single TB observation provided extend dims to 2
    if len(TBs.shape) == 1:
        TBs = TBs[np.newaxis,:]
    
    # if given array of TBs represents TBs not part of the training database
    # standardize them with respect to the mean and std of the training database TBs
    
        
    TBs_train = xr.open_dataset(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/train_test_data/train_test_data_nn_24-32-1_reg_{nn_level}m_v{version}.nc').tb_train.values
    
    if TBs.shape[1] != TBs_train.shape[1]:
        print("Given TB array has to have same frequencies as provided in the respective training database of the NN.")
        return 
        
    TBs_centered = np.zeros(TBs.shape)
    for channel in range(TBs.shape[1]):

        mu_train = np.nanmean(TBs_train[:,channel])
        sigma_train = np.nanstd(TBs_train[:,channel])   

        TBs_centered[:,channel] = (TBs[:,channel] - mu_train)/sigma_train
            
    return TBs_centered

def standardize_nn_input_data_v2(TBs, mu, sigma):
    
    TBs = np.asarray(TBs)
    # if single TB observation provided extend dims to 2
    if len(TBs.shape) == 1:
        TBs = TBs[np.newaxis,:]
    
    TBs_centered = np.zeros(TBs.shape)
    for channel in range(TBs.shape[1]):

        TBs_centered[:,channel] = (TBs[:,channel] - mu[channel])/sigma[channel]
    
    return TBs_centered
    

def add_noise(clean_signal,sigma=0.75):
    
    mu = 0
    
    noise = np.random.normal(mu, sigma, clean_signal.shape) 
    noisy_signal = clean_signal + noise
    
    return noisy_signal

def get_RF_date_of(RF):
    """
    Function to return date of specified HALO-AC3 research flight.
    """
    
    RF_dates = {'RF02':'20220312',
                'RF03':'20220313',
                'RF04':'20220314',
                'RF05':'20220315',
                'RF06':'20220316',
                'RF07':'20220320',
                'RF08':'20220321',
                'RF09':'20220328',
                'RF10':'20220329',
                'RF11':'20220330',
                'RF12':'20220401',
                'RF13':'20220404',
                'RF14':'20220407',
                'RF15':'20220408',
                'RF16':'20220410',
                'RF17':'20220411',
                'RF18':'20220412',
               }
    
    return RF_dates[RF]

 
def get_ICON_along_RF(RF,sfc_type='all'):
    """
    Returns the collocated ICON profiles as well as the collocated ICON surface properties
    of specified HALO research flight.
    """
    
    # Get date of research flight
    DATE = get_RF_date_of(RF)
    
    # Read in collocated ICON profile data
    ICON_profiles = xr.open_dataset(
        f'/work/um0203/u301238/ICON/HALO-ICON_collocations/collocation_HALO-ICON_AC3_{DATE}_{RF}_delta-t_1_v5.nc')

    # Convert ICON numerical timestamps to np.datetime64 timestamps and use these
    # to replace the numerical timestamps in the xr.dataset
    timestamps0 = np.array(
        [datetime.datetime.utcfromtimestamp(ICON_profiles.time_hamp.values[t]) 
        for t in range(len(ICON_profiles.time_hamp))]).astype(datetime.datetime)
    ICON_profiles = ICON_profiles.update({'time_hamp': ('time_hamp',timestamps0)})
    
    timestamps0 = np.array(
        [datetime.datetime.utcfromtimestamp(ICON_profiles.time_icon.values[t]) 
        for t in range(len(ICON_profiles.time_icon))]).astype(datetime.datetime)
    ICON_profiles = ICON_profiles.update({'time_icon': ('time_icon',timestamps0)})
    
    # Read in collocated ICON surface data
    ICON_surface = xr.open_dataset(
        f'/work/um0203/u301238/ICON/HALO-ICON_collocations/collocation_HALO-ICON_AC3_{DATE}_{RF}_sfc_properties_v2.nc')
    
    ICON_surface = ICON_surface.drop(['time_icon'])
    ICON_surface = ICON_surface.update({'time_hamp': ('time_hamp',unix_to_dt64_timestamps(ICON_surface.time_hamp))})

    # Extract land- and sea-ice-mask from surface file
    landmask = xr.DataArray(ICON_surface.fr_land.values,dims=['time_hamp'])
    seaicemask = xr.DataArray(ICON_surface.fr_seaice.values,dims=['time_hamp'])
    
    # select only measurements over desired surface type
    if sfc_type == 'ocean':
        ICON_profiles = ICON_profiles.where((landmask==0)&(seaicemask==0),drop=True)
        ICON_surface = ICON_surface.where((landmask==0)&(seaicemask==0),drop=True)
    if sfc_type == 'seaice':
        ICON_profiles = ICON_profiles.where((landmask==0)&(seaicemask==1),drop=True)
        ICON_surface = ICON_surface.where((landmask==0)&(seaicemask==1),drop=True)
    if sfc_type == 'land':
        ICON_profiles = ICON_profiles.where((landmask==1)&(seaicemask==0),drop=True)
        ICON_surface = ICON_surface.where((landmask==1)&(seaicemask==0),drop=True)
    if sfc_type == 'all':
        ICON_profiles = ICON_profiles
        ICON_surface = ICON_surface

    return ICON_profiles, ICON_surface

def plot_scatterdensity(a,b,xlabel,ylabel,log=False):
    """
    Plots a scatterdensity plot for two parameters a,b.
    """

    fig, axs = plt.subplots(figsize=(9,8))
    
    if log == True:
        a = np.log10(a)
        b = np.log10(b)
        
        a[a<0.] = np.nan
        b[b<0.] = np.nan
    if log == False:
        a[a<1.] = np.nan
        b[b<1.] = np.nan

    nans = np.logical_or(np.isnan(a), np.isnan(b))
    a = a[~nans]
    b = b[~nans]

    maxval = max(np.max(a),np.max(b))
    minval = max(np.min(a),np.min(b))
    
    # Calculate the point density
    ab = np.vstack([a,b])
    c = scipy.stats.gaussian_kde(ab)(ab)

    # Sort the points by density, so that the densest points are plotted last
    idx = c.argsort()
    a, b, c = a[idx], b[idx], c[idx]

    bias = np.round((np.mean(b) - np.mean(a)),2)
    corr = np.round((scipy.stats.pearsonr(a,b)[0]),2)
    rmse = np.round(np.sqrt(np.nanmean((b-a)**2)),2)

    axs.grid(alpha=0.5)
    if log == True:
        sc = axs.scatter(10**a,10**b,c=c,s=25,cmap='viridis',vmin=np.min(c),vmax=np.max(c))
        axs.plot(np.arange(10**0,10**maxval),np.arange(10**0,10**maxval),linewidth=2,color='black',alpha=0.5)
        axs.set_xscale('log')
        axs.set_yscale('log')
    if log == False:
        sc = axs.scatter(a,b,c=c,s=25,cmap='viridis',vmin=np.min(c),vmax=np.max(c))
        axs.plot(np.arange(0,maxval),np.arange(0,maxval),linewidth=2,color='black',alpha=0.5)
    
    #axs.scatter(y_test, y_test_predictions**2)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)

    cbar_ax = fig.add_axes([1.0, 0.15, 0.02, 0.85])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Point density',size=20)

    axs.text(0.05, 0.95, 'Bias: '+str(bias), 
             transform=axs.transAxes, fontsize=20,
             verticalalignment='top')
    axs.text(0.05, 0.885, 'RMSE: '+str(rmse), 
             transform=axs.transAxes, fontsize=20,
             verticalalignment='top')
    axs.text(0.05, 0.81, 'Corr: '+str(corr), 
             transform=axs.transAxes, fontsize=20,
             verticalalignment='top')

    #props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    #axs.text(1.05, 0.95, , 
    #         transform=axs.transAxes, fontsize=20,
    #         fontweight='bold',verticalalignment='top', bbox=props)


    #plt.savefig(f'/home/u/u301238/master_thesis/plots/NN/NN_scatterdensity_130322.jpg',
    #            bbox_inches='tight',dpi=300)
    
def plot_hist_of_input(input_vector):
    """
    Plots histograms of brightness temperatures of each given frequency.
    """
    fig, axs = plt.subplots(nrows=input_vector.shape[1],ncols=1,figsize=(20,20),sharex=True)
    bins = np.arange(np.nanmin(input_vector), np.nanmax(input_vector) + 1, 1)
    for channel in range(input_vector.shape[1]):

        axs[channel].grid(alpha=0.5)
        axs[channel].hist(input_vector[:,channel],ec='black',bins=bins)
        axs[channel].axvline(np.nanmean(input_vector[:,channel]),color='black')
        
def create_pamtra_TB_vector(pamtra_ds,outlevels):

    # select "nadir looking" BTs
    pamtra_ds = pamtra_ds.sel(angles=180,grid_y=0)
    pamtra_ds = pamtra_ds.drop(['grid_y','angles'])
    # average over v and h polarisation
    pamtra_ds = pamtra_ds.mean(dim='passive_polarisation')
    # get indices of specified altitudes
    level_inds = [np.where(pamtra_ds.outlevels.values[0,:].squeeze() == level)[0][0] for level in outlevels]
    # select pamtra dataset at specified altitudes
    pamtra_ds = pamtra_ds.sel(outlevel=xr.DataArray(level_inds,dims=['outlevel']))
    
    # select arrays of BTs of K,V,W band
    K_band = pamtra_ds.tb.sel(frequency=get_HAMP_freqs_of('K_band')).values[:,:,:]
    V_band = pamtra_ds.tb.sel(frequency=get_HAMP_freqs_of('V_band')).values[:,:,:]
    W_band = pamtra_ds.tb.sel(frequency=get_HAMP_freqs_of('W_band')).values[:,:].reshape(pamtra_ds.tb.values.shape[0],len(outlevels),1)
    
    # average over doubleside frequencies of F_band
    TB_120_mean = np.mean(pamtra_ds.tb.sel(frequency=xr.DataArray([118.75-1.4, 118.75+1.4],dims='frequency')),axis=2)
    TB_121_mean = np.mean(pamtra_ds.tb.sel(frequency=xr.DataArray([118.75-2.3, 118.75+2.3],dims='frequency')),axis=2)
    TB_122_mean = np.mean(pamtra_ds.tb.sel(frequency=xr.DataArray([118.75-4.2, 118.75+4.2],dims='frequency')),axis=2)
    TB_127_mean = np.mean(pamtra_ds.tb.sel(frequency=xr.DataArray([118.75-8.5, 118.75+8.5],dims='frequency')),axis=2)
    # create array of BTs of F_band
    F_band = np.empty([K_band.shape[0],len(outlevels),4])
    F_band[:,:,0] = TB_120_mean
    F_band[:,:,1] = TB_121_mean
    F_band[:,:,2] = TB_122_mean
    F_band[:,:,3] = TB_127_mean

    # average over doubleside frequencies of G_band
    TB_183_mean = np.mean(pamtra_ds.tb.sel(frequency=xr.DataArray([183.31-0.6, 183.31+0.6],dims='frequency')),axis=2)
    TB_184_mean = np.mean(pamtra_ds.tb.sel(frequency=xr.DataArray([183.31-1.5, 183.31+1.5],dims='frequency')),axis=2)
    TB_185_mean = np.mean(pamtra_ds.tb.sel(frequency=xr.DataArray([183.31-2.5, 183.31+2.5],dims='frequency')),axis=2)
    TB_186_mean = np.mean(pamtra_ds.tb.sel(frequency=xr.DataArray([183.31-3.5, 183.31+3.5],dims='frequency')),axis=2)
    TB_188_mean = np.mean(pamtra_ds.tb.sel(frequency=xr.DataArray([183.31-5.0, 183.31+5.0],dims='frequency')),axis=2)
    TB_190_mean = np.mean(pamtra_ds.tb.sel(frequency=xr.DataArray([183.31-7.5, 183.31+7.5],dims='frequency')),axis=2)
    TB_195_mean = np.mean(pamtra_ds.tb.sel(frequency=xr.DataArray([183.31-12.5, 183.31+12.5],dims='frequency')),axis=2)
    # create array of BTs of G_band
    G_band = np.empty([K_band.shape[0],len(outlevels),5])
    G_band[:,:,0] = TB_183_mean
    G_band[:,:,1] = TB_185_mean
    G_band[:,:,2] = TB_186_mean
    G_band[:,:,3] = TB_188_mean
    G_band[:,:,4] = TB_190_mean
    #G_band = replace_outliers_with_CHmean(G_band,lower_thrs=230)

    TB_vector = np.concatenate((
        K_band,
        V_band,
        W_band,
        F_band,
        G_band),
        axis=2)
    
    #print("\nCreated ",TB_vector.shape, " TB input vector")
    return TB_vector

def load_nn_training_data(altitude=12500):
    
    #print("Loading PAMTRA training data (TBs)...")
    # create list of all pamtra simulations of retrieval database
    # create list of all pamtra simulations of retrieval database
    pamtra_files = sorted(glob('/work/um0203/u301238/PAMTRA/PAMTRA_NN_training_data/PAMTRA-ICON_*_4000rndm-profiles_all_hamp_freqs_v3.nc'))

    # open them as one concatenated multifile dataset
    pamtra = xr.open_mfdataset(
        pamtra_files,
        combine='nested',
        concat_dim='grid_x')

    # create a (profile,frequency) TB input vector out of the PAMTRA simulated TBs 
    # by averaging over all doubleside frequencies
    TB_input_vector = create_pamtra_TB_vector(pamtra,outlevels=[altitude])
    TB_input_vector = TB_input_vector[:,0,:]

    # Add random noise to the simulated TBs
    for channel in range(TB_input_vector.shape[1]):
        if (channel >= 0) & (channel <= 6): # K-Band
            TB_input_vector[:,channel] = add_noise(TB_input_vector[:,channel],sigma=0.1)
        if (channel >= 7) & (channel <= 13): # V-Band
            TB_input_vector[:,channel] = add_noise(TB_input_vector[:,channel],sigma=0.2)
        if channel == 14: # W-Band
            TB_input_vector[:,channel] = add_noise(TB_input_vector[:,channel],sigma=0.25)
        if (channel >= 15) & (channel <= 18): # F-Band
            TB_input_vector[:,channel] = add_noise(TB_input_vector[:,channel],sigma=0.6)
        if (channel >= 19) & (channel <= 23): # G-Band
            TB_input_vector[:,channel] = add_noise(TB_input_vector[:,channel],sigma=0.6)

    # Load in numpy arrays containing ICON hydrometeor contents of PAMTRA simulations       
    ICON_array_list = sorted(glob('/home/u/u301238/master_thesis/ICON/ICON_NN_training_data/ICON_NWP_HALOAC3-domain_*_4000rndm-profiles_profiles.npy'))

    ICON_arrays = np.load(ICON_array_list[0])
    for i in range(1,len(ICON_array_list)):
        ICON_arrays = np.concatenate((ICON_arrays,np.load(ICON_array_list[i])),axis=0)

    # calculate hydrometeor integrals
    frozen_water, liquid_water, IWV = integrate_hydm(
        hym_c = ICON_arrays[:,:,4:]*1000,
        p = ICON_arrays[:,:,1],
        T = ICON_arrays[:,:,2],
        RH = ICON_arrays[:,:,3]/100,
        z = ICON_arrays[:,:,0],
        z_max = None,
        eq_distant = False,
        rho_moist=True,
        axis=1)

    return TB_input_vector, frozen_water, liquid_water, IWV

def split_nn_training_data(TBs,IWP,LWP=None,IWV=None,split_ratio=0.75):
    
    # create a random choice of profile indices of the training dataset
    # (as profiles are sorted by time otherwise, splitting wouldn't make sense then)
    rndm_choice = np.random.permutation(len(IWP))

    # get split index at which data hast to be splitted to fullfill the specified split_ratio
    split_ind = int(split_ratio*len(IWP))

    # X
    # reorder X-vectors (atm. condition) according to random choice of profile indices
    IWP_rndm_srt = IWP[rndm_choice]
    # split into train and test data
    IWP_train = IWP_rndm_srt[0:split_ind]
    IWP_test = IWP_rndm_srt[split_ind:]
    
    if LWP is not None:
        LWP_rndm_srt = LWP[rndm_choice]
        # split into train and test data
        LWP_train = LWP_rndm_srt[0:split_ind]
        LWP_test = LWP_rndm_srt[split_ind:]
        
    if IWV is not None:
        IWV_rndm_srt = IWV[rndm_choice]
        # split into train and test data
        IWV_train = IWV_rndm_srt[0:split_ind]
        IWV_test = IWV_rndm_srt[split_ind:]

    # Y
    # reorder Y-vector (TBs) according to random choice of profile indices
    TBs_rndm_srt = TBs[rndm_choice,:]
    # split into train and test data
    TBs_train = TBs_rndm_srt[0:split_ind,:]
    TBs_test = TBs_rndm_srt[split_ind:,:]

    return TBs_train, TBs_test, IWP_train, IWP_test, LWP_train, LWP_test, IWV_train, IWV_test

def clip_nn_output(nn_prediction,truth=None):
    
    negative_predictions = np.round((len(nn_prediction[nn_prediction<0.])/len(nn_prediction))*100,2)
    print(f"Negative predictions: {negative_predictions} %")

    # clip nn output / predictions (= set negative values to zero)
    nn_prediction_cliped = nn_prediction.copy()
    nn_prediction_cliped[nn_prediction_cliped<0.]=0.
    
    if truth is not None:
        # calculate bias before and after cliping
        bias_before = np.mean(nn_prediction) - np.mean(truth)
        bias_after = np.mean(nn_prediction_cliped) - np.mean(truth)

        print("Bias before cliping: ",np.round(bias_before,2))
        print("Bias after cliping: ",np.round(bias_after,2))

    return nn_prediction_cliped

def save_nn_train_test_retrieved_data(TBs_train, TBs_test, IWP_train, IWP_test, IWP_retrieved, nn_v):
    
    filename = (f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_{nn_v}/train_test_data_nn_{nn_v}.nc')
    ds = Dataset(filename,'w','NETCDF4')

    # profile and frequency dimensions
    ds.createDimension('train_cases',len(IWP_train))
    ds.createDimension('test_cases',len(IWP_test))
    ds.createDimension('freq',24)

    # TBs
    tb_train_var = ds.createVariable('tb_train','float64',('train_cases','freq'))
    tb_train_var.long_name = 'ICON-PAMTRA simulated brightness temperatures of training data set'
    tb_train_var.setncattr('units','K')
    tb_train_var[:,:] = TBs_train[:,:]
    tb_test_var = ds.createVariable('tb_test','float64',('test_cases','freq'))
    tb_test_var.long_name = 'ICON-PAMTRA simulated brightness temperatures of test data set'
    tb_test_var.setncattr('units','K')
    tb_test_var[:,:] = TBs_test[:,:]

    # IWP
    iwp_train_var = ds.createVariable('iwp_train','float64',('train_cases'))
    iwp_train_var.long_name = 'ICON simulated IWP values of training data set'
    iwp_train_var.setncattr('units','g m2')
    iwp_train_var[:] = IWP_train[:]
    iwp_test_var = ds.createVariable('iwp_test','float64',('test_cases'))
    iwp_test_var.long_name = 'ICON simulated IWP values of test data set'
    iwp_test_var.setncattr('units','g m2')
    iwp_test_var[:] = IWP_test[:]
    iwp_retr_var = ds.createVariable('iwp_retrieved','float64',('test_cases'))
    iwp_retr_var.long_name = 'Retrieved IWP values of test data set'
    iwp_retr_var.setncattr('units','g m2')
    iwp_retr_var[:] = IWP_retrieved[:]

    ds.close()
    
    print(f"Data saved to /home/u/u301238/master_thesis/nn/dnn_model_iwp_{nn_v}/train_test_data_nn_{nn_v}.nc")

def flatten_1D_array_manually(ds):
    
    arr = np.zeros([4000])
    arr[0:1000] = ds.isel(time=0).values[:]
    arr[1000:2000] = ds.isel(time=1).values[:]
    arr[2000:3000] = ds.isel(time=2).values[:]
    arr[3000:4000] = ds.isel(time=3).values[:]
    
    return arr

def flatten_2D_array_manually(ds):
    
    arr = np.zeros([4000,150])
    arr[0:1000,:] = ds.isel(time=0).values[:,:]
    arr[1000:2000,:] = ds.isel(time=1).values[:,:]
    arr[2000:3000,:] = ds.isel(time=2).values[:,:]
    arr[3000:4000,:] = ds.isel(time=3).values[:,:]
    
    return arr

def median_frac_error(true,retrieved):
    return np.nanmedian(10**np.absolute(np.log10(retrieved/true)) - 1)

def absolute_error(true,retrieved):
    return retrieved-true

def relative_error(true, retrieved, abs=False):
    
    if abs==False:
        RE = (retrieved-true)/true
    if abs==True:
        RE = np.abs((retrieved-true))/true
    
    return RE

def retrieve_IWP(TBs,altitude):
    
    NN_13000_V1 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_13000m_v1')
    NN_13000_V2 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_13000m_v2')
    NN_13000_V3 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_13000m_v3')
    
    NN_12500_V1 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_12500m_v1')
    NN_12500_V2 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_12500m_v2')
    NN_12500_V3 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_12500m_v3')
    
    NN_12000_V1 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_12000m_v1')
    NN_12000_V2 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_12000m_v2')
    NN_12000_V3 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_12000m_v3')
    
    NN_11500_V1 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_11500m_v1')
    NN_11500_V2 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_11500m_v2')
    NN_11500_V3 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_11500m_v3')
    
    NN_11000_V1 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_11000m_v1')
    NN_11000_V2 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_11000m_v2')
    NN_11000_V3 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_11000m_v3')
    
    NN_10500_V1 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_10500m_v1')
    NN_10500_V2 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_10500m_v2')
    NN_10500_V3 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_10500m_v3')
    
    NN_10000_V1 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_10000m_v1')
    NN_10000_V2 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_10000m_v2')
    NN_10000_V3 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_10000m_v3')
    
    NN_9500_V1 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_9500m_v1')
    NN_9500_V2 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_9500m_v2')
    NN_9500_V3 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_9500m_v3')
    
    NN_9000_V1 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_9000m_v1')
    NN_9000_V2 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_9000m_v2')
    NN_9000_V3 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_9000m_v3')
    
    NN_8500_V1 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_8500m_v1')
    NN_8500_V2 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_8500m_v2')
    NN_8500_V3 = keras.models.load_model(f'/home/u/u301238/master_thesis/nn/dnn_model_iwp_all_levels_v2/dnn_model_iwp_24-32-1_reg_8500m_v3')
    
    NN_levels = np.arange(8500,13500,500)
    levels = np.array([find_nearest_value(NN_levels,altitude[i]) for i in range(len(altitude))])
    
    IWP_V1 = np.zeros(len(TBs))
    IWP_V1[:] = np.nan
    IWP_V2 = np.zeros(len(TBs))
    IWP_V2[:] = np.nan
    IWP_V3 = np.zeros(len(TBs))
    IWP_V3[:] = np.nan
    
    if len(levels[levels==13000]) != 0:
        print(f"Retrieving from 13000m ({len(levels[levels==13000])} TBs)")
        IWP_V1[np.where(levels==13000)] = NN_13000_V1.predict(standardize_nn_input_data_v1(TBs[levels==13000],nn_level=13000,version=1))[:,0]**2
        IWP_V2[np.where(levels==13000)] = NN_13000_V2.predict(standardize_nn_input_data_v1(TBs[levels==13000],nn_level=13000,version=2),verbose=0)[:,0]**2
        IWP_V3[np.where(levels==13000)] = NN_13000_V3.predict(standardize_nn_input_data_v1(TBs[levels==13000],nn_level=13000,version=3),verbose=0)[:,0]**2
    if len(levels[levels==12500]) != 0:
        print(f"Retrieving from 12500m ({len(levels[levels==12500])} TBs)")
        IWP_V1[np.where(levels==12500)] = NN_12500_V1.predict(standardize_nn_input_data_v1(TBs[levels==12500],nn_level=12500,version=1))[:,0]**2
        IWP_V2[np.where(levels==12500)] = NN_12500_V2.predict(standardize_nn_input_data_v1(TBs[levels==12500],nn_level=12500,version=2),verbose=0)[:,0]**2
        IWP_V3[np.where(levels==12500)] = NN_12500_V3.predict(standardize_nn_input_data_v1(TBs[levels==12500],nn_level=12500,version=3),verbose=0)[:,0]**2
    if len(levels[levels==12000]) != 0:
        print(f"Retrieving from 12000m ({len(levels[levels==12000])} TBs)")
        IWP_V1[np.where(levels==12000)] = NN_12000_V1.predict(standardize_nn_input_data_v1(TBs[levels==12000],nn_level=12000,version=1))[:,0]**2
        IWP_V2[np.where(levels==12000)] = NN_12000_V2.predict(standardize_nn_input_data_v1(TBs[levels==12000],nn_level=12000,version=2),verbose=0)[:,0]**2
        IWP_V3[np.where(levels==12000)] = NN_12000_V3.predict(standardize_nn_input_data_v1(TBs[levels==12000],nn_level=12000,version=3),verbose=0)[:,0]**2
    if len(levels[levels==11500]) != 0:
        print(f"Retrieving from 11500m ({len(levels[levels==11500])} TBs)")
        IWP_V1[np.where(levels==11500)] = NN_11500_V1.predict(standardize_nn_input_data_v1(TBs[levels==11500],nn_level=11500,version=1))[:,0]**2
        IWP_V2[np.where(levels==11500)] = NN_11500_V2.predict(standardize_nn_input_data_v1(TBs[levels==11500],nn_level=11500,version=2),verbose=0)[:,0]**2
        IWP_V3[np.where(levels==11500)] = NN_11500_V3.predict(standardize_nn_input_data_v1(TBs[levels==11500],nn_level=11500,version=3),verbose=0)[:,0]**2
    if len(levels[levels==11000]) != 0:
        print(f"Retrieving from 11000m ({len(levels[levels==11000])} TBs)")
        IWP_V1[np.where(levels==11000)] = NN_11000_V1.predict(standardize_nn_input_data_v1(TBs[levels==11000],nn_level=11000,version=1))[:,0]**2
        IWP_V2[np.where(levels==11000)] = NN_11000_V2.predict(standardize_nn_input_data_v1(TBs[levels==11000],nn_level=11000,version=2),verbose=0)[:,0]**2
        IWP_V3[np.where(levels==11000)] = NN_11000_V3.predict(standardize_nn_input_data_v1(TBs[levels==11000],nn_level=11000,version=3),verbose=0)[:,0]**2
    if len(levels[levels==10500]) != 0:
        print(f"Retrieving from 10500m ({len(levels[levels==10500])} TBs)")
        IWP_V1[np.where(levels==10500)] = NN_10500_V1.predict(standardize_nn_input_data_v1(TBs[levels==10500],nn_level=10500,version=1))[:,0]**2
        IWP_V2[np.where(levels==10500)] = NN_10500_V2.predict(standardize_nn_input_data_v1(TBs[levels==10500],nn_level=10500,version=2),verbose=0)[:,0]**2
        IWP_V3[np.where(levels==10500)] = NN_10500_V3.predict(standardize_nn_input_data_v1(TBs[levels==10500],nn_level=10500,version=3),verbose=0)[:,0]**2
    if len(levels[levels==10000]) != 0:
        print(f"Retrieving from 10000m ({len(levels[levels==10000])} TBs)")
        IWP_V1[np.where(levels==10000)] = NN_10000_V1.predict(standardize_nn_input_data_v1(TBs[levels==10000],nn_level=10000,version=1))[:,0]**2
        IWP_V2[np.where(levels==10000)] = NN_10000_V2.predict(standardize_nn_input_data_v1(TBs[levels==10000],nn_level=10000,version=2),verbose=0)[:,0]**2
        IWP_V3[np.where(levels==10000)] = NN_10000_V3.predict(standardize_nn_input_data_v1(TBs[levels==10000],nn_level=10000,version=3),verbose=0)[:,0]**2
    if len(levels[levels==9500]) != 0:
        print(f"Retrieving from 9500m ({len(levels[levels==9500])} TBs)")
        IWP_V1[np.where(levels==9500)] = NN_9500_V1.predict(standardize_nn_input_data_v1(TBs[levels==9500],nn_level=9500,version=1))[:,0]**2    
        IWP_V2[np.where(levels==9500)] = NN_9500_V2.predict(standardize_nn_input_data_v1(TBs[levels==9500],nn_level=9500,version=2),verbose=0)[:,0]**2    
        IWP_V3[np.where(levels==9500)] = NN_9500_V3.predict(standardize_nn_input_data_v1(TBs[levels==9500],nn_level=9500,version=3),verbose=0)[:,0]**2    
    if len(levels[levels==9000]) != 0:
        print(f"Retrieving from 9000m ({len(levels[levels==9000])} TBs)")
        IWP_V1[np.where(levels==9000)] = NN_9000_V1.predict(standardize_nn_input_data_v1(TBs[levels==9000],nn_level=9000,version=1))[:,0]**2
        IWP_V2[np.where(levels==9000)] = NN_9000_V2.predict(standardize_nn_input_data_v1(TBs[levels==9000],nn_level=9000,version=2),verbose=0)[:,0]**2
        IWP_V3[np.where(levels==9000)] = NN_9000_V3.predict(standardize_nn_input_data_v1(TBs[levels==9000],nn_level=9000,version=3),verbose=0)[:,0]**2
    if len(levels[levels==8500]) != 0:
        print(f"Retrieving from 8500m ({len(levels[levels==8500])} TBs)")
        IWP_V1[np.where(levels==8500)] = NN_8500_V1.predict(standardize_nn_input_data_v1(TBs[levels==8500],nn_level=8500,version=1))[:,0]**2
        IWP_V2[np.where(levels==8500)] = NN_8500_V2.predict(standardize_nn_input_data_v1(TBs[levels==8500],nn_level=8500,version=2),verbose=0)[:,0]**2
        IWP_V3[np.where(levels==8500)] = NN_8500_V3.predict(standardize_nn_input_data_v1(TBs[levels==8500],nn_level=8500,version=3),verbose=0)[:,0]**2
    
    
    IWP = np.nanmean(np.array([IWP_V1,IWP_V2,IWP_V3]),axis=0)
    
    print("")
    print((len(IWP[IWP<0])/len(IWP)),"% negative predictions (=clipped to 0).")
    print("")
    IWP[IWP<0] = 0.

    return IWP, levels
    